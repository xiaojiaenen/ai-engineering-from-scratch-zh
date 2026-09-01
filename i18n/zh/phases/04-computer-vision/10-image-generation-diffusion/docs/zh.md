# 图像生成 — 扩散模型

> 扩散模型学习的是去噪。训练它从含噪图像中移除一小部分噪声，重复该反向过程一千次，你就有了一个图像生成器。

**类型：** 构建
**语言：** Python
**前置知识：** 第4阶段第07课（U-Net）、第1阶段第06课（概率）、第3阶段第06课（优化器）
**时间：** ~75 分钟

## 学习目标

- 推导前向加噪过程 `x_0 -> x_1 -> ... -> x_T`，并解释为何对任意 `t` 都存在闭式 `q(x_t | x_0)`
- 实现 DDPM 风格的训练目标（对每一步加入的噪声进行回归），以及一个从纯噪声逐步回到图像的采样器
- 构建时间条件 U-Net（足够小可在 CPU 上训练），使其能预测任意时间步的噪声
- 解释 DDPM 与 DDIM 采样的区别，以及各自的适用场景（第23课将深入讲解 flow matching 与 rectified flow）

## 问题所在

GAN 是一次生成：噪声输入，图像输出，单次前向传播。它们速度快但训练困难。扩散模型是迭代生成：从纯噪声开始，逐步去噪，图像逐渐浮现。它们速度慢但训练容易。过去五年，后一种特性占据了主导：任何小型团队都能训练一个扩散模型并得到合理的样本；而 GAN 训练是一门需要多年失败经验才能掌握的手艺。

除了训练稳定性之外，扩散模型的迭代结构也是解锁现代图像生成所有能力的关键：文本条件控制、图像修复、图像编辑、超分辨率、可控风格。采样循环中的每一步都是注入新约束的地方。这正是 Stable Diffusion、Imagen、DALL-E 3、Midjourney，以及你未来会使用的所有可控图像模型都基于扩散的原因。

本课构建最简 DDPM：前向加噪、反向去噪、训练循环。下一课（Stable Diffusion）将把它接入生产系统，搭配 VAE、文本编码器和无分类器引导。

## 核心概念

### 前向过程

取一张图像 `x_0`。加入少量高斯噪声得到 `x_1`。再加入少量噪声得到 `x_2`。重复 T 步，直到 `x_T` 几乎与纯高斯噪声无法区分。

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
```

`beta_t` 是一个小方差调度，通常在 T=1000 步内从 0.0001 线性增长到 0.02。每一步略微削弱信号并注入新噪声。

### 闭式跳跃

逐步加噪是一个马尔可夫链，但数学可以折叠：你可以一步直接从 `x_0` 采样 `x_t`。

```
定义 alpha_t = 1 - beta_t
定义 alpha_bar_t = prod_{s=1..t} alpha_s

则：
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

等价地：
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  其中 epsilon ~ N(0, I)
```

这个单一方程是整个扩散模型可行的根本原因。训练时你随机选取一个 `t`，直接从 `x_0` 采样 `x_t`，一步完成训练——无需模拟完整马尔可夫链。

### 反向过程

前向过程是固定的。反向过程 `p(x_{t-1} | x_t)` 才是神经网络所学习的内容。扩散模型不直接预测 `x_{t-1}`，而是预测第 t 步加入的噪声 `epsilon`，然后由数学推导出 `x_{t-1}`。

```mermaid
flowchart LR
    X0["x_0<br/>(干净图像)"] --> Q1["q(x_t|x_0)<br/>加噪"]
    Q1 --> XT["x_t<br/>(含噪)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["预测的 epsilon"]
    EPS --> LOSS["与<br/>真实 epsilon 的 MSE"]

    XT -.->|采样| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|重复1000次| X0S["x_0 (采样结果)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### 训练损失

每个训练步：

1. 采样真实图像 `x_0`。
2. 在 [1, T] 内均匀采样时间步 `t`。
3. 采样噪声 `epsilon ~ N(0, I)`。
4. 计算 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`。
5. 用网络预测 `epsilon_theta(x_t, t)`。
6. 最小化 `|| epsilon - epsilon_theta(x_t, t) ||^2`。

就这些。神经网络学习预测任意时间步的噪声。损失是 MSE。没有对抗博弈，没有崩溃，没有振荡。

### 采样器（DDPM）

生成时：从 `x_T ~ N(0, I)` 开始，逐步反向走每一步。

```
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    其中 z ~ N(0, I) 若 t > 1，否则 z = 0
return x_0
```

关键在于，尽管一般情形下反向条件分布没有闭式，但对于这种特定的高斯前向过程，它是存在的。那些看似复杂的系数就是贝叶斯法则给出的结果。

### 为什么需要 1000 步

前向噪声调度被设计为每一步只加入刚好足够的噪声，使反向步近似高斯。步数太少，反向步偏离高斯太远，网络难以建模。步数太多，采样昂贵而收益递减。T=1000 配合线性调度是 DDPM 的默认设置。

### DDIM：20 倍更快的采样

训练完全相同。采样改变。DDIM（Song 等，2020）定义了一个确定性反向过程，可在不重新训练的情况下跳过时间步。使用 DDIM 在 50 步内采样可得到接近 1000 步 DDPM 的质量。所有生产系统都使用 DDIM 或更快的变体（DPM-Solver、Euler ancestral）。

### 时间条件

网络 `epsilon_theta(x_t, t)` 需要知道当前是哪一个时间步在去噪。现代扩散模型通过正弦时间嵌入（与 Transformer 中的位置编码思路相同）将 `t` 注入到 U-Net 各层特征图中。

```
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

若没有时间条件，网络必须从图像本身猜测噪声水平，效果可以但样本效率要低得多。

```figure
cv-diffusion-image
```

## 动手构建

### 步骤 1：噪声调度

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

一次性预计算，训练和采样时按索引取值。

### 步骤 2：前向扩散（q_sample）

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

一行闭式。`t` 是一个批次的时间步，批次中每个图像对应一个。

### 步骤 3：微小的时间条件 U-Net

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

两层 U-Net，时间条件注入瓶颈处。处理真实图像时扩展深度和宽度即可。

### 步骤 4：训练循环

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

这就是完整的训练循环。没有 GAN 博弈，没有特殊损失，只有一行 MSE 调用。

### 步骤 5：采样器（DDPM）

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

1000 次前向传播产生一批样本。实际代码中应替换为 DDIM 50 步采样器。

### 步骤 6：DDIM 采样器（确定性，约 20 倍更快）

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0` 完全确定性（相同噪声输入始终产生相同输出）。`eta=1` 还原为 DDPM。

## 使用它

生产环境请使用 `diffusers`：

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

该库提供了现成的调度器（DDPM、DDIM、DPM-Solver、Euler、Heun）、可配置的 U-Net、文本到图像和图像到图像的 pipeline，以及 LoRA 微调辅助工具。

研究用途推荐使用 `k-diffusion`（Katherine Crowson），它拥有最忠实于原论文的参考实现和最好的采样变体。

## 交付成果

本课产出：

- `outputs/prompt-diffusion-sampler-picker.md` — 一个根据质量目标、延迟预算和条件类型选择 DDPM / DDIM / DPM-Solver / Euler 的提示模板。
- `outputs/skill-noise-schedule-designer.md` — 一个技能，给定 T 和目标噪声水平，生成线性、余弦或 Sigmoid beta 调度，并提供信噪比随时间变化的诊断图。

## 练习

1. **(简单)** 可视化前向过程：取一张图像，绘制 `t in [0, 100, 250, 500, 750, 1000]` 时的 `x_t`。验证 `x_1000` 看起来像纯高斯噪声。
2. **(中等)** 在合成圆形数据集上训练 TinyUNet 20 个 epoch，采样 16 个圆形。比较 DDPM（1000 步）和 DDIM（50 步）采样——使用相同噪声种子时，它们是否产生相似的图像？
3. **(困难)** 实现余弦噪声调度（Nichol & Dhariwal, 2021）：`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`。在同一模型上分别用线性和余弦调度训练，展示余弦调度在低步数时产生更好的样本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 前向过程 | "随时间加噪" | 固定的马尔可夫链，在 T 步内将图像腐蚀为高斯噪声 |
| 反向过程 | "逐步去噪" | 学习到的分布，从噪声逐步走回图像 |
| 噪声预测 | "预测噪声" | 训练目标：`epsilon_theta(x_t, t)` 预测第 t 步加入的噪声 |
| Beta 调度 | "噪声量" | T 个小的方差序列，定义每一步注入多少噪声 |
| alpha_bar_t | "累积保留因子" | 从 1 到 t 的 (1 - beta_s) 的乘积；t 越大，信号剩余越少 |
| DDPM 采样器 | "祖先型、随机" | 从条件高斯中采样每个 x_{t-1}；1000 步 |
| DDIM 采样器 | "确定性、快速" | 将采样重写为确定性 ODE；20-100 步质量相近 |
| 时间条件 | "告诉模型 t 是哪个" | t 的正弦嵌入注入 U-Net，使模型知道噪声水平 |

## 延伸阅读

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — 让扩散模型变得实用并在 FID 上击败 GAN 的论文
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) — 余弦调度和 v-参数化
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) — 使实时推理成为可能的确定性采样器
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) — 对所有扩散设计选择的统一视图；当前最佳参考
