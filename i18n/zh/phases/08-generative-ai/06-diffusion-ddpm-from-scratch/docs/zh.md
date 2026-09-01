# 扩散模型 — 从零实现 DDPM

> Ho、Jain、Abbeel（2020）为这个领域提供了一道令人无法抗拒的配方：用上千个微小步骤将数据逐步加入噪声，训练一个神经网络来预测噪声，然后在推理时逆转这个过程。如今，每一个主流的图像、视频、3D 和音频模型都运行在这个循环之上，可能还会叠加 flow matching 或一致性技巧。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播）、Phase 8 · 02（VAE）
**时间：** 约 75 分钟

## 问题

你需要一个来自 `p_data(x)` 的采样器。GAN 的极小极大博弈常常发散，VAE 的 Gaussian 解码器会产出模糊的样本。你真正想要的是一个训练目标，它具备 (a) 一个单一稳定的损失（没有鞍点、没有极小极大博弈）、(b) `log p(x)` 的下界（这样你就能得到似然），以及 (c) 达到 SOTA 质量的样本。

Sohl-Dickstein 等人（2015）给出了一个理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})` 逐步加入高斯噪声，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 来进行去噪。Ho、Jain、Abbeel（2020）证明这个损失可以简化为一行代码——预测噪声——并且推导更加干净。2020 年这还只是一个小众研究，2021 年它产出了 SOTA 样本，2022 年它变成了 Stable Diffusion，到 2026 年它已成为基础架构。

## 核心概念

![DDPM：正向加噪，反向去噪](../assets/ddpm.svg)

**正向过程 `q`。** 在 `T` 个微小步骤中加入高斯噪声。解析解——让数学变得可处理的原因——在于累积步也仍是高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 是一个调度序列。将 `β_t` 从 1e-4 线性变化到 0.02，共 T=1000 步，则 `x_T` 近似服从 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测被加入的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 可以是 `sqrt(β_t)` 或一个学习到的方差。这个表达式看起来很丑，但它只是代数运算——根据后验 `q(x_{t-1} | x_t, x_0)` 解出 `x_{t-1}`，并用噪声预测值替换 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据采样 `x_0`，随机选取 `t`，采样 `ε ~ N(0, I)`，通过解析式一次性计算出加噪后的 `x_t`，然后对噪声进行回归。一个损失，没有极小极大博弈，没有 KL 散度，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。将反向步骤从 `t = T` 迭代到 `1`。完成。

## 为什么有效

三个直觉：

1. **去噪很容易；生成很难。** 在 `t=T` 时，数据是纯噪声——网络只需解决一个平凡问题。在 `t=0` 时，网络只需清理几个像素。在中间 `t`，问题更难，但网络可以通过所有噪声水平共享的同一组权重获得大量梯度。

2. **伪装后的分数匹配。** Vincent（2011）证明了预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即 *score*。反向 SDE 使用这个 score 沿着密度梯度行走——一条向高概率区域移动的引导随机游走。

3. **ELBO 简化为简单 MSE。** 完整的变分下界在每个时间步都有一个 KL 项。使用 DDPM 的参数化，这些 KL 项简化为带特定系数的噪声预测 MSE；Ho 丢弃了系数（称之为"简单"损失），质量反而 *提升* 了。

```figure
diffusion-denoise
```

## 构建

`code/main.py` 实现了一个一维 DDPM。数据是一个双模式混合分布。"网络"是一个小型 MLP，接收 `(x_t, t)` 并输出预测噪声。训练使用单行损失，采样则迭代反向链。

### 步骤 1：正向调度（解析形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤 2：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤 3：一个训练步

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤 4：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个 40 步、使用 24 单元 MLP 的一维问题，这大约需要 200 个 epoch 就能学习到双模式混合分布。

## 时间条件

网络需要知道它正在去噪的时间步。两种标准选项：

- **正弦嵌入。** 类似 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP，广播到网络中。
- **FiLM / 组归一化条件。** 在每个模块处将嵌入投影为逐通道缩放/偏置（FiLM）。

我们的玩具代码使用正弦嵌入 → 拼接。生产环境的 U-Net 使用 FiLM。

## 陷阱

- **调度非常重要。** 线性 `β` 是 DDPM 的默认选择，但余弦调度（Nichol & Dhariwal, 2021）在相同计算量下能给出更好的 FID。如果质量出现瓶颈，切换调度方案。
- **时间步嵌入很脆弱。** 将原始 `t` 作为浮点数传入在玩具一维问题中可以工作，但在图像上会失败；务必使用合适的嵌入。
- **ε 预测 vs V 预测。** 对于窄范围（非常小或非常大的 t），`ε` 的信噪比很差。V 预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **无条件引导。** 在推理时，同时计算条件和非条件 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。详见第 08 课。
- **1000 步太多了。** 生产环境使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。详见第 12 课。

## 用途

| 角色 | 2026 年典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小型、玩具） | DDPM + U-Net |
| 图像潜空间扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜空间扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜空间扩散 | Encodec + 扩散 Transformer |
| 科学领域（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散模型是通用的生成式基础架构。Flow matching（第 13 课）是 2024-2026 年的竞争者，通常在相同质量下赢得推理速度。

## 交付

保存 `outputs/skill-diffusion-trainer.md`。技能输入数据集 + 计算预算，输出：调度（线性/余弦/sigmoid）、预测目标（ε/v/x）、步数、引导强度、采样器族以及评估协议。

## 练习

1. **简单。** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的可视化直方图）会如何下降？T 降到多少时双模式结构会崩溃？
2. **中等。** 从 ε 预测切换到 v 预测。重新推导反向步骤。比较最终样本质量。
3. **困难。** 添加无条件引导。对类标签 `c ∈ {0, 1}` 进行条件训练，训练时 10% 的概率丢弃标签，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量 `w = 0, 1, 3, 7` 时的条件模式命中率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------------|-----------------------|
| 正向过程 | "加噪" | 固定的马尔可夫链 `q(x_t \| x_{t-1})`，用于破坏数据。 |
| 反向过程 | "去噪" | 学习的链 `p_θ(x_{t-1} \| x_t)`，用于重建数据。 |
| β 调度 | "噪声阶梯" | 每步的方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；由 `x_0` 给出 `x_t` 的解析解。 |
| 简单损失 | "噪声上的 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都坍缩为此形式。 |
| ε 预测 | "预测噪声" | 输出是被加入的噪声；标准 DDPM。 |
| V 预测 | "预测速度" | 输出为 `α·ε - σ·x`；在各时间步有更好的条件性。 |
| DDPM | "那篇论文" | Ho 等人 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同的训练目标。 |
| Classifier-free 引导 | "CFG" | 混合条件和非条件噪声预测，以放大条件性。 |

## 生产备注：扩散推理是一个步数问题

DDPM 论文运行 T=1000 个反向步。没有人会在生产中这样部署。每个真正的推理栈都会选择三种策略之一——每种都清晰地映射到生产环境中"延迟来自哪里"的框架：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。直接替换反向循环；训练的 `ε_θ` 权重保持不变。延迟降低 20-50 倍。
2. **蒸馏。** 训练学生网络以更少的步数匹配教师：Progressive Distillation（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。延迟再降低 5-10 倍，需要重新训练。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟降低约 2 倍。可与 (1) 和 (2) 叠加。

对于生产扩散服务器，预算讨论与生产环境中 LLM 的描述相同：延迟为 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等价物是整个响应时间，因为图像生成对用户而言是"一次性"的。

## 延伸阅读

- [Sohl-Dickstein 等人 (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) —— 扩散论文，领先其时代。
- [Ho、Jain、Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM。
- [Song、Meng、Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) —— DDIM，更少步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) —— 余弦调度，学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) —— 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Karras 等人 (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) —— 统一符号，最干净的配方。
