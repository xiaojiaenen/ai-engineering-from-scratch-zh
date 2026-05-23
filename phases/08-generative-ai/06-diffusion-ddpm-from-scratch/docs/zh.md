# 从零开始构建扩散模型 — DDPM

> Ho、Jain、Abbeel（2020年）为该领域提供了一套难以舍弃的范式：通过一千个微小步骤，用噪声逐步破坏数据；训练一个神经网络来预测噪声；在推理时逆转该过程。如今，所有主流的图像、视频、3D 和音乐模型都运行于这一循环之上，可能还结合了流匹配或一致性技巧。

**类型:** 构建
**语言:** Python
**先决条件:** 阶段 3 · 02 (反向传播), 阶段 8 · 02 (VAE)
**时间:** 约 75 分钟

## 问题所在

你想要一个 `p_data(x)` 的采样器。GAN 进行着一个常常发散的极小极大博弈。VAE 通过高斯解码器生成模糊的样本。你真正需要的是一个训练目标，它应满足：(a) 是一个单一的稳定损失（无鞍点，无极小极大），(b) 是 `log p(x)` 的下界（这样你就有似然估计），(c) 样本质量达到当前最佳水平。

Sohl-Dickstein 等人（2015年）给出了一个理论答案：定义一个逐步添加高斯噪声的马尔可夫链 `q(x_t | x_{t-1})`，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020年）表明损失可以简化为一行——预测噪声——并清理了数学推导。在2020年，这只是一个新奇想法。在2021年，它产生了最先进的样本。在2022年，它成为了 Stable Diffusion。在2026年，它已成为基础技术。

## 核心概念

![DDPM：前向加噪，反向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步骤中添加高斯噪声。其闭合形式（数学可处理性的原因）是累积步骤也是高斯的：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)` 用于一个 `β_t` 的调度。从 1e-4 到 0.02 线性选择 `β_t`，共 T=1000 步，且 `x_T` 约等于 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 可以是 `sqrt(β_t)` 或一个学习到的方差。这个表达式看起来复杂，但它只是代数运算——根据后验 `x_{t-1}` 求解 `q(x_{t-1} | x_t, x_0)`，并用其噪声预测估计值 `x_0` 进行代换。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选择一个 `t`，采样 `ε ~ N(0, I)`，通过闭合形式一次性计算带噪的 `x_t`，然后对噪声进行回归。一个损失，无极小极大，无KL散度，无重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步骤。完成。

## 为何有效

三个直觉：

1.  **去噪容易；生成困难。** 在 `t=T` 时，数据是纯噪声——网络需要解决一个简单问题。在 `t=0` 时，网络只需清理少量像素。在中间 `t` 时，问题很困难，但网络从每个噪声等级流经相同权重的梯度很多。

2.  **伪装的分数匹配。** Vincent（2011年）证明，预测噪声等价于估计 *分数* `∇_x log q(x_t | x_0)`。反向SDE使用这个分数沿密度梯度行走——一种向高概率区域的有导向随机游走。

3.  **ELBO简化为简单的MSE。** 完整的变分下界在每个时间步有一个KL项。在DDPM的参数化下，这些KL项简化为带有特定系数的噪声预测MSE；Ho去掉了系数（称之为"简单"损失），质量*反而*提高了。

## 动手构建

`code/main.py` 实现了一个一维DDPM。数据是双峰混合分布。"网络"是一个小型MLP，接受 `(x_t, t)` 并输出预测的噪声。训练使用单行损失。采样迭代反向链。

### 第1步：前向调度（闭合形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 第2步：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 第3步：一个训练步骤

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 第4步：反向采样

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

对于一个具有40个时间步和24单元MLP的一维问题，这可以在约200个周期内学习到双峰混合分布。

## 时间条件化

网络需要知道它正在去噪的是哪个时间步。两种标准选项：

*   **正弦嵌入。** 类似于Transformer的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过一个MLP，广播到网络中。
*   **FiLM / 组归一化条件化。** 在每个块中将嵌入投影为每通道的缩放/偏置（FiLM）。

我们的示例代码使用正弦嵌入→拼接。生产级的U-Net使用FiLM。

## 注意事项

*   **调度至关重要。** 线性 `β` 是DDPM的默认设置，但余弦调度（Nichol & Dhariwal, 2021）在相同计算量下提供更好的FID分数。如果质量停滞，请更换调度。
*   **时间步嵌入很脆弱。** 将原始 `t` 作为浮点数传递对玩具一维问题有效，但对图像会失败；始终使用合适的嵌入。
*   **v预测 vs ε预测。** 对于狭窄区间（非常小或非常大的t），`ε` 信噪比差。v预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3和Flux使用它。
*   **无分类器指导。** 在推理时，同时计算条件 `ε` 和无条件 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，然后用 `w ≈ 3-7` 进行混合。在课程08中涵盖。
*   **1000步太多了。** 生产环境使用DDIM（20-50步）、DPM-Solver（10-20步）或蒸馏（1-4步）。参见课程12。

## 应用场景

| 角色 | 2026年的典型技术栈 |
|------|-------------------|
| 图像像素空间扩散（小型，玩具） | DDPM + U-Net |
| 图像潜在空间扩散 | VAE编码器 + U-Net 或 DiT (课程07) |
| 视频潜在空间扩散 | 时空DiT (Sora, Veo, WAN) |
| 音频潜在空间扩散 | Encodec + 扩散Transformer |
| 科学（分子、蛋白质、物理） | 等变扩散 (EDM, RFdiffusion, AlphaFold3) |

扩散是通用的生成基础。流匹配（课程13）是2024-2026年的竞争对手，在相同质量下通常在推理速度上更胜一筹。

## 部署上线

保存 `outputs/skill-diffusion-trainer.md`。该技能接受一个数据集+计算预算，并输出：调度（线性/余弦/sigmoid）、预测目标（ε/v/x）、步数、指导尺度、采样器系列和评估协议。

## 练习

1.  **简单。** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的可视化直方图）如何下降？在 T 为多少时双峰结构会崩溃？
2.  **中等。** 从 ε预测 切换到 v预测。重新推导反向步骤。比较最终的样本质量。
3.  **困难。** 添加无分类器指导。以类别标签 `c ∈ {0, 1}` 为条件，在训练期间有10%的时间丢弃它，在采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 处测量条件模式命中率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 前向过程 | "添加噪声" | 固定的马尔可夫链 `q(x_t | x_{t-1})`，用于破坏数据。 |
| 反向过程 | "去噪" | 学习的链 `p_θ(x_{t-1} | x_t)`，用于重建数据。 |
| β 调度 | "噪声阶梯" | 每步的方差；线性、余弦或sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；提供了从 `x_0` 到 `x_t` 的闭合形式。 |
| 简单损失 | "对噪声的MSE" | `||ε - ε_θ(x_t, t)||²`；所有变分推导都归结为此。 |
| ε预测 | "预测噪声" | 输出是添加的噪声；标准DDPM。 |
| V预测 | "预测速度" | 输出是 `α·ε - σ·x`；在不同t值下条件更好。 |
| DDPM | "那篇论文" | Ho等人 2020；线性β，1000步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50步，相同训练目标。 |
| 无分类器指导 | "CFG" | 混合条件与无条件噪声预测以放大条件性。 |

## 生产注意事项：扩散推理是一个步数问题

DDPM论文运行 T=1000 个反向步骤。生产中没人会这样部署。每个实际的推理栈都从以下三种策略中选择一种——每一种都清晰地映射到关于"延迟来自哪里"的生产框架：

1.  **更快的采样器，相同的模型。** DDIM（20-50步），DPM-Solver++（10-20），UniPC（8-16）。反向循环的直接替换；训练好的 `ε_θ` 权重不变。延迟降低20-50倍。
2.  **蒸馏。** 训练一个学生网络，用更少的步骤匹配教师：渐进式蒸馏（2→1），一致性模型（任意→1-4），LCM，SDXL-Turbo，SD3-Turbo。延迟再降低5-10倍，需要重新训练。
3.  **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`，TensorRT-LLM的扩散后端，`xformers`/SDPA注意力，bf16权重。单步延迟降低约2倍。可与(1)和(2)叠加。

对于一个生产扩散服务器，预算讨论与生产文献中描述的LLM是一样的：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。首词元时间（TTFT）很小（一步）；等效的每词元时间（TPOT）是整个响应时间，因为从用户角度看，图像是"一次性"生成的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散模型论文，具有前瞻性。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少的步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器指导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一的符号体系，最清晰的配方。