# 流匹配与整流流

> 扩散模型需要20-50个采样步骤，因为它们从噪声到数据走的是一条弯曲路径。流匹配（Lipman 等人，2023）和整流流（Liu 等人，2022）训练的是直线路径。路径越直，所需步骤越少，推理速度越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 在2024年都转向了流匹配。

**类型：** 构建  
**语言：** Python  
**前提知识：** 阶段 8 · 06 (DDPM)，阶段 1 · 微积分  
**时间：** 约45分钟

## 问题所在

DDPM的反向过程是一个从 `N(0, I)` 回到数据分布的1000步随机游走。DDIM将其压缩到20-50个确定性步骤。你需要更少的步骤——理想情况下是一步。阻碍在于求解反向过程的ODE是刚性的；路径是弯曲的。

如果你能训练模型，使得从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的单个欧拉步骤就能奏效。流匹配直接实现了这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练一个向量场 `v_θ(x, t)` 来匹配其时间导数，在推理时进行积分。

整流流（Liu 2022）更进一步：通过一个重流程序迭代地拉直路径，该程序产生一个逐渐更接近线性的ODE。经过两次重流迭代，2步采样器的质量就能匹配50步DDPM。

## 核心概念

![流匹配：噪声与数据之间的直线插值](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 和 `x_1 ~ N(0, I)`。沿此直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)` 并训练它以匹配该导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配**损失（Lipman 2023）。训练是无模拟的：你永远不需要展开ODE。只需采样 `(x_0, x_1, t)` 并进行回归。

### 采样

在推理时，将学到的向量场沿时间*反向*积分：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，欧拉步进至 `t=0`。

### 整流流（Liu 2022）

直线流有效，但学到的路径*实际上并不笔直*——它们会弯曲，因为许多 `x_0` 可以映射到同一个 `x_1`。整流流的重流步骤如下：

1. 使用随机配对训练流模型 v_1。
2. 通过将 v_1 从 `(x_1, x_0)` 积分到其终点 `x_1` 来采样N对 `x_0`。
3. 在这些配对样本上训练 v_2。由于这些配对现在是“ODE匹配”的，它们之间的直线插值实际上更加平坦。
4. 重复上述步骤。

在实践中，2次重流迭代即可使其接近线性，从而实现2-4步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配模型蒸馏而来的。

### 为什么在2024年它在图像领域胜出

三个原因：

1. **无模拟训练** —— 训练过程中无需展开ODE，实现简单。
2. **更好的损失几何结构** —— 直线路径具有一致的信噪比，而DDPM的ε损失在调度计划的边界处信噪比很差。
3. **更快的推理** —— SDXL-Turbo质量下仅需4-8步；通过一致性蒸馏可达到1步。

## 流匹配与 DDPM 的精确联系

使用高斯条件路径的流匹配就是*具有特定噪声调度的扩散*。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度计划，流匹配就能恢复使用 `v = α'·x_0 - σ'·x_1` 进行Stratonovich重述的扩散。对于高斯路径，两者在代数上是等价的。

流匹配的贡献在于：目标（一个简单的速度场）的*清晰性*、一个更干净的损失函数，以及允许尝试非高斯插值器的自由。

## 动手实现

`code/main.py` 在一个双模高斯混合上实现了1维流匹配。向量场 `v_θ(x, t)` 是一个用直线目标训练的小型MLP。在推理时，分别进行1、2、4和20步欧拉积分，并比较样本质量。

### 步骤 1：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 步骤 2：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 步骤 3：比较不同步数

预期4步采样器的质量将已经匹配20步的质量——这对延迟来说意义重大。

## 陷阱

- **时间参数化。** 流匹配使用 `t ∈ [0, 1]`，其中 `t=0` 对应数据，`t=1` 对应噪声。DDPM使用 `t ∈ [0, T]`，其中 `t=0` 对应数据，`t=T` 对应噪声。方向相同，尺度不同。论文经常搞错这一点。
- **调度计划选择。** 整流流的直线是“标准”的流匹配调度计划，但你可以使用余弦或logit-normal t采样（SD3这样做的）来获得更好的尺度覆盖。
- **重流成本。** 为重流生成配对数据集，每个样本都需要一次完整的推理过程。只有在你确实需要1-2步推理时才进行重流。
- **无分类器指导仍然适用。** 只需在线性组合中将ε替换为v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 应用场景

| 用例 | 2026年技术栈 |
|------|--------------|
| 文本到图像，最佳质量 | 流匹配：SD3, Flux.1-dev |
| 文本到图像，1-4步 | 蒸馏流匹配：Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 实时推理 | 基于流匹配基础模型的一致性蒸馏（LCM, PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5, AudioCraft 2 |
| 视频生成 | 流匹配与扩散混合（Sora, Veo, Stable Video） |
| 科学/物理（粒子轨迹、分子） | 流匹配 + 等变向量场 |

当一篇论文在2025-2026年说“比扩散更快”时，它几乎总是指流匹配 + 蒸馏。

## 交付

保存 `outputs/skill-fm-tuner.md`。该技能接受一个扩散风格的模型规格，并将其转换为流匹配训练配置：调度计划选择、时间采样分布（均匀/logit-normal）、优化器、重流计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较1步与20步的MSE与真实数据分布。
2. **中等。** 从均匀的 `t` 采样切换到logit-normal（集中在中间t处采样）。模型质量是否提高？
3. **困难。** 实现一次重流迭代：通过对第一个模型积分来生成配对的 (x_0, x_1)，在配对数据上训练第二个模型，并比较1步样本质量。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|----------|------------|
| 流匹配 (Flow matching) | “直线扩散” | 训练 `v_θ(x, t)` 以匹配沿插值路径的 `x_1 - x_0`。 |
| 整流流 (Rectified flow) | “重流 (Reflow)” | 一种迭代过程，用于拉直学到的流。 |
| 速度场 (Velocity field) | “v_θ” | 模型的输出——移动 `x_t` 的方向。 |
| 直线插值器 (Straight-line interpolant) | “路径” | `x_t = (1-t)·x_0 + t·x_1`；平凡的目标导数。 |
| 欧拉采样器 (Euler sampler) | “1阶ODE求解器” | 最简单的积分器；当路径笔直时效果良好。 |
| Logit-normal t采样 | “SD3采样” | 将 `t` 采样集中在梯度最强的中间值附近。 |
| 一致性蒸馏 (Consistency distillation) | “1步采样器” | 训练一个学生模型，直接将任何 `x_t` 映射到 `x_0`。 |
| 使用速度的CFG (CFG with velocity) | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同样的技巧，新的变量。 |

## 生产说明：Flux.1-schnell 是流匹配的最快形态

流匹配在生产中的胜利是 Flux.1-schnell —— 一个流匹配的DiT模型，被蒸馏到1-4个推理步骤，同时保持了Flux-dev级别的质量。Niels的“在8GB显存机器上运行Flux”笔记本是参考部署方案：T5 + CLIP编码，量化MMDiT去噪（schnell用4步，dev用50步），VAE解码。成本核算如下：

| 变体 | 步骤 | L4上1024²延迟 | 总FLOPs（相对） |
|------|------|--------------|----------------|
| Flux.1-dev (原版) | 50 | ~15 秒 | 1.0× |
| Flux.1-schnell | 4 | ~1.2 秒 | 0.08× (快12倍) |
| SDXL-base | 30 | ~4 秒 | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 秒 | 0.03× |

生产规则：**流匹配基础模型 + 蒸馏 = 2026年快速文本到图像的默认方案。** 每个主要供应商都提供这种组合：SD3-Turbo (SD3 + 流匹配 + 蒸馏), Flux-schnell (Flux-dev + 整流流拉直), CogView-4-Flash。纯扩散基础模型仅存在于旧版检查点中。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) —— 整流流。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) —— 流匹配。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3，大规模整流流。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) —— 涵盖流匹配与扩散的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) —— 扩散/流匹配的1步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) —— turbo变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) —— 生产中的流匹配。