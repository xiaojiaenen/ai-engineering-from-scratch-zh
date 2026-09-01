# Flow Matching 与 Rectified Flows

> Diffusion models take 20-50 sampling steps because they walk a curved path from noise to data. Flow matching (Lipman et al., 2023) and rectified flow (Liu et al., 2022) trained straight paths. Straighter paths mean fewer steps mean faster inference. Stable Diffusion 3, Flux.1, and AudioCraft 2 all switched to flow matching in 2024.

> 扩散模型需要 20-50 个采样步数，因为它们沿着从噪声到数据的弯曲路径行走。Flow matching（Lipman 等人，2023）和 rectified flow（Liu 等人，2022）训练的是直线路径。路径越直，步数越少，推理越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了 flow matching。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 06（DDPM），Phase 1 · 微积分
**时间：** 约 45 分钟

## 问题所在

DDPM 的反向过程是一个从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其压缩为 20-50 步确定性步数。你想要更少的步数——理想情况下只要一步。阻碍在于求解反向过程的 ODE 是刚性的；路径是弯曲的。

如果你能够训练模型使得从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的一个 Euler 步就足够了。Flow matching 直接构建这个：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练向量场 `v_θ(x, t)` 来匹配其时间导数，在推理时积分。

Rectified flow（Liu 2022）更进一步：通过 reflow 过程迭代地拉直路径，产生逐渐接近线性的 ODE。经过两次 reflow 迭代后，一个 2 步采样器就能达到 50 步 DDPM 的质量。

## 概念

![Flow matching：噪声与数据之间的直线插值](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 且 `x_1 ~ N(0, I)`。沿这条直线的时
间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义神经向量场 `v_θ(x_t, t)` 并训练它以匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这是**条件流匹配**损失（Lipman 2023）。训练是无模拟的：你从不展开 ODE。只需采样 `(x_0, x_1, t)` 并回归即可。

### 采样

在推理时，沿学习到的向量场*反向*时间积分：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，Euler 步进到 `t=0`。

### Rectified flow（Liu 2022）

直线流可以工作，但学习到的路径*实际上并不直*——它们会弯曲，因为多个 `x_0` 可以映射到同一个 `x_1`。Rectified flow 的 reflow 步骤：

1. 用随机配对训练 flow 模型 v_1。
2. 通过从 `x_1` 积分 v_1 到其落点的 `x_0`，采样 N 对 `(x_1, x_0)`。
3. 在这些配对示例上训练 v_2。由于配对现在是"ODE 匹配"的，它们之间的直线插值确实更平坦。
4. 重复。

实践中 2 次 reflow 迭代就能达到近线性，实现 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是基于 flow matching 蒸馏的模型。

### 为什么这在 2024 年赢得了图像生成

三个原因：

1. **无模拟训练** — 训练期间无需 ODE 展开，实现简单。
2. **更好的损失几何** — 直线路径具有一致的信噪比，而 DDPM ε-loss 在调度边缘有不良 SNR。
3. **更快的推理** — SDXL-Turbo 质量下 4-8 步；一致性蒸馏下 1 步。

## Flow matching 与 DDPM — 精确连接

带高斯条件路径的 flow matching 是*带有特定噪声调度*的扩散模型。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，flow matching 恢复为 Stratonovich 重构的扩散模型，其中 `v = α'·x_0 - σ'·x_1`。两者对高斯路径在代数上等价。

Flow matching 带来的是：目标的*清晰性*（一个普通的速度），更干净的损失，以及对非高斯插值的实验许可。

```figure
normalizing-flow
```

## 构建它

`code/main.py` 实现了双模高斯混合的 1-D flow matching。向量场 `v_θ(x, t)` 是一个微型 MLP，用直线目标训练。在推理时，积分 1、2、4 和 20 个 Euler 步并比较样本质量。

### 步骤 1：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 反向传播 + 更新
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

### 步骤 3：比较步数

预期 4 步采样器已经能达到 20 步的质量——这对延迟意义重大。

## 陷阱

- **时间参数化。** Flow matching 使用 `t ∈ [0, 1]`，其中 `t=0` 在数据端，`t=1` 在噪声端。DDPM 使用 `t ∈ [0, T]`，其中 `t=0` 在数据端，`t=T` 在噪声端。方向相同，尺度不同。论文经常搞错这个。
- **调度选择。** Rectified flow 的直线是"标准"的 flow-matching 调度，但你可以使用余弦或 logit-normal t 采样（SD3 就是这样做的）来获得更好的尺度覆盖。
- **Reflow 成本。** 生成 reflow 的配对数据集是对每个样本的一次完整推理传递。只有当你真正需要 1-2 步推理时才做 reflow。
- **Classifier-free guidance 仍然适用。** 只需在线性组合中将 ε 替换为 v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 使用它

| 用例 | 2026 技术栈 |
|------|------------|
| 文生图，最佳质量 | Flow matching：SD3、Flux.1-dev |
| 文生图，1-4 步 | 蒸馏 flow matching：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从 flow-matched 基础的一致性蒸馏（LCM、PCM） |
| 音频生成 | Flow matching：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | Flow matching 与扩散混合（Sora、Veo、Stable Video） |
| 科学/物理（粒子轨迹、分子） | Flow matching + 等变向量场 |

当 2025-2026 年的论文说"比扩散更快"时，几乎总是 flow matching + 蒸馏。

## 交付

保存 `outputs/skill-fm-tuner.md`。Skill 接收一个扩散风格的模型规格，并将其转换为 flow-matching 训练配置：调度选择、时间采样分布（uniform / logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py` 并比较 1 步与 20 步 MSE 相对于真实数据分布。
2. **中等。** 从 uniform `t` 采样切换到 logit-normal（将采样集中在中间 t）。模型质量是否会提升？
3. **困难。** 实现一次 reflow 迭代：通过积分第一个模型生成配对 (x_0, x_1)，在配对上训练第二个模型，并比较 1 步样本质量。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Flow matching | "直线扩散" | 训练 `v_θ(x, t)` 在插值上匹配 `x_1 - x_0`。 |
| Rectified flow | "Reflow" | 迭代拉直学习流的过程。 |
| Velocity field | "v_θ" | 模型输出 — 移动 `x_t` 的方向。 |
| Straight-line interpolant | "路径" | `x_t = (1-t)·x_0 + t·x_1`；平凡的导数目标。 |
| Euler sampler | "一阶 ODE 求解器" | 最简单的积分器；路径直时效果很好。 |
| Logit-normal t | "SD3 采样" | 将 `t` 采样集中在梯度最强的中间值。 |
| Consistency distillation | "1 步采样器" | 训练学生网络直接将任何 `x_t` 映射到 `x_0`。 |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；同样的技巧，新的变量。 |

## 生产笔记：Flux.1-schnell 是速度最快的 flow matching

Flow matching 的生产成果是 Flux.1-schnell —— 一个 flow-matched DiT，蒸馏到 1-4 个推理步数，同时保持 Flux-dev 级质量。Niels 的"在 8GB 机器上运行 Flux"笔记本是参考部署配方：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 用 4 步 vs dev 的 50 步），VAE 解码。成本核算：

| 变体 | 步数 | L4 上 1024² 延迟 | 总 FLOPs（相对） |
|------|------|------------------|-----------------|
| Flux.1-dev（原始） | 50 | ~15 秒 | 1.0× |
| Flux.1-schnell | 4 | ~1.2 秒 | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 秒 | 0.25× |
| SDXL-Lightning 2 步 | 2 | ~0.3 秒 | 0.03× |

生产规则：**flow-matched 基础 + 蒸馏 = 2026 年快速文生图的默认方案。** 每个主要厂商都提供这种组合：SD3-Turbo（SD3 + flow + 蒸馏）、Flux-schnell（Flux-dev + rectified-flow 拉直）、CogView-4-Flash。纯扩散基础仅存在于遗留 checkpoint 中。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模 rectified flow。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 涵盖 FM + diffusion 的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散/flow 的 1 步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching 在生产中的应用。
