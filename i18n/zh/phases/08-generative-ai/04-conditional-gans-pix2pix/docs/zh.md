# 条件 GAN 与 Pix2Pix

> 2014–2017 年间最大的突破之一，是能够控制 GAN 生成什么内容。附上一个标签、一张图片，或一句话。Pix2Pix 实现了图像版本，即使在今天，它在特定的图像到图像任务上依然能击败所有通用文本到图像模型。

**类型：** 动手实践
**语言：** Python
**前置知识：** 第 8 阶段 · 03（GAN）、第 4 阶段 · 06（U-Net）、第 3 阶段 · 07（CNN）
**预计用时：** 约 75 分钟

## 问题所在

无条件 GAN 会采样任意人脸。用作演示很有趣，但在生产中毫无用处。你想要的是：*将草图映射为照片*、*将地图映射为航拍图*、*将白天场景映射为夜晚*、*对灰度图像进行上色*。在所有这些任务中，你都会得到一个输入图像 `x`，并需要输出与之语义对应的 `y`。对于同一个 `x`，存在许多合理的 `y`。均方误差（MSE）会将这些合理结果"平均"成一团糊状物。而对抗损失不会，因为"看起来真实"是尖锐的判别标准。

条件 GAN（Mirza & Osindero，2014）将一个条件 `c` 同时作为 `G` 和 `D` 的输入。Pix2Pix（Isola et al.，2017）对此做了专门化：条件是一张完整的输入图像，生成器是 U-Net，判别器是*基于 patch 的*分类器（PatchGAN），损失函数为对抗损失 + L1。即使在 2026 年，这种方案在特定的图像到图像领域仍优于从零训练的文本到图像模型，因为它是用*成对数据*训练的——你拥有所需的全部信号。

## 核心概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件生成器。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（无输入噪声——Isola 发现显式噪声会被忽略掉）。

**条件判别器。** `D(x, y) → [0, 1]`。输入是*配对*（条件，输出）。这是关键区别：D 需要判断 `y` 是否与 `x` 一致，而不只是判断 `y` 是否看起来真实。

**U-Net 生成器。** 带跳连的编码器-解码器结构，跨越瓶颈层。对于输入和输出共享低层结构（边缘、轮廓）的任务而言至关重要。没有跳连，高频细节就会消失。

**PatchGAN 判别器。** D 不输出单一的真实/虚假分数，而是输出一个 `N×N` 网格，每个单元格评判约 70×70 像素的感受野。最终取平均。这基于马尔可夫随机场假设：真实性是局部属性。训练更快，参数更少，输出更锐利。

**损失函数。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项能稳定训练，并推动 G 向已知目标靠近。L1 比 L2 产生更锐利的边缘（中位数而非均值）。`λ = 100` 是 Pix2Pix 的默认值。

## CycleGAN —— 当你没有成对数据时

Pix2Pix 需要成对的 `(x, y)` 数据。CycleGAN（Zhu et al.，2017）放弃了这一要求，代价是引入额外的损失：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你可以将马翻译成斑马、夏季转换为冬季，而无需成对样本。

在 2026 年，非配对图像到图像翻译主要借助扩散模型（ControlNet、IP-Adapter）完成，而非 CycleGAN，但循环一致性思想几乎存在于每一篇非配对域适应论文中。

```figure
gx-patchgan
```

## 动手实现

`code/main.py` 在 1-D 数据上实现了一个简单的条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务是为给定类别从条件分布中生成样本。

### 步骤 1：将条件同时附加到 G 和 D 的输入中

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot 编码是最简单的方式。更大的模型会使用学习嵌入、FiLM 调制或交叉注意力。

### 步骤 2：条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件*下的真实分布，而非边缘分布。

### 步骤 3：验证逐类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** G 学会了边缘化，而 D 从未进行惩罚，因为条件信号太弱。解决方法：更激进地对 D 施加条件（在早期层，而非仅后期），使用投影判别器（Miyato & Koyama，2018）。
- **L1 权重过低。** G 会漂移到任意看起来真实的输出，而非忠实于目标的输出。针对 Pix2Pix 风格的任务，从 λ≈100 开始。
- **L1 权重过高。** G 会产生模糊输出，因为 L1 仍然是 L_p 范数。训练稳定后逐步降低权重。
- **D 中出现Ground-truth泄漏。** 将 `(x, y)` 拼接为 D 的输入，而非仅输入 `y`。没有这一点，D 无法检查一致性。
- **逐类模式崩溃。** 每个类别可能独立崩溃。运行类别条件多样性检查。

## 如何使用

2026 年图像到图像任务的最佳方案：

| 任务 | 最佳方案 |
|------|----------|
| 草图 → 照片，同域，有成对数据 | Pix2Pix / Pix2PixHD（速度快，输出锐利） |
| 草图 → 照片，非配对 | 配合 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2，或 SD + ControlNet-Seg |
| 风格迁移 | 配合 IP-Adapter 或 LoRA 的扩散模型；GAN 方法已是过时方案 |
| 深度图 → 照片 | 基于 Stable Diffusion 的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus，或 SD-Upscale（扩散） |
| 图像上色 | ColTran、基于扩散的上色模型，或 Pix2Pix-color |
| 白天→夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方案 |

当你满足以下三个条件时，Pix2Pix 仍是正确选择：（a）有成千上万条成对样本，（b）任务狭窄且可重复，（c）需要快速推理。在通用开放域任务上，扩散模型胜出。

## 交付物

保存 `outputs/skill-img2img-chooser.md`。该技能接收任务描述、数据可用性（成对还是非配对、样本数量 N）、以及延迟/质量预算，然后输出：方案选择（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本，以及评估协议（LPIPS、FID、任务专用指标）。

## 练习题

1. **简单。** 修改 `code/main.py` 以添加第三个类别。确认 G 仍能将每个类别的噪声映射到正确的模式。
2. **中等。** 在 1-D 设置中将 L1 替换为感知风格损失（例如，一个作为特征提取器的小型冻结 D）。这会改变条件分布的锐利程度吗？
3. **困难。** 在 1-D 设置中设计一个 CycleGAN：两个分布、两个生成器、循环损失。证明它能在无配对数据的情况下学习两者之间的映射。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| 条件 GAN | "带标签的 GAN" | G(z, c)，D(x, c)。两个网络都能看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 使用 U-Net G 和 PatchGAN D + L1 损失的成对条件 GAN。 |
| U-Net | "带跳连的编码器-解码器" | 对称卷积网络；跳连保留高频信息。 |
| PatchGAN | "局部真实性分类器" | D 输出每块分数而非全局分数。 |
| CycleGAN | "非配对图像翻译" | 两个 G + 循环一致性损失；无需成对数据。 |
| SPADE | "GauGAN" | 使用语义图对中间激活进行归一化；分割到图像的翻译。 |
| FiLM | "逐特征线性调制" | 来自条件的逐特征仿射变换；低成本的条件注入方式。 |

## 生产笔记：Pix2Pix 作为延迟受限的基线

当你有成对数据且任务狭窄（草图→渲染、语义图→照片、白天→夜晚）时，Pix2Pix 的一次前向推理在延迟上比扩散模型快一个数量级。生产环境中的典型对比如下：

| 方案 | 步数 | L4 单卡 512² 典型延迟 |
|------|------|----------------------|
| Pix2Pix（U-Net 前向） | 1 | 约 30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | 约 1.2 s |
| SDXL-Turbo Img2Img | 1–4 | 约 0.15–0.35 s |
| ControlNet + SDXL base | 20–30 | 约 3–5 s |

在静态批处理中，Pix2Pix 凭借吞吐量胜出（每个请求的 FLOPs 相同）。扩散模型在质量和泛化性上胜出。现代常见做法是：为狭窄任务部署一个 Pix2Pix 风格的蒸馏模型，同时为长尾输入保留扩散模型作为备选。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —— 条件 GAN 开山之作。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) —— Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) —— CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) —— Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) —— SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) —— 投影判别器。
