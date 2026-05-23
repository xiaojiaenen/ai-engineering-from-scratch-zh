# 条件GAN与Pix2Pix

> 2014-2017年的第一个重大突破是控制GAN的生成内容。附加一个标签、一张图片或一句话。Pix2Pix实现了图像版本，并在狭窄的图像到图像任务中，至今仍击败所有通用的文本到图像模型。

**类型：** 构建
**语言：** Python
**先修阶段：** 阶段 8 · 03 (GANs)， 阶段 4 · 06 (U-Net)， 阶段 3 · 07 (CNNs)
**时间：** ~75分钟

## 问题

一个无条件GAN生成任意人脸。对演示有用，在生产中无用。你想要的是：*将草图映射为照片*、*将地图映射为航拍照片*、*将白天场景映射为夜晚*、*为灰度图像上色*。在所有这些场景中，你被给定一个输入图像 `x`，必须输出 `y`，且两者之间存在某种语义对应。每个 `x` 对应许多可能的 `y`。均方误差会将它们模糊成一团。对抗性损失则不会，因为“看起来真实”这个概念是锐利的。

条件GAN（Mirza & Osindero, 2014）添加了一个条件 `c` 作为 `G` 和 `D` 的输入。Pix2Pix（Isola et al., 2017）对此进行了专门化：条件是一个完整的输入图像，生成器是一个U-Net，判别器是一个*基于块*的分类器（PatchGAN），损失函数是对抗性损失 + L1损失。这个方法甚至在2026年，在狭窄的图像到图像领域中仍优于从零开始的文本到图像模型，因为它是在*配对数据*上训练的——你恰好拥有你需要的信号。

## 概念

![Pix2Pix: U-Net生成器， PatchGAN判别器](../assets/pix2pix.svg)

**条件G.** `G(x, z) → y`。在Pix2Pix中，`z` 是G内部的Dropout（没有输入噪声——Isola发现显式噪声会被忽略）。

**条件D.** `D(x, y) → [0, 1]`。输入是*配对*的（条件， 输出）。这是关键区别：D必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 是否看起来真实。

**U-Net生成器。** 带有跨越瓶颈层的跳跃连接的编码器-解码器结构。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN判别器。** D不输出单个真/假分数，而是输出一个 `N×N` 网格，其中每个单元格判断约70×70像素的感受野。取平均值。这是一个马尔可夫随机场假设：真实性是局部的。训练速度更快，参数更少，输出更锐利。

**损失函数。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1项稳定训练，并将G推向已知目标。L1比L2（中位数， 而非均值）产生更锐利的边缘。`λ = 100` 是Pix2Pix的默认设置。

## CycleGAN——当你没有配对数据时

Pix2Pix需要配对的 `(x, y)` 数据。CycleGAN（Zhu et al., 2017）以额外的损失为代价取消了这个要求：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 和 `G(F(y)) ≈ y`。这使你能够将马映射到斑马，夏天映射到冬天，而无需配对样本。

在2026年，无配对图像到图像翻译主要通过扩散模型（ControlNet、IP-Adapter）完成，而不是CycleGAN，但循环一致性的思想几乎存在于每一篇无配对领域适应论文中。

## 动手构建

`code/main.py` 在一维数据上实现一个微型条件GAN。条件 `c` 是一个类别标签（0或1）。任务：为给定类别生成一个来自条件分布的样本。

### 步骤1：将条件附加到G和D的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码是最简单的方法。更大的模型使用可学习的嵌入、FiLM调制或交叉注意力。

### 步骤2：训练条件

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件*下的真实分布，而不是边缘分布。

### 步骤3：验证每个类别的输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **条件被忽略。** G学会了边缘化，D从未惩罚，因为条件信号很弱。解决方法：更积极地为D添加条件（在早期层， 而不仅仅是后期层），使用投影判别器（Miyato & Koyama 2018）。
- **L1权重过低。** G会漂移到任意看起来真实的输出，而非忠实的输出。对于Pix2Pix风格的任务，建议从 λ≈100 开始。
- **L1权重过高。** G产生模糊输出，因为L1仍然是L_p范数。一旦训练稳定，就逐渐减小权重。
- **D中存在真实数据泄漏。** 将 `(x, y)` 作为D输入进行拼接，而不仅仅是 `y`。没有这个，D无法检查一致性。
- **每个类别的模式崩溃。** 每个类别可能独立崩溃。运行类别条件多样性检查。

## 使用它

2026年图像到图像任务的状态：

| 任务 | 最佳方法 |
|------|----------|
| 草图 → 照片，相同领域，配对数据 | Pix2Pix / Pix2PixHD（仍然快速，仍然锐利） |
| 草图 → 照片，无配对 | 带有涂鸦条件模型的ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 使用IP-Adapter或LoRA的扩散模型；GAN方法是遗留方案 |
| 深度图 → 照片 | 基于Stable Diffusion的ControlNet-Depth |
| 超分辨率 | Real-ESRGAN (GAN)， ESRGAN-Plus， 或 SD-Upscale (扩散) |
| 上色 | ColTran，基于扩散的上色器， 或 Pix2Pix-color |
| 白天 → 夜晚，季节，天气 | CycleGAN 或 基于ControlNet的方法 |

当 (a) 你有数千个配对样本，(b) 任务是狭窄且可重复的，并且 (c) 你需要快速推理时，Pix2Pix仍然是正确的工具。在通用的开放域任务上，扩散模型获胜。

## 交付

保存 `outputs/skill-img2img-chooser.md`。该技能需要一个任务描述、数据可用性（配对 vs 无配对， N个样本）、延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet变体、SDXL + IP-Adapter）、训练数据需求、推理成本和评估协议（LPIPS、FID、特定任务指标）。

## 练习

1. **简单。** 修改 `code/main.py` 以添加第三个类别。确认G仍然将每个类别的噪声映射到正确的模式。
2. **中等。** 在一维设置中用感知风格损失（例如，一个小的冻结D作为特征提取器）替换L1。它会改变条件分布的锐度吗？
3. **困难。** 在一维环境中勾勒一个CycleGAN：两个分布，两个生成器，循环损失。展示它如何在没有配对数据的情况下学习在它们之间进行映射。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 条件GAN | “带标签的GAN” | G(z, c), D(x, c)。两个网络都看到条件。 |
| Pix2Pix | “图像到图像GAN” | 配对cGAN， 使用U-Net G和PatchGAN D + L1损失。 |
| U-Net | “带跳跃连接的编码器-解码器” | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | “局部真实性分类器” | D输出每个块的分数，而不是全局分数。 |
| CycleGAN | “无配对图像翻译” | 两个G + 循环一致性损失；无需配对数据。 |
| SPADE | “GauGAN” | 使用语义图对中间激活进行归一化；分割到图像。 |
| FiLM | “特征级线性调制” | 基于条件的逐特征仿射变换；廉价的条件化方法。 |

## 生产注释：Pix2Pix作为延迟受限的基线

当你有配对数据和狭窄任务（草图 → 渲染， 语义图 → 照片， 白天 → 夜晚）时，Pix2Pix的一次性推理在延迟上比扩散模型快一个数量级。生产环境中的比较通常是：

| 路径 | 步数 | 在单个L4上512²的典型延迟 |
|------|------|--------------------------|
| Pix2Pix (U-Net前向传播) | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix在静态批次的吞吐量上获胜（每个请求的FLOPs相同）。扩散模型在质量和泛化能力上获胜。现代做法通常是为狭窄任务部署一个Pix2Pix风格的蒸馏模型，并为尾部输入保留一个扩散模型作为后备。

## 扩展阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器。