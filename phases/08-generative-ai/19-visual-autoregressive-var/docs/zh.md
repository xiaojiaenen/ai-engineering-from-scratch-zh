# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步骤）。VAR 则在尺度上迭代采样——它先预测一个 1x1 的 token，然后是 2x2，接着是 4x4，直到最终分辨率，每个尺度都以之前的尺度为条件。2024 年的论文表明，VAR 在图像生成方面遵循与 GPT 风格相同的缩放定律，并在相同计算预算下击败了 DiT。本课将构建其核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**先决知识：** 第 7 阶段第 03 课（多头注意力），第 8 阶段第 06 课（DDPM）
**时长：** 约 90 分钟

## 问题

自回归生成在语言建模中占据主导地位，因为它具有可预测的扩展性：计算量越大，参数越多，困惑度越低，输出质量越好。2024 年之前，图像生成领域主要进行过两种自回归尝试：PixelRNN/PixelCNN（逐像素生成）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 码本上逐 token 生成）。

两者都受困于生成顺序问题。像素和 token 排列在二维网格中，但自回归模型必须按照一维光栅顺序访问它们。一个早期角落的像素对最终图像的样子一无所知。生成质量的扩展性比文本上的 GPT 更差，并且在匹配的计算量下从未达到扩散模型的质量。

VAR 通过改变生成的对象来解决生成顺序问题。VAR 不是在空间上逐个预测图像 token，而是以递增的分辨率预测整个图像。步骤 1：预测一个 1x1 的 token（整体图像的“摘要”）。步骤 2：预测一个 2x2 的 token 网格（更粗略的特征）。步骤 3：预测一个 4x4 的网格。步骤 K：预测最终的 (H/8)x(W/8) 网格。

每个尺度都因果地（按“尺度顺序”）关注所有之前的尺度，并且在其自身的尺度内并行进行。顺序问题消失了：尺度 k 的整个图像在一次 Transformer 前向传播中生成。

## 概念

### VQ-VAE 多尺度分词器

VAR 需要一个**多尺度离散分词器**。对于图像 x，它生成一系列分辨率逐步提高的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的码本（典型大小为 4096-16384）。每个尺度的分词并非独立——它经过训练，使得将每个尺度的残差相加可以重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一种**残差 VQ** 的变体。尺度 k 捕获了尺度 1..k-1 所遗漏的内容。解码器接收所有尺度嵌入的总和并生成图像。

多尺度 VQ 分词器只需训练一次（类似 VQGAN），然后冻结。所有的生成工作都由其上的自回归模型完成。

### 下一尺度预测

生成模型是一个 Transformer，它接收来自所有先前尺度的 token，并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入编码了尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k 中位置 (i, j) 的 token 可以关注尺度 1..k 的所有 token，以及尺度 k 内部在使用的内部顺序（VAR 使用固定的位置注意力，没有尺度内因果性——尺度内的所有位置是并行预测的）中较早出现的 token。

训练损失：在每个尺度 k，给定所有先前尺度的 token 来预测 token z_k。对离散的 VQ 码本应用交叉熵损失。结构与 GPT 相同，只是现在的“序列”是按尺度结构化的。

### 生成

在推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K=10 个尺度，生成过程是 10 次 Transformer 前向传播。每次前向传播并行产生其整个尺度——在尺度内没有逐 token 的自回归。对于 256x256 的图像，这大约需要 10 次前向传播，而 DiT 需要 28-50 次。

### 为什么下一尺度优于下一 token

三个结构性优势：
1.  **从粗到细与自然图像统计对齐。** 人类视觉感知和图像数据集都表现出与尺度相关的规律：低频结构稳定且可预测；高频细节取决于低频内容。下一尺度预测利用了这一点。
2.  **尺度内并行生成。** 与 GPT 风格的 token 自回归不同，VAR 在一个步骤中生成一个尺度的所有 token。有效的生成长度是对数级的，而不是线性的。
3.  **无生成顺序偏差。** 尺度 k 的 token 能看到所有尺度 k-1 的内容；不存在迫使早期 token 在后期上下文可用之前就做出承诺的“左侧”或“上方”偏差。

### 缩放定律

Tian 等人证明，VAR 在 ImageNet 上的 FID 遵循幂律缩放曲线——就像 GPT 在困惑度上遵循的那样。将参数或计算量加倍可以可靠地将误差减半。这是第一个展现出这种与语言模型一样清晰的缩放行为的图像生成模型。结果是，VAR 的缩放预测变得可以从计算量预测，而不是每个架构的经验猜测。

### 与扩散模型的关系

VAR 和扩散模型共享相同的数据压缩思想：两者都将生成问题分解为一系列更容易的子问题。
- 扩散模型：逐渐添加噪声，学习撤销一步。
- VAR：逐渐添加分辨率，学习预测下一个尺度。
它们是问题的不同切入角度。两者都产生易于处理的条件分布。经验证明，VAR 在推理时更快（前向传播更少，尺度内全部并行），并且在类条件 ImageNet 上匹配或击败了 DiT。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 构建它

在 `code/main.py` 中，你将：
1.  在合成的“图像”数据（2D 高斯环）上构建一个小型**多尺度 VQ 分词器**。
2.  训练一个**VAR 风格的 Transformer** 进行下一尺度预测。
3.  通过调用 Transformer 4 次（4 个尺度）并进行解码来进行采样。
4.  验证按尺度顺序的训练使得在尺度内生成是并行的。

这是一个玩具实现。重点是观察尺度结构化的注意力掩码和尺度内并行生成的实际工作情况。

## 交付它

本课产生 `outputs/skill-var-tokenizer-designer.md` —— 一项设计多尺度分词器的技能：尺度数量、尺度比率、码本大小、残差共享、解码器架构。

## 练习

1.  **尺度数量消融。** 使用 4、6、8、10 个尺度训练 VAR。衡量重建质量与自回归传播次数的关系。更多尺度 = 更精细的残差 = 更好的质量，但需要更多传播。
2.  **码本大小。** 使用大小为 512、4096、16384 的码本训练分词器。更大的码本提供更好的重建效果，但预测更难。找到拐点。
3.  **尺度内并行检查。** 对于一个训练好的 VAR，显式测量其注意力模式。在尺度 k 内，模型是否关注跨尺度位置但不关注尺度内位置？验证掩码的实现。
4.  **VAR 与 DiT 缩放对比。** 对于相同的 ImageNet 类条件任务，在匹配的参数预算（例如 33M、130M、458M）下训练 VAR 和 DiT。绘制 FID 与计算量的关系图。VAR 应在每个规模上领先于 DiT —— 在小规模上复现论文的结果。
5.  **文本条件。** 通过 adaLN 扩展 VAR 以接受文本嵌入（CLIP 池化后的）作为额外的条件输入。这是 HART 的配方。在文本对齐的采样中，FID 能改善多少？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| VAR | “视觉自回归” | 通过下一尺度预测在 VQ token 网格金字塔上进行图像生成 |
| 下一尺度预测 | “先预测粗略的，再预测精细的” | 模型以所有先前尺度为条件，预测分辨率递增的尺度上的 token |
| 多尺度 VQ 分词器 | “残差 VQ” | 生成 K 个分辨率递增的 token 网格的 VQ-VAE，解码器将所有尺度相加 |
| 尺度 k | “金字塔层级 k” | K 个分辨率级别中的一个，从 k=1 时的 1x1 到 k=K 时的 (H/p)x(W/p) |
| 尺度内并行 | “每个尺度一次前向传播” | 尺度 k 的所有 token 在一次 Transformer 前向传播中预测，而非自回归地 |
| 跨尺度因果 | “尺度有序注意力” | 尺度 k 的 token 可以关注尺度 1..k 的所有内容，但不能关注尺度 k+1..K |
| 残差 VQ | “加性分词化” | 每个尺度的 token 编码了较低尺度留下的残差；解码器将所有尺度嵌入相加 |
| VAR 缩放定律 | “图像 GPT 缩放” | FID 在计算量上遵循可预测的幂律，就像语言模型的困惑度一样 |
| HART | “混合 VAR + 文本” | 文本条件 VAR 变体，结合了 MaskGIT 风格的迭代解码与 VAR 的尺度结构 |
| 尺度位置嵌入 | “(尺度, 行, 列) 三元组” | 位置编码同时携带尺度索引和该尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — “Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction”](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考文献
- [Peebles and Xie, 2022 — “Scalable Diffusion Models with Transformers”](https://arxiv.org/abs/2212.09748) — DiT，扩散模型比较基线
- [Esser et al., 2021 — “Taming Transformers for High-Resolution Image Synthesis”](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器所扩展的分词器家族
- [van den Oord et al., 2017 — “Neural Discrete Representation Learning”](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词的基础
- [Tang et al., 2024 — “HART: Efficient Visual Generation with Hybrid Autoregressive Transformer”](https://arxiv.org/abs/2410.10812) — 文本条件 VAR