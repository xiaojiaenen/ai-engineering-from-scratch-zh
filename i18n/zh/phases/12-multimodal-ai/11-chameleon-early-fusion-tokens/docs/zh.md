# Chameleon 与早期融合 Token-Only 多模态模型

> 我们迄今见过的所有 VLM 都将图像和文本分开处理。视觉 token 来自视觉编码器，流入投影层，然后在 LLM 内部与文本相遇。视觉词表和文本词表从不重叠。Chameleon（Meta，2024年5月）提出了一个问题：如果它们重叠呢？训练一个 VQ-VAE，将图像转换为来自共享词表的离散 token 序列。每个多模态文档现在都是一个序列——文本 token 和图像 token 交错排列，使用单一自回归损失。副作用：模型可以生成混合模态的输出——在单次推理调用中交替输出文本 token 和图像 token。本课程阅读早期融合论文并从头构建一个玩具版本。

**类型：** 实践构建
**语言：** Python（标准库、VQ-VAE tokenizer + 交错解码器）
**先决条件：** 第12阶段 · 05，第8阶段（生成式 AI）
**时间：** 约180分钟

## 学习目标

- 解释为什么共享词表 + 单一损失改变了模型的能力边界。
- 描述 VQ-VAE 如何将图像 token 化为与 transformer 的下一个 token 预测目标兼容的离散序列。
- 列举 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置位置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并描述每种方法何时是正确选择。

## 问题所在

基于 Adapter 的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的事物。一个文本 token 经过 `embed(text_token)`；一张图像经过 `visual_encoder(image) → projector → ...伪token`。模型有两条输入路径，在中间某处合并。

三个后果：

1. LLM 只能消费图像，不能生成图像。输出只能是文本。
2. 混合模态文档（段落和图像交替，如文章中那样）很别扭——你要么在模型外部解析多模态输入，要么进行链式生成。
3. 分布不匹配。视觉 token 和文本 token 存在于隐藏空间的不同区域，造成微妙的对齐问题。

Chameleon 拒绝了这一前提：图像只是一系列来自共享词表的离散 token。在交错文档上训练模型，一个损失，一个自回归解码器，你就能免费获得混合模态生成能力。

## 核心概念

### VQ-VAE 作为图像 tokenizer

Tokenizer 是一个矢量量化变分自编码器。架构如下：

- 编码器：CNN + ViT，将图像映射为空间特征图，例如 32x32 维 256 的特征。
- 码本：学习到的 K 个向量词表（Chameleon 使用 8192），同样维 256。
- 量化：对每个空间特征，通过 L2 距离查找最近的码本条目。用整数索引替换连续特征。
- 解码器：CNN，将量化特征还原回像素。

训练：VAE 重建损失 + 承诺损失 + 码本损失。码本索引构成了图像的离散字母表。

对于 Chameleon：一张图像变成 32*32 = 1024 个 token，从 8192 的词表中选取。与文本 token（来自 LLM 的 BPE 词表，假设为 32000）拼接。最终词表：40192。transformer 看到一个序列，一个损失。

### 共享词表

Chameleon 的词表组合了文本 token、图像 token 和模态分隔符。每个 token 都有一个唯一的 ID。输入嵌入层将每个 ID 映射到 D 维隐藏向量。输出投影层将隐藏向量映射回词表 logit。Softmax 选择下一个 token，无论其模态。

分隔符很重要：`<image>` 和 `</image>` 标签包裹图像 token 序列。在生成时，如果模型发出 `<image>`，下游软件就知道接下来的 1024 个 token 是发送给解码器进行像素渲染的 VQ 索引。

### 混合模态生成

推理是共享词表中的下一个 token 预测。示例 prompt："Draw a cat and describe it." Chameleon 输出：

```
<image> 4821 1029 2891 ...（1024个图像token）</image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——它可以先产生图像再文本、先文本再图像，或交错。同一个解码器，同一个损失。

与 Adapter VLM 对比，其中生成只能是纯文本。Chameleon 重新打开了关于模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 顺序

早期融合的规模训练不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在注意力内部对 query 和 key 投影应用 LayerNorm，在点积之前。防止深层中 logit 幅度爆炸。被多个 2024 年后的巨型模型采用。
- Dropout 放置位置。在每个残差加法后都放置 dropout，不仅是在 attention 和 MLP 之后。当图像 token 的梯度可能占主导时需要更多正则化。
- LayerNorm 顺序。残差分支上使用 Pre-LN（标准做法），并在最后一个块的跳跃连接上额外添加 LN。稳定最后一层的梯度流动。

没有这些技巧，34B 参数的 Chameleon 训练在多个 checkpoint 时发散。有了它们，它就能收敛。训练配方与架构一样都是贡献的一部分。

### Tokenizer 的重建上限

VQ-VAE 是有损的。在 8192 个码本条目和每 512x512 图像 1024 个 token 的情况下，重建 PSNR 上限约为 26-28 dB。这对于可识别的图像生成是足够的，但明显差于连续空间的扩散模型（Stable Diffusion 3 可达 32+ dB）。

Tokenizer 是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）能提升上限。Emu3（第12.12课）仅通过更好的 tokenizer 就实现了 SDXL 质量的生成。

### Chameleon 与 BLIP-2 / LLaVA 对比

Chameleon（早期融合，共享词表）：
- 一个损失，一个解码器。
- 生成混合模态输出。
- Tokenizer 是质量上限。
- 昂贵：推理路径上每次生成图像都需要 VQ-VAE 解码。

BLIP-2 / LLaVA（后期融合，独立塔）：
- 视觉输入，仅文本输出。
- 复用预训练 LLM。
- 理解方面无 tokenizer 瓶颈。
- 廉价：单次前向传播。

根据任务选择。如果需要图像生成，选 Chameleon 家族。如果只需要理解，Adapter-VLM 更简单且复用更多预训练算力。

### Fuyu 和 AnyGPT

Fuyu（Adept，2023）是一种相关方法：完全跳过独立的视觉编码器，将原始图像 patch 直接通过 LLM 的输入投影层，就像它们是 token 一样，无需 tokenizer。比 Chameleon 更简单，但失去了共享词表的输出生成能力。

AnyGPT（Zhan 等人，2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种模态使用相同的 VQ-VAE 技巧，共享 transformer。在 Lesson 12.16 中有更多覆盖。

```figure
vq-codebook
```

## 动手实践

`code/main.py` 构建了一个玩具端到端早期融合模型：

- 一个微小的 VQ-VAE 风格量化器，将 8x8 patch 映射到码本索引（K=16）。
- 一个共享词表（文本 id 0..31 + 图像 id 32..47 + 分隔符 48、49）。
- 一个在合成标题 + 图像 token 序列上训练的玩具自回归解码器（bigram 表）。
- 采样循环，在给定 prompt 下发出交替的文本 + 图像 token。

代码刻意保持 transformer 微小（bigrams），以便你可以端到端追踪信号流。

## 交付成果

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅需理解 vs 理解 + 生成、所需的图像质量、成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（后期融合）之间做出选择，并用定量经验法则进行论证。

## 练习

1. Chameleon 使用 K=8192 个码本条目和每 512x512 图像 1024 个 token。估算相对于 24-bit RGB 图像的压缩比。它是有损的吗？有多有损？

2. 相同 VQ-VAE 密度下，一张 4K 图像（3840x2160）会产生多少图像 token？Chameleon 风格的模型能否在一次推理调用中生成 4K 图像？最先崩溃的是什么——上下文、tokenizer 质量还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定一个 64 维的 query 和 key，展示 LayerNorm 前后点积的变化。为什么幅度控制在深层很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性的内容。描述论文在 34B 规模下未使用 QK-Norm 时观察到的具体失效模式。"norm explosion" 的特征是什么？

5. 扩展现有的玩具解码器，使其能够根据纯文本 prompt 发出混合模态响应。测量模型在训练数据分布为 60% 文本优先 / 40% 图像优先的情况下，选择图像优先与文本优先的频率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 早期融合 | "统一 token" | 图像从一开始就转换为与 transformer 词表共享的离散 token |
| VQ-VAE | "图像 tokenizer" | CNN + ViT + 码本，将图像映射为 transformer 可以预测的整数索引 |
| 共享词表 | "一个字典" | 一个涵盖文本 + 图像 + 模态分隔符的统一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 在 query 和 key 的点积之前对其应用 LayerNorm，防止 norm 爆炸 |
| 混合模态生成 | "文本 + 图像输出" | 推理时自主在一次传递中产生交错的文本和图像 token |
| 码本大小 | "K 个条目" | VQ-VAE 可以量化的离散向量数量；在压缩和保真度之间权衡 |
| Tokenizer 上限 | "重建极限" | 解码 VQ token 可实现的最高 PSNR；限制模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等人 — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu 等人 — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan 等人 — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
