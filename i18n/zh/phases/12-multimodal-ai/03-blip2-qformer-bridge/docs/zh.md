# 从 CLIP 到 BLIP-2 — Q-Former 作为模态桥梁

> CLIP 对齐图像和文本，但无法生成描述、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥梁解决了这个问题：32 个可学习查询向量通过交叉注意力在冻结的 ViT 特征上进行注意力操作，然后直接插入到冻结 LLM 的输入流中。188M 参数的桥梁将 11B LLM 与 ViT-g/14 连接起来。到 2026 年为止的每个基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的表亲——都是它的后代。本课阅读 Q-Former 的架构，解释其两阶段训练，并构建一个玩具版本，将视觉 token 送入冻结的文本解码器。

**类型：** 构建
**语言：** Python（stdlib、交叉注意力 + 可学习查询演示）
**先决条件：** 阶段 12 · 02（CLIP）、阶段 7（Transformers）
**时间：** 约 180 分钟

## 学习目标

- 解释为什么冻结视觉编码器和冻结 LLM 之间的可训练瓶颈比端到端微调在成本与稳定性上更优。
- 实现一个交叉注意力块，其中固定数量的可学习查询对外部图像特征进行注意力操作。
- 了解 BLIP-2 的两阶段预训练流程：表示学习（ITC + ITM + ITG）然后是生成式学习（冻结解码器上的 LM 损失）。
- 比较 Q-Former 与 LLaVA 中使用的简单 MLP 投影器，并论证每种选择的适用场景。

## 问题描述

你有一个冻结的 ViT，它每张图像产生 256 个 patch token，维度为 1408。你有一个冻结的 7B LLM，它期望维度为 4096 的 token 嵌入。显而易见的桥梁——从 1408 到 4096 的线性层——可行，但将所有 256 个 patch token 送入 LLM 上下文意味着每张图像增加 256 个 token。对于 32 张图像的批次，光是视觉模态就消耗了 8192 个 token。

BLIP-2 的问题：你能否将 256 个 token 的图像表示压缩为少得多（比如 32 个）的 token，同时保留足够信息让 LLM 能够生成描述、回答问题并对图像进行推理？并且你能在不触碰冻结骨干网络的情况下训练这个桥梁吗，使训练成本仅来自桥梁参数？

答案：Q-Former。32 个可学习的"查询"向量对 ViT 的 patch token 进行交叉注意力，产生一个 32 个 token 的视觉摘要供 LLM 消费。总共 188M 参数。在触及 LLM 之前，使用对比、匹配和生成目标进行训练。

## 概念

### 可学习查询

Q-Former 的核心技巧：不让 LLM 的文本 token 去关注图像 patch，而是引入一组新的 32 个可学习查询向量 `Q`，让它们去关注图像 patch。这些查询是模型的参数——在训练期间学习，且相同的 32 个查询用于每张图像。

交叉注意力之后，每个查询持有图像的压缩摘要——"描述主要对象"、"描述背景"、"计数对象"等。查询不会字面上在语义标签上专业化；它们学会任何能使下游损失降低的编码。

### 架构

Q-Former 是一个小型 transformer（12 层，约 100M 参数），具有两条路径：

1. 查询路径：32 个查询向量流过自注意力（在它们之间），然后对冻结 ViT 的 patch token 进行交叉注意力，然后 FFN。
2. 文本路径：类 BERT 文本编码器与查询路径共享自注意力和 FFN 权重。文本路径禁用交叉注意力。

训练时两条路径同时运行。查询和文本通过共享自注意力交互，这意味着查询可以对需要它的任务（ITM、ITG）以文本为条件。在 VLM 交接时的推理期间，只有查询流过，产生 32 个视觉 token。

### 两阶段训练

BLIP-2 分两个阶段预训练：

阶段 1：表示学习（无 LLM）。三种损失：
- ITC（图像-文本对比）：池化查询 token 与文本 CLS token 之间的 CLIP 风格对比。
- ITM（图像-文本匹配）：二元分类器——这张图像-文本对是否匹配？采用困难负样本挖掘。
- ITG（图像 grounding 文本生成）：文本上的因果 LM 头，以查询为条件。迫使查询编码可生成文本的内容。

只有 Q-Former 训练。ViT 冻结。不涉及 LLM。

阶段 2：生成式学习。附加一个冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过小型线性层将 32 个查询输出投影到 LLM 的嵌入维度。将它们前置到文本提示前。仅在拼接后的提示 + 图像 + 描述序列的 LM 损失上训练线性投影和 Q-Former。

阶段 2 之后，Q-Former + 投影即为完整的视觉适配器。推理时：图像 → ViT → Q-Former → 线性投影 → 前置到文本 → 冻结 LLM 输出结果。

### 参数经济

BLIP-2 带 ViT-g/14（1.1B，冻结）+ OPT-6.7B（6.7B，冻结）+ Q-Former（188M，训练）= 8B 总计，188M 训练。Q-Former 单独仅占整个堆栈参数的约 2.4%。训练成本反映这一点：在少数 A100 上训练数天 vs 端到端的数周。

质量：BLIP-2 在零样本 VQA 上匹配或超越 Flamingo-80B，而体积为其 1/50。桥梁有效。

### InstructBLIP 与指令感知 Q-Former

InstructBLIP（2023）通过额外输入扩展了 Q-Former：指令文本本身。在交叉注意力时，查询现在可以同时访问图像 patch 和指令。查询可以按指令专业化（"数汽车"、"描述情绪"），而不是学习单一固定摘要。在保留任务上的基准提升。

### MiniGPT-4 与仅投影器方法

MiniGPT-4 保留了 Q-Former，但仅训练输出线性投影，冻结其余部分。便宜，但以质量为代价——查询是 BLIP-2 的，不是你的。适合快速迭代，但不是最佳架构。

### 为何 LLaVA 选择更简单方案

LLaVA（2023，课程 12.05）用简单的 2 层 MLP 替换了 Q-Former，将每个 ViT patch token 投影到 LLM 空间——每张图像 576 个 token（24x24 网格），全部送入 LLM。压缩效果较差，但允许 LLM 在原始 patch 上进行注意力操作。当时这曾引发争议；到 2023 年底成为主流，因为视觉指令数据（LLaVA-Instruct-150k）证明 MLP 可以被训练以保留足够信号。权衡：LLaVA 的上下文填充更快，但自然扩展到多图像和视频。

到 2026 年，该领域分化为两类：Q-Former 在 token 预算关键场景存活（长视频、多图像）；MLP 投影器在每 token 原始质量优先的场景占主导。

### 门控交叉注意力：Flamingo，祖先

Flamingo（课程 12.04）早于 BLIP-2，使用相同的交叉注意力思想，但在每个冻结 LLM 层上应用，而非作为单一桥梁。BLIP-2 表明你可以仅在输入层压缩即可仍然有效。Gemini 和 Idefics 两者结合：交错输入 token 加上可选的门控交叉注意力用于上下文中 few-shot。

### 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数视频-语言模型（出于 token 预算原因）。
- Perceiver resampler：Flamingo 的变体（课程 12.04）；Idefics 系列、Eagle、OmniMAE。
- MLP 投影器：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- 注意力池化：VILA、PaliGemma。

四种方法都有效。决定性问题是你是受 token 预算约束还是每 token 质量优先。

```figure
modality-projection
```

## 使用方式

`code/main.py` 构建一个 stdlib Q-Former 风格交叉注意力：

1. 模拟 256 个图像 patch token（维度 128）。
2. 实例化 32 个可学习查询（维度 128）。
3. 运行缩放点积交叉注意力（Q 来自查询，K/V 来自 patches）。
4. 通过线性层投影到 LLM 维度（512）。
5. 输出 32 个 LLM 就绪的视觉 token。

所有数学运算在纯 Python 中（向量的嵌套循环）。玩具但形状正确。注意力权重矩阵被打印出来，以便查看每个查询从哪些 patch 中提取。

## 交付物

本课生成 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（视觉编码器 token 数、LLM 上下文预算、部署约束、质量目标），它推荐 Q-Former vs MLP vs Perceiver resampler，附带简短理由和每种桥梁的参数数量估算。

## 练习

1. 在 PyTorch 中实现交叉注意力块。验证当使用 32 个查询和 256 个 key/value 时，注意力权重矩阵为 32 x 256，且每行在 softmax 后求和为 1。

2. 在 BLIP-2 阶段 1 中，Q-Former 同时运行三种损失：ITC、ITM、ITG。用伪代码写出每个的前向签名。哪个需要文本编码器路径处于活动状态？

3. 比较参数数量：Q-Former（12 层，768 隐藏维度）vs 2 层 MLP 投影器（1408 → 4096，两层）。在何种 LLM 规模下，188M Q-Former 的成本通过训练效率获得回报？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 如何初始化。解释为何从 BERT-base（而非随机）初始化能加速收敛。

5. 对于每秒 1 帧采样到 60 帧的 10 分钟视频，计算每帧 token 成本在（Q-Former → 每帧 32 token）vs（MLP 投影器 → 每帧 576 token）。哪种能放入 128k token 的 LLM 上下文窗口？

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|------------|----------|
| Q-Former | "查询 transformer" | 小型 transformer，带有 32 个可学习查询向量，对冻结 ViT 特征进行交叉注意力 |
| 可学习查询 | "视觉软提示" | 一组固定参数，作为交叉注意力的查询侧；每个模型独立学习，跨所有输入共享 |
| 交叉注意力 | "Q 从这里来，K/V 从那里来" | 查询、键和值来自不同来源的注意力；查询如何从 ViT patch 中提取信息 |
| ITC | "图像-文本对比" | 应用于 Q-Former 池化查询与文本 CLS 的 CLIP 风格损失 |
| ITM | "图像-文本匹配" | 在困难负样本挖掘对上的二元分类器；迫使查询区分细粒度不匹配 |
| ITG | "图像 grounding 文本生成" | 以查询为条件的因果 LM 损失；迫使查询编码可解码为文本的内容 |
| 两阶段预训练 | "先表示后生成" | 阶段 1 仅训练 Q-Former（ITC/ITM/ITG）；阶段 2 附加冻结 LLM 并仅训练投影 + Q-Former |
| 冻结骨干 | "不微调" | 视觉编码器和 LLM 权重固定；仅训练桥梁 |
| 投影头 | "线性映射到 LLM 维度" | 最终线性层，将 Q-Former 输出映射到 LLM 的嵌入维度 |
| Perceiver resampler | "Flamingo 的版本" | 类似的可学习查询交叉注意力，由 Flamingo 在每个层上应用而非作为单一桥梁 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 前身，包含 ITC/ITM/ITG 三元组。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "先对齐后融合"——阶段 1 训练的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 指令感知的 Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅投影器方法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学习查询交叉注意力的通用架构。
