# Flamingo 与门控交叉注意力用于小样本 VLM

> DeepMind 的 Flamingo（2022）做了两件前所未有的事。它证明了一个模型可以处理任意交错的图像、视频和文本序列；它还证明了 VLM 可以学会上下文学习——给出包含三个（图像、描述）对的小样本提示，模型就能为一张新图像生成描述，无需任何梯度更新。其机制是：门控交叉注意力层，插入到冻结 LLM 的既有层之间，配有一个从零点起始的 `tanh` 门，以便在初始化时保留 LLM 的纯文本能力。本课将讲解 Flamingo 的 Perceiver 重采样器和门控交叉注意力架构——它们是 Gemini 交错输入和 Idefics2 视觉标记的祖先。

**类型：** Learn
**语言：** Python（stdlib、门控交叉注意力 + Perceiver 重采样器演示）
**前置要求：** Phase 12 · 03（BLIP-2 Q-Former）
**时长：** ~120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 `tanh(gate) = 0` 在初始化时保留冻结 LLM 的文本能力。
- 逐步理解 Perceiver 重采样器：N 个图像 patch → 通过交叉注意力得到 K 个固定"潜在"查询向量。
- 描述 Flamingo 如何处理图像-文本交错序列，并使用尊重图像位置的因果掩码。
- 复现小样本多模态提示结构（3 组图像-描述示例，后接一个查询图像）。

## 问题所在

BLIP-2 将 32 个视觉 token 送入冻结 LLM 的输入层。这对每个提示只用一张图像的情况效果不错。但如果你想把*多张*图像与文本交错输入，例如"这是图像 A，描述它；这是图像 B，描述它；现在这是图像 C，描述它"？LLM 的自注意力就需要在一个流中同时处理图像 token 和文本 token，而"哪些位置能注意力到哪些图像"会变得非常棘手。

Flamingo 的答案是：完全不改动 LLM 的输入流。在既有 LLM 模块之间插入额外的交叉注意力层。文本 token 仍然像往常一样流过 LLM 的因果自注意力；在每 M 个 LLM 模块之间，文本 token 还会通过一个新的门控层交叉注意力到图像特征。该门控（初始化为零）意味着在第一步时，新层等价于恒等变换——模型行为与预训练 LLM 完全相同。随着训练推进，门控逐渐打开，视觉信息开始流入。

Flamingo 回答的第二个问题是：如何处理每个提示中数量不等的图像（0、1 或多张）？Perceiver 重采样器——一个小巧的交叉注意力模块，接收任意数量的 patch，产出固定数量的视觉潜在 token。LLM 交叉注意力层无论提示中包含多少张图像，都看到相同的形状。

## 概念详解

### 冻结的 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 出发。全部 70B 权重均不更新。既有文本自注意力和 FFN 正常运行。

### Perceiver 重采样器

对于提示中的每张图像，ViT 产出 N 个 patch token。Perceiver 重采样器拥有 K 个固定的可学习潜在向量（Flamingo 使用 K=64）。每个重采样器模块由两个子步骤组成：

1. **交叉注意力**：K 个潜在向量对 N 个 patch token 进行注意力（Q 来自潜在向量，K/V 来自 patch）。
2. **潜在向量内部的自注意力 + FFN**。

经过 6 个重采样器模块后，输出为 K=64 个维度 1024 的视觉 token，无论 ViT 产生了多少 patch。224×224 图像（196 个 patch）和 480×480 图像（900 个 patch）都会以 64 个重采样器 token 结束。

对于视频，重采样器沿时间维度应用：每帧的 patch 产生 64 个潜在向量，时间位置编码让模型能够区分 t=0 与 t=N。整个视频变成 T × 64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力模块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个从零初始化的可学习标量。
- `tanh(0) = 0`，因此初始化时门控分支贡献为零。
- 随着 `alpha` 偏离零点，交叉注意力贡献平滑增长。
- 残差连接意味着即使门控完全打开，也不会覆盖 LLM 的文本表示；只是在之上叠加视觉信息。

这是 Flamingo 中最重要的设计选择：视觉条件注入是加法式的、带门控的，且在初始化时为零。初始化阶段的 Flamingo 在处理纯文本输入时就是一个完美的 Chinchilla 70B。

### 用于交错输入的掩码交叉注意力

在一个形如 `<image A> 描述A <image B> 描述B <image C> ?` 的提示中，每个文本 token 只能看到序列中出现在它之前的图像。交叉注意力掩码确保：位置为 `t` 的文本 token 只能注意力到图像索引 `i < i_t` 的图像重采样器 token，其中 `i_t` 是位置 `t` 之前最近的图像索引。"只看到最近的前序图像"和"看到所有前序图像"都是合理的选择；Flamingo 选择了前者。

### 上下文小样本学习

一个 Flamingo 提示看起来像：

```
<image1> 一张猫的照片。<image2> 一张狗的照片。<image3> 一张
```

模型看到补全模式后输出"鸟"（或图像 3 所示的任何内容）。无需梯度步骤。冻结 LLM 的上下文学习能力通过门控交叉注意力得以延续——这是论文的核心结论，也是其意义所在。

### 训练数据

Flamingo 在三个数据集上训练：

1. 多模态 MassiveWeb（M3W）：4300 万网页，含交错图像与文本，按阅读顺序重建。
2. 图像-文本对（ALIGN + LTIP）：44 亿对。
3. 视频-文本对（VTP）：2700 万短视频片段。

OBELICS（2023）是交错网页语料的开源复现版本，Idefics、Idefics2 以及多数开源的"类 Flamingo"模型均基于此训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开源复现版。架构完全一致（Perceiver 重采样器 + 在冻结 LLaMA 或 MPT 上的门控交叉注意力）。有 3B、4B、9B 检查点。因基础 LLM 更小、数据更少，质量不及 Flamingo。

Otter（2023）在 OpenFlamingo 基础上，利用 MIMIC-IT（一组多模态指令数据集）进行指令微调，证明门控交叉注意力同样适用于指令遵循。

### 后辈模型

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力演进线，逐步简化（Idefics2 放弃了重采样器，改用直接 patch token + 自适应池化）。
- Flamingo 到 Chameleon 的过渡：到 2024 年，许多团队转向早期融合（第 12.11 课）；Flamingo 式的门控交叉注意力仍保留在需要冻结主干模型的生产场景中。
- Gemini 的交错输入：概念上继承了 Flamingo 交错格式的灵活性，尽管具体机制是专有的。

### 与 BLIP-2 的比较

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接器 | 输入处一次性使用 Q-Former | 每隔 M 层使用门控交叉注意力 |
| 视觉 token | 每张图像 32 个 | 每张图像每层交叉注意力 64 个 |
| 冻结 LLM | 是 | 是 |
| 上下文小样本 | 弱 | 强——论文核心卖点 |
| 交错输入 | 无原生支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数数量 | 1.88 亿可训练 | ~100 亿可训练（交叉注意力层） |
| 计算量 | 8 卡 A100 数天 | 数千块 TPUv4 数周 |

预算有限的单图 VQA 选 BLIP-2。需要交错、小样本或多图推理则选 Flamingo/Idefics2。

```figure
cross-attention-fusion
```

## 动手实践

`code/main.py` 演示了以下内容：

1. 在 36 个假 patch token 上使用 8 个可学习潜在向量的 Perceiver 重采样器（纯 Python 交叉注意力实现）。
2. 一个门控交叉注意力步骤：`alpha = 0` 时输出等于输入（LLM 不变），随后 `alpha = 2.0` 时混合视觉贡献。
3. 一个交错掩码构建器，为 `（图像 1）（文本 1）（图像 2）（文本 2）` 序列生成二维注意力掩码。

## 交付成果

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开源 VLM 的配置（是否有重采样器、交叉注意力频率、门控方案），它能识别出 Flamingo 谱系中的元素并解释冻结策略。适用于调试微调后文本性能下降的问题（答案通常是：门控开得太快太宽）。

## 练习

1. 计算 Flamingo-9B 的视觉参数数量：9B LLM + 14 亿门控交叉注意力层 + 6400 万重采样器。可训练参数占总参数的比例是多少？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。通过实验展示当 `alpha=0` 时，初始化时刻 `y==x` 严格成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解它们在每个提示包含不同图像数量时如何处理批次中的多张图像。描述其填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让一个文本 token 只能注意力到*最近的前序图像*，而非所有前序图像？阅读 Flamingo 论文第 2.4 节并解释其中的权衡。

5. 上下文小样本：构造一个提示，包含 4 个"图像 → 主物体颜色"的示例，用于一个新的 Flamingo 变体。描述随着示例数量从 0 增加到 8，预期准确率的变化模式。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| Perceiver 重采样器 | "固定潜在向量的交叉注意力" | 将可变数量的输入 patch 转换为 K 个固定 token 的模块 |
| 门控交叉注意力 | "tanh 门控桥接" | 残差层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始化值为 0 |
| 交错输入 | "混合序列" | 图像和文本按阅读顺序自由混合的提示格式 |
| 冻结 LLM | "不更新 LLM 梯度" | 文本 LLM 的权重不更新；只有重采样器和交叉注意力层被训练 |
| 小样本 | "上下文示例" | 在提示中给出若干（图像、答案）对；模型无需微调即可泛化 |
| OBELICS | "交错网页语料库" | 包含 1.41 亿网页的开源数据集，图像和文本按阅读顺序排列 |
| Chinchilla | "70B 冻结基座" | Flamingo 的冻结文本 LLM，源自 DeepMind 的 Chinchilla 论文 |
| 门控调度 | "alpha 如何变化" | 训练过程中交叉注意力门控的打开速率 |
| 交叉注意力频率 | "每隔 M 层" | 门控交叉注意力模块的插入频率；Flamingo 使用 M=4 |
| OpenFlamingo | "开源复现" | MosaicML/LAION 的开源检查点，3-9B；架构与 Flamingo 完全一致 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 经过指令微调的 Flamingo 后辈。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代化简化。
