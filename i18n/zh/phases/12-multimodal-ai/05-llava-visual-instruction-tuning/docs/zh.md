# LLaVA 与视觉指令微调

> LLaVA（2023年4月）是当今被借鉴最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素 token 拼接替换了 Flamingo 的门控交叉注意力，并使用 GPT-4 从纯文本描述中生成的 15.8 万条视觉指令轮次进行训练。2023 至 2026 年间构建 VLM 的实践者都或多或少实现了 LLaVA 的某个变体。LLaVA-1.5 引入了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 将单图、多图和视频统一到一个配方中。本课将阅读该配方、实现投影器，并解释为何"更简洁反而更胜"。

**类型：** 构建
**语言：** Python（标准库，投影器 + 指令模板构建器）
**先修知识：** 第 12 阶段 · 02（CLIP）、第 11 阶段（LLM 工程——指令微调）
**时间：** 约 180 分钟

## 学习目标

- 构建一个 2 层 MLP 投影器，将 ViT patch 嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 熟悉 LLaVA 的两阶段配方：（1）在 55.8 万条图文对上的投影器对齐，（2）在 15.8 万条 GPT-4 生成的轮次上的视觉指令微调。
- 构建包含图像 token 占位符、系统提示以及用户/助手回合的 LLaVA 格式提示。
- 解释社区为何从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上占优。

## 问题所在

BLIP-2 的 Q-Former（第 12.03 课）将图像压缩为 32 个 token。干净、高效，对基准测试表现良好。但它有两个问题。

首先，Q-Former 是可训练的，但其损失并非最终任务损失。第 1 阶段训练 ITC+ITM+ITG。第 2 阶段训练 LM 损失。查询向量学到了一些中间表示，LLM 随后需要解码它们。信息在瓶颈处流失。

其次，Q-Former 占用 1.88 亿参数，而在 LLaVA 2023 年的规模下，你必须与目标 LLM 共同设计它。更换 LLM，重新训练 Q-Former。更换视觉编码器，重新训练。每种组合都是一个独立的研发项目。

LLaVA 的答案简单得令人尴尬：取 ViT 的 576 个 patch token，让它们通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后将全部 576 个 dump 进 LLM 的输入序列。没有瓶颈，没有基于奇怪目标的第 1 阶段预训练，直接用 LM 损失训练 MLP。

数据从何而来？LLaVA 的第二个洞察：使用 GPT-4（纯文本版）生成指令数据。将 COCO  caption 和边界框数据喂给 GPT-4，让它生成对话、描述和复杂推理问题。免费获得 15.8 万条指令-回复轮次，无需人工标注。

结果：一个在 8 块 A100 上训练一天的 VLM，在 MMMU 上击败了 Flamingo，并发布了一个开源 checkpoint，社区可以在此基础上扩展。到 2023 年底，已涌现出 50 多个分支。

## 概念解析

### 架构

LLaVA-1.5 在 13B 规模下：
- 视觉编码器：CLIP ViT-L/14 @ 336（第 1 阶段冻结，第 2 阶段可选择解冻）。
- 投影器：带 GELU 激活的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后期使用 Llama-3.1-8B）。

图像 + 文本提示的前向流程：

```
img -> ViT -> 576 个 1024 维的 patches
patches -> MLP -> 576 个 4096 维的 tokens
提示：system + "<image>" 占位符 + 用户问题
将 <image> token 替换为 576 个投影后的 tokens
将完整序列输入 LLM
解码回复
```

图像占据 LLM 上下文中的 576 个 token。在 2048 上下文长度下，还剩 1472 个 token 供文本使用。在 32k 上下文下，这只是一个微不足道的开销。

### 第 1 阶段：投影器对齐

冻结 ViT，冻结 LLM，仅训练 2 层 MLP。数据集：55.8 万条图像-描述对（LAION-CC-SBU）。损失：在投影图像 tokens 条件下对 caption 进行语言建模。

在 batch size 128 下，一个 epoch 只需数小时即可完成。投影器学会了将 ViT 空间映射到 LLM 空间，无需任务特定的监督。

### 第 2 阶段：视觉指令微调

解冻投影器（仍保持可训练）。解冻 LLM（通常完全解冻，有时使用 LoRA）。在 15.8 万条视觉指令轮次上训练。

指令数据是关键。Liu 等人通过以下步骤生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工 caption + 边界框列表）。
3. 使用三种提示模板发送给 GPT-4：
   - 对话："生成用户与助手之间关于该图像的来回对话。"
   - 详细描述："给出丰富详细的图像描述。"
   - 复杂推理："提出一个需要推理图像的问题，然后回答它。"
4. 将 GPT-4 的输出解析为（指令，回复）对。

这些步骤都不直接触及图像本身——仅使用文本描述。GPT-4 会"幻觉"出合理的图像内容。有一些噪声，但奏效了：15.8 万条轮次足以解锁对话能力。

### 社区为何大量借鉴此方法

- 无需调整第 1 阶段的特定损失函数，全程使用 LM 损失。
- 投影器训练只需数小时，而非数天。
- LLM 可替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重新训练投影器。
- 视觉指令数据流水线使用 GPT-4，针对新领域可低成本重新生成。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023年10月）新增了：
- 学术任务数据（VQA、OKVQA、RefCOCO）混合进指令微调。
- 更好的系统提示。
- 上下文从 2048 扩展到 32k。

LLaVA-NeXT（2024年1月）新增了：
- AnyRes：将高分辨率图像拆分为 2x2 或 1x3 网格的 336x336 裁片，再加一个全局低分辨率缩略图。每个裁片成为 576 个 token；每图总计约 2880 个视觉 token。OCR 和图表任务显著提升。
- 更好的指令数据混合，使用 ShareGPT4V（高质量 GPT-4V caption）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课详细介绍了 OneVision。简言之：相同的投影器，但使用课程学习训练，在一个模型中覆盖单图、多图和视频，共享视觉 token 预算。

### 与 Q-Former 的对比

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每图视觉 token 数 | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 1.88 亿 + LM | 4000 万 + LM |
| 第 1 阶段损失 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需重新训练 | 替换后最小化重训即可 |
| 多图支持 | 别扭 | 自然（拼接） |
| 视频支持 | 别扭 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简洁性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是瓶颈（LLM 上下文增长到 32k-128k+），简洁性主导。

### 提示格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位 token。在分词前，它会被替换为 576 个视觉 token（AnyRes 时为 2880 个）。分词器看到的是比训练时稍长的序列，但 LLM 能够处理这个新颖输入，因为第 1 阶段已教会它这样做。

### 参数量经济账

LLaVA-1.5-7B 的参数分解：
- CLIP ViT-L/14 @ 336：3.03 亿（第 1 阶段冻结，第 2 阶段通常解冻）。
- 投影器（2 个线性层）：约 2200 万可训练参数。
- Llama-7B：70 亿。
- 总计：73 亿参数。第 2 阶段可训练：完整 70 亿 + 2200 万投影器。

第 2 阶段训练成本：8xA100 上约 20 小时。这是关键数字——一天、一个节点、可复现。这正是 LLaVA 得以广泛传播的原因。

```figure
mm-llava-projector
```

## 动手实践

`code/main.py` 实现了：

1. 纯 Python 实现的 2 层 MLP 投影器（为了 toy 规模使用 dim 16 → 32 → 32）。
2. 提示构建流水线：系统提示 + `<image>` 替换为 N 个投影 token + 用户回合 + 助手生成占位符。
3. 可视化 576-token 视觉块在 LLM 上下文中的占比（消耗 2k / 32k / 128k 上下文的百分比）。

## 交付成果

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列 checkpoint，它会运行一个包含 10 个提示的 vibes-eval 套件（3 个 captioning、3 个 VQA、2 个推理、2 个拒绝），并输出人类可读的评分表。这不是基准测试；而是一个冒烟测试，用于确认投影器和 LLM 连接良好。

## 练习题

1. 计算 2 层 MLP 投影器在 `1024 → 4096 → 4096` 下的可训练参数数量。若含 GELU 和 bias，它占 LLaVA-13B 的比例是多少？

2. 为"拒绝"场景构建一个 LLaVA 提示——图像中包含私密人物。写出预期的助手回复。为何 LLaVA 应在零样本下拒绝此类请求？需要什么训练数据来强化这种拒绝行为？

3. 阅读 LLaVA-NeXT 博客中的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉 token 数量。与 336x336 基础模型的 576 token 对比。

4. LLaVA 第 1 阶段投影器使用 caption 上的 LM 损失训练。若跳过第 1 阶段直接进入第 2 阶段（视觉指令微调），会发生什么？引用 Prismatic VLMs 的消融实验（arXiv:2402.07865）给出答案。

5. LLaVA-Instruct-150k 使用 GPT-4 配合 COCO caption 生成指令。针对新领域（医疗 X 光片、卫星图像），描述生成领域指令的四步数据流水线。每一步可能出现什么问题？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 投影器 | "MLP 桥接器" | 带 GELU 的 2 层 MLP，将 ViT 维度映射到 LLM 维度 |
| 图像 token | "<image> 占位符" | 推理前被 N 个投影视觉 token 替换的提示标记 |
| 视觉指令微调 | "LLaVA 第 2 阶段" | 在 GPT-4 生成的（图像，指令，回复）三元组上训练 |
| 第 1 阶段对齐 | "投影器预训练" | 冻结 ViT 和 LLM，使用 caption 上的 LM 损失训练投影器 |
| AnyRes | "多裁片拼接" | 将高分辨率图像拆分为平铺网格，拼接每个裁片的视觉 token |
| LLaVA-Instruct | "GPT-4 生成" | 从 COCO caption + GPT-4 合成的 15.8 万条指令-回复对 |
| 视觉编码器冻结 | "骨干网络锁定" | CLIP 权重在第 1 阶段不更新，第 2 阶段有时也不更新 |
| ShareGPT4V | "更好的 caption" | 由 GPT-4V 生成的 100 万条密集描述，用于更高质量的对齐 |
| VQA | "视觉问答" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 的消融研究，系统性地测试投影器和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 原论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集描述数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一的单图、多图、视频模型。
