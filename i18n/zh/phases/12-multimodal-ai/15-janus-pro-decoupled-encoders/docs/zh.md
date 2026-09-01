# Janus-Pro：解耦编码器用于统一多模态模型

> 统一多模态模型存在不可避免的张力。理解需要语义特征——SigLIP 或 DINOv2 的输出向量富含概念级信息。生成需要便于重建的编码——VQ token 可组合回清晰像素。这两个目标在一个编码器中并不兼容。Janus（DeepSeek，2024年10月）和 Janus-Pro（DeepSeek，2025年1月）认为解决方案是放弃尝试：解耦两个编码器。在任务间共享 transformer 主体，但理解走 SigLIP，生成走 VQ tokenizer。在 7B 参数规模下，Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上与 LLaVA 持平。本课程解读为何两个编码器能在一处失败时取得成功。

**类型：** Build
**语言：** Python（stdlib、双编码器路由 + 共享主体信号）
**前置要求：** 阶段 12 · 13（Transfusion）、阶段 12 · 14（Show-o）
**时长：** 约 120 分钟

## 学习目标

- 解释为何单一共享编码器会牺牲理解或生成质量。
- 描述 Janus-Pro 的路由机制：输入侧使用 SigLIP 特征进行理解，VQ token 用于输入的生成路径和输出的生成路径。
- 追踪使 Janus-Pro 在 Janus 失败之处成功的「数据混排扩展」策略。
- 比较解耦架构（Janus-Pro）、耦合连续架构（Transfusion）和耦合离散架构（Show-o）。

## 问题所在

统一模型在理解和生成之间共享同一个 transformer 主体。此前的尝试（Chameleon、Show-o、Transfusion）都使用一个视觉 tokenizer 同时处理两个方向。这个 tokenizer 是一个妥协方案：

- 针对重建优化（生成）：VQ-VAE 能捕获细粒度像素细节，但产生的 token 语义连贯性较弱。
- 针对语义优化（理解）：SigLIP 嵌入将"cat"图像与"cat" token 聚集在一起，但不支持良好的重建。

Show-o 和 Transfusion 在某一方向上的可见质量损失为这种妥协付出了代价。Janus-Pro 提出：既然任务需求不同，为何要依赖一个 tokenizer？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构将两个编码器分离：

- **理解路径**。输入图像 → SigLIP-SO400m → 2 层 MLP → transformer 主体。
- **生成路径**。输入图像（若对已有图像进行条件生成） → VQ tokenizer → token ID → transformer 主体。
- **输出生成**。transformer 预测的图像 token → VQ 解码器 → 像素。

transformer 主体是共享的。主体上游和下游的一切都是任务特定的。

输入通过提示格式区分：`<understand>` 标签路由到 SigLIP；`<generate>` 路由到 VQ。或者路由由任务隐式决定。

### 为何有效

理解损失接收 SigLIP 特征，CLIP 风格预训练已将其调优至语义相似度。模型在感知基准测试上的表现优于 Show-o / Transfusion，因为输入特征更适合该任务。

生成损失接收 VQ token，tokenizer 已将其调优至重建能力。图像质量优于 Show-o，因为 VQ 编码可干净地组合回像素。

共享 transformer 主体同时接触两种输入分布（SigLIP 和 VQ），并学会同时处理两者。其主张是：足够多的数据 + 足够多的参数，主体能够吸收这种切换开销。

### 数据扩展——Janus 与 Janus-Pro

Janus（初版，arXiv 2410.13848）引入了这种解耦方案，但规模较小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 7B 参数（vs 1.3B）。
- 第一阶段（对齐）使用 90M 图像-文本对，较之前的 72M 提升。
- 第二阶段（统一）使用 72M，较之前的 26M 提升。
- 第三阶段新增 200k 图像生成指令样本。

成果：Janus-Pro-7B 在 MMMU 上与 LLaVA 持平（60.3 vs ~58），并在 GenEval 上超越 DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一谱系的两端均具竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）用整流流生成路径替换了 VQ 生成路径（连续）。拆分变为：理解用 SigLIP + 生成用整流流。质量上限进一步提升。架构仍保持「解耦编码器 + 共享主体」。

### 共享主体的职责

transformer 主体处理统一序列，但面对两种输入分布。其职责是：

- **理解**：消费 SigLIP 特征 + 文本 token → 自回归输出生成文本。
- **生成**：消费文本 token + （可选的图像 VQ token）→ 自回归输出生成图像 VQ token。

主体的每个块中没有模态特定的权重。它是你在 Qwen 或 Llama 内部会预期的那种文本风格 transformer，外加两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这一选择很关键：LLM 贡献的推理能力是纯从头训练的统一模型难以企及的。

### 与 InternVL-U 的比较

InternVL-U（第 12.10 课）是 2026 年的后续工作。它结合了：

- 原生多模态预训练（InternVL3 骨干）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一的 理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入更大的框架。解耦编码器思想现在已成为大规模统一模型的默认方案。

### 局限性

解耦编码器增加了架构复杂度。需要训练两个 tokenizer，维护两条输入路径，应对两套故障模式。对于不需要生成的产品，Janus-Pro 属于过度设计——选择 LLaVA 家族的理解模型即可。

对于不需要理解的产品，Janus-Pro 属于杀鸡用牛刀——选择 Stable Diffusion 3 / Flux 模型即可。

对于需要两者的产品，Janus-Pro 现在是参考开源架构。

```figure
l5-janus-decouple
```

## 实践使用

`code/main.py` 模拟了 Janus-Pro 的路由机制：

- 两个模拟编码器：类 SigLIP（产生 256 维语义向量）和类 VQ（产生整数编码）。
- 一个提示路由器，根据任务标签选择编码器。
- 一个共享主体（占位），无论哪种编码器产生 token，都能处理 token 序列。
- 从第一阶段（对齐）到第三阶段（指令微调）的加权采样调度切换。

打印三个示例的路由路径：图像 QA、文生图、图像编辑。

## 产出交付

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望在前沿级别质量上同时实现统一生成 + 理解的产品，它会选择 Janus-Pro、JanusFlow 或 InternVL-U，并给出具体数据规模建议。

## 练习题

1. Janus-Pro-7B 在 GenEval 上超越 DALL-E 3。解释为何一个 7B 开源模型能在生成上匹敌前沿闭源模型，却在理解上不能。

2. 实现一个路由器函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理像"描述并随后绘制草图"这样的模糊提示？

3. JanusFlow 用整流流替换了 VQ 路径。transformer 主体现在输出什么，损失函数发生哪些变化？

4. 提出一个 Janus-Pro 架构可通过增加一个解耦编码器处理的新任务。示例：图像分割（类 DINO 风格）、深度估计（类 MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据扩展的内容。哪个数据阶段对 T2I 质量提升贡献最大？对比 Janus 而言。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| 解耦编码 | "两个视觉编码器" | 按方向分离 tokenizer 或编码器：理解用语义，生成用重建 |
| 共享主体 | "一个 transformer" | 单个 transformer 处理任一编码器的输出；无模态特定权重 |
| SigLIP 用于理解 | "语义特征" | CLIP 家族视觉塔，提供丰富概念特征但重建能力差 |
| VQ 用于生成 | "重建编码" | 向量量化 token，可干净解码回像素 |
| JanusFlow | "整流流变体" | Janus-Pro 采用连续流匹配生成头而非 VQ |
| 路由标签 | "任务标签" | 提示标记（`<understand>` / `<generate>`）用于选择输入编码器 |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
