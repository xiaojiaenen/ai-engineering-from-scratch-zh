# 开放权重 VLM 配方：哪些真正重要

> 2024-2026 年的开放权重 VLM 文献是一座消融实验表组成的森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合的组合。Allen AI 的 Molmo 证明详细的人工标注比 GPT-4V 蒸馏更优。Cambrian-1 运行了 20+ 种编码器对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在控制基准上比较了 27 种训练配方。在所有噪音中，一组结果跨论文保持一致：图像编码器比连接器架构更重要，数据混合比两者都更重要，详细的人工标注优于蒸馏合成数据。这篇笔记帮你读懂这些表格，你就不用再读了。

**类型：** 学习 + 实验
**语言：** Python（标准库、消融表解析器 + 配方选择器）
**前置条件：** Phase 12 · 05（LLaVA 基线）
**时间：** 约 180 分钟

## 学习目标

- 说出五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表并预测哪个旋钮能推动给定基准。
- 给定计算预算和任务组合，为新的 VLM 选择配方（编码器、连接器、数据、分辨率）。
- 解释为什么详细人工标注在同 token 数下优于 GPT-4V 蒸馏。

## 问题所在

存在数百个开放权重 VLM。大多数"好"与"最强"之间的差距不是架构，而是数据、分辨率调度和编码器选择。当你的模型表现不佳时知道该先调哪个旋钮，能避免一个价值 500 万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）基于 caption-pair 预训练 + LLaVA-Instruct-150k 运行。良好的基线。在 MMMU 上止步于 35% 左右。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）运行了 exhaustive ablations。结果出人意料且实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）命名了各轴：

1. **图像编码器。** CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上有所不同。
2. **连接器。** MLP（2-4 层）、Q-Former（32 个 query + cross-attn）、Perceiver Resampler（64 个 query）、C-Abstractor（卷积 + 双线性池化）。
3. **语言模型。** Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 规模是主要的参数成本。
4. **训练数据。** Caption pairs（CC3M、LAION）、交错数据（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. **分辨率调度。** 固定 224/336/448、AnyRes、原生动态。训练中递增或保持不变。

每个生产级 VLM 都会在各轴上做出选择。MMMU 分数的方差主要由第 1、4、5 轴解释——而非你选择了哪种连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节显示：将 CLIP ViT-L/14 换为 SigLIP SO400m/14 带来了 3+ 分的 MMMU 提升。将连接器从 MLP 换为 Perceiver Resampler 增加不到 1 分。Idefics2 复现了相同结论：SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver（在相同 token 数下）。

Cambrian-1 的"Cambrian Vision Encoders Match-Up"（Tong 等，2024）在视觉中心基准（CV-Bench）上运行了 20+ 种编码器。排行榜顶部由 DINOv2 和 SigLIP 混合占据；CLIP 处于中游；ImageBind 和 ViT-MAE 较低。CLIP ViT-L 与 DINOv2 ViT-g/14 在 CV-Bench 上的差距约为 5-7 分。

2026 年开放 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于语义 + 密集特征；必要时可与 DINOv2 ViT-g/14 特征拼接（Cambrian 的"Spatial Vision Aggregator"即采用此方案）。

### 轴 2：连接器设计差异不大

MM1、Idefics2、Prismatic 和 MM-Interleaved 均得出相同结论：在固定视觉 token 数下，连接器架构几乎无影响。基于 mean-pooled patch 的 2 层 MLP 与 32-query Q-Former 在相同 token 预算下性能差距小于 1 分。

真正重要的是 token 数量。更多视觉 token = 更多 LLM 计算 = 更好的性能直到某一点，随后边际收益递减。每张图 64 个 token 对 OCR 来说太少。576-1024 个 token 是大多数开放 VLM 的最佳区间。2048+ 仅在文档和图表场景有增益。

Q-Former vs MLP 是成本问题，而非质量问题：Q-Former 无论图像分辨率如何都将 token 限制在 32-64；MLP 输出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率，差异可以忽略。

### 轴 3：LLM 规模决定上限

将 LLM 从 7B 翻倍到 13B 在各种 VLM 论文中稳定带来 2-4 分的 MMMU 提升。达到 70B 时你几乎在所有基准上饱和。VLM 的多模态推理上限即 LLM 的文本推理上限——视觉编码器只能为其提供输入，无法替其推理。

这就是 Qwen2.5-VL-72B 和 Claude Opus 4.7 碾压 MMMU-Pro 和 ScreenSpot-Pro 的原因：语言大脑足够大。一个 7B VLM 无法通过精巧的连接器设计替代 70B VLM。

### 轴 4：数据——详细人工标注优于蒸馏

Molmo + PixMo（Deitke 等，2024）是 2024 年每人都应阅读的结果。Allen AI 的标注者以 1-3 分钟的密集语音转文字方式描述图像，得到 712K 张密集标注图像。训练数据中无任何 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中的 11 个上击败了 Llama-3.2-90B-Vision。差距不在架构——而在标注质量。详细人工标注每张图片包含 5-10 倍于简短网络标注的信息量，且在 GPT-4V 蒸馏发生幻觉时仍保持事实准确性。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）遵循了相同的策略，使用混合人工 + GPT-4V 标注。趋势很清晰：对于 2026 前沿，标注密度 > 标注数量 > 蒸馏便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融：384 → 448 带来 1-2 分提升。448 → 980（配合图像分块 AnyRes）在 OCR 基准上额外带来 3-5 分。固定分辨率训练在中等精度处 plateau；分辨率递增（从 224 起步，到 448 或原生结束）训练更快且最终更高。

Cambrian-1 运行了分辨率与 token 数量的权衡：在固定计算下，你可以选择更多 token 但更低分辨率，或更少 token 但更高分辨率。更高分辨率在 OCR 上胜出；更低分辨率 + 更多 token 在通用场景理解上胜出。

2026 年生产配方：Stage 1 以 384 固定训练，Stage 2 对 OCR 密集型任务使用动态分辨率最高至 1280。

### Prismatic 控制对比

Prismatic VLMs（Karamcheti 等，2024）是控制所有轴的论文。相同的 13B LLM、相同的指令数据、相同评估——仅一次改变一个轴。结果：

- 每张图视觉 token 数解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其他一切（数据混合、调度器、学习率）占剩余约 15%。

这是粗略分解，但它是文献中对"我应该先消融哪个"最干净的答案。

### 2026 配方选择器

基于上述证据，2026 年新项目的默认开放 VLM 配方：

- **编码器：** 原生分辨率下的 SigLIP 2 SO400m/14 + NaFlex；如需分割/定位则拼接 DINOv2 ViT-g/14 密集特征。
- **连接器：** 作用于 patch token 的 2 层 MLP。除非受 token 限制否则跳过 Q-Former。
- **LLM：** Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于成本控制，70B 用于质量，根据目标延迟选择。
- **数据：** PixMo + ShareGPT4V + Cauldron，叠加任务特定指令数据。
- **分辨率：** 动态（长边最小 256，最大 1280 像素）。
- **调度：** Stage 1 对齐（仅投影器）、Stage 2 全量微调、Stage 3 任务特定微调。

以上每一项默认值都追溯到本课末尾所列论文中的测量消融实验。

```figure
l5-vlm-recipe-knobs
```

## 动手实践

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表（精简版），允许你查询：

- "给定预算 X 和任务 Y，什么配方获胜？"
- "如果我在 7B Llama 上用 SigLIP 替换 CLIP，预期 MMMU 差值是多少？"
- "对 80% 置信度答案，我应该先消融哪个轴？"

输出为带预期基准差值的排名配方列表，以及"先消融"推荐。

## 交付

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，输出完整配方（编码器、连接器、LLM、数据混合、分辨率调度），并附上论证每项选择所依据的消融引用。避免工程师每次启动新的 VLM 项目时都重新发明 Idefics2 消融表。

## 练习题

1. 阅读 MM1 第 3.2 节。对于预算 50M 图片的固定 2B LLM，哪种编码器获胜？在 13B LLM 下答案会翻转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在视觉中心基准上优于任一单独使用，但在 MMMU 上无增益。预测哪些基准会提升，哪些保持不变。

3. 你的目标是基于 2B LLM 的移动端 UI 智能体。选择编码器、连接器、分辨率和数据混合。用具体消融表论证每项选择。

4. Molmo 提供 4B 和 72B 两种模型。4B 可与闭源 7B VLM 竞争；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这说明了什么关于 LLM 规模 plateau 的假设？

5. 设计一个消融表以在 7B VLM 上将数据混合质量与编码器质量隔离开。最少需要多少次训练运行？提出四个轴的设定值。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------|---------|
| Ablation（消融） | "拧一个旋钮" | 训练多个运行，仅在一个设计空间轴上不同，其余保持不变 |
| Connector（连接器） | "桥" / "投影器" | 将图像编码器输出映射到 LLM token 空间的 trainable 模块（MLP、Q-Former、Perceiver） |
| Detailed human caption（详细人工标注） | "密集标注" | 多句人工撰写的描述（通常 80-300 token），比网络 alt text 信息更丰富 |
| Distillation（蒸馏） | "GPT-4V 标注" | 由更强闭源 VLM 生成的训练数据；方便但易继承幻觉 |
| AnyRes / dynamic res（动态分辨率） | "高分路径" | 通过分块或 M-RoPE 将大于编码器原生分辨率的图像输入的策略 |
| Resolution ramp（分辨率递增） | "课程学习" | 从低分辨率开始并逐步提升的训练调度，加速对齐学习 |
| Vision-centric bench（视觉中心基准） | "CV-Bench / BLINK" | 强调细粒度视觉感知而非重型语言推理的评测 |
| PixMo | "Molmo 的数据" | Allen AI 的 712K 密集标注图像数据集；人工语音转录为密集标注 |

## 延伸阅读

- [McKinzie 等 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等 — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等 — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
