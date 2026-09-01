# Self-Refine 与 CRITIC：迭代式输出改进

> Self-Refine（Madaan 等人，2023）使用一个 LLM 以三种角色——生成、反馈、精炼——在循环中运行。在 7 个任务上的平均提升为 +20（绝对值）。CRITIC（Gou 等人，2023）通过将验证步骤路由到外部工具来强化反馈环节。到 2026 年，这一模式已作为"评估器-优化器"（Anthropic）或护栏循环（OpenAI Agents SDK）出现在每个框架中。

**类型：** Build
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent 循环）、Phase 14 · 03（Reflexion）
**耗时：** 约 60 分钟

## 学习目标

- 阐述 Self-Refine 的三个提示词（生成、反馈、精炼），并解释历史上下文为何对 refine 提示词至关重要。
- 解释 CRITIC 的关键洞见：在没有外部依据的情况下，LLM 在自我验证方面不可靠。
- 实现带有历史上下文的 stdlib Self-Refine 循环，以及可选的外部验证器。
- 将此模式映射到 Anthropic 的"评估器-优化器"工作流和 OpenAI Agents SDK 的输出护栏。

## 问题描述

一个 agent 给出了一个接近正确的答案。可能是一行代码有语法错误，可能是一个摘要过长，也可能是一个计划遗漏了边界情况。你需要的是：让 agent 自我审查其输出，然后修复它。

Self-Refine 表明这种方法可以在单个模型上工作，无需训练数据，也无需 RL。但有一个问题：LLM 在硬事实问题上不擅长自我验证。CRITIC 给出了修复方案——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认模式：生成、验证（尽可能使用外部工具）、精炼、直到验证器通过为止。

## 概念

### Self-Refine（Madaan 等人，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当 feedback 说"没有问题"或预算耗尽时停止。
```

关键细节：`refine` 能看到完整的历史记录——所有先前的输出和批评——因此不会重犯同样的错误。论文进行了消融实验：去掉历史上下文后，质量急剧下降。

核心结论：在 7 个任务（数学、代码、缩略词、对话）上平均获得 +20 的绝对提升（包括 GPT-4）。无需训练，无需外部工具，仅使用单个模型。

### CRITIC（Gou 等人，arXiv:2305.11738，v4 2024年2月）

Self-Refine 的弱点：反馈步骤是一个 LLM 给自己打分。对于事实性声明，这是不可靠的（模型产生的幻觉往往对产生它的模型自身来说看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性声明的搜索引擎
- 用于代码正确性的代码解释器
- 用于算术计算的计算器
- 领域特定的验证器（单元测试、类型检查器、linter）

验证器基于工具结果生成结构化的批评。精炼器则基于此批评进行改写。

核心结论：由于批评具有外部依据，CRITIC 在事实性任务上优于 Self-Refine。在缺少外部验证器的任务（创意写作、格式化）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。在可用时优先使用（单元测试、类型检查器、护栏断言）。
2. **无反馈发出。** 模型说"输出没问题"。成本更低但不可靠；需配合最大迭代次数上限使用。

2026 年的默认做法：将它们组合使用。"当验证器通过 OR 模型说没问题 AND 迭代次数 >= 2 OR 迭代次数 >= 最大迭代次数时停止。"

### 评估器-优化器（Anthropic，2024）

Anthropic 在 2024 年 12 月的文章中将其列为五种工作流模式之一。两个角色：

- 评估器：对输出评分并生成批评。
- 优化器：根据批评修改输出。

循环直到评估器通过。这就是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 增加的关键工程细节是：评估器和优化器的提示词应当显著不同，以防止模型只是简单盖章通过。

### OpenAI Agents SDK 输出护栏

OpenAI Agents SDK 将此模式作为"输出护栏"提供。护栏是一个在 agent 最终输出上运行的验证器。如果护栏触发（抛出 `OutputGuardrailTripwireTriggered`），则拒绝该输出，agent 可以重试。护栏可以调用工具（CRITIC 风格）或作为纯函数（Self-Refine 风格）。

### 2026 年注意事项

- **盖章循环。** 同一个模型用相同风格的提示词同时做生成和批评，会收敛于"看起来不错"。使用结构上不同的提示词，或用一个更小、更便宜的模型来做批评。
- **过度精炼。** 每次精炼都会增加延迟和 token 消耗。限制 1-3 次精炼；超过后应升级给人工审查。
- **在琐碎任务上使用 CRITIC。** 如果没有外部验证器，CRITIC 会退化为 Self-Refine；不要为一个占位验证器支付延迟代价。

```figure
self-refine
```

## 动手实现

`code/main.py` 在一个玩具任务上实现了 Self-Refine 和 CRITIC：根据主题生成简短的项目符号列表。验证器检查格式（3 个项目符号，每项不超过 60 个字符）。CRITIC 增加了一个外部"事实验证器"，对已知幻觉进行惩罚。

组件：

- `generate` —— 脚本化生成器。
- `feedback` —— LLM 风格的自我批评。
- `verify_external` —— CRITIC 风格的有依据验证器。
- `refine` —— 根据历史上下文改写输出。
- 停止条件 —— 验证器通过或最多 4 次迭代。

运行方式：

```
python3 code/main.py
```

比较 Self-Refine 与 CRITIC 的运行结果。CRITIC 捕捉到了一个 Self-Refine 遗漏的事实错误，因为外部验证器拥有自我批评者所缺乏的外部依据。

## 实际使用

Anthropic 的评估器-优化器模式就是用 Claude 友好的语言描述的这个模式。OpenAI Agents SDK 的输出护栏是 CRITIC 形状的（护栏可以调用工具）。LangGraph 提供了一个 reflection 节点，其形态类似于 Self-Refine。Google 的 Gemini 2.5 Computer Use 增加了一个每步安全评估器，这是一种 CRITIC 变体：每个动作在执行前都会被验证。

## 上线部署

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算来配置一个评估器-优化器循环。它会生成用于生成器、评估器/验证器和优化器的提示词，以及一个停止策略。

## 练习

1. 将玩具任务设为 max_iterations=1 运行。CRITIC 还有帮助吗？
2. 将外部验证器替换为一个有噪声的验证器（随机 30% 的假阳性）。循环会怎样表现？这是 2026 年大多数护栏堆栈的现实情况。
3. 实现一个"不同模型上生成器-批评者"的变体：大模型负责生成，小模型负责批评。它是否优于同模型方案？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。列出三类验证工具的类别并为每类各举一例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。该 SDK 哪里做得不对，哪里做得对？

## 关键术语

| 术语 | 人们通常怎么说 | 实际上是什么意思 |
|------|----------------|------------------------|
| Self-Refine | "能自我修复的 LLM" | 单模型内的生成→反馈→精炼循环，附带历史上下文 |
| CRITIC | "工具接地验证" | 用外部验证器（搜索、代码、计算、测试）替代反馈步骤 |
| 评估器-优化器 | "Anthropic 工作流模式" | 两个角色——评估器评分，优化器改写——循环收敛 |
| 输出护栏 | "事后检查" | OpenAI Agents SDK 中在 agent 产生输出后运行的验证器 |
| 验证步骤 | "批评阶段" | 核心决策点：基于外部依据还是自我评分 |
| 精炼历史 | "模型已经尝试过的内容" | 追加到 refine 提示词中的先前输出+批评；去掉后质量骤降 |
| 盖章循环 | "自我认同失败" | 相同提示词的批评返回"看起来不错"；解决方案是使用结构上不同的提示词 |
| 停止条件 | "收敛测试" | 验证器通过 OR 无反馈 AND 迭代上限；绝不能只用单一条件 |

## 延伸阅读

- [Madaan 等人，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) —— 该领域的经典论文
- [Gou 等人，CRITIC（arXiv:2305.11738）](https://arxiv.org/abs/2305.11738) —— 工具接地验证
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 评估器-优化器工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) —— 输出护栏作为 CRITIC 形状的验证器
