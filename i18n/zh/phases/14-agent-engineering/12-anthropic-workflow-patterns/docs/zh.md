# Anthropic 的工作流模式：简单优于复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）与智能体（动态工具使用）。五种工作流模式覆盖了大多数场景。从直接 API 调用开始，只有在步骤无法预测时才引入智能体。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环）
**时间：** 约 60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：提示链、路由、并行化、编排器-工作者、评估器-优化器。
- 解释智能体与工作流的差异及各自的工程成本。
- 判断何时选择工作流而非智能体（反之亦然）。
- 针对脚本化 LLM，用标准库实现全部五种模式。

## 问题所在

团队常常为只需一次函数调用的问题去套用多智能体框架。这种选择的代价是真实的：框架增加了层层抽象，模糊了提示词，隐藏了控制流，并诱发了过早的复杂度。Schluntz 和 Zhang 在 2024 年 12 月的这篇文章是该行业引用最多的反驳观点：从简单起步，仅在复杂度证明其价值时才引入它。

## 概念

### 工作流 vs 智能体

- **工作流。** 通过预定义代码路径编排 LLM 和工具的协作。工程师掌控整个图结构。
- **智能体。** LLM 动态地指挥自己的工具并执行自己的步骤。模型掌控整个图结构。

两者各有适用场景。工作流成本更低、速度更快、更易调试。智能体可以解决开放式问题，但失败模式更难推理。

### 增强的 LLM

五种模式的共同基础：一个具备三种能力的 LLM——搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用都可以使用这些能力。

### 五种模式

1. **提示链（Prompt Chaining）。** 第 1 次调用的输出作为第 2 次调用的输入。适用于任务有清晰线性分解的场景。步骤之间可加入程序化门控。

2. **路由（Routing）。** 一个分类 LLM 决定调用哪个下游 LLM 或工具。适用于类别截然不同的输入需要不同处理方式的情况（如一级客服、退款、 Bug、销售）。

3. **并行化（Parallelization）。** 并发运行 N 次 LLM 调用，聚合结果。两种形式：分块（不同片段）和投票（相同提示，N 次运行，取多数/合成）。

4. **编排器-工作者（Orchestrator-Workers）。** 一个编排器 LLM 动态决定运行哪些工作者（也是 LLM），并合成它们的输出。与智能体循环类似，但编排器不会无限循环。

5. **评估器-优化器（Evaluator-Optimizer）。** 一个 LLM 提出答案，另一个 LLM 对其进行评估，迭代直到评估通过。这是对自我精炼（Lesson 05）的泛化。

### 工作流胜过智能体的场景

- **可预测的任务。** 如果你能枚举步骤，就应当这么做。
- **成本受限的任务。** 工作流步骤数有界；智能体可能陷入螺旋。
- **合规受限的任务。** 审计人员希望阅读图结构，而不是从轨迹中推断它。

### 智能体胜过工作流的场景

- **开放式研究。** 下一步取决于上一步的返回结果。
- **变长任务。** 耗时数分钟到数小时、步骤数未知的任务。
- **新领域。** 当你还不确定哪种工作流合适时——先探索，后固化。

### 上下文工程的互补

"Effective context engineering for AI agents"（Anthropic，2025）将相邻领域形式化：20 万 token 窗口是一个预算，而不是容器。该讲什么、何时压缩、何时让上下文增长，将在 Phase 14 关于上下文压缩的课程中详细展开（在本课程重新编号之前的 Phase 14 较早课时 06）。

```figure
workflow-chain
```

## 动手实现

`code/main.py` 针对 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合结果。
- `orchestrator_workers(task, workers)` — 编排器选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直至通过。

运行方式：

```
python3 code/main.py
```

每种模式都会打印其执行轨迹。每种模式的代码约 10–15 行；而一个框架的成本则是数千行。

## 如何使用

- 大多数任务直接使用 API 调用。
- 仅在模式确实需要持久化状态（LangGraph）、Actor 模型并发（AutoGen v0.4）或角色模板化（CrewAI）时才使用框架。
- 当你需要 Claude Code 的编排器形态但不想自行构建时，使用 Claude Agent SDK。

## 交付物

`outputs/skill-workflow-picker.md` 根据给定任务描述选择最合适的工作流模式，包括决策依据和当工作流不足以胜任时重构为智能体的路径。

## 练习

1. 为路由添加置信度阈值。低于阈值则升级给人工。在一級客服场景中，阈值应该设在哪里？
2. 为 `parallel_vote` 添加超时。当一个调用挂起时会发生什么？如何在有缺失投票的情况下聚合？
3. 将 `evaluator_optimizer` 改造成贪心算法：跨迭代保留前 2 名输出，避免一个后期出现的劣质结果覆盖掉后期出现的优质结果。
4. 将提示链与路由结合：一个路由器选择三条链中的一条。衡量其 token 成本与单一大型提示方案之间的差异。
5. 选择你生产环境中的一个功能，画出工作流图，统计步骤数。智能体在这里真的会更好吗？

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|--------------|---------|
| Workflow | "预定义的流程" | 由工程师掌控的 LLM 和工具调用图 |
| Agent | "自主 AI" | 由模型掌控的图；动态工具调度 |
| Augmented LLM | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；基本原子单元 |
| Prompt chaining | "顺序调用" | 第 N 次调用的输出作为第 N+1 次调用的输入 |
| Routing | "分类器分发" | 决定由哪条链/哪个模型处理输入 |
| Parallelization | "扇出" | N 次并发调用；按分块或投票聚合 |
| Orchestrator-workers | "调度智能体" | 编排器 LLM 动态选择专家 LLM |
| Evaluator-optimizer | "提议者 + 评判者" | 迭代至评估通过；自我精炼的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (2024 年 12 月)](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 互补领域
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 何时状态化图值得其成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 编排器-工作者模式的工程化产品
