# Supervisor / Orchestrator-Worker 模式

> 一个主控代理负责规划和委派；专门的工人在并行上下文中执行并向主控汇报。这是 Anthropic Research 系统的核心模式（Claude Opus 4 作为主控，Sonnet 4 作为子代理），在内部研究评测中较单代理 Opus 4 提升了 +90.2%。Anthropic 的工程博客指出，BrowseComp 上 80% 的方差仅由 token 使用量解释——多代理胜出的主要原因是每个子代理都获得了独立的上下文窗口。本课从基础原语构建 supervisor 模式，并涵盖 2026 年生产部署中的工程经验。

**类型：** 学习 + 构建
**语言：** Python（stdlib，`threading`）
**前置条件：** Phase 16 · 04（Primitive Model）
**时间：** 约 75 分钟

## 问题

研究是单代理系统典型地失败的任务类型。你问"2023 到 2026 年间多代理系统发生了什么变化？"一个单代理按顺序阅读五篇论文，用其一半上下文填充论文文本，然后必须同时对所有论文进行推理。等到读到第五篇时，它已经忘记了第一篇的内容。它无法并行化。

supervisor 模式解决了这个问题：一个主控代理规划搜索，将每个子问题委派给工人，然后综合结果。每个工人为一个狭窄的问题获得自己的 200k token 窗口。主控从不查看原始论文——只看工人摘要。

Anthropic 的生产 Research 系统在内部研究评测中比单 Opus 4 报告了 +90.2% 的提升。同一篇博客指出，BrowseComp 80% 的方差仅由 *token 使用量* 解释。每个子代理获得全新的上下文是主要机制。

## 概念

### 该模式

```
                 ┌──────────────┐
                 │   Lead       │  规划、分解、
                 │  (Opus 4)    │  综合
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
        全新        全新         全新
        上下文      上下文       上下文
```

主控从不读取原始材料。工人在主控综合之前互不查看彼此的工作。每条箭头是一个带有狭窄产物的交接。

### 为何能取胜

三个机制：

1. **每个子代理的新上下文。** 探索"FIPA-ACL 传承"的工人不会携带主控规划花费的 40k token。它为一个问题获得 200k 窗口。
2. **通过提示专业化。** 主控的提示是"分解和综合"，而非"研究"。每个工人的提示很窄："找出 X 中发生了什么变化。"聚焦的提示产生聚焦的输出。
3. **并行化。** 工人并发运行。实际耗时约为 `max(worker_times) + plan + synthesis`，而非 `sum(worker_times)`。

### 工程经验（Anthropic 2025）

Anthropic 博客列出了几条至今仍与 2026 年相关的生产经验：

- **根据查询复杂度调整工作量。** 简单查询：一个代理，3-10 次工具调用。复杂查询：10+ 代理。主控必须自己估算这个，而非调用方。
- **先广后深。** 先将问题分解为广泛的子问题，然后根据答案是否需要深度为每个子问题生成更多工人。
- **彩虹部署。** 代理是长运行且带状态的。传统的蓝绿部署不适用。Anthropic 使用彩虹部署：新版本的渐进式发布，同时旧版本逐渐退役。
- **Token 使用占主导。** 多代理的 token 用量约为单代理的 15 倍。仅在任务价值能证明成本合理时才使用。

### 图原生转向

LangGraph 最初提供了一个 `langgraph-supervisor` 库和一个高级的 `create_supervisor` 辅助函数。2025 年 LangChain 将推荐改为通过工具调用直接实现 supervisor 模式，因为工具调用给你更多对 *主控所见内容* 的控制（上下文工程）。该库仍然可用；文档现在推荐工具调用形式。

### 失效模式

- **主控幻觉出错误计划。** 如果主控生成的子问题不能分解真正的问题，工人会在错误目标上进行精确研究。
- **工人过度探索。** 没有明确的范围边界，工人会偏离其分配的子问题并污染综合步骤。
- **综合冲突。** 两个工人返回矛盾的事实。主控必须要么重新询问（增加一轮），要么明确记录分歧。悄悄选择一个方面是最差的失败：用户永远不会知道发生了分歧。

### 何时 supervisor 是错误的

- **顺序任务。** 如果步骤 2 字面上需要步骤 1 的输出，并行化毫无益处。使用管道（CrewAI Sequential、LangGraph 线性图）。
- **简单查询。** 单代理处理它们更快更便宜。在生成工人前使用主控的"调整工作量"检查。
- **严格确定性。** Supervisor 使用 LLM 选择的委派。当审计/回放比适应性更重要时，静态图更好。

```figure
supervisor-hierarchy
```

## 构建它

`code/main.py` 使用 `threading` 实现了包含三个并行工人的 supervisor。主控将查询分解为子问题，工人在每个子问题上并发运行，主控进行综合。没有使用真正的 LLM——工人被脚本化为模拟 fetch-and-summarize。

关键结构：

- `Lead.plan(query)` 将查询拆分为 3 个子问题。
- `Worker.run(sub_q)` 返回一个假摘要（在生产中可以是任何使用工具的代理）。
- `Lead.run(query)` 在线程中启动工人，等待并综合。

运行：

```
python3 code/main.py
```

输出显示计划、带有开始/结束时间戳的并行工人追踪，以及最终综合结果。你可以看到实际耗时的收益：三个 0.3 秒的工人运行约需 0.35 秒，而非 0.9 秒。

## 使用它

`outputs/skill-supervisor-designer.md` 接受用户查询并生成一个 supervisor 模式设计：主控系统提示、工人角色、子问题分解规则和综合模板。在构建新的研究类代理系统之前使用它。

## 部署它

部署 supervisor 模式前的检查清单：

- **模型配对。** 主控使用推理级模型（Opus 类、`o3` 类）。工人使用更快更便宜的模型（Sonnet、`o4-mini`）。
- **工人超时。** 任何超过中位运行时间 2 倍的工人都会被终止；主控要么以较窄范围重新生成，要么跳过它继续。
- **每个工人的 token 上限。** 硬性限制（如预期综合输入的 10 倍）可防止失控的工人耗尽预算。
- **可观测性。** 追踪主控的计划、每个工人的工具调用和综合结果。这是任何事后调试的基础。
- **彩虹部署。** 带状态的长运行代理需要渐进的版本过渡，而非热切换。

## 练习

1. 运行 `code/main.py`，然后将主控修改为生成 5 个工人而非 3 个。观察实际耗时的影响。在这个演示中，多少工人数量时生成开销超过了并行节省？
2. 实现一个工人超时：终止任何运行超过 0.5 秒的工人，并让主控综合剩余结果。你需要什么可观测性才能知道一个工人被截断了？
3. 为主控的综合添加一个冲突检测步骤：如果两个工人返回矛盾的答案，主控记录分歧而非选择其中一个。如何在不调用 LLM 的情况下检测矛盾？
4. 阅读 Anthropic 的 Research 系统工程博客。列出三个这个玩具演示需要在生产中采用的实践。
5. 比较 LangGraph 的 `create_supervisor`（旧版）与新的工具调用推荐。哪个给你更好的对主控所见内容的控制？为什么 Anthropic 明确只传递子答案而非原始工人上下文到综合中？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Supervisor | "主控代理" | 一个负责规划、委派和综合的编排代理。本身不执行工作。 |
| Worker | "子代理" | 一个由 supervisor 调用的聚焦代理，具有狭窄范围和自己的上下文窗口。 |
| Orchestrator-worker | "Supervisor 模式" | 同一件事，不同名称。2026 年文献两种都用。 |
| Fresh context | "干净窗口" | 工人的上下文从其系统提示和分配的问题开始，而非主控的历史。 |
| Rainbow deployment | "渐进式发布" | 长运行带状态代理需要版本化的渐退替换，而非蓝绿部署。 |
| Token dominance | "上下文是变量" | 根据 Anthropic，80% 的研究评测方差来自总 token 使用量，而非模型选择。 |
| Scale effort | "匹配代理数量与复杂度" | 主控估算查询难度，相应地生成 1 个 vs 10+ 个工人。 |
| Synthesis conflict | "工人分歧" | 两个工人返回矛盾的事实；主控必须公开分歧，而非悄悄选择一个。 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor 模式的生产参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 工具调用 supervisor 现在是推荐形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 旧版辅助函数，在 2026 年生产中仍在使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 基于交接的 supervisor 变体
