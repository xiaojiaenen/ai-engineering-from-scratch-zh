# 交接与例行流程 — 无状态编排

> OpenAI 的 Swarm（2024年10月）将多智能体编排精炼为两个原语：**例行流程**（instructions + tools 作为 system prompt）和**交接**（返回另一个 Agent 的工具）。没有状态机，没有分支 DSL——LLM 通过调用正确的交接工具来完成路由。OpenAI Agents SDK（2025年3月）是其生产级继任者。Swarm 本身仍是概念上最清晰的参考——其全部源码仅数百行。该模式之所以流行，是因为 API 表面大致为 'agent = prompt + tools；handoff = 返回 agent 的函数'。局限性：无状态，因此记忆由调用方负责。

**类型：** 学习 + 实践
**语言：** Python（标准库）
**前置知识：** 第 16 阶段 · 04（原始模型）
**时间：** 约 60 分钟

## 问题

每个多智能体框架都希望你学习它的 DSL：LangGraph 的节点与边、CrewAI 的 crew 与 task、AutoGen 的 GroupChat 与 manager。这些 DSL 确实是真正的抽象，但它们让事情变得比实际需要更重。

Swarm 反其道而行之：利用模型已有的工具调用能力。交接变为工具调用。编排器是当前持有对话的那个智能体。状态机隐式存在于智能体的 system prompt 中。

## 概念

### 两个原语

**例行流程。** 定义智能体角色和可用工具的 system prompt。可以将其视为一个作用域内的指令集："你是分诊智能体；如果用户询问退款，则交接给退款智能体。"

**交接。** 智能体可调用的、返回新 Agent 对象的工具。Swarm 运行时检测到 Agent 返回值，并在下一轮切换活跃智能体。

这就是全部抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm 检测到 Agent 返回值 → 切换活跃智能体

triage_agent = Agent(
    name="triage",
    instructions="将用户路由至正确的专家。",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊智能体的 system prompt 使其根据用户消息选择合适的交接。LLM 的工具调用完成路由。

### 为何流行

- **API 简洁。** 只需学习两个概念。
- **利用模型已有的能力。** 工具调用在各厂商已具备生产级质量。
- **无需状态机负担。** 你不需要描述图；智能体的 prompt 描述了它们会交接给谁。

### 无状态的权衡

Swarm 在运行之间明确是无状态的。框架在运行期间保留消息历史，但不持久化任何内容。记忆、连续性、长时任务——全是调用方的问题。

在生产环境中（OpenAI Agents SDK，2025年3月），这是主要变化之一：SDK 在保留交接原语的同时，增加了内置的会话管理、护栏和追踪功能。

### Swarm/交接适用的场景

- **分诊模式。** 一线智能体将用户路由至专家。
- **基于技能的交接。** "如果任务需要写代码，调用编程智能体；如果需要研究，调用研究智能体。"
- **简短、有边界的对话。** 客户支持、FAQ 转工单、简单工作流。

### Swarm 不适用的场景

- **具有共享记忆的长会话。** 交接会将对话状态重置为新智能体的 prompt 加历史。在没有调用方管理记忆的情况下，跨智能体没有持久状态。
- **并行执行。** 交接是串行的——活跃智能体逐个切换。并行需要调用方编排多个 Swarm 运行。
- **审计与回放。** 无状态运行难以精确回放；LLM 的交接选择是非确定性的。

### OpenAI Agents SDK（2025年3月）

生产级继任者增加了：

- **会话状态。** 跨运行的持久线程。
- **护栏。** 输入/输出验证钩子。
- **追踪。** 每次工具调用和交接都会被记录。
- **交接过滤器。** 控制交接时传递什么上下文。

交接原语得以保留；围绕它增加了生产环境所需的便利性。

### Swarm vs GroupChat

两者都使用 LLM 驱动的路由，但在**谁来选择下一个**上存在差异：

- GroupChat：选择器（函数或 LLM）从外部选择下一个发言者。
- Swarm：当前智能体通过调用交接工具来选择其继任者。

Swarm 是"智能体决定下一步"；GroupChat 是"管理器决定下一步"。Swarm 的决策存在于活跃智能体的工具调用中；GroupChat 的决策存在于 `GroupChatManager` 中。

```figure
sw-handoff-routing
```

## 构建它

`code/main.py` 从零实现了 Swarm：一个 Agent dataclass、一个交接机制（工具返回 Agent）和一个检测智能体切换的运行循环。

演示：一个分诊智能体路由至退款、销售或支持专家。每位专家各有自己的工具。运行循环会打印每次交接。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-handoff-designer.md` 为给定任务设计交接拓扑：哪些智能体存在、它们可以调用哪些交接、什么上下文会被传递。

## 交付

检查清单：

- **交接日志。** 每次交接都会写入一条追踪事件，包含来源智能体、目标智能体、上下文快照。
- **上下文传递规则。** 决定交接时传递什么：完整历史（成本高）、最近 N 条消息、或摘要。
- **交接护栏。** 交接给具有不同工具权限的专家时必须经过认证——否则提示注入可以强制发起不需要的交接。
- **循环检测。** 两个智能体来回交接是常见故障；通过简单的最近 K 轮环形检查来检测。
- **备用智能体。** 如果交接目标不存在，则回退到安全的默认值。

## 练习

1. 运行 `code/main.py`，分诊至退款智能体。确认第二轮的活跃智能体是 refund。
2. 添加循环检测规则：如果相同两个智能体连续交接了 3 次，强制退出。设计回退方案。
3. 阅读 OpenAI Agents SDK 关于交接过滤器的文档。实现一个"交接时摘要"版本：出站智能体在入站智能体接管之前将上下文压缩为要点摘要。
4. 比较 Swarm 交接与 GroupChatManager 选择器。哪种模式更容易受到提示注入的影响，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。指出 Swarm 做出的一个明确设计决策，OpenAI Agents SDK 是改变了它还是保留了它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Routine（例行流程） | "智能体的 prompt" | System prompt + 工具列表。定义角色和可用的交接。 |
| Handoff（交接） | "转移给另一个智能体" | 活跃智能体可调用的、返回新 Agent 的工具。运行时切换活跃智能体。 |
| Stateless（无状态） | "运行之间没有记忆" | Swarm 不持久化任何内容；记忆由调用方负责。 |
| Active agent（活跃智能体） | "现在谁在说话" | 当前持有对话的智能体。交接会改变这个状态。 |
| Context transfer（上下文传递） | "交接时传递什么" | 入站智能体能看到什么历史的策略：完整、最近 N 条、或已摘要。 |
| Handoff loop（交接循环） | "智能体来回切换" | 故障模式，两个智能体不断相互交接。 |
| OpenAI Agents SDK | "生产级 Swarm" | 2025年3月继任者；在交接原语之上增加了会话、护栏、追踪。 |
| Handoff filter（交接过滤器） | "转移时的关卡" | SDK 功能，用于在交接边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考性阐述
- [OpenAI Swarm 仓库](https://github.com/openai/swarm) — 原始实现，作为概念参考保留
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 具有会话和追踪的生产级继任者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 子智能体如何通过 `Task` 使用类似交接的模式
