# 有状态图编排 —— 持久化执行与检查点

> Agent 是一个状态机；节点是函数；边是状态转换；每个节点后都会对状态进行检查点保存。在任何故障发生时，可从上一个成功的检查点恢复。LangGraph 是该低层有状态编排模型的 2026 参考实现。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 01（Agent 循环），阶段 14 · 12（工作流模式）
**时间：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带类型化状态的 state machine，函数式节点，条件边，以及节点后的检查点机制。
- 列举文档强调的四大能力：持久化执行、流式输出、人工介入、完整记忆。
- 解释 LangGraph 支持的三种编排拓扑：监督者、对等网络（蜂群）、分层（嵌套子图）。
- 实现一个带类型化状态、条件边、以及检查点/恢复循环的标准库 state graph。

## 问题

Agent 和工作流面临共性问题：当一个 40 步的执行在第 38 步失败时，你希望从第 38 步继续，而不是重新开始。次等的 state 模型让操作者不得不在一个假设从头运行的库上打补丁来实现重试。

LangGraph 的设计答案：state 是一等公民的类型化对象，mutation 是显式的，每个节点后都会持久化检查点。恢复只需调用 `load_state(session_id)`。

## 概念

### 图结构

图的定义包含：

- **State 类型。** 每个节点读取和修改的类型化 dict（或 Pydantic model）。
- **节点。** 纯函数 `(state) -> state_update`。返回后将更新合并到 state 中。
- **边。** 节点间的条件或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个具有 `classify`、`refund`、`bug`、`sales`、`done` 节点的 agent —— 一个作为图的路由工作流。

### 持久化执行

每个节点返回后，运行时将 state 序列化并写入 checkpointer（SQLite、Postgres、Redis 或自定义）。在第 N 步发生故障时，运行时可以 `resume(session_id)` 并以精确的 state 从第 N+1 步继续。

LangGraph 文档明确强调了在生产环境中重视此特性的用户：Klarna、Uber、J.P. Morgan。其主张并非图结构本身，而是图结构加上检查点使得恢复成本极低。

### 流式输出

每个节点都可以产出部分输出。图会将每个节点的增量事件流式传输给调用方，使 UI 能够随着图的执行而实时更新。

### 人工介入

在节点之间检查和修改 state。实现方式：在关键节点前暂停，将 state 暴露给人工，接受修改后恢复。Checkpointer 使得这变得容易，因为 state 已经是序列化状态。

### 记忆

短期（在单次运行内——状态中的对话历史）和长期（跨运行——通过 checkpointer 和独立长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者。** 中央路由 LLM 分发给专业子 agent。`langgraph-supervisor` 中的 `create_supervisor()`（尽管 LangChain 团队在 2026 年建议直接通过工具调用来实现以获得更好的上下文控制）。
2. **蜂群 / 对等网络。** Agent 通过共享工具接口直接交接。无需中央路由器。
3. **分层。** 监督者管理子监督者，实现为嵌套子图。

### 该模式的常见错误

- **检查点过于精简。** 仅检查点保存对话轮次会导致工具状态和记忆写入无法恢复。必须序列化完整 state。
- **非确定性节点。** 恢复假定节点输入会产生相同的 state 更新。随机种子、墙钟时间、外部 API 都必须被捕获。
- **过度使用条件边。** 每条边都是条件性的图是一个无法推理的状态机。更偏好线性链配合偶尔的分叉。

```figure
langgraph-state
```

## 构建

`code/main.py` 实现了一个标准库有状态图：

- `State` —— 带 `messages`、`step`、`route`、`output`、`human_approval` 的类型化 dict。
- `Node` —— 接收 state 并返回更新 dict 的可调用对象。
- `StateGraph` —— 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存模拟）—— 每个节点后序列化 state；`load(session_id)` 恢复。
- 演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

追踪日志展示了首次运行在 human gate 处失败，持久化，然后恢复并产生最终输出。

## 使用

- **LangGraph** —— 参考实现，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建自己的图。
- **AutoGen v0.4**（第 14 课）—— 高并发场景下的 actor 模型替代方案。
- **Claude Agent SDK**（第 17 课）—— 内置 session store 的托管框架。
- **自定义** —— 当你需要精确控制 state 形状或 checkpointer 后端时。

## 交付

`outputs/skill-state-graph.md` 生成一个 LangGraph 风格的 state graph，在任何目标运行时中集成检查点和恢复功能。

## 练习

1. 当分类置信度低于阈值时，添加从 `classify` 到 `end` 的条件边。人工手动设置 `route` 后恢复运行。
2. 将类 SQLite 模拟替换为真实的 SQLite checkpointer。测量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义 reducer 合并。不可变 state 在此带来了什么优势？
4. 阅读 `langgraph-supervisor` 参考文档。将玩具程序迁移到 `create_supervisor`。比较追踪日志形状。
5. 添加流式输出：每个节点运行期间产出部分 state。打印到达的增量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| State graph | "Agent as state machine" | 类型化 state + 节点 + 边 + reducer |
| Checkpointer | "Persistence backend" | 每个节点后序列化 state；支持恢复 |
| Reducer | "State merger" | 将当前 state 与节点更新合并的函数 |
| Conditional edge | "Branch" | 由 state 函数选择的边 |
| Subgraph | "Nested graph" | 作为节点在另一个图中使用的图 |
| Durable execution | "Resume from failure" | 以精确 state 在最后一个成功节点重启 |
| Supervisor | "Router LLM" | 专业子 agent 的中央分发器 |
| Swarm | "P2P agents" | Agent 通过共享工具交接；无中央路由器 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— 参考文档
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/) —— 监督者模式 API
- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— actor 模型替代方案
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) —— session store 和子 agent
