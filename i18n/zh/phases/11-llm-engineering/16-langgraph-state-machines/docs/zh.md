# Agent 状态机：图、节点与检查点

> 手写的 ReAct 循环是一个 `while True`。以显式图形式编写的同一个循环，是可以进行checkpoint、中断、分支和时间旅行的。Agent 本身没有变化，围绕它的框架变了。

**类型：** 实践
**语言：** Python
**前置知识：** 第 11 阶段 · 09（函数调用），第 11 阶段 · 14（模型上下文协议）
**时间：** 约 75 分钟

## 问题所在

你交付了一个支持函数调用的 Agent。它能正常工作三个回合，然后出了问题：模型尝试调用一个返回 500 的工具，用户在中途改变了主意，或者 Agent 决定在执行退款操作前未经人工审批就执行。这个 `while True:` 循环没有任何钩子。你无法暂停它、回退它，也无法分支进入"如果模型当时选择了另一个工具会怎样"的假设场景。一旦越过演示阶段把这个 Agent 上线，它就成了一个黑盒——要么能工作，要么不能。

一旦你看穿了这一层，下一步就很明显了。Agent 本身已经是一个状态机了 —— 系统提示词、消息历史、待处理的工具调用、以及下一步动作。把状态机显式化：为"模型思考"、"工具运行"、"人工审批"等创建节点，为它们之间的条件转换创建边。图一旦显式化，框架就免费获得了四样东西：检查点（在步骤之间保存状态）、中断（暂停等待人工介入）、流式传输（流式输出 token 和中间事件），以及时间旅行（回退到之前的状态并尝试不同的分支）。

LangGraph 是这个抽象的参考实现。它不是 LangChain 意义上的 Agent 框架（"这是 AgentExecutor，祝你好运"）。它是一个具有**一等公民**状态、**一等公民**持久化和**一等公民**中断支持的图运行时。Agent 循环是你画出来的，不是手写出来的。

## 核心概念

![LangGraph StateGraph：节点、边和检查点](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三样东西。

1. **状态 (State)**。一个类型化的字典（TypedDict 或 Pydantic 模型），流经整个图。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的 *reducer（归约器）* 进行合并 —— 列表字段默认使用 `operator.add` 追加，其他字段默认覆盖。
2. **节点 (Nodes)**。Python 函数 `state -> partial_state`。每个节点是一个离散步骤："调用模型"、"运行工具"、"总结"。
3. **边 (Edges)**。节点之间的转换。静态边固定走向一个地方。条件边接受一个路由器函数 `state -> next_node_name`，以便图可以根据模型输出进行分支。

你编译这个图。编译过程绑定拓扑结构，附加一个检查点器（可选，但对生产环境至关重要），并返回一个可运行对象。你用初始状态和 `thread_id` 调用它。每次执行步骤都会基于 `(thread_id, checkpoint_id)` 键持久化一个检查点。

### 四大核心能力

**检查点 (Checkpointing)。** 每次节点转换都会将新状态写入存储（测试环境用内存，生产环境用 Postgres/Redis/SQLite）。通过再次调用相同的 `thread_id` 来恢复执行。图会从暂停的地方继续。

**中断 (Interrupts)。** 标记一个节点为 `interrupt_before=["human_review"]`，执行会停止在该节点运行之前。状态会持久化。你的 API 向用户响应"等待审批"。随后针对同一 `thread_id` 发出带有 `Command(resume=...)` 的请求即可恢复执行。

**流式传输 (Streaming)。** `graph.stream(state, mode="updates")` 按发生时输出状态变更。`mode="messages"` 流式传输模型节点内的 LLM token。`mode="values"` 输出完整快照。你选择要在 UI 中呈现的内容。

**时间旅行 (Time-travel)。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任意先前的 `checkpoint_id` 传递给 `graph.invoke`，即可从该点分叉。这对于调试（"如果模型当时选了工具 B 会怎样？"）和回放生产痕迹的回归测试非常有用。

### Reducer 是关键

每个状态字段都有一个 reducer。大多数默认值都够用 —— 新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息才会追加而不是替换。并行边通过 reducer 合并它们的更新。如果两个节点都更新了 `messages`，而你忘了使用 `Annotated[list, add_messages]`，后者会静默胜出，你会丢失一半的对话轮次。reducer 是库中最微妙的地方；处理正确，其余部分自然组合。

### 四节点 ReAct 图

一个生产级的 ReAct Agent 由四个节点和两条边组成：

1. `agent` —— 使用当前消息历史调用 LLM。返回 assistant 消息（可能包含 tool_calls）。
2. `tools` —— 执行最后一条 assistant 消息中的任何 tool_calls，将工具结果作为 tool 消息追加。
3. 从 `agent` 出发的条件边：如果最后一条消息有 tool_calls 则路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就是这样。你获得了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → …），附带检查点、中断和流式传输功能，代码大约 40 行。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 允许一个节点分发并行子图。示例：Agent 决定同时查询三个检索器。每个 `Send` 都会为目标节点生成一个并行执行；它们的输出通过状态 reducer 合并。这就是 LangGraph 表达编排器-工作者模式的方式，无需线程原语。

### 子图 (Subgraphs)

编译后的图可以作为另一个图中的节点。外层图看到一个节点；内层图有自己独立的状态和检查点。这就是团队构建主管-工作者 Agent 的方式：主管图根据用户意图将请求路由到各个领域的子图。

```figure
l5-state-graph-ledger
```

## 动手构建

### 步骤 1：定义状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    # 消息列表，使用 add_messages 归约器进行合并
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    # 调用 LLM 获取响应
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    # 检查最后一条消息是否包含 tool_calls
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# 编译图并附加内存检查点器
app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表追加而不是覆盖的 reducer。忘记它是 LangGraph 最常见的 bug。

### 步骤 2：带线程运行

```python
# 配置线程 ID
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("查找 Anthropic 总部地址")]},
    config,
    stream_mode="updates",  # 以状态变更形式流式传输
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，让用户看到"agent 正在思考… 调用 search_web… 得到结果… 正在回答。"

### 步骤 3：添加人工干预中断

标记一个节点，使其在执行前暂停。

```python
# 编译时指定在 tools 节点之前中断
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每个工具调用之前暂停
)

state = app.invoke({"messages": [HumanMessage("删除生产数据库")]}, config)
# state["__interrupt__"] 已被设置。检查拟议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写入拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("被人工审核员阻止。")]})
```

状态、检查点和线程都在中断期间持久化。除了执行期间，内存中没有任何残留数据。

### 步骤 4：用于调试的时间旅行

```python
# 获取历史检查点
history = list(app.get_state_history(config))
for snapshot in history:
    # 打印最后一条消息的前 80 个字符
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从之前的检查点分叉
target = history[3].config  # 回退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点向前回放
```

将 `None` 作为输入传入会从给定检查点回放；传入值会将其作为更新追加到该检查点的状态后再恢复。这就是你重现失败 Agent 运行轨迹而不必重放整个对话的方法。

### 步骤 5：为生产环境切换检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 使用 Postgres 作为持久化存储
with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()  # 初始化数据库表
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已内置。`MemorySaver` 仅用于测试。任何需要跨重启持久化的场景都需要真实的存储。

## 技能要点

> 你要把 Agent 构建为图，而不是 `while True` 循环。

在使用 LangGraph 之前，先做一个 60 秒的设计：

1. **命名节点。** 每个离散的决策或副作用操作都是一个节点。"Agent 思考"、"工具运行"、"审核员批准"、"响应流式传输"。如果你无法列出它们，说明这个任务还不足以构成 Agent 形态。
2. **声明状态。** 使用最小化 TypedDict，每个列表字段都要配一个 reducer。不要把什么都塞进 `messages`；把任务特定的字段（如工作 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升到顶层。
3. **绘制边。** 静态边除非下一步依赖模型输出。每个条件边都需要一个带有命名分支的路由器函数。
4. **提前选择检查点器。** `MemorySaver` 用于测试，Postgres/Redis/SQLite 用于其他所有场景。没有检查点器就不要上线 —— 没有检查点器意味着没有恢复、没有中断、没有时间旅行。
5. **在工具执行前决定中断点。** 审批放在副作用节点**入边**上，以便你可以在造成损害前取消；校验放在模型**出边**上，以便你可以低成本地拒绝错误调用。
6. **默认启用流式传输。** `mode="updates"` 用于 UI，`mode="messages"` 用于模型节点内的 token 级别流式传输，`mode="values"` 用于评估时的完整快照。

拒绝交付没有检查点器的 LangGraph Agent。拒绝交付在副作用**之后**才中断的 Agent。拒绝交付 `messages` 字段没有使用 `add_messages` 作为 reducer 的 Agent。

## 练习

1. **简单。** 用计算器工具和网页搜索工具实现上述四节点 ReAct 图。验证 `list(app.get_state_history(config))` 在两回合对话中至少返回四个检查点。
2. **中等。** 添加一个 `planner` 节点，在 `agent` 之前运行并将结构化 `plan: list[str]` 写入状态。让 `agent` 标记计划步骤为已完成。如果 `plan` 在检查点恢复后丢失（归约器配置错误），则测试失败。
3. **困难。** 构建一个主管图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图都有自己独立的状态和检查点。在外层图上添加 `interrupt_before=["writer"]`，以便人工可以审批研究简报。确认从先前检查点的时间旅行只会重放分叉分支。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------------------|---------|
| StateGraph | "LangGraph 图" | 你在编译前添加节点和边的构建器对象。 |
| Reducer | "字段如何合并" | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认覆盖，`add_messages` 追加。 |
| Thread | "对话 ID" | 一个 `thread_id` 字符串，用于限定一个会话的所有检查点。 |
| Checkpoint | "暂停的状态" | 节点转换后完整图状态的持久化快照，键为 `(thread_id, checkpoint_id)`。 |
| Interrupt | "暂停等待人工" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；通过 `Command(resume=...)` 恢复。 |
| Time-travel | "从之前的步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | "并行子图分发" | 节点可返回的构造器，用于生成目标节点的 N 个并行执行。 |
| Subgraph | "作为节点的编译图" | 用作另一个图中节点的编译 StateGraph；保留其自身状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、reducer、检查点器和中断的权威参考。
- [LangGraph 概念：状态、归约器、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的思维模型，源自官方文档。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — 关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的细节。
- [LangGraph 人工干预](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 LangGraph Agent 都实现的 pattern；阅读它以理解推理轨迹的合理性。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 应该优先选用哪些图结构（链、路由器、编排器-工作者、评估器-优化器）以及何时选用。
- 第 11 阶段 · 09（函数调用）— 每个 LangGraph Agent 节点复用的工具调用原语。
- 第 11 阶段 · 14（模型上下文协议）— 通过 MCP 适配器接入 LangGraph `ToolNode` 的外部工具发现机制。
- 第 11 阶段 · 17（Agent 框架权衡）— 何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
