# LangGraph — 代理的状态机

> 手写的ReAct循环是一个`while True`。用LangGraph编写的ReAct循环则是一个可以设置检查点、中断、分支和时间回溯的图。代理本身没有改变，改变的是围绕它的运行框架。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 11 · 09（函数调用），Phase 11 · 14（模型上下文协议）
**时间：** 约75分钟

## 问题所在

你部署了一个函数调用代理。它运行了三个回合，然后就出了问题：模型尝试调用一个返回500错误的工具；用户在任务中途改变了主意；或者代理决定退款一个订单，却没有人类的审批。`while True:`循环没有钩子。你无法暂停它，无法回退它，也无法分支到“如果模型选了另一个工具会怎样”的路径。一旦你将它投入演示之外的环境，这个代理就变成了一个非黑即白的黑盒——要么成功，要么失败。

一旦你看清这一点，下一步就显而易见了。代理本身已经是一个状态机——系统提示词加上消息历史加上待处理的工具调用加上下一个动作。让这个状态机显式化：为“模型思考”、“工具运行”、“人类审批”设置节点，为它们之间的条件转换设置边。一旦图变得显式，框架就能免费获得四样东西：检查点（在步骤间保存状态）、中断（为人类暂停）、流式传输（流式输出token和中间事件）以及时间回溯（回退到先前状态并尝试不同分支）。

LangGraph就是封装了这种抽象的库。它并非LangChain意义上的代理框架（“这是个AgentExecutor，祝你好运”）。它是一个具有**一等公民状态、一等公民持久化和一等公民中断**的图形运行时。代理循环是你**画出来**的，而不是你手写的。

## 核心概念

![LangGraph StateGraph: 节点、边和检查点器](../assets/langgraph-stategraph.svg)

一个`StateGraph`包含三样东西。

1.  **状态。** 一个类型化字典（TypedDict或Pydantic模型），在图中流动。每个节点接收完整的状态，并返回一个部分更新，LangGraph使用每个字段的*归约器*来合并更新——`operator.add`用于需要累积的列表，默认是覆盖。
2.  **节点。** 用Python函数`state -> partial_state`实现。每个节点代表一个离散步骤：“调用模型”、“运行工具”、“总结”。
3.  **边。** 节点之间的转换。静态边指向一个固定目标。条件边使用一个路由器函数`state -> next_node_name`，以便图能根据模型输出进行分支。

你编译这个图。编译操作会绑定拓扑结构，附加一个检查点器（可选但在生产环境中必不可少），并返回一个可运行对象。你用一个初始状态和一个`thread_id`来调用它。执行的每一步都会基于`(thread_id, checkpoint_id)`持久化一个检查点。

### 四大超能力

**检查点。** 每次节点转换都将新状态写入存储（测试时用内存，生产环境用Postgres/Redis/SQLite）。通过使用相同的`thread_id`再次调用图来恢复。图会从暂停的地方继续。

**中断。** 用`interrupt_before=["human_review"]`标记一个节点，执行就会在该节点运行前停止。状态被持久化。你的API响应用户“等待审批”。稍后使用同一个`thread_id`和`Command(resume=...)`的请求会恢复执行。

**流式传输。** `graph.stream(state, mode="updates")`在事件发生时产生状态增量。`mode="messages"`在模型节点内部流式输出LLM的token。`mode="values"`产生完整的快照。你可以在UI中选择要呈现的内容。

**时间回溯。** `graph.get_state_history(thread_id)`返回完整的检查点日志。将任何先前的`checkpoint_id`传递给`graph.invoke`，你就能从该点进行分叉。非常适合调试（“如果模型选了工具B会怎样？”）以及重放生产轨迹的回归测试。

### 归约器是关键

每个状态字段都有一个归约器。大多数默认值没问题——新值覆盖旧值。但消息列表需要`operator.add`，这样新消息才会追加而不是替换。并行边通过归约器合并它们的更新。如果两个节点都更新了`messages`而你忘了`Annotated[list, add_messages]`，后者会静默地覆盖前者，你就会丢失半个回合的记录。归约器是这个库中唯一需要留心的地方；处理好它，其余部分就可组合了。

### 四节点ReAct图

一个生产级的ReAct代理包含四个节点和两条边：

1. `agent` —— 使用当前消息历史调用LLM。返回助手消息（可能包含工具调用）。
2. `tools` —— 执行最后一条助手消息中的任何工具调用，并将工具结果作为工具消息追加。
3. 一条从`agent`出发的条件边：如果最后一条消息包含工具调用，则路由到`tools`，否则路由到`END`。
4. 一条从`tools`回到`agent`的静态边。

就这样。你就得到了完整的ReAct循环（思考 → 行动 → 观察 → 思考 → …），并带有检查点、中断和流式传输功能，代码量大约只有40行。

### StateGraph 与 Send（扇出）

`Send(node_name, state)`允许一个节点派发并行子图。例如：代理决定同时查询三个检索器。每个`Send`都会为目标节点生成一个并行执行；它们的输出通过状态归约器合并。这就是LangGraph在不使用线程原语的情况下表达编排者-工作者模式的方式。

### 子图

一个编译后的图可以作为另一个图中的节点。外部图看到一个单一节点；内部图拥有自己的状态和自己的检查点。这就是团队构建监督者-工作者代理的方式：监督者图将用户意图路由到每个领域的工作者子图。

## 动手构建

### 第一步：定义状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`是使消息列表累积而不是覆盖的归约器。忘记它是最常见的LangGraph错误。

### 第二步：带线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典`{node_name: state_delta}`。你的前端可以将这些流式传输到UI，让用户看到“代理正在思考…调用search_web…收到结果…正在回答。”

### 第三步：添加人在回路中断

标记一个节点，使其在运行前暂停执行。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程都会在中断期间持久化。除了执行期间，没有任何数据存在于内存中。

### 第四步：用于调试的时间回溯

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

传递`None`作为输入会从给定的检查点开始回放；传递一个值则会将其作为更新追加到该检查点的状态中，然后再恢复执行。这就是你在不重新运行整个对话的情况下复现一次糟糕代理运行的方法。

### 第五步：为生产环境更换检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis和Postgres已内置支持。`MemorySaver`用于测试。任何需要跨重启持久化的场景都需要真正的存储。

## 技能要点

> 你要以图的形式构建代理，而不是作为`while True`循环。

在使用LangGraph之前，先进行60秒设计：

1.  **命名节点。** 每一个离散的决策或有副作用的动作都是一个节点。“代理思考”、“工具运行”、“审核者批准”、“响应流式输出”。如果你无法列出它们，那这个任务还不是代理形态的。
2.  **声明状态。** 使用最小化的TypedDict，为每个列表字段指定归约器。不要把所有东西都塞进`messages`；将任务特定字段（一个工作`plan`、一个`budget`计数器、一个`retrieved_docs`列表）提升到顶层。
3.  **绘制边。** 除非下一步依赖于模型输出，否则使用静态边。每个条件边都需要一个带有命名分支的路由器函数。
4.  **预先选择检查点器。** `MemorySaver`用于测试，Postgres/Redis/SQLite用于其他任何情况。不要在没有检查点器的情况下部署——没有检查点器意味着无法恢复、无法中断、无法时间回溯。
5.  **在工具运行前决定中断，而非之后。** 审批放在进入有副作用节点的边上，这样你可以在造成损害前取消；验证放在从模型出来的边上，这样你可以廉价地拒绝错误的调用。
6.  **默认流式传输。** `mode="updates"`用于UI，`mode="messages"`用于模型节点内部的token级流式传输，`mode="values"`用于评估期间的完整快照。

拒绝部署没有检查点器的LangGraph代理。拒绝部署在副作用*之后*才中断的代理。拒绝部署没有将`add_messages`作为`messages`字段归约器的代理。

## 练习

1.  **简单。** 使用计算器工具和网页搜索工具，实现上面的四节点ReAct图。验证`list(app.get_state_history(config))`在两次对话后返回至少四个检查点。
2.  **中等。** 添加一个在`agent`之前运行的`planner`节点，将结构化的`plan: list[str]`写入状态。让`agent`将计划步骤标记为完成。如果`plan`在检查点恢复后丢失（归约器错误），则测试失败。
3.  **困难。** 构建一个监督者图，使用`Send`在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的状态和检查点器。在外部图上添加一个`interrupt_before=["writer"]`，以便人类可以批准研究摘要。确认从先前检查点进行的时间回溯只重新运行了分叉的分支。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|--------------|
| StateGraph | “LangGraph的图” | 在编译前添加节点和边的构建器对象。 |
| Reducer | “字段如何合并” | 当节点返回该字段的更新时应用的函数`(old, new) -> merged`；默认是覆盖，`add_messages`是追加。 |
| Thread | “一个对话ID” | 一个`thread_id`字符串，用于限定一次会话中所有检查点的范围。 |
| Checkpoint | “一个暂停的状态” | 节点转换后完整图状态的持久化快照，以`(thread_id, checkpoint_id)`为键。 |
| Interrupt | “为人类暂停” | `interrupt_before` / `interrupt_after`在节点边界停止执行；用`Command(resume=...)`恢复。 |
| Time-travel | “从先前步骤分叉” | `graph.invoke(None, config_with_old_checkpoint_id)`从该检查点向前回放。 |
| Send | “并行子图派发” | 一个节点可以返回的构造器，用于生成目标节点的N个并行执行。 |
| Subgraph | “一个编译后的图作为节点” | 一个编译后的StateGraph，在另一个图中用作节点；保留其自身的作用域。 |

## 扩展阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— StateGraph、归约器、检查点器和中断的规范参考。
- [LangGraph 概念：状态、归约器、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/) —— 本课程使用的思维模型，直接来自源头。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) —— 关于Postgres/SQLite/Redis存储、检查点命名空间和线程ID的细节。
- [LangGraph 人在回路](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) —— `interrupt_before`、`interrupt_after`、`Command(resume=...)`和编辑状态模式。
- [Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models” (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个LangGraph代理实现的模式；阅读以了解推理轨迹的理由。
- [Anthropic — 构建有效的代理 (2024年12月)](https://www.anthropic.com/research/building-effective-agents) —— 偏好使用哪些图形状（链、路由器、编排者-工作者、评估器-优化器）以及何时使用。
- Phase 11 · 09（函数调用） —— 每个LangGraph代理节点复用的工具调用原语。
- Phase 11 · 14（模型上下文协议） —— 通过MCP适配器插入LangGraph `ToolNode`的外部工具发现。
- Phase 11 · 17（代理框架权衡） —— 何时选择LangGraph而非CrewAI、AutoGen或Agno。