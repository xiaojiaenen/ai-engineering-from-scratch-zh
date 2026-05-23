# AutoGen v0.4：演员模型与智能体框架

> AutoGen v0.4（微软研究院，2025年1月）围绕演员模型重新设计了智能体编排。支持异步消息交换、事件驱动的智能体、故障隔离和天然并发。该框架目前处于维护模式，微软智能体框架（2025年10月公开预览版）将成为其继任者。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置课程：** 阶段14 · 01（智能体循环）、阶段14 · 12（工作流模式）  
**时长：** 约75分钟

## 学习目标

- 描述演员模型：智能体作为演员，消息作为唯一的进程间通信，每个演员独立故障隔离。
- 说出AutoGen v0.4的三个API层——Core、AgentChat、Extensions——及其各自用途。
- 解耦消息传递与处理如何带来故障隔离和天然并发。
- 用Python实现一个标准库演员运行时，并将一个双智能体代码审查流程移植到该运行时上。

## 问题所在

大多数智能体框架是同步的：一个智能体生产，一个智能体消费，在一个调用栈中。失败会崩溃整个栈。并发性是额外附加的。分布式需要重写。

AutoGen v0.4的解决方案是：演员模型。每个智能体是一个拥有私有收件箱的演员。消息是唯一的交互方式。运行时将传递与处理解耦。失败隔离在单个演员内。并发是原生的。分布式只是不同的传输方式。

## 核心概念

### 演员

一个演员拥有：

- 一个私有状态（外部无法直接访问）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，其中的效果可以是“回复”、“发送给其他演员”、“生成新演员”、“更新状态”、“停止自身”。

两个演员不能共享内存。它们只能发送消息。

### AutoGen v0.4的三个API层

1.  **Core。** 底层演员框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2.  **AgentChat。** 任务驱动的高层API（v0.2版ConversableAgent的替代品）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3.  **Extensions。** 集成层——支持OpenAI、Anthropic、Azure、工具、记忆。

### 解耦为何重要

在v0.2模型中，调用`agent_a.chat(agent_b)`会同步阻塞agent_a，直到agent_b返回。在v0.4中，`send(agent_b, msg)`将消息放入agent_b的收件箱并立即返回。运行时稍后进行传递。带来三个结果：

-   **故障隔离。** Agent B崩溃不会导致Agent A崩溃——运行时捕获B处理器中的失败并决定如何处理（记录日志、重试、死信队列）。
-   **天然并发。** 多个消息同时在途；演员并发处理其收件箱。
-   **为分布式准备就绪。** 收件箱 + 传输是相同的抽象，无论演员是在进程内还是在其他主机上。

### 拓扑结构

-   **RoundRobinGroupChat（轮流群聊）。** 智能体按固定顺序轮流发言。
-   **SelectorGroupChat（选择器群聊）。** 一个选择器智能体根据对话上下文决定下一个发言者。
-   **Magentic-One。** 参考多智能体团队，用于网络浏览、代码执行、文件处理。基于AgentChat构建。

### 可观测性

内置OpenTelemetry支持。每条消息都发出一个span；工具调用携带`gen_ai.*`属性，符合2026年OTel GenAI语义规范（第23课）。

### 状态：维护模式

2026年初：AutoGen v0.7.x在研究和原型设计方面保持稳定。微软已将活跃开发转向微软智能体框架（2025年10月1日公开预览版；1.0正式版目标2026年第一季度末）。AutoGen的模式可以轻松移植——演员模型是持久的核心思想。

## 动手构建

`code/main.py`实现了一个标准库演员运行时：

-   `Message` — 带有`sender`、`recipient`、`topic`、`body`的类型化负载。
-   `Actor` — 抽象类，带有`receive(message, runtime)`。
-   `Runtime` — 事件循环，带有共享队列、传递、故障隔离。
-   一个双演员演示：`ReviewerAgent`审查代码，`ChecklistAgent`运行检查清单；他们交换消息直到达成共识。

运行它：

```
python3 code/main.py
```

跟踪显示了消息传递，一个演员的模拟失败并未导致另一个崩溃，以及最终收敛到共同的结论。

## 实际应用

-   **AutoGen v0.4/v0.7**（维护中）——适用于研究、原型设计、多智能体模式。
-   **微软智能体框架**（公开预览版）——前进的方向；同样的演员模型思想，刷新的API。
-   **LangGraph swarm 拓扑**（第13课）——通过共享工具移交实现的类似模式。
-   **自定义演员运行时**——当您需要特定传输（如NATS、RabbitMQ、gRPC）时。

## 部署应用

`outputs/skill-actor-runtime.md`为给定的多智能体任务生成一个最小演员运行时加上一个团队模板（轮询或选择器）。

## 练习

1.  添加一个死信队列：当处理器抛出异常时，将失败的消息暂存以供人工检查。在您的玩具程序中，DLQ被命中的频率如何？
2.  实现`SelectorGroupChat`：一个选择器演员根据对话状态选择谁处理下一条消息。
3.  添加分布式传输：将进程内队列替换为一个基于JSON-over-HTTP的服务器，以便演员可以在独立进程中运行。
4.  为每条消息连接一个OTel span（或一个无操作替代品）。按照第23课发出`gen_ai.agent.name`、`gen_ai.operation.name`。
5.  阅读AutoGen v0.4的架构文章。将您的玩具程序移植到真正的`autogen_core` API。您跳过了哪些在生产中重要的东西？

## 关键术语

| 术语         | 常见说法       | 实际含义                                     |
|--------------|----------------|----------------------------------------------|
| Actor（演员）     | "Agent"        | 私有状态 + 收件箱 + 处理器；无共享内存         |
| Message（消息）   | "Event"        | 类型化负载；演员间唯一的交互方式               |
| Inbox（收件箱）   | "Mailbox"      | 每个演员的待处理消息队列                       |
| Runtime（运行时） | "Agent host"   | 路由消息并隔离故障的事件循环                   |
| Topic（主题）     | "Channel"      | 演员间的具名发布-订阅路由                      |
| Fault isolation（故障隔离） | "Let it crash" | 一个演员失败不会导致其他演员崩溃               |
| RoundRobinGroupChat（轮流群聊） | "Fixed-rotation team" | 智能体按顺序轮流       |
| SelectorGroupChat（选择器群聊） | "Context-routed team" | 选择器决定下一个发言者   |
| Magentic-One     | "Reference team" | 用于网络 + 代码 + 文件的多智能体小队           |

## 延伸阅读

-   [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重构文章
-   [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 图形结构替代方案
-   [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen默认发出的span