# 智能体的 Actor 模型 — 异步消息与类型化运行时

> 智能体即 Actor：异步消息交换、事件驱动处理器、故障隔离、原生并发。AutoGen v0.4（微软研究院，2025年1月）围绕此模型重新设计了智能体编排；该框架现已进入维护模式，Microsoft Agent Framework（2025年10月公共预览）是其生产级继任者。

**类型：** 学习 + 实践
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（智能体循环），阶段 14 · 12（工作流模式）
**时间：** 约 75 分钟

## 学习目标

- 描述 Actor 模型：智能体即 Actor，消息是唯一 IPC，每个 Actor 故障隔离。
- 说出 AutoGen v0.4 的三层 API —— Core、AgentChat、Extensions —— 及各层的用途。
- 解释为什么将消息投递与处理解耦能带来故障隔离和原生并发。
- 用 Python stdlib 实现一个 Actor 运行时，并将双智能体代码审查流程移植到其上。

## 问题所在

大多数智能体框架是同步的：一个智能体产生，一个智能体消费，在调用栈中运行。故障会崩溃整个栈。并发是外挂的。分布式需要重写。

AutoGen v0.4 的答案：Actor 模型。每个智能体是一个 Actor，拥有私有收件箱。消息是唯一交互方式。运行时将投递与处理解耦。故障隔离到单个 Actor。并发是原生的。分布式只是不同的传输。

## 核心概念

### Actor

一个 Actor 拥有：

- 私有状态（外部无法直接访问）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中 effects 可以是"回复"、"发送给其他 Actor"、"派生新 Actor"、"更新状态"、"停止自身"。

两个 Actor 不能共享内存。它们只能发送消息。

### 三层 API

AutoGen v0.4 将其接口分为三层：

1. **Core。** 底层 Actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成层 —— OpenAI、Anthropic、Azure、工具、记忆。

### 解耦为何重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱后立即返回。运行时稍后投递。三个后果：

- **故障隔离。** Agent B 崩溃不会导致 Agent A 崩溃 —— 运行时在 B 的处理器中捕获故障并决定如何处理（记录日志、重试、转入死信队列）。
- **原生并发。** 多条消息同时在飞行中；Actor 并发处理其收件箱。
- **分布式就绪。** 无论 Actor 是进程内还是另一台主机上，收件箱 + 传输是相同的抽象。

### 拓扑结构

- **RoundRobinGroupChat。** Actor 按固定顺序轮流执行。
- **SelectorGroupChat。** 选择器 Agent 根据对话上下文决定下一个由谁处理。
- **Magentic-One。** 网页浏览、代码执行、文件处理的参考型多智能体团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息都会产生一个 span；工具调用按 2026 年 OTel GenAI 语义约定携带 `gen_ai.*` 属性（第 23 课）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型开发稳定可用。微软已将活跃开发转移到 Microsoft Agent Framework，作为生产级继任者（2025年10月1日公共预览；1.0 GA 目标为 2026年 Q1 末）。AutoGen 的模式可平滑移植 —— Actor 模型是持久有效的思想。

```figure
actor-mailbox
```

## 动手实现

`code/main.py` 实现了一个 stdlib Actor 运行时：

- `Message` —— 带 `sender`、`recipient`、`topic`、`body` 的类型化负载。
- `Actor` —— 抽象基类，包含 `receive(message, runtime)`。
- `Runtime` —— 事件循环，带共享队列、投递、故障隔离。
- 双 Actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 执行检查清单；两者通过消息交换直至达成一致。

运行方式：

```
python3 code/main.py
```

输出轨迹展示消息投递、某一 Actor 中模拟的故障不影响另一个 Actor，以及最终收敛到共享裁决。

## 如何使用

- **AutoGen v0.4/v0.7**（维护中）—— 适合研究、原型开发、多智能体模式。
- **Microsoft Agent Framework** —— 生产级继任者（2025年10月公共预览）；相同的 Actor 模型思想，全新的 API。
- **LangGraph swarm 拓扑**（第 13 课）—— 通过共享工具交接实现的类似模式。
- **自定义 Actor 运行时** —— 当你需要特定传输协议时（NATS、RabbitMQ、gRPC）。

## 交付物

`outputs/skill-actor-runtime.md` 生成一个最小 Actor 运行时以及针对给定多智能体任务的团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败消息暂存供人工检查。在你的玩具系统中，死信队列多久被触发一次？
2. 实现 `SelectorGroupChat`：一个选择器 Actor 根据对话状态决定下一个由谁来处理消息。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务，使 Actor 可在不同进程中运行。
4. 为每条消息接入一个 OTel span（或使用无操作的占位实现）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具系统移植到真正的 `autogen_core` API。在生产环境中你省略了哪些重要的东西？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Actor | "智能体" | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | "事件" | 类型化负载；Actor 交互的唯一方式 |
| Inbox | "信箱" | 每个 Actor 的待处理消息队列 |
| Runtime | "智能体宿主" | 路由消息并隔离故障的事件循环 |
| Topic | "通道" | Actor 间命名发布-订阅路由 |
| Fault isolation | "Let it crash" | 一个 Actor 故障不会导致其他 Actor 崩溃 |
| RoundRobinGroupChat | "固定轮转团队" | Actor 按顺序轮流执行 |
| SelectorGroupChat | "上下文路由团队" | 选择器决定下一个由谁处理 |
| Magentic-One | "参考团队" | 面向网页 + 代码 + 文件的多智能体小组 |

## 延伸阅读

- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 重新设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 图结构的替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— AutoGen 默认发出的 spans
