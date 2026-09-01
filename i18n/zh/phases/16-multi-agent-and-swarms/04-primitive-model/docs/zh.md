# 多智能体原语模型

> 四个原语，仅此而已——智能体（agent）、交接（handoff）、共享状态（shared state）、编排器（orchestrator）——这四个维度构成了设计空间的全部，而 2026 年主流发布的多智能体框架（AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework）都是这个空间中的点。本课从零构建它们，在一个玩具系统上运行四种模式，然后将每个主流框架映射到相同的坐标轴上，让你用一段话读懂任何新发布的框架。

**类型：** Learn
**语言：** Python (stdlib)
**前置要求：** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**时间：** ~60 分钟

## 问题

每六个月就有一个新的多智能体框架发布。AutoGen 在 2023 年。CrewAI 在 2024 年。LangGraph 和 OpenAI Swarm 在 2024 年。Google ADK 在 2025 年 4 月。Microsoft Agent Framework RC 在 2026 年 2 月。每个新闻稿都声称自己是"正确的抽象"。

如果你试图逐个学习它们，你会崩溃。API 看起来不同。文档对"智能体"的定义各执一词。一个框架把共享内存称为"黑板"（blackboard），另一个称为"消息池"（message pool），第三个称为"StateGraph"。你开始怀疑这个领域只是在炒作。

事实并非如此。抛开营销，这四个原语是稳定的。学一次，用一段话读懂每个新框架。

## 概念

### 四个原语

1. **智能体（Agent）**——系统提示词加工具列表。无状态；每次运行从系统提示词和当前消息历史开始。
2. **交接（Handoff）**——从一个智能体到另一个智能体的结构化控制转移。机械实现上，是一个返回新智能体的工具调用，或是一个基于条件的图边。
3. **共享状态（Shared state）**——多个智能体都可以读取（有时写入）的数据结构。消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）**——决定谁下一个发言的存在。选项包括：显式图（确定性）、LLM 说话者选择器（软性）、上一个发言者的交接调用（OpenAI Swarm），或基于队列的调度器（swarm 架构）。

这就是整个设计空间。每个框架为每个轴选择默认值；其余的都是表面语法。

### 2026 年每个框架的映射

| 框架 | 智能体 | 交接 | 共享状态 | 编排器 |
|------|--------|------|----------|--------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一次交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 串联的 Task 输出 | manager LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图，确定性 |
| Microsoft Agent Framework | agent + 编排模式 | 模式特定 | thread / context | 模式特定 |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host 决定 |

表面差异看起来巨大。底层：相同的四个旋钮。

### 为什么这很重要

一旦看到原语，框架比较就变成了简短的检查清单：

- 编排器是否信任 LLM 来路由（Swarm），还是在代码中固定路由（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影（StateGraph reducer）？
- 智能体能否修改彼此的提示词（CrewAI manager）还是只能交接（Swarm）？

这三个问题能回答 80% 的框架适配问题。你不再寻找"最好的多智能体框架"，而是开始为你真正关心的轴设计。

### 无状态洞察

除共享状态外，每个原语都是无状态的。Agent 是 (prompt, tools) 的函数。Handoff 是函数调用。Orchestrator 是调度器。**系统中唯一的有状态部分是共享状态。** 所有有趣的 bug 都住在这里：内存污染（Lesson 15）、消息顺序、版本控制、写竞争。

隐藏共享状态的框架（Swarm）将问题推给调用者。集中化共享状态的框架（LangGraph checkpoint、AutoGen pool）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的解剖

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

无内存。无状态。具有相同系统提示词和工具的两个智能体是可互换的。所有看起来像 per-agent 状态的东西实际上都在共享状态或交接协议中。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导地位：

- **函数返回**——工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在工具模式中携带路由。
- **图边**——LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
- **说话者选择**——AutoGen GroupChat。选择器函数（有时本身是 LLM 调用）读取池并选择下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常还有更多：结构化产物（CrewAI Task 输出）、类型化上下文（LangGraph reducers）、外部记忆（MCP、向量 DB）。

两种拓扑：**全量池**（每个智能体看到所有消息）和**投影**（智能体看到角色作用域视图）。全量池简单但扩展性差。投影池可扩展但需要前置 schema 设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种变体：

- **静态**——图在构建时固定（LangGraph 确定性、CrewAI Sequential）。
- **LLM 选择**——LLM 读取池并选择下一个说话者（AutoGen、CrewAI Hierarchical）。
- **交接驱动**——当前智能体通过调用交接工具决定（Swarm）。
- **队列驱动**——工作者从共享队列拉取；没有显式的下一个说话者（swarm 架构、Matrix）。

### 框架之间什么会变化

一旦原语固定，剩下的设计决策是：

- **内存策略**——临时与持久 checkpointing（LangGraph checkpointer）。
- **安全边界**——谁能批准交接（human-in-the-loop）。
- **成本核算**——per-agent token 预算。
- **可观测性**——追踪交接、持久化状态以进行回放。

所有这些都可在原语之上实现。它们都不是新的原语。

```figure
a5-primitive-radar
```

## 构建它

`code/main.py` 用约 150 行 stdlib Python 实现了四个原语。没有真正的 LLM——每个智能体都是脚本策略，所以焦点保持在协调结构上。

该文件导出：

- `Agent`——name、system prompt、tools、policy function 的数据类。
- `Handoff`——返回新智能体的函数。
- `SharedState`——线程安全的消息池。
- `Orchestrator`——三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示运行相同的三个智能体管道（research → write → review）通过所有三种编排器类型，并在结束时打印消息池。你可以看到输出仅在*谁选择下一个*上不同；智能体和共享状态在所有运行中是相同的。

运行它：

```
python3 code/main.py
```

预期输出：三种编排器运行，每种一种模式。每种都打印最终消息池。交接驱动的运行如果研究者决定提前完成会到达更少的智能体——这是微型化的 LLM 路由权衡。

## 使用它

`outputs/skill-primitive-mapper.md` 是一个技能，读取任何多智能体代码库或框架文档并返回四原语映射。在新框架发布时运行它以在深入阅读文档前获得一段话级别的理解。

## 交付它

在采用新框架之前，为其编写原语映射。如果你做不到，文档不完整或框架正在发明第五个原语（罕见——检查你是否未见过的一种共享状态变体）。

将映射固定在架构文档中。当新团队成员加入时，在 API 文档之前发送映射。当框架版本更改时，diff 映射而不是 changelog。

## 练习

1. 用不同的智能体策略运行 `code/main.py` 三次。观察编排器选择如何改变哪些智能体运行。
2. 实现第四种编排器类型：一个队列驱动的，其中智能体轮询共享状态以获取工作。可能发生什么死锁，你如何检测它？
3. 取 LangGraph quickstart（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并将其重写为四个原语。LangGraph 的哪些抽象 1:1 映射，哪些是便利封装？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 使哪个原语最易用，以及它将哪个原语推给调用者。
5. 在表中找到一个完全隐藏共享状态的框架。解释当智能体需要在交接间协调而不重读历史时什么会崩溃。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Agent | "带工具 LLM" | `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff | "控制转移" | 命名下一个智能体和可选 payload 的结构化调用。三种实现：函数返回、图边、说话者选择。 |
| Shared state | "记忆" / "上下文" | 多智能体系统中唯一的有状态部分。消息池或黑板。 |
| Orchestrator | "协调器" | 决定谁下一个运行的存在。静态图、LLM 选择器、交接驱动或队列驱动。 |
| Primitive | "抽象" | 每个框架参数化的四个轴之一。不是框架功能。 |
| Message pool | "共享聊天历史" | 完整历史的共享状态。易于推理，扩展性差。 |
| Projected state | "作用域视图" | 共享状态的角色特定视图。可扩展，需要 schema 设计。 |
| Speaker selection | "谁接下来说话" | 编排器模式，其中函数（通常是 LLM）从一组中选择下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)——交接驱动编排的最清晰阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/)——GroupChat + 说话者选择是 LLM 选择编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)——图边编排和基于 reducer 的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction)——role-goal-backstory 智能体、Sequential / Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2)——Microsoft 将 v0.4 移入维护后 live 的 AutoGen v0.2 线
