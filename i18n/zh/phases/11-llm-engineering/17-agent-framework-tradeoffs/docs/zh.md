# Agent 框架权衡 —— Graph、Role 与 Actor 编排

> 每个框架都演示同一个样例（研究代理生成报告），同时隐藏同一个缺陷（状态 schema 与编排层冲突）。选择抽象方式与你问题形状相匹配的框架；其余都是你需要写两遍的胶水代码。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 11 · 09（Function Calling）、Phase 11 · 16（LangGraph）
**时间：** 约 45 分钟

## 问题所在

你有一个任务，需要不止一次 LLM 调用。可能是一个研究工作流（规划、搜索、总结、引用）。可能是一个代码审查流水线（解析 diff、评论、打补丁、验证）。可能是一个多轮助手，负责订票、写邮件、提交报销。你挑选一个框架。

三天后，你发现框架的抽象泄漏了。CrewAI 给了你角色，但当"研究员"需要把结构化计划交给"写作者"时它就开始跟你作对。AutoGen 给了代理间的对话，但没有原生状态支持，所以你的检查点是会话日志的 pickle。LangGraph 给了你一个状态图，但在你不知道代理会做什么之前就迫使你命名每一个转换。Agno 给了你一个单代理抽象，当你尝试分支到三个并发工作者时它就开始报错。

解决方案不是"选择最好的框架"，而是将框架的核心抽象与你的问题形状匹配。本课就是绘制那张地图。

## 概念

![Agent 框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

四个框架主导了 2026 年的格局。它们的核心抽象各不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|------|----------|--------|----------|
| **LangGraph** | `StateGraph` —— 类型化状态、节点、条件边、检查点器。 | 具有显式状态和人工介入中断的工作流；需要时间旅行调试的生产级代理。 | 松散的、角色驱动头脑风暴，拓扑未知。 |
| **CrewAI** | `Crew` —— 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 角色驱动或人格驱动的工作流，具有简短的顺序/层级计划。 | 超出 Crew 轮次历史的任何状态性内容；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或多个代理交替发言，直到满足退出条件。 | 多代理*对话*（教师-学生、提议者-批评者、执行者-审核者），思维从对话中涌现。 | 已知 DAG 的确定性工作流；需要在重启间保持持久状态的任何内容。 |
| **Agno** | `Agent` —— 单个 LLM + 工具 + 记忆，可组合为团队。 | 快速构建的单代理和轻量级团队；强大的多模态支持和内置存储驱动。 | 具有自定义 reducer 的深度显式分支图。 |

### "抽象"实际含义什么

框架的核心抽象是你向他人介绍架构时在白板上画的东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转换，每一点的状态对象都是类型化的。心智模型是一个状态机。
- **CrewAI** → 你画一张组织架构图。每个角色有职位描述，经理分配任务。心智模型是一组小型专家团队。
- **AutoGen** → 你画一个 Slack 私聊窗口。两个代理互相发消息；如果需要调解员第三个加入。心智模型是对话。
- **Agno** → 你画一个带工具附件的单框。把框放在一起就是团队。心智模型是"自带电池的代理"。

### 状态问题

状态是大多数框架选择在生产中出问题的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型），每字段 reducer，一等公民检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行开箱即用。*（参见 Phase 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段在任务间以字符串形式流动，或通过 `output_pydantic` 结构化。开箱没有持久的每 Crew 存储；如果 Crew 必须在重启后存活，你需要自己附加。
- **AutoGen。** 状态是对话历史和任何用户定义的 `context`。会话转录持久化；任意工作流状态除非你编写适配器否则不持久。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附加到 `Agent`——会话和用户记忆自动持久化。不是完整的图检查点器；是会话存储。

### 分支问题

每个非平凡代理都会分支。谁决定分支很重要。

- **LangGraph** —— 你通过条件边决定。路由是用命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点器记录选择了哪条分支。
- **CrewAI** —— 管理器在层级模式中决定；在顺序模式中你在构建时决定。路由隐式存在于任务列表中；管理器提示词之外没有第一公民的"if"。
- **AutoGen** —— 代理通过对话决定。分支是从下一个说话者涌现出来的。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** —— 代理通过决定下一个调用哪个工具来分支。团队有协调器/路由器/协作者模式；超出此范围的分支是开发者的责任。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任何 OTel 导出器的 OpenTelemetry。每次节点转换都是一个追踪跨度；检查点同时可作为可重放的追踪。LangSmith 是官方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** —— 自 2025 年末起支持一等公民 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** —— 通过 `autogen-core` 的 OpenTelemetry 集成；AgentOps 和 Opik 有连接器。追踪粒度是每代理消息，而非每节点。
- **Agno** —— 内置 `monitoring=True` 标志加上 OpenTelemetry 导出器；与 Langfuse 的会话追踪紧密集成。

### 成本与延迟

所有四个框架都会增加每次调用的开销（框架逻辑、验证、序列化）。开销大致递增顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架额外执行的 LLM 路由量决定。CrewAI 的层级管理器花费 token 来决定下一个是谁；AutoGen 的 `GroupChatManager` 同理。LangGraph 只在你编写 `llm.invoke` 的地方花费 token。Agno 的单代理路径很轻。

当每次运行成本很重要时，优先使用显式路由（LangGraph 边、AutoGen `speaker_selection_method`）而非 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等公民 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配进来。通过 `allow_delegation=True` 支持 Crew 到 Crew 委派。
- **AutoGen** → `FunctionTool` 包装任意 Python 可调用对象；MCP 适配器可用。与 AG2 生态系统的代理到代理模式紧密耦合。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可以在代理和团队间共享。

## 技能

> 你能用一句话解释为什么某个框架适合某个代理问题。

预构建清单：

1. **画出形状。** 这是图（类型化状态、命名转换）？角色扮演的形式（专家交接工作）？对话（代理交谈直到完成）？还是带工具的单个代理？
2. **决定谁分支。** 开发者决定分支 → LangGraph。管理器代理决定 → CrewAI 层级模式。对话涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复？时间旅行？运行中人工中断？如果是，LangGraph 是默认选择；Agno 会话覆盖会话范围内的状态。
4. **检查成本预算。** LLM 选择的路由每轮消耗额外 token。如果代理每天运行数千次，优先选择显式路由。
5. **预算框架开销。** 每个框架都是另一个依赖项。如果任务只是两次 LLM 调用加一个工具，写 30 行裸 Python；没有框架比任何框架都便宜。

在你能够画出图、组织架构图、对话或代理框之前，不要随手抓一个框架。不要选择一个迫使你与你实际需要的事物对抗其状态模型的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|----------|----------|------|
| 具有类型化状态、人工审批、长运行的工作流 DAG | LangGraph | 一等公民状态、检查点器、中断、时间旅行。 |
| 具有不同角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 角色对应任务在 CrewAI 中表达成本低；分支变复杂时用 LangGraph 扩展。 |
| 提议者-批评者或教师-学生对话 | AutoGen | 双代理对话是其原生形状。 |
| 带工具、会话、记忆的单代理 | Agno | 最轻量的设置，内置存储和记忆。 |
| 带 reducer 的数千次并行扇出 | LangGraph + `Send` | 唯一具有并行分发 API 的框架。 |
| 快速原型，无框架承诺 | 裸 Python + 提供商 SDK | 没有框架是最快的框架。 |

```figure
l5-framework-fit
```

## 练习

1. **简单。** 将同一个任务——"研究 Anthropic 总部，撰写 200 字简报，引用来源"——分别在 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）中实现。报告每次运行的 token 消耗和代码行数。
2. **中等。** 在 AutoGen（researcher ↔ writer 对话，editor 通过 `GroupChat` 加入）和 Agno（单个代理带 `search_tools` 和 `write_tools`，加上会话存储）中构建相同任务。对四个实现在以下方面排名：(a) 每次运行成本，(b) 崩溃后恢复能力，(c) 在 write 步骤前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接收简短问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回带一句话理由的建议。用你自己设计的六个用例验证它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Orchestration（编排） | "代理如何协调" | 决定下一个运行哪个节点/角色/代理的层。 |
| Durable state（持久状态） | "重启后恢复" | 存活于进程死亡的状态，附加到检查点或会话存储。 |
| LLM-selected routing（LLM 选择路由） | "让模型决定" | 规划 LLM 每轮选择下一步；灵活但每次决策都消耗 token。 |
| Explicit routing（显式路由） | "开发者决定" | Python 函数或静态边选择下一步；廉价且可审计。 |
| Crew | "一个 CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级）绑定为单个可运行单元。 |
| GroupChat | "AutoGen 的多代理对话" | 带发言选择器的 N 个代理之间的受管对话。 |
| Team (Agno) | "多代理 Agno" | 一组代理上的路由/协调/协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点器抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— StateGraph、检查点器、中断、时间旅行。
- [CrewAI 文档](https://docs.crewai.com/) —— Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/) —— ConversableAgent、GroupChat、团队、工具。
- [Agno 文档](https://docs.agno.com/) —— Agent、Team、Workflow、存储、记忆。
- [Anthropic —— 构建有效代理（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) —— 模式库（提示链、路由、并行化、编排器-工作者、评估器-优化器），与框架无关。
- [Yao 等，《ReAct：协同推理与行动》（ICLR 2023）](https://arxiv.org/abs/2210.03629) —— 每个框架包装的循环。
- [Wu 等，《AutoGen：通过多代理对话启用下一代 LLM 应用》（2023）](https://arxiv.org/abs/2308.08155) —— AutoGen 设计论文。
- [Park 等，《生成代理：人类行为的交互拟像》（UIST 2023）](https://arxiv.org/abs/2304.03442) —— CrewAI 风格人格堆栈建立的角色扮演基础。
- Phase 11 · 16（LangGraph）—— 本课用来基准比较的框架。
- Phase 11 · 19（Reflexion）—— 一种可干净映射到 LangGraph 但映射到 CrewAI 时较为别扭的模式。
- Phase 11 · 22（生产可观测性）—— 如何 instrumentation 你选择的任何框架。
