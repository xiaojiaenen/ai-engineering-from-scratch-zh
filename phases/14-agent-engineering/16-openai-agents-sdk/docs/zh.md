# OpenAI Agents SDK：移交、护栏、追踪

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体框架。包含五大基本要素：Agent、Handoff、Guardrail、Session、Tracing。移交本质上是名为 `transfer_to_<agent>` 的工具。护栏会在输入或输出时触发。追踪默认开启。

**类型：** 学习 + 实践
**语言：** Python (标准库)
**前置要求：** 第14章第01节（智能体循环），第14章第06节（工具使用）
**时长：** 约75分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五大基本要素。
- 解释移交：为何被建模为工具，模型看到的名称格式，以及上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式。
- 实现一个使用标准库运行时的方案，包含移交、护栏和基于 span 的追踪。

## 问题所在

无法清晰委派的智能体会将所有内容塞入一个提示词。没有护栏的智能体会泄露 PII、输出违反政策的内容或陷入无限循环。OpenAI 的 SDK 将实现高效多智能体协作的三大基本要素规范化。

## 核心概念

### 五大基本要素

1. **智能体。** 大语言模型 + 指令 + 工具 + 移交能力。
2. **移交。** 将任务委派给另一个智能体。对模型呈现为名为 `transfer_to_<agent_name>` 的工具。
3. **护栏。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（针对每个函数工具）进行验证。
4. **会话。** 自动管理跨轮次的对话历史。
5. **追踪。** 为 LLM 生成、工具调用、移交、护栏内置 span。

### 移交作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它会向运行时发出信号：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 版本压缩）。
2. 使用目标智能体的指令初始化该智能体。
3. 由目标智能体继续运行。

这是将监督者模式（第13课 / 第28课）产品化。

### 护栏

三种类型：

- **输入护栏。** 在第一个智能体的输入上运行。在调用任何 LLM 之前拒绝不安全或超出范围的请求。
- **输出护栏。** 在最后一个智能体的输出上运行。捕获 PII 泄露、政策违规、格式错误的响应。
- **工具护栏。** 针对每个函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 同时运行。降低尾部延迟。如果触发，主 LLM 的工作会被丢弃（浪费 token）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，则不会在主调用上浪费 token。

触发时会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次 LLM 生成、工具调用、移交和护栏都会发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可选择退出。`add_trace_processor(processor)` 可将 span 同时发送到您自己的后端和 OpenAI 的后端。

### 会话

`Session` 将对话历史存储在后端（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 此模式可能出现的问题

- **移交漂移。** 智能体 A 移交给智能体 B，而智能体 B 又移交回智能体 A。添加跳转计数器。
- **护栏绕过。** 工具护栏仅对函数工具触发；内置工具（文件读取、网页获取）需要单独的策略。
- **过度追踪。** span 中包含敏感内容。与 OTel GenAI 内容捕获规则（第23课）配合使用——存储在外部，通过 ID 引用。

## 动手构建

`code/main.py` 使用标准库实现了 SDK 的基本结构：

- `Agent`、`FunctionTool`、`Handoff`（作为具有转移语义的函数工具）。
- `Runner`，包含输入/输出/工具护栏、移交调度和跳转计数器。
- 一个简单的 span 发射器，用于展示追踪结构。
- 一个分诊智能体，根据用户查询移交到计费或支持部门；其中一个输入会触发护栏。

运行它：

```
python3 code/main.py
```

追踪显示了两次成功的移交、一次输入护栏触发，以及一个与真实 SDK 发出的结构相似的 span 树。

## 选择使用

- **OpenAI Agents SDK** 用于以 OpenAI 为核心的产品。
- **Claude Agent SDK**（第17课）用于以 Claude 为核心的产品。
- **LangGraph**（第13课）当您需要显式状态和持久化恢复时。
- **自定义实现**当您需要精确控制时（语音、多提供商、联邦部署）。

## 部署上线

`outputs/skill-agents-sdk-scaffold.md` 为 Agents SDK 应用生成脚手架，包含分诊智能体、移交、输入/输出/工具护栏、会话存储和追踪处理器。

## 练习

1. 添加移交跳转计数器：在 N 次转移后拒绝。追踪此行为。
2. 实现 `nest_handoff_history` 作为选项——在转移前将先前消息压缩为单个摘要。
3. 编写一个阻塞式输出护栏。比较会触发它的提示词和不会触发它的提示词的延迟。
4. 将 `add_trace_processor` 连接到 JSON 日志记录器。它为每个 span 发出什么结构？
5. 阅读 SDK 文档。将您的标准库玩具版本移植到 `openai-agents-python`。您哪里建模错了？

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| Agent | "大语言模型 + 指令" | SDK 中的智能体类型；拥有工具和移交能力 |
| Handoff | "转移" | 模型调用的工具，用于委派给另一个智能体 |
| Guardrail | "策略检查" | 对输入 / 输出 / 工具调用的验证 |
| Tripwire | "护栏触发" | 护栏拒绝时抛出的异常 |
| Session | "历史存储" | 跨运行持久化的对话记忆 |
| Tracing | "Spans" | 对 LLM + 工具 + 移交 + 护栏的内置可观测性 |
| Blocking guardrail | "顺序检查" | 护栏先运行；触发时无 token 浪费 |
| Parallel guardrail | "并发检查" | 护栏同时运行；延迟更低，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 基本要素、移交、护栏、追踪
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应方案
- [Anthropic, 构建高效智能体](https://www.anthropic.com/research/building-effective-agents) — 何时需要使用移交
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span 映射的标准