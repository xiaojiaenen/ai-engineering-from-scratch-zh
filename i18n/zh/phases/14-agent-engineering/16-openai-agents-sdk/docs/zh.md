# OpenAI Agents SDK：转移、护栏、追踪

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体框架。五个核心原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 是命名为 `transfer_to_<agent>` 的工具。Guardrail 在输入或输出时触发。Tracing 默认开启。

**类型：** 学习 + 构建
**语言：** Python (标准库)
**前置知识：** 阶段 14 · 01（智能体循环）、阶段 14 · 06（工具使用）
**预计时间：** ~75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个核心原语。
- 解释 Handoff：为何被建模为工具、模型看到的命名模式，以及上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式的区别。
- 实现一个带有 Handoff + 护栏 + 跨度式追踪的标准库运行时。

## 问题所在

无法清晰委托任务的智能体会把一切塞进同一个提示词。没有护栏的智能体会泄露 PII、产生违反策略的输出，或陷入无限循环。OpenAI 的 SDK 将让多智能体工作变得可控的三个核心原语加以规范化。

## 概念讲解

### 五个核心原语

1. **Agent**。LLM + 指令 + 工具 + Handoff。
2. **Handoff**。向另一个智能体的委托。向模型呈现为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail**。对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）的验证。
4. **Session**。跨轮次的自动对话历史管理。
5. **Tracing**。内置 LLM 生成、工具调用、Handoff、护栏的跨度（span）。

### Handoff 作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它时，运行时执行以下操作：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 功能压缩上下文）。
2. 用该智能体的指令初始化目标智能体。
3. 以目标智能体继续运行。

这是将主管模式（参见第 13 课/第 28 课）产品化的体现。

### 护栏（Guardrails）

三种类型：

- **输入护栏**。在第一个智能体的输入上运行。在任何 LLM 调用前拒绝不安全或超范围的请求。
- **输出护栏**。在最后一个智能体的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具护栏**。按每个函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 并行运行。降低尾部延迟。若触发，主 LLM 的工作将被丢弃（浪费 token）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。若触发，不会在主调用上浪费 token。

触发时会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` 异常。

### 追踪（Tracing）

默认开启。每次 LLM 生成、工具调用、Handoff 和护栏都会产生一个跨度。设置 `OPENAI_AGENTS_DISABLE_TRACING=1` 可关闭。通过 `add_trace_processor(processor)` 可将跨度转发到你自己的后端，与 OpenAI 的追踪并存。

### 会话（Sessions）

`Session` 将对话历史存储在后端（SQLite、Redis 或自定义）。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 常见陷阱

- **Handoff 漂移**。智能体 A 转移给 B，B 又转移回 A。加入跳数计数器。
- **护栏绕过**。工具护栏仅在函数工具上触发；内置工具（文件读取、网页抓取）需要单独的策略。
- **过度追踪**。跨度中包含敏感内容。结合 OTel GenAI 内容捕获规则（参见第 23 课）—— 外部存储，通过 ID 引用。

```figure
ae-agent-handoff
```

## 动手构建

`code/main.py` 用标准库实现了 SDK 的核心形态：

- `Agent`、`FunctionTool`、`Handoff`（作为具有转移语义的函数工具）。
- 带输入/输出/工具护栏、Handoff 分发和跳数计数器的 `Runner`。
- 一个简单的跨度发射器，展示追踪形态。
- 一个分诊智能体：根据用户查询将任务转移给计费或客服智能体；其中一个输入会触发护栏。

运行：

```
python3 code/main.py
```

追踪结果显示两次成功的 Handoff、一次输入护栏触发，以及一个与现实 SDK 发出结构相似的跨度树。

## 如何使用

- **OpenAI Agents SDK** 用于以 OpenAI 为首选产品的场景。
- **Claude Agent SDK**（第 17 课）用于以 Claude 为首选产品的场景。
- **LangGraph**（第 13 课）在你需要明确状态和持久化恢复时使用。
- **自定义实现** 当需要精确控制（语音、多提供商、联邦部署）时使用。

## 交付物

`outputs/skill-agents-sdk-scaffold.md` 脚手架一个包含分诊智能体、Handoff、输入/输出/工具护栏、会话存储和追踪处理器的 Agents SDK 应用。

## 练习

1. 添加 Handoff 跳数计数器：超过 N 次转移则拒绝。追踪其行为。
2. 实现 `nest_handoff_history` 选项——在转移前将先前消息压缩为一个摘要。
3. 编写一个阻塞式输出护栏。对比会触发它的提示词与通过它的提示词之间的延迟。
4. 将 `add_trace_processor` 接入 JSON 日志记录器。每个跨度会输出什么形状的数据？
5. 阅读 SDK 文档。将你的标准库玩具项目移植到 `openai-agents-python`。你有哪些建模错误？

## 关键术语

| 术语 | 人们常说的意思 | 实际含义 |
|------|----------------|----------|
| Agent | "LLM + 指令" | SDK 中的智能体类型；拥有工具和 Handoff |
| Handoff | "转移" | 模型调用的用于委托给另一个智能体的工具 |
| Guardrail | "策略检查" | 对输入 / 输出 / 工具调用的验证 |
| Tripwire | "护栏触发" | 护栏拒绝时抛出的异常 |
| Session | "历史存储" | 跨运行持久化的对话记忆 |
| Tracing | "跨度" | 针对 LLM + 工具 + Handoff + 护栏的内置可观测性 |
| Blocking guardrail | "顺序检查" | 护栏先运行；触发时不浪费 token |
| Parallel guardrail | "并发检查" | 护栏并行运行；延迟更低，但触发时会浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 核心原语、Handoff、护栏、追踪
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — 以 Claude 为特色的对应产品
- [Anthropic，《构建有效智能体》](https://www.anthropic.com/research/building-effective-agents) — 何时应该考虑使用 Handoff
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 标准 Agents SDK 跨度所映射到的规范
