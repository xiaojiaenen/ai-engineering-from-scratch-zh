# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体。你需要一条覆盖所有操作的完整链路追踪。OpenTelemetry GenAI 语义约定（v1.37 及以上版本的稳定属性）是 2026 年的标准，被 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课将指出必需的属性、梳理跨度层次结构（智能体 → LLM → 工具），并提供一个可与任意 OTel 导出器配合使用的标准库跨度发射器。

**类型：** 构建
**语言：** Python（标准库、OTel 跨度发射器）
**前置条件：** 阶段 13 · 07（MCP 服务器）、阶段 13 · 08（MCP 客户端）
**预计时间：** 约 75 分钟

## 学习目标

- 命名 LLM 跨度和工具执行跨度的必需 OTel GenAI 属性。
- 构建覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发的链路层次。
- 决定哪些内容应捕获（选择加入）vs 脱敏（默认设置）。
- 向本地收集器（Jaeger、Langfuse）发射跨度，而无需重写工具代码。

## 问题所在

2026 年 2 月的一个调试案例：用户报告"我的智能体有时响应需要 30 秒；有时只需 3 秒。"无链路追踪。日志显示了 LLM 调用，但未显示工具分发、MCP 服务器往返、子智能体调用。你只能猜测。最终你发现：一个 MCP 服务器在冷启动时偶尔会挂起。

没有端到端链路追踪，你无法定位此问题。OTel GenAI 可解决此问题。

相关约定于 2025-2026 年在 OpenTelemetry 语义约定组下确定。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 均能解析相同的跨度。一次插桩；推送至任意后端。

## 概念解析

### 跨度层次结构

```
agent.invoke_agent  （顶层，INTERNAL 跨度）
 ├── llm.chat       （CLIENT 跨度）
 ├── tool.execute   （INTERNAL）
 │    └── mcp.call  （CLIENT 跨度）
 ├── llm.chat       （CLIENT 跨度）
 └── subagent.invoke （INTERNAL）
```

所有内容嵌套在同一 trace id 下。span id 用于链接父子关系。

### 必需属性

根据 2025-2026 年的语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用于关联的提供商响应 id。

对于工具跨度：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体调用 id。
- `gen_ai.tool.description` — 工具描述（可选）。

对于智能体跨度：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度类型（SpanKind）

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL` 用于智能体自身的循环步骤和工具执行。

### 选择加入的内容捕获

默认情况下，跨度仅携带指标和时序数据——不包含提示词或补全结果。大负载数据和个人身份信息（PII）默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 及特定的内容捕获环境变量以启用内容捕获。在生产环境启用前请仔细审查。

### 跨度上的事件

可在跨度上添加 token 级别的事件：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在跨度内按时间顺序排列，便于详细回放。

### 导出器

OTel 跨度可导出至：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** 专为 LLM 可观测性设计；可视化 token 使用情况。
- **Arize Phoenix。** 结合评估与追踪。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；便于查询。

所有导出器均使用 OTLP 作为传输格式。你的代码无需关心差异。

### 跨 MCP 的传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入请求中。Streamable HTTP 支持标准头。stdio 无法原生携带 HTTP 头；规范 2026 年路线图讨论在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在此功能上线前：手动将 traceparent 包含在每个请求的 `_meta` 中。服务器记录 trace id。

### 指标

除跨度外，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

用于不需要逐调用细节的仪表盘。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它封装了流行框架（LangGraph、Pydantic AI、CrewAI）以自动发射 OTel 跨度。如果你的技术栈使用受支持的框架，这是一个不错的选择；否则请使用手动插桩。

```figure
t3-span-waterfall
```

## 实践使用

`code/main.py` 将 OTel 格式的跨度发射到 stdout（采用 OTLP-JSON 类似格式），展示一个调用 LLM、分发两个工具并进行一次 MCP 往返的智能体。没有真实的导出器——本课重点在于跨度形状和属性集。将输出粘贴到兼容 OTLP 的查看器或直接阅读即可。

关注点：

- 所有跨度共享同一 trace id。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；其中一个场景通过环境变量开启。

## 交付成果

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能将生成插桩计划：在哪里添加跨度、填充哪些属性、以及目标导出器。

## 练习

1. 运行 `code/main.py`。数出跨度数量并识别哪些是 CLIENT vs INTERNAL。

2. 开启内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意其对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并为每次调用发射为直方图样本。

4. 从父智能体跨度向 MCP 请求的 `_meta.traceparent` 字段传播 traceparent。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出规范中列出但本课代码未发射的一个属性。补充它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OTel | "OpenTelemetry" | 用于追踪、指标、日志的开箱标准 |
| GenAI semconv | "GenAI 语义约定" | LLM/工具/智能体跨度的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "带时间的操作" | 具有开始、结束和属性的工作单元 |
| Trace | "跨跨度血缘" | 共享 trace id 的跨度树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于跨度方向的提示 |
| OTLP | "OpenTelemetry Line Protocol" | 导出器的传输格式 |
| Opt-in content | "提示词/补全捕获" | 默认关闭；环境变量启用 |
| traceparent | "W3C 头" | 跨服务传播追踪上下文 |
| Exporter | "后端专用推送组件" | 将跨度发送至 Jaeger/Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 跨度、指标和事件的权威约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行跨度属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级 `invoke_agent` 跨度
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的真相来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成指南
