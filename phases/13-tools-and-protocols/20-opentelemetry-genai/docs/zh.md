# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体调用五个工具、三个MCP服务器和两个子智能体。你需要一条贯穿所有环节的追踪链路。OpenTelemetry GenAI语义约定（v1.37及以上版本的稳定属性）是2026年标准，被Datadog、Langfuse、Arize Phoenix、OpenLLMetry和AgentOps原生支持。本课程将命名所需属性，讲解跨度层级（智能体→LLM→工具），并提供一个可插入任何OTel导出器的标准库跨度发射器。

**类型：** 构建
**语言：** Python（标准库、OTel跨度发射器）
**前置要求：** 阶段13 · 07（MCP服务器）、阶段13 · 08（MCP客户端）
**时间：** ~75分钟

## 学习目标

- 命名LLM跨度和工具执行跨度所需的OTel GenAI属性。
- 构建覆盖智能体循环、LLM调用、工具调用和MCP客户端分发的追踪层级。
- 决定捕获（选择启用）与编辑（默认）哪些内容。
- 将跨度发射到本地收集器（Jaeger、Langfuse）而无需重写工具代码。

## 问题

2026年2月的一次调试：用户报告“我的智能体有时需要30秒响应，有时只需要3秒”。没有追踪链路。日志显示了LLM调用，但没有工具分发，没有MCP服务器往返，没有子智能体。你只能猜测。最终发现：一个MCP服务器在冷启动时偶尔会卡住。

没有端到端追踪，你无法发现这个问题。OTel GenAI解决了它。

这些约定在2025-2026年由OpenTelemetry语义约定小组确定。它们定义了稳定的属性名称，因此Datadog、Langfuse、Phoenix、OpenLLMetry和AgentOps都能解析相同的跨度。一次插桩；可输出到任何后端。

## 概念

### 跨度层级

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

所有内容都嵌套在一个追踪ID下。跨度ID链接了父子关系。

### 必需属性

根据2025-2026语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如`"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 提供商响应ID，用于关联。

对于工具跨度：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用ID。
- `gen_ai.tool.description` — 工具描述（可选）。

对于智能体跨度：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度种类

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM提供商、MCP服务器）。
- `SpanKind.INTERNAL` 用于智能体自身的循环步骤和工具执行。

### 选择启用的内容捕获

默认情况下，跨度携带指标和计时信息——而非提示或补全内容。大型负载和PII默认关闭。设置`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`和特定的内容捕获环境变量以包含内容。在生产环境启用前请仔细审查。

### 跨度上的事件

令牌级事件可作为跨度事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 工具调用记录。

事件在跨度内按时间排序，以便详细回放。

### 导出器

OTel跨度可导出到：

- **Jaeger / Tempo.** 开源，本地部署。
- **Langfuse.** 专注于LLM可观测性；可视化令牌使用情况。
- **Arize Phoenix.** 评估与追踪结合。
- **Datadog.** 商业产品；原生解析`gen_ai.*`属性。
- **Honeycomb.** 列式存储；查询友好。

都使用OTLP，即有线格式。你的代码无需关心。

### 跨MCP传播

当MCP客户端调用服务器时，将W3C traceparent头注入请求。可流式传输的HTTP支持标准头部。Stdio本身不携带HTTP头部；规范的2026年路线图讨论在JSON-RPC调用上添加`_meta.traceparent`字段。

在实现之前：在每个请求的`_meta`中手动包含traceparent。服务器记录追踪ID。

### 指标

除了跨度，GenAI语义约定还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

用于不需要逐次调用详情的仪表板。

### AgentOps层

AgentOps（成立于2024年）专注于GenAI可观测性。它封装流行框架（LangGraph、Pydantic AI、CrewAI）以自动发射OTel跨度。如果你的技术栈使用了支持的框架，这很有用；否则请使用手动插桩。

## 使用它

`code/main.py` 将调用LLM、分发两个工具并进行一次MCP往返的智能体的OTel格式跨度发射到标准输出（类似OTLP-JSON格式）。没有真正的导出器——课程重点在于跨度形状和属性集。将输出粘贴到OTLP兼容的查看器中，或者直接阅读。

需关注的点：

- 追踪ID在所有跨度间共享。
- 父子关系通过`parentSpanId`编码。
- 所需`gen_ai.*`属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量开启。

## 交付它

本课程产出`outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能将产出一个插桩计划：在哪里添加跨度、填充哪些属性以及针对哪些导出器。

## 练习

1. 运行`code/main.py`。计算跨度数量并识别哪个是CLIENT vs INTERNAL。

2. 开启内容捕获（环境变量）并确认`gen_ai.content.prompt`和`gen_ai.content.completion`事件出现。注意其对PII的影响。

3. 添加工具执行指标`gen_ai.tool.execution.duration`并作为每次调用的直方图样本发射。

4. 将父智能体跨度的traceparent传播到MCP请求的`_meta.traceparent`字段中。验证MCP服务器会看到相同的追踪ID。

5. 阅读OTel GenAI语义约定规范。找出一个语义约定中列出但本课程代码**未**发射的属性。将其添加进来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准 |
| GenAI语义约定 | "GenAI semantic conventions" | LLM / 工具 / 智能体跨度的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有GenAI属性共享此前缀 |
| 跨度 | "计时操作" | 具有开始、结束和属性的工作单元 |
| 追踪 | "跨跨度祖先" | 共享追踪ID的跨度树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于跨度方向的提示 |
| OTLP | "OpenTelemetry Line Protocol" | 导出器的有线格式 |
| 选择启用内容 | "提示/补全捕获" | 默认关闭；环境变量启用 |
| traceparent | "W3C头部" | 跨服务传播追踪上下文 |
| 导出器 | "后端特定传输组件" | 将跨度发送到Jaeger / Datadog等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI跨度、指标和事件的权威约定
- [OpenTelemetry — GenAI跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM和工具执行跨度属性列表
- [OpenTelemetry — GenAI智能体跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级别的`invoke_agent`跨度
- [open-telemetry/semantic-conventions — GenAI跨度](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub托管的真相源
- [Datadog — LLM OTel语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成演练