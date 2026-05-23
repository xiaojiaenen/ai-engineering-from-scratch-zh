# OpenTelemetry GenAI 语义规范

> OpenTelemetry 的 GenAI 特别兴趣小组（于 2024 年 4 月启动）定义了智能体遥测的标准模式。Span 名称、属性和内容捕获规则在各厂商间趋于一致，从而使智能体追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同的含义。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**先决条件：** Phase 14 · 13 (LangGraph)， Phase 14 · 24 (可观测性平台)
**时间：** 约 60 分钟

## 学习目标

- 说出 GenAI span 的分类：模型/客户端、智能体、工具。
- 区分 `invoke_agent` CLIENT 与 INTERNAL span 以及各自适用的场景。
- 列出顶层 GenAI 属性：提供者名称、请求模型、数据源 ID。
- 解释内容捕获契约：需明确选择启用、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题所在

每个厂商都自创 span 名称。运维团队最终需要为每个框架构建独立的仪表盘。OpenTelemetry 的 GenAI 特别兴趣小组通过定义整个生态系统共同遵循的单一标准来解决这个问题。

## 核心概念

### Span 分类

1.  **模型/客户端 span。** 涵盖原始 LLM 调用。由提供者 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2.  **智能体 span。** `create_agent`（构建智能体时）和 `invoke_agent`（运行时）。
3.  **工具 span。** 每次工具调用一个；通过父子关系连接到智能体 span。

### 智能体 span 命名

- Span 名称：如果已命名则为 `invoke_agent {gen_ai.agent.name}`；回退为 `invoke_agent`。
- Span 类型：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 实际解析的模型（可能因路由而异于请求）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个语料库或存储。

存在针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的技术特定规范。

### 内容捕获

默认规则：默认情况下，工具不应捕获输入/输出。捕获需通过以下方式明确选择启用：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容外部存储（S3、您的日志存储），在 span 上记录引用（指针 ID，而非正文）。这是第 27 课中集成到可观测性中的内容污染防御策略。

### 稳定性

截至 2026 年 3 月，大多数规范仍为实验性。可通过以下方式选择启用稳定预览版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生地将 GenAI 属性映射到其 LLM 可观测性模式中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 此模式的常见问题

- **在 span 中捕获完整的提示词。** 追踪中包含运维人员可读的 PII、密钥、客户数据。应外部存储。
- **缺少 `gen_ai.provider.name`。** 当归属信息缺失时，多提供商仪表盘会失效。
- **没有父链接的 span。** 孤立的工具 span。务必传播上下文。
- **未设置稳定性选项。** 后端升级时，您的属性可能被重命名。

## 构建它

`code/main.py` 实现了一个符合 GenAI 规范的标准库 span 发射器：

- `Span`，包含 GenAI 属性模式。
- `Tracer`，包含 `start_span`，嵌套上下文。
- 一个脚本化的智能体运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 span、用于 LLM 调用的 `chat` span。
- 一种内容捕获模式，将提示词外部存储，并在 span 上记录 ID。

运行它：

```
python3 code/main.py
```

输出：一个包含所有必需 GenAI 属性的 span 树，以及一个显示选择启用的内容引用的“外部存储”。

## 使用它

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）— 为生态系统自动插桩。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；基于 GenAI 属性构建仪表盘。
- **自托管** — 运行带有 GenAI 处理器的 OTel Collector。

## 部署它

`outputs/skill-otel-genai.md` 将 OTel GenAI span 集成到现有智能体中，采用内容捕获默认值和外部引用存储。

## 练习

1.  使用 `invoke_agent`（INTERNAL）+ 每个工具的 span 为您的第 01 课 ReAct 循环插桩。发送到一个 Jaeger 实例。
2.  以“仅引用”模式添加内容捕获：提示词存储到 SQLite，span 属性仅携带行 ID。
3.  阅读 `gen_ai.data_source.id` 的规范。将其集成到您的第 09 课 Mem0 搜索中。
4.  设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证您的属性不会被收集器重命名。
5.  构建一个仪表盘：“哪些工具错误与哪些模型相关联”，仅基于 GenAI 属性。

## 关键术语

| 术语               | 人们怎么说             | 实际含义                                  |
|--------------------|------------------------|-------------------------------------------|
| GenAI SIG          | "OpenTelemetry GenAI 小组" | 定义该模式的 OTel 工作组                  |
| invoke_agent       | "智能体 span"          | 表示一次智能体运行的 span 名称            |
| CLIENT span        | "远程调用"             | 用于调用远程智能体服务的 span             |
| INTERNAL span      | "进程内"               | 用于进程内智能体运行的 span               |
| gen_ai.provider.name | "提供者"             | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 源"            | 检索命中了哪个语料库/存储                 |
| 内容捕获           | "提示词日志"           | 明确选择捕获消息；生产环境中应外部存储    |
| 稳定性选项         | "预览模式"             | 用于固定实验性规范的环境变量              |

## 延伸阅读

- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范本身
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认包含 GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 跟踪上下文传播