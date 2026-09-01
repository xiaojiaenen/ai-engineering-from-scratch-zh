# OpenTelemetry GenAI 语义规范

> OpenTelemetry 的 GenAI SIG（于 2024 年 4 月启动）定义了智能体遥测的标准 Schema。Span 名称、属性和内容捕获规则在各厂商间收敛，使得智能体追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有统一含义。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 24（可观测性平台）
**时间：** ~60 分钟

## 学习目标

- 说出 GenAI Span 的类别：模型/客户端、智能体、工具。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL Span 及各自适用场景。
- 列出顶层 GenAI 属性：提供者名称、请求模型、数据源 ID。
- 解释内容捕获契约：需显式启用、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题

每个厂商都发明自己的 span 名称。运维团队最终不得不为每个框架构建独立的看板。OpenTelemetry 的 GenAI SIG 通过定义整个生态系遵循的统一标准来解决这个问题。

## 概念

### Span 类别

1. **模型/客户端 spans。** 覆盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **智能体 spans。** `create_agent`（当智能体被构造时）和 `invoke_agent`（当智能体运行时）。
3. **工具 spans。** 每次工具调用一个；通过父子关系连接到智能体 span。

### 智能体 Span 命名

- Span 名称：`invoke_agent {gen_ai.agent.name}`（如果已命名）；否则回退到 `invoke_agent`。
- Span 类型：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由而与请求模型不同）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 对于 RAG：查询了哪个语料库或存储。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 有技术特定的规范。

### 内容捕获

默认规则：仪器化不应默认捕获输入/输出。捕获需通过以下方式显式启用：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储到外部（S3、日志存储），在 spans 中记录引用（指针 ID，而非文本）。这是 Lesson 27 中内容投毒防御与可观测性的结合。

### 稳定性

截至 2026 年 3 月，大多数规范仍处于实验阶段。通过以下方式启用稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM 可观测性 Schema。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 此模式常见错误

- **在 spans 中捕获完整提示。** PII、密钥、客户数据出现在运维可读的追踪中。应外部存储。
- **缺少 `gen_ai.provider.name`。** 当缺少归属信息时，多提供商看板会失效。
- **缺少父级链接的 spans。** 孤立的工具 spans。务必传播上下文。
- **未设置稳定性启用。** 你的属性可能在后端升级时被重命名。

```figure
ae-genai-span-tree
```

## 构建它

`code/main.py` 实现了一个匹配 GenAI 规范的 stdlib span 发射器：

- 带有 GenAI 属性 Schema 的 `Span`。
- 带有 `start_span`、嵌套上下文的 `Tracer`。
- 一个脚本化智能体运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 spans、用于 LLM 调用的 `chat` spans。
- 一种内容捕获模式，将提示存储到外部并在 spans 中记录 ID。

运行：

```
python3 code/main.py
```

输出：一个带有所有必需 GenAI 属性的 span 树，以及显示启用内容的"外部存储"。

## 使用它

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（Lesson 24）— 自动仪器化生态系。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；从 GenAI 属性构建看板。
- **自建部署** — 运行带有 GenAI 处理器的 OTel Collector。

## 交付

`outputs/skill-otel-genai.md` 将 OTel GenAI spans 接入现有智能体，包含内容捕获默认设置和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每个工具的 spans 仪器化你的 Lesson 01 ReAct 循环。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：提示存入 SQLite，span 属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入你的 Lesson 09 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性未被 collector 重命名。
5. 构建看板：仅从 GenAI 属性中分析"哪些工具错误与哪些模型相关"。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义 Schema 的 OTel 工作组 |
| invoke_agent | "智能体 span" | 表示智能体运行的 span 名称 |
| CLIENT span | "远程调用" | 对远程智能体服务的调用的 span |
| INTERNAL span | "进程内" | 进程内智能体运行的 span |
| gen_ai.provider.name | "提供者" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 源" | 检索命中了哪个语料库/存储 |
| 内容捕获 | "提示日志" | 可选择性地捕获消息；在生产中外部存储 |
| 稳定性启用 | "预览模式" | 固定实验规范的 Env var |

## 延伸阅读

- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范文档
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认包含 GenAI spans
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel spans
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
