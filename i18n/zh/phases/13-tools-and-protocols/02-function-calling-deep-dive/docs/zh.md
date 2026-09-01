# Function Calling 深度解析 — OpenAI、Anthropic、Gemini

> 三家前沿提供商在 2024 年收敛于相同的工具调用循环，随后在其他方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课并排对比三者的差异，以便在一家提供商上开发的代码在移植时不会出错。

**类型：** 构建
**语言：** Python（stdlib、schema 转换器）
**前置条件：** Phase 13 · 01（工具接口）
**时间：** 约 75 分钟

## 学习目标

- 阐述 OpenAI、Anthropic 和 Gemini 函数调用负载的三个形状差异（声明、调用、结果）。
- 将一份工具声明翻译成三种提供商格式，并预测严格模式约束的差异点。
- 在各提供商中使用 `tool_choice` 强制、禁止或自动选择工具调用。
- 了解各提供商的硬性限制（工具数量、schema 深度、参数长度）以及违反限制时各自的错误签名。

## 问题所在

函数调用请求的形状因提供商而异。以下是 2026 年生产栈中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传递 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你必须解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码来强制执行 schema 合规性。

**Anthropic Messages API。** 你传递 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经过解析（是对象而非字符串）。你用包含 `{type: "tool_result", tool_use_id, content}` 块的新的 `user` 消息进行回复。

**Google Gemini API。** 你传递 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 进行回复。

同样的循环。不同的字段名、不同的嵌套、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写天气代理的团队，移植到 Anthropic 需要两天，再移植到 Gemini 需要一天，仅为了处理管道逻辑。

本课构建一个转换器，将三种格式统一为一个规范的工具声明，并在边缘进行路由。Phase 13 · 17 将同一模式泛化为 LLM 网关。

## 概念

### 公共结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制调用特定工具、禁止工具，或让模型自行决定。
3. **调用发出。** 结构化输出去调用工具并传入参数。
4. **调用 ID。** 将响应与正确的调用关联（并行调用时尤为重要）。
5. **结果注入。** 一条消息或块，将结果与调用关联起来。

### 字段级形状差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明包装 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | JSON 字符串化 | 已解析的对象 | 已解析的对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | role `tool`，`tool_call_id` | 包含 `tool_result`、`tool_use_id` 的 `user` | 包含匹配 `id` 的 `functionResponse` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式 | `strict: true` | schema 即合同（始终强制执行） | 请求级别的 `responseSchema` |

### 你会实际遇到的限制

- **OpenAI。** 每次请求最多 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式要求不包含 `$ref`，`oneOf`/`anyOf`/`allOf` 不能重叠，每个属性都必须列在 `required` 中。
- **Anthropic。** 每次请求最多 64 个工具。Schema 深度实际上无限制，但实践限制为 10。没有严格模式标志；schema 是一份合同，模型倾向于遵守。
- **Gemini。** 每次请求最多 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有不同）。自 Gemini 3 起，并行调用使用唯一 ID。

### `tool_choice` 行为

三方都支持的三种模式，命名不同。

- **Auto（自动）。** 模型选择工具或文本。默认模式。
- **Required（必需）/ Any（任意）。** 模型必须调用至少一个工具。
- **None（无）。** 模型不得调用工具。

加上各家独有的一个模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` 标志区分单调用与多调用。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何，都会通过 schema 验证器路由每个响应。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条 assistant 消息中发出多个调用。你运行所有调用，然后用一个包含每个 `tool_call_id` 一条条目的批处理 tool-role 消息进行回复。Anthropic 历史上只支持单调用；`disable_parallel_tool_use: false`（截至 Claude 3.5 的默认设置）启用多调用。Gemini 2 允许并行调用，但不提供稳定 ID；Gemini 3 增加了 UUID，以便乱序响应能够干净地关联。

### 流式传输

三方都支持流式工具调用。线格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的 delta 块增量到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，以便多个并行调用可以交错。

Phase 13 · 03 深入讲解并行 + 流式重聚合。本课专注于声明和单调用形状。

### 错误与修复

无效参数错误看起来也不同。

- **OpenAI（非严格模式）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格模式）。** 验证发生在解码期间；无效 JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务器端验证。
- **Gemini。** OpenAPI 3.0 的一个怪癖：对象字段上的 `enum` 被静默忽略；自己验证。

### 转换器模式

代码中的规范工具声明如下所示（形状由你选择）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小型函数将它翻译成三种提供商形状。`code/main.py` 中的 harness 正是这样做的，然后在一个假工具调用上对三种提供商的响应形状进行往返测试。不需要网络——本课教授的是形状，而非 HTTP。

生产团队将此转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 提供了一个网关，在任何一种提供商前暴露 OpenAI 形状的 API。

```figure
function-call-args
```

## 使用它

`code/main.py` 定义了一个规范的 `Tool` 数据类，以及三个生成 OpenAI、Anthropic 和 Gemini 声明 JSON 的转换器。然后它将手工制作的每种形状的提供商响应解析为同一个规范调用对象，证明皮肤下的语义是相同的。运行它并并排对比三种声明的差异。

值得关注的点：

- 三种声明块仅在不同在包装和字段名上。
- 三种响应块差异在于调用所在的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 交付成果

本课产出 `outputs/skill-provider-portability-audit.md`。给定针对一家提供商的函数调用集成，该技能会生成一份可移植性审计报告：它依赖哪些提供商限制、哪些字段需要重命名，以及移植到其他提供商时会破坏什么。

## 练习

1. 运行 `code/main.py`，验证三种提供商声明 JSON 都序列化了同一个底层 `Tool` 对象。修改规范工具以添加枚举参数，并确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每种提供商添加 `ListToolsResponse` 解析器，以提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有此功能；注意这种不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到三种提供商形状。然后映射 `mode="any"` 和 `mode="none"`。核对课的差异表。

4. 选择三家提供商之一，从头到尾阅读其函数调用指南。在它的 schema 规范中找到一个其他两家不支持的字段。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每种提供商的验证器（本课 01 的 stdlib 即可作为代理）运行它，并记录哪些错误触发。记录你在生产中会使用哪家提供商以获取严格性保证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 函数调用 | "工具使用" | 结构化工具调用发出的提供商级 API |
| 工具声明 | "工具规范" | 名称 + 描述 + JSON Schema 输入负载 |
| `tool_choice` | "强制 / 禁止" | 自动 / 必需 / 无 / 特定名称模式 |
| 严格模式 | "Schema 强制执行" | OpenAI 标志，约束解码以匹配 schema |
| `tool_use` 块 | "Anthropic 的调用形状" | 包含 id、name、input 的内联内容块 |
| `functionCall` 部分 | "Gemini 的调用形状" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数-as-字符串 | "JSON 字符串化" | OpenAI 以 JSON 字符串形式返回参数，而非对象 |
| 并行工具调用 | "单次回合内的扇出" | 一条 assistant 消息中的多个工具调用 |
| 拒绝 | "模型拒绝" | 仅严格模式的拒绝块，而非调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用一种带有微小差异的 JSON-Schema 类似方言 |

## 延伸阅读

- [OpenAI — Function calling 指南](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的规范参考
- [Anthropic — Tool use 概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — Function calling 参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行细节
