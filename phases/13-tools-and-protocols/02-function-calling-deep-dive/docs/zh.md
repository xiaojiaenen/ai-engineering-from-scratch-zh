# 函数调用深度解析 — OpenAI、Anthropic、 Gemini

> 2024年，三大前沿提供商在相同的工具调用循环上趋同，然后在所有其他方面走向分歧。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一ID关联。本课程将三者并排进行差异比较，以确保在一个提供商上发布的代码在移植到另一个提供商时不会中断。

**类型：** 构建
**语言：** Python (标准库、Schema 转换器)
**前置条件：** 第13阶段 · 第01课 (工具接口)
**时间：** ~75分钟

## 学习目标

- 阐述 OpenAI、Anthropic 和 Gemini 函数调用负载（声明、调用、结果）的三个形状差异。
- 将一个工具声明在所有三种提供商格式之间进行转换，并预测严格模式约束的差异之处。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬性限制（工具数量、Schema深度、参数长度）以及当超出限制时各自发出的错误签名。

## 问题所在

函数调用请求的形状因提供商而异。以下是来自2026年生产栈的三个具体示例：

**OpenAI 聊天补全 / 响应 API。** 你传递 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你必须解析的JSON字符串。严格模式 (`strict: true`) 通过约束解码来强制执行Schema合规性。

**Anthropic 消息 API。** 你传递 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 形式返回。`input` 已经是解析好的对象，而非字符串。你使用一个包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息进行回复。

**Google Gemini API。** 你传递 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你使用 `{functionResponse: {name, id, response}}` 进行回复。

相同的循环。不同的字段名、不同的嵌套结构、不同的字符串与对象约定、不同的关联机制。一个在OpenAI上编写天气代理的团队，仅仅为了适配底层逻辑，就需要花费两天移植到Anthropic，再花一天移植到Gemini。

本课程构建一个转换器，将三种格式统一为一种规范的工具声明，并在边缘进行路由。第13阶段 · 第17课将相同模式泛化为一个LLM网关。

## 概念解析

### 公共结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入Schema。
2. **工具选择。** 强制指定特定工具、禁止工具使用，或让模型决定。
3. **调用发出。** 命名工具并包含参数的结构化输出。
4. **调用ID。** 将响应关联到正确的调用（对于并行调用很重要）。
5. **结果注入。** 一条消息或一个块，将结果与调用关联起来。

### 字段级形状差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化JSON | 解析后的对象 | 解析后的对象 |
| ID格式 | `call_...` (OpenAI生成) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| 结果块 | 角色 `tool`, `tool_call_id` | 包含 `tool_result`, `tool_use_id` 的 `user` | 包含匹配 `id` 的 `functionResponse` |
| 强制调用某工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式Schema | `strict: true` | Schema即合约（始终强制） | 请求级别的 `responseSchema` |

### 你将实际遇到的限制

- **OpenAI。** 每个请求128个工具。Schema深度5层。参数字符串 <= 8192 字节。严格模式要求无 `$ref`，无 `oneOf`/`anyOf`/`allOf` 的重叠，`required` 中列出的每个属性。
- **Anthropic。** 每个请求64个工具。Schema深度实际无限制，但实际限制为10层。无严格模式标志；Schema即合约，模型倾向于遵守。
- **Gemini。** 每个请求64个函数。Schema类型是OpenAPI 3.0子集（与JSON Schema 2020-12略有分歧）。自Gemini 3起并行调用使用唯一ID。

### `tool_choice` 行为

所有人都支持三种模式，但名称不同。

- **自动。** 模型选择工具或文本。默认模式。
- **必需/任意。** 模型必须调用至少一个工具。
- **无。** 模型必须不调用工具。

另外每个提供商有一个独特模式：

- **OpenAI。** 通过名称强制调用特定工具。
- **Anthropic。** 通过名称强制调用特定工具；`disable_parallel_tool_use` 标志区分单次与多次调用。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何，都会将每个响应路由通过Schema验证器。

### 并行调用

OpenAI的 `parallel_tool_calls: true`（默认）在一个助手消息中发出多个调用。你运行所有这些调用，并用一个包含每个 `tool_call_id` 对应条目的批处理工具角色消息进行回复。Anthropic 历史上是单次调用；`disable_parallel_tool_use: false`（自Claude 3.5起默认启用）允许多次调用。Gemini 2 允许并行调用但未提供稳定ID；Gemini 3 添加了UUID，因此无序的响应也能干净地关联。

### 流式传输

三家都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块增量到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 块开始 / 块增量 / 块停止事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3新增）发出带有 `functionCallId` 的块，因此多个并行调用可以交错进行。

第13阶段 · 第03课深入探讨并行+流式重组。本课程聚焦于声明和单次调用的形状。

### 错误与修复

无效参数错误看起来也不一样。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的JSON解析失败，你注入一条错误消息并重新调用。
- **OpenAI（严格）。** 验证发生在解码期间；无效JSON不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；Schema是建议性的。需要在服务器端验证。
- **Gemini。** OpenAPI 3.0特性：`enum` 在对象字段上被静默忽略；需要自行验证。

### 转换器模式

你代码中的规范工具声明看起来是这样的（你选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三种提供商的形状。`code/main.py` 中的工具链正是这样做的，然后通过每个提供商的响应形状往返一个模拟的工具调用。不需要网络 — 本课程教授的是形状，而非HTTP。

生产团队将此转换器封装在 `AbstractToolset` (Pydantic AI)、`UniversalToolNode` (LangGraph) 或 `BaseTool` (LlamaIndex) 中。第13阶段 · 第17课提供了一个网关，在任何这三家提供商前端提供一个OpenAI形状的API。

## 使用方法

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个转换器，它们生成OpenAI、Anthropic和Gemini的声明JSON。然后，它将每种形状手工制作的提供商响应解析成同一个规范调用对象，证明其内部语义是相同的。运行它并并排比较三种声明。

需关注点：

- 三种声明块仅在信封和字段名称上不同。
- 三个响应块在调用存在的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）上不同。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 交付产物

本课程生成 `outputs/skill-provider-portability-audit.md`。给定一个针对某个提供商的函数调用集成，该技能生成一个可移植性审计：它依赖于哪些提供商限制，哪些字段需要重命名，以及当移植到其他每个提供商时什么会中断。

## 练习题

1. 运行 `code/main.py` 并验证三种提供商声明JSON是否都序列化了同一个底层 `Tool` 对象。修改规范工具以添加一个枚举参数，并确认只有Gemini转换器需要处理OpenAPI特性。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，该解析器在模型返回后提取工具列表，或在 `list_tools` 或发现调用后提取。OpenAI原生没有；注意这种不对称性。

3. 实现 `tool_choice` 转换：将一个规范的 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供商形状。然后映射 `mode="any"` 和 `mode="none"`。参考课程的差异表。

4. 选择三个提供商之一，通读其函数调用指南。在其Schema规范中找出一个其他两家不支持的字段。候选字段：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个工具调用，其参数违反了声明的Schema。通过每个提供商的验证器运行它（课程01中的标准库验证器即可作为代理），并记录哪些错误被触发。记录你在生产中为了严格性会选择使用哪个提供商。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 | “工具使用” | 提供商级别的API，用于发出结构化工具调用 |
| 工具声明 | “工具规范” | 名称 + 描述 + JSON Schema 输入负载 |
| `tool_choice` | “强制/禁止” | 自动 / 必需 / 无 / 特定名称模式 |
| 严格模式 | “Schema强制” | OpenAI的标志，约束解码以匹配Schema |
| `tool_use` 块 | “Anthropic的调用形状” | 带有id、name、input的内联内容块 |
| `functionCall` 部分 | “Gemini的调用形状” | 包含name、args和id的 `parts[]` 条目 |
| 参数作为字符串 | “字符串化JSON” | OpenAI将args作为JSON字符串返回，而非对象 |
| 并行工具调用 | “单次扇出” | 一个助手消息中的多个工具调用 |
| 拒绝 | “模型拒绝” | 仅限严格模式的拒绝块，而非调用 |
| OpenAPI 3.0子集 | “Gemini Schema特性” | Gemini使用类似JSON Schema的方言，但有细微差异 |

## 延伸阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的规范参考
- [Anthropic — 工具使用概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一ID和OpenAPI子集
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini的企业级接口
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式Schema强制详情