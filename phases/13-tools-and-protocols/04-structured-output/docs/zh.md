# 结构化输出 —— JSON Schema、Pydantic、Zod、约束解码

> "礼貌地要求模型返回 JSON" 有 5% 到 15% 的失败率，即便在顶尖模型上也是如此。结构化输出通过约束解码弥合了这一差距：模型实际上被阻止发出任何违反 schema 的 token。OpenAI 的严格模式、Anthropic 的 schema 类型化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一思想的五种表面形式。本课将构建 schema 验证器和严格模式合约，学习者将在每个生产环境提取管线中使用它们。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12 子集）
**前置要求：** 阶段 13 · 02（函数调用深入）
**时间：** ~75 分钟

## 学习目标

- 为提取目标编写符合 JSON Schema 2020-12 规范的 schema，并使用正确的约束（枚举、最小/最大值、必填、模式）。
- 解释为什么严格模式和约束解码与"生成后验证"提供的保证不同。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 部署一个包含类型化修复和类型化拒绝处理的提取管线。

## 问题所在

一个读取采购订单邮件的代理需要将自由文本转换为 `{customer, line_items, total_usd}`。有三种方法。

**方法一：提示输出 JSON。** "以 JSON 格式回复，包含字段 customer、line_items、total_usd。" 这在顶尖模型上 85% 到 95% 的情况下有效。失败有六种方式：缺少花括号、尾随逗号、类型错误、幻觉字段、在 token 限制处截断、泄露"这是你的 JSON："等散文内容。

**方法二：生成后验证。** 自由生成，解析，对照 schema 验证，失败时重试。可靠但昂贵——你为每次重试付费，并且截断错误每次发生都会额外消耗一轮交互。

**方法三：约束解码。** 提供商在解码时强制执行 schema。无效 token 被从采样分布中屏蔽。输出保证可解析且保证可验证。失败收敛为一种模式：拒绝（模型决定输入不符合 schema）。

每个 2026 年的顶尖提供商都提供了方法三的某种形式。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上 `refusal`（如果模型拒绝则包含在响应中）。
- **Anthropic。** 对 `tool_use` 输入进行 schema 强制执行；`stop_reason: "refusal"` 并非现实，但 `end_turn`（没有工具调用）是拒绝信号。
- **Gemini。** `responseSchema` 在请求级别；2026 年 Gemini 为选定类型提供 token 级别的语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出一个类型为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod (TypeScript)。** 运行时解析器，根据 Zod schema 验证提供商输出；与 OpenAI 的 `beta.chat.completions.parse` 配对使用。

共同点：声明一次 schema，端到端强制执行。

## 核心概念

### JSON Schema 2020-12 —— 通用语言

每个提供商都接受 JSON Schema 2020-12。你最常用到的结构：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名称到子 schema 的映射。
- `required`：必须出现的字段名称列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI 严格模式增加了三个要求：每个属性都必须列在 `required` 中，`additionalProperties: false` 无处不在，并且没有未解析的 `$ref`。如果你违反这些，API 会在请求时返回 400 错误。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类风格的模型生成 JSON Schema。Pydantic AI 封装了这一点，所以你可以这样写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

代理框架会在边缘层将 schema 翻译成 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出作为类型化的 `Invoice` 实例返回。验证错误会抛出带有类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod (`z.object({customer: z.string(), ...})`) 是 TypeScript 的等价物。OpenAI 的 Node SDK 提供了 `zodResponseFormat(Invoice)`，它会翻译成 API 的 JSON Schema 负载。

### 拒绝

严格模式无法强制模型回答。如果输入无法适配 schema（"邮件是一首诗，不是发票"），模型会发出一个包含原因的 `refusal` 字段。你的代码必须将此作为一等结果来处理，而不是失败。拒绝也作为一个安全信号很有用：当模型被要求从受保护内容的邮件中提取信用卡号时，它会返回一个附带安全原因的拒绝。

### 开源中的约束解码

开源权重实现使用三种技术。

1. **基于语法的解码** (`outlines`、`guidance`、`lm-format-enforcer`)：从 schema 构建一个确定性有限自动机；在每一步，屏蔽那些会违反 FSM 的 token 的 logits。
2. **带有 JSON 解析器的 logit 屏蔽**：与模型同步运行流式 JSON 解析器；在每一步，计算有效的下一个 token 集合。
3. **带有验证器的推测解码**：廉价的草稿模型提议 token，验证器强制执行 schema。

商业提供商在幕后选择其中一种。2026 年的技术水平对于短结构化输出比普通生成更快，对于长输出则大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效的 JSON。在严格模式下不会发生。在非严格模式提供商上仍可能发生。
2. **Schema 违规。** 输出可以解析但违反了 schema。在严格模式下不会发生。在其之外很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你不在严格模式下（Anthropic 工具使用、非严格 OpenAI、旧版 Gemini），恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常足够。三次重试能捕获弱模型的波动。超过三次则表明 schema 有问题：模型无法为某些输入满足它，提示或 schema 需要修复。

### 小模型支持

约束解码也适用于小模型。一个 30 亿参数的开源模型配合语法强制执行，在结构化任务上的表现优于一个使用原始提示的 700 亿参数模型。这是结构化输出对生产环境至关重要的主要原因：它将可靠性与模型大小解耦。

## 使用它

`code/main.py` 在标准库中提供了一个最小的 JSON Schema 2020-12 验证器（类型、必填、枚举、最小/最大值、模式、元素、附加属性）。它封装一个 `Invoice` schema，并通过验证器运行一个伪造的 LLM 输出，演示解析错误、schema 违规和拒绝路径。在生产环境中，可以将伪造输出替换为任何提供商的真实响应。

观察要点：

- 验证器返回一个类型化的 `[ValidationError]` 列表，包含路径和消息。这是你希望暴露给重试提示的格式。
- 拒绝分支不会重试。它记录日志并返回一个类型化的拒绝。阶段 14 · 09 使用拒绝作为安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示了严格模式如何杜绝幻觉字段。

## 部署它

本课生成 `outputs/skill-structured-output-designer.md`。给定一个自由文本的提取目标（发票、支持工单、简历等），该技能生成一个严格模式兼容的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，并已内嵌类型化拒绝和重试处理桩代码。

## 练习

1.  运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器使用 `minimum` 约束路径拒绝它。

2.  扩展验证器以支持带有鉴别器的 `oneOf`。常见情况：`line_item` 要么是产品要么是服务，由 `kind` 标记。严格模式在这里有微妙规则；请查看 OpenAI 的结构化输出指南。

3.  将相同的发票 schema 写为 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手动编写的 schema 进行比较。找出 Pydantic 默认设置但手动版本遗漏的一个字段。

4.  测量拒绝率。构建十个不应可提取的输入（歌词、数学证明、空白邮件），并使用严格模式通过真实提供商运行它们。统计拒绝次数与幻觉输出次数。这是你进行拒绝感知重试的真实依据。

5.  从头到尾阅读 OpenAI 的结构化输出指南。找出它在严格模式下明确禁止而普通 JSON Schema 允许的一个结构。然后设计一个非本质性地使用该禁止结构的 schema，并将其重构为严格模式兼容。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| JSON Schema 2020-12 | "The schema spec" (Schema 规范) | IETF 草案 schema 方言，每个现代提供商都支持 |
| Strict mode | "Guaranteed schema" (保证的 schema) | 通过约束解码强制执行 schema 的 OpenAI 标志 |
| Constrained decoding | "Logit masking" (Logit 屏蔽) | 解码时强制执行，屏蔽无效的下一个 token |
| Refusal | "Model declines" (模型拒绝) | 当输入无法适配 schema 时的类型化结果 |
| Parse error | "Invalid JSON" (无效 JSON) | 输出未解析为 JSON；严格模式下不可能发生 |
| Schema violation | "Wrong shape" (形状错误) | 已解析但违反了类型/必填/枚举/范围 |
| `additionalProperties: false` | "No extras allowed" (不允许额外字段) | 禁止未知字段；OpenAI 严格模式必需 |
| Pydantic BaseModel | "Typed output" (类型化输出) | 发出并验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript output type" (TypeScript 输出类型) | 用于提供商输出验证的 TypeScript 运行时 schema |
| Grammar enforcement | "Open-weights constrained decode" (开源权重约束解码) | 基于 FSM 的 logit 屏蔽，如在 outlines / guidance 中 |

## 延伸阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和 schema 要求
- [OpenAI — 引入 API 中的结构化输出](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布的解释解码保证的博文
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 可序列化到每个提供商的类型化 output_type 绑定
- [JSON Schema — 2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes) — 规范原文
- [Microsoft — Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和严格模式注意事项