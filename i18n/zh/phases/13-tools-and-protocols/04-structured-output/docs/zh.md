# 结构化输出 —— JSON Schema、Pydantic、Zod、约束解码

> "礼貌地请求模型返回 JSON"即使在前沿模型上也会有 5% 到 15% 的失败率。结构化输出通过约束解码弥合了这一差距：模型在解码时就被阻止输出违反 schema 的 token。OpenAI 的 strict mode、Anthropic 的 schema-typed 工具调用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 以及 Zod 的 `.parse` 是同一理念的五个表面形式。本课程构建 schema 验证器和 strict mode 契约，学员将把这套机制应用于每一条生产级抽取管线。

**类型：** 实操构建
**语言：** Python（stdlib、JSON Schema 2020-12 子集）
**前置知识：** Phase 13 · 02（函数调用深入解析）
**时长：** 约 75 分钟

## 学习目标

- 使用合适的约束（enum、min/max、required、pattern）为抽取目标编写 JSON Schema 2020-12。
- 解释为什么 strict mode 和约束解码提供的保证不同于"生成后验证"。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 交付一条带有类型化修复和拒绝处理的抽取管线。

## 问题背景

一个读取采购订单邮件的 agent 需要将自由文本转为 `{customer, line_items, total_usd}` 结构。有三种方案。

**方案一：让 prompt 指定 JSON。** "以 JSON 回复，包含字段 customer、line_items、total_usd。"在前沿模型上 85% 到 95% 的情况下有效。失败场景有六种：缺少花括号、尾部逗号、类型错误、幻觉字段、在 token 上限处截断、输出泄漏出如"以下是你的 JSON："之类的自然语言。

**方案二：生成后验证。** 自由生成、解析、对照 schema 校验，失败则重试。可靠但昂贵——你要为每次重试买单，截断类 bug 每次都会消耗额外一轮对话。

**方案三：约束解码。** 供应商在解码时强制执行 schema。无效 token 被屏蔽出采样分布。输出既保证能解析也保证符合 schema。失败模式收敛为一种：拒绝（模型判定输入无法匹配 schema）。

2026 年的前沿供应商全部提供了某种形式的方案三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，若模型拒绝则在响应中包含 `refusal` 字段。
- **Anthropic。** 在 `tool_use` 输入上执行 schema 强制；不会出现 `stop_reason: "refusal"`，但空工具调用的 `end_turn` 是信号。
- **Gemini。** 请求级 `responseSchema`；2026 年的 Gemini 对部分类型提供 token 级语法规则约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 在运行时校验 provider 输出是否符合 Zod schema；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同点：声明一次 schema，端到端强制执行。

## 概念

### JSON Schema 2020-12 —— 通用语

每个 provider 都接受 JSON Schema 2020-12。你最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字）、`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用到每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认值随模式而异）。

OpenAI strict mode 额外要求三条：每个属性都必须在 `required` 中列出、`additionalProperties: false` 全局生效、不存在未解析的 `$ref`。违反这些会在请求时收到 400 错误。

### Pydantic —— Python 绑定

Pydantic v2 通过 `model_json_schema()` 从 dataclass 形态的模型生成 JSON Schema。Pydantic AI 对此做了封装，让你只需写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent 框架在边缘侧把 schema 转换为 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出会以类型化的 `Invoice` 实例返回。验证错误会抛出带有类型化错误路径的 `ValidationError`。

### Zod —— TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 的等价实现。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，将其翻译为 API 的 JSON Schema payload。

### 拒绝（Refusal）

Strict mode 无法强迫模型回答。如果输入无法适配 schema（"那封邮件是一首诗歌，不是发票"），模型会返回包含理由的 `refusal` 字段。你的代码必须把它当作一等公民来处理，而不是失败。拒绝同时也是安全信号：被要求从受保护内容邮件中提取信用卡号时，模型会返回附带安全理由的拒绝。

### 开放权重的约束解码

开源实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：根据 schema 构建确定有限状态机（DFA）；在每个步骤中屏蔽会违反 FSM 的 token 的 logits。
2. **带 JSON 解析器的 logit 屏蔽**：与模型同步运行流式 JSON 解析器；在每个步骤计算合法下一个 token 集合。
3. **带验证器的投机解码**：廉价草稿模型提议 token，验证器强制执行 schema。

商业供应商在后台选择其中一种。2026 年的 SOTA 对短结构化输出比裸生成更快，对长输出速度大致相当。

### 三种失败模式

1. **解析错误。** 输出不是合法 JSON。strict mode 下不可能发生。非 strict provider 仍可能发生。
2. **schema 违规。** 输出能解析但违反了 schema。strict mode 下不可能发生。在非 strict 环境下很常见。
3. **拒绝。** 模型选择不回答。必须作为类型化结果处理。

### 重试策略

当你处于 strict mode 之外时（Anthropic 工具调用、非 strict OpenAI、旧版 Gemini），恢复模式为：

```
generate -> parse -> validate -> 失败则注入错误并重试，最多 3 次
```

一次重试通常就够。三次重试能捕获弱模型的偶发问题。超过三次是 schema 有问题的信号：模型在某些输入上无法满足它，需要修复 prompt 或 schema。

### 小模型支持

约束解码在小模型上也有效。带语法强制的 3B 参数开源模型在结构化任务上胜过带裸 prompt 的 70B 参数模型。这是结构化输出对生产至关重要的主要原因：它将可靠性与模型规模解耦。

```figure
constrained-decoding
```

## 使用方式

`code/main.py` 提供一个基于 stdlib 的最小 JSON Schema 2020-12 验证器（涵盖 type、required、enum、min/max、pattern、items、additionalProperties）。它包裹一个 `Invoice` schema 并让伪造 LLM 输出通过验证器，演示解析错误、schema 违规和拒绝三种路径。在实际生产中把伪造输出换成任意 provider 的真实响应即可。

值得关注的点：

- 验证器返回类型化的 `[ValidationError]` 列表，包含 path 和 message。这正是你应该在重试 prompt 中暴露的形状。
- 拒绝分支不做重试。它会记录日志并返回类型化的拒绝结果。Phase 14 · 09 会把拒绝作为安全信号使用。
- `additionalProperties: false` 检查会在对抗测试输入上触发，展示为什么 strict mode 能阻断幻觉字段。

## 交付物

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本抽取目标（发票、工单、简历等），skill 会生成一个兼容 strict mode 的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，同时预留类型化的拒绝和重试处理桩代码。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器通过 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带判别字的 `oneOf`。典型场景：`line_item` 要么是商品要么是服务，由 `kind` 字段标记。strict mode 在此处有微妙规则；查阅 OpenAI 的结构化输出指南。

3. 将同一个 Invoice schema 写为 Pydantic BaseModel，并对比 `model_json_schema()` 输出与你手写的 schema。找出 Pydantic 默认设置而你手写版本遗漏的那个字段。

4. 测量拒绝率。构造十个不可抽取的输入（一首歌词、一道数学证明、一封空白邮件）并以 strict mode 跑真实 provider。统计拒绝数与幻觉输出数。这是你拒绝感知重试的地面真相。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在 strict mode 中明确禁止但普通 JSON Schema 允许的某个构造。然后设计一个非必需使用该构造的 schema 并将其重构为 strict 兼容。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| JSON Schema 2020-12 | "schema 规范" | IETF 草案 schema 方言，现代 provider 都支持 |
| Strict mode | "保证 schema" | OpenAI 的标志位，通过约束解码强制执行 schema |
| 约束解码 | "logit 屏蔽" | 解码时强制屏蔽非法下一个 token |
| 拒绝 | "模型不答" | 输入无法匹配 schema 时的类型化结果 |
| 解析错误 | "无效 JSON" | 输出未解析为 JSON；strict mode 下不可能发生 |
| schema 违规 | "形状不对" | 已解析但违反了类型/必填/枚举/范围 |
| `additionalProperties: false` | "不允许多余字段" | 禁止未知字段；OpenAI strict 模式必需 |
| Pydantic BaseModel | "类型化输出" | 能发射并验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | TS 运行时 schema，用于校验 provider 输出 |
| 语法强制 | "开放权重约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、拒绝处理和 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布帖，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 类型化 `output_type` 绑定，序列化到各 provider
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 权威规范
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和 strict mode 注意事项
