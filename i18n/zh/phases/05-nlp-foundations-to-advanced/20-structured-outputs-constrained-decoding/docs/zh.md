# 结构化输出与约束解码

> 让 LLM 输出 JSON。大多数时候你能得到 JSON。但在生产环境中，"大多数"就是问题所在。约束解码通过在采样前修改 logit，将"大多数"变为"始终"。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段 · 17（聊天机器人），第5阶段 · 19（子词分词）
**时间：** 约 60 分钟

## 问题所在

一个分类器提示 LLM："从 {positive, negative, neutral} 中返回一个。"模型返回了："情感是积极的——这个评论 overwhelmingly favorable，因为客户明确表示他们……"。你的解析器崩溃了。你的分类器的 F1 分数为 0.0。

自由格式生成不是一份契约，它只是一个建议。生产系统需要一份契约。

2026年存在三个层次。

1. **提示工程。** 礼貌地请求。"只返回 JSON 对象。"对前沿模型大约有效 ~80%，对小模型则更差。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。在支持的 schema 上可靠。但供应商锁定。
3. **约束解码。** 在每个生成步骤修改 logits，使模型*无法*发出无效 token。100% 由构造保证合法。适用于任何本地模型。

本课程建立对所有三种方法的直觉，并指明何时使用哪一种。

## 概念

![约束解码在每个步骤屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每个生成步骤，LLM 会对完整词汇表（约 10 万 token）产生一个 logit 向量。*logit 处理器*位于模型和采样器之间。它根据目标语法（JSON Schema、正则表达式、上下文无关语法）在当前位置的合法性计算出哪些 token 是有效的，并将所有无效 token 的 logits 设为负无穷。对剩余 logits 做 softmax，概率质量仅分配给合法的后继 token。

2026年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机。每个 token 都有 O(1) 合法后继 token 查询。基于 FSM，因此递归 schema 需要展平。
- **XGrammar / llguidance。** 上下文无关语法引擎。处理递归 JSON Schema。解码开销近乎为零。OpenAI 在 2025 年的结构化输出实现中致谢了 llguidance。
- **vLLM guided decoding。** 通过 Outlines、XGrammar 或 lm-format-enforcer 后端内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`。
- **Instructor。** 基于 Pydantic 的任意 LLM 封装层。在验证失败时重试。跨供应商，但不修改 logits——依赖重试 + 感知结构化输出的提示。

### 反直觉的结果

约束解码往往比无约束生成*更快*。两个原因。首先，它缩小了下一个 token 的搜索空间。其次，聪明的实现在强制 token（如 `{"name": "` 这样的脚手架——每个字节都已确定）上完全跳过 token 生成。

### 代价高昂的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 之前，模型会在思考之前就 commit 一个答案。JSON 是合法的。但答案是错的。没有验证能捕捉到这一点。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

```figure
constrained-decoder
```

## 构建它

### 步骤 1：从零开始实现正则约束生成

参见 `code/main.py` 中的独立 FSM 实现。核心思想在 30 行代码中：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪我们迄今为止满足的语法部分。`valid_tokens(state, tokenizer)` 计算哪些词汇表 token 可以推进 FSM 而不脱离接受路径。

### 步骤 2：使用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远如此。FSM 使无效输出不可达。

### 步骤 3：使用 Instructor 实现供应商无关的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同的机制。Instructor 不触碰 logits。它将 schema 格式化到提示中，解析输出，并在验证失败时重试（默认 3 次）。适用于任何供应商。重试会增加延迟和成本。跨供应商可移植性是它的卖点。

### 步骤 4：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。对支持的 schema 可靠性与 Outlines 持平。无需本地模型管理。将你锁定在供应商生态中。

## 陷阱

- **递归 schema。** Outlines 将递归展平为固定深度。树状结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举。** 10,000 个选项的枚举编译缓慢或超时。改用检索器：先预测 top-k 候选，再约束到这些候选。
- **语法过于严格。** 强制 `date: "YYYY-MM-DD"` 正则，模型就无法为缺失日期输出 `"unknown"`。模型会通过编造一个日期来补偿。允许 `null` 或哨兵值。
- **过早 commit。** 参见字段顺序陷阱。始终先放推理，再放答案。
- **无 schema 的供应商 JSON mode。** 纯 JSON mode 只保证 JSON 语法合法，不保证对你的用例合法。始终提供完整的 schema。

## 如何使用

2026 技术栈：

| 场景 | 选择 |
|------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任意供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 合法性，扁平 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 批量处理，可接受重试 | Instructor + 最便宜的模型 |

## 交付物

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择结构化输出方法、schema 设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

根据使用场景（供应商、延迟预算、schema 复杂度、失败容忍度），输出：

1. 机制。原生供应商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。一句话说明理由。
2. Schema 设计。字段顺序（先推理后答案）、"unknown" 的 nullable 字段、enum vs 正则、必填字段。
3. 失败策略。最大重试次数、降级模型、优雅 `null` 处理、分布外拒绝。
4. 验证计划。schema 合规率（目标 100%）、语义合法性（LLM-judge）、字段覆盖率、延迟 p50/p99。

拒绝任何将 `answer` 或 `decision` 置于推理字段之前的设计。拒绝在无 schema 时使用裸 JSON mode。标记在仅支持 FSM 的库后面的递归 schema。
```

## 练习

1. **简单。** 提示一个小尺寸开源模型（如 Llama-3.2-3B），不使用约束解码来生成 `Review(sentiment, confidence, evidence_span)`。在 100 条评论上测量能解析为合法 JSON 的比例。
2. **中等。** 使用 Outlines JSON mode 对同一语料进行处理。比较合规率、延迟和语义准确性。
3. **困难。** 从零实现一个用于电话号码（`\d{3}-\d{3}-\d{4}`）的正则约束解码器。在 1000 个样本上验证输出无效率为 0。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 约束解码 | 强制合法输出 | 在每个生成步骤屏蔽无效 token 的 logits。 |
| Logit 处理器 | 约束的那东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；O(1) 合法后继 token 查询。 |
| CFG | 上下文无关语法 | 能处理递归的语法；比 FSM 更慢但表达力更强。 |
| Schema 字段顺序 | 有影响吗？ | 有——第一个字段会触发 commit；始终将推理放在答案之前。 |
| Guided decoding | vLLM 的叫法 | 同一概念，集成到推理服务器中。 |
| JSON mode | OpenAI 的早期版本 | 保证 JSON 语法合法；不保证与 schema 匹配。 |

## 延伸阅读

- [Willard, Louf（2023）。Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) —— Outlines 论文。
- [XGrammar 论文（2024）](https://arxiv.org/abs/2411.15100) —— 快速基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) —— 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) —— API 参考 + 注意事项。
- [Instructor 库](https://python.useinstructor.com/) —— 跨供应商的 Pydantic + 重试。
- [JSONSchemaBench（2025）](https://arxiv.org/abs/2501.10868) —— 6 种约束解码框架的基准测试。
