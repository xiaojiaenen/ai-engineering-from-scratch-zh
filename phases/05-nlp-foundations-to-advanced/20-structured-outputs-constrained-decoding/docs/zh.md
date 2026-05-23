# 结构化输出与约束解码

> 向大语言模型请求 JSON，大多数时候都能得到 JSON。但在生产环境中，"大多数"就是问题所在。约束解码通过在采样前编辑对数几率，将"大多数"变成"始终"。

**类型：** 构建  
**语言：** Python  
**前置课程：** 第5阶段·第17课（聊天机器人），第5阶段·第19课（子词分词）  
**时长：** 约60分钟

## 问题所在

一个分类器提示大语言模型："返回 {积极，消极，中性} 中的一个。" 模型返回："这种情绪是积极的——这篇评论非常正面，因为客户明确表示他们……"。你的解析器崩溃了。你的分类器的 F1 分数是 0.0。

自由生成不是契约。它只是一种建议。生产系统需要一份契约。

在 2026 年，存在三个层面。

1.  **提示工程。** 礼貌地请求。"只返回 JSON 对象。" 在前沿模型上大约 80% 的情况下有效，在较小的模型上效果更差。
2.  **原生结构化输出 API。** OpenAI `response_format`、Anthropic 工具使用、Gemini JSON 模式。对于支持的架构是可靠的。供应商锁定。
3.  **约束解码。** 在每个生成步骤修改对数几率，使模型*无法*输出无效 token。通过构建保证 100% 有效。适用于任何本地模型。

本课旨在为这三种方式建立直觉，并说明何时该选择哪一种。

## 概念解析

![约束解码在每一步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每个生成步骤，大语言模型会基于整个词汇表（约 10 万个 token）产生一个对数几率向量。一个*对数几率处理器*位于模型和采样器之间。它根据当前在目标语法（JSON Schema、正则表达式、上下文无关文法）中的位置，计算哪些 token 是有效的，并将所有无效 token 的对数几率设置为负无穷大。对剩余对数几率进行 softmax 操作后，概率质量只分配给有效的后续 token。

2026 年的实现：

-   **Outlines。** 将 JSON Schema 或正则表达式编译成有限状态机。每个 token 都有 O(1) 的有效下一 token 查找。基于 FSM，因此递归架构需要扁平化处理。
-   **XGrammar / llguidance。** 上下文无关文法引擎。处理递归 JSON Schema。解码开销接近于零。OpenAI 在其 2025 年结构化输出实现中提到了 llguidance。
-   **vLLM 引导式解码。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端实现。
-   **Instructor。** 基于 Pydantic 的、适用于任何大语言模型的包装器。在验证失败时重试。跨供应商，但不修改对数几率——它依赖于重试和结构化输出感知提示。

### 反直觉的结果

约束解码通常比无约束生成*更快*。有两个原因。第一，它缩小了下一 token 的搜索空间。第二，巧妙的实现会为强制 token（如 `{"name": "` 这样的脚手架——每个字节都是确定的）完全跳过 token 生成。

### 代价高昂的陷阱

字段顺序很重要。如果把 `answer` 放在 `reasoning` 之前，模型就会在思考之前就给出答案。JSON 是有效的。答案是错误的。没有验证能捕获到这一点。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

架构字段顺序是逻辑，不是格式。

## 动手构建

### 第 1 步：从零开始实现正则表达式约束生成

参见 `code/main.py` 了解一个独立的 FSM 实现。30 行代码的核心思想如下：

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

FSM 跟踪我们到目前为止满足了语法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词汇表 token 可以推进 FSM 而不会离开接受路径。

### 第 2 步：使用 Outlines 实现 JSON Schema 约束

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

零验证错误。永远没有。FSM 使无效输出无法达到。

### 第 3 步：使用 Instructor 实现供应商无关的 Pydantic

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

机制不同。Instructor 不接触对数几率。它将架构格式化到提示中，解析输出，并在验证失败时重试（默认 3 次）。适用于任何供应商。重试会增加延迟和成本。跨供应商可移植性是其卖点。

### 第 4 步：使用原生供应商 API

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

服务器端约束解码。对于支持的架构，可靠性与 Outlines 相当。无需本地模型管理。将你锁定在供应商上。

## 陷阱提示

-   **递归架构。** Outlines 将递归展平到固定深度。树状结构输出（嵌套注释、AST）需要 XGrammar 或 llguidance（基于 CFG）。
-   **大型枚举。** 包含 10,000 个选项的枚举编译缓慢或超时。切换到检索器：先预测 top-k 候选项，然后约束到这些候选项。
-   **语法过于严格。** 强制使用 `date: "YYYY-MM-DD"` 正则表达式，模型就无法为缺失的日期输出 `"unknown"`。模型会通过编造一个日期来补偿。允许 `null` 或使用哨兵值。
-   **过早承诺。** 参见上面的字段顺序陷阱。始终把推理放在前面。
-   **供应商无架构的 JSON 模式。** 纯 JSON 模式只保证是有效的 JSON，不保证*对你的用例*有效。始终提供完整的架构。

## 使用场景

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单架构 | 原生供应商结构化输出 |
| 任何供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平架构 | Outlines (FSM) |
| 本地模型，递归架构 | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导式解码 |
| 可接受重试的批量处理 | Instructor + 最便宜的模型 |

## 部署上线

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1.  **简单。** 使用一个小的开源模型（例如 Llama-3.2-3B），在不使用约束解码的情况下，为 `Review(sentiment, confidence, evidence_span)` 生成提示。在 100 条评论上测量能解析为有效 JSON 的比例。
2.  **中等。** 在相同语料库上使用 Outlines JSON 模式。比较合规率、延迟和语义准确性。
3.  **困难。** 从零开始为电话号码 (`\d{3}-\d{3}-\d{4}`) 实现一个正则表达式约束解码器。在 1000 个样本上验证无效输出为 0。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码 | 强制输出有效结果 | 在每个生成步骤屏蔽无效 token 的对数几率。 |
| 对数几率处理器 | 约束的那个东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；O(1) 的有效下一 token 查找。 |
| CFG | 上下文无关文法 | 能处理递归的语法；比 FSM 慢但表达能力更强。 |
| 架构字段顺序 | 这有关系吗？ | 是的——第一个字段就会做出承诺；始终把推理放在答案之前。 |
| 引导式解码 | vLLM 对它的叫法 | 相同概念，集成到推理服务器中。 |
| JSON 模式 | OpenAI 的早期版本 | 保证 JSON 语法；不保证符合架构。 |

## 延伸阅读

-   [Willard, Louf (2023). 用于大语言模型的高效引导式生成](https://arxiv.org/abs/2307.09702) —— Outlines 论文。
-   [XGrammar 论文 (2024)](https://arxiv.org/abs/2411.15100) —— 快速的基于 CFG 的约束解码。
-   [vLLM — 结构化输出](https://docs.vllm.ai/en/latest/features/structured_outputs.html) —— 推理服务器集成。
-   [OpenAI — 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs) —— API 参考和注意事项。
-   [Instructor 库](https://python.useinstructor.com/) —— 跨供应商的 Pydantic + 重试。
-   [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) —— 6 种约束解码框架的基准测试。