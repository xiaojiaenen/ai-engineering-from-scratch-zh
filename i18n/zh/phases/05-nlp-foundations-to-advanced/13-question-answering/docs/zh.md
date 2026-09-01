# 问答系统

> 三种系统塑造了现代 QA。抽取式找到答案片段。检索增强将其锚定在文档中。生成式产出答案。每个现代 AI 助手都是这三者的混合。

**类型:** 构建
**语言:** Python
**前置条件:** 阶段 5 · 11（机器翻译），阶段 5 · 10（注意力机制）
**耗时:** 约 75 分钟

## 问题描述

用户输入"初代 iPhone 是什么时候发布的？"，期望得到"2007 年 6 月 29 日"这样的直接答案。而不是"苹果的历史悠久而多样。"也不是孤零零的"2007"。需要一个直接、有据、准确的答案。

在过去十年中，三种架构主导了 QA 领域。

- **抽取式 QA。** 给定一个问题和一个已知包含答案的段落，找出答案片段在段落中的起始和结束索引。SQuAD 是标准评测基准。
- **开放域 QA。** 段落未给出。先检索相关段落，然后抽取或生成答案。这是当前每个 RAG 管道的基石。
- **生成式 / 闭卷 QA。** 大型语言模型从其参数记忆中作答。无需检索。推理速度最快，但事实可靠性最低。

2026 年的趋势是混合架构：检索最相关的几个段落，然后提示生成式模型基于这些段落作答。这就是 RAG，第 14 课将深入讲解检索部分。本课构建 QA 部分。

## 概念解析

![QA 架构：抽取式、检索增强、生成式](../assets/qa.svg)

**抽取式。** 使用 Transformer（BERT 系列）联合编码问题和段落。训练两个头节点预测答案的起始和结束 token 索引。损失函数为有效位置上的交叉熵。输出是段落中的一个片段。永远不会幻觉（由设计保证），也无法处理段落无法回答的问题（由设计保证）。

**检索增强（RAG）。** 两个阶段。首先，检索器从语料库中找到 top-k 个段落。其次，阅读器（抽取式或生成式）使用这些段落生成答案。检索器与阅读器的分离使得两者可以独立训练和评估。现代 RAG 通常在两者之间加入重排序器。

**生成式。** 仅解码器的 LLM（GPT、Claude、Llama）从学习到的权重中作答。无检索步骤。在常识上表现优秀，但在罕见或近期事实上表现灾难性。幻觉率与预训练数据中事实的出现频率呈负相关。

```figure
qa-span
```

## 动手实现

### 步骤 1：使用预训练模型进行抽取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 是在 SQuAD 2.0 上训练的，其中包含无法回答的问题。默认情况下，`question-answering` 管道即使模型的无效分数最高，也会返回得分最高的片段——它不会自动返回空答案。要获得显式的"无答案"行为，请在管道调用中传入 `handle_impossible_answer=True`：此时管道仅在无效分数超过所有片段分数时才返回空答案。无论如何都要检查 `score` 字段。

### 步骤 2：检索增强管道（示意图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    # 对问题进行编码
    q_emb = encoder.encode([question], normalize_embeddings=True)
    # 计算余弦相似度
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    # 取 top-k
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    # 检索相关段落
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管道。密集检索器（Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的 top 段落中提取答案片段。适用于小规模语料库。对于百万级文档的语料库，请使用 FAISS 或向量数据库。

### 步骤 3：带 RAG 的生成式

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式很关键。明确告诉模型基于上下文作答，并在上下文不足时返回"I don't know"，相比朴素提示可将幻觉率降低 40-60%。更复杂的模式会添加引用、置信度分数和结构化提取。

### 步骤 4：反映真实世界的评估

SQuAD 使用 **精确匹配（EM）** 和 **token 级 F1**。EM 是在标准化后（小写、去除标点、去除冠词）的严格匹配——预测完全匹配则得分，否则为 0。F1 基于预测与参考之间的 token 重叠计算，给出部分分数。两者都会低估同义 paraphrase："June 29, 2007"与"June 29th, 2007"通常得到 0 EM（序数词破坏了标准化），但仍能通过重叠 token 获得显著的 F1。

对于生产环境 QA：

- **答案准确性**（由 LLM 或人工判断，因为指标无法捕捉语义等价性）。
- **引用准确性。** 引用的段落是否真正支持答案？通过生成的引用与检索段落之间的字符串匹配可轻松自动检查。
- **拒绝校准。** 当答案不在检索到的段落中时，系统是否正确地说"我不知道"？测量虚假置信率。
- **检索召回率。** 在评估阅读器之前，先测量检索器是否将正确段落放入 top-k。阅读器无法修复缺失的段落。

### RAGAS：2026 生产评估框架

`RAGAS` 专为 RAG 系统设计，是 2026 年的发货默认选择。它在不需要金标准参考的情况下对四个维度进行评分：

- **忠实度。** 答案中的每个主张是否来自检索到的上下文？通过基于 NLI 的蕴含关系测量。这是主要的幻觉指标。
- **答案相关性。** 答案是否回答了问题？通过从答案生成假设性问题并与真实问题比较来测量。
- **上下文精确度。** 在检索到的片段中，有多少比例是真正相关的？低精确度 = 提示中有噪声。
- **上下文召回率。** 检索集合是否包含了所有必要信息？低召回率 = 阅读器无法成功。

无参考评分让你能够在没有精心策划的金标准答案的情况下评估实时生产流量。针对精确匹配指标无用的开放式问题，在此基础上叠加 LLM-as-judge。

`pip install ragas`。接入你的检索器 + 阅读器。为每个查询获得四个标量。对回归告警。

## 如何使用

2026 技术栈。

| 用例 | 推荐方案 |
|------|---------|
| 给定段落，找答案片段 | `deepset/roberta-base-squad2` |
| 在固定语料库上，不接受闭卷回答 | RAG：密集检索器 + LLM 阅读器 |
| 在文档库上实时查询 | 带混合检索器（BM25 + 密集）和重排序器的 RAG（第 14 课） |
| 对话式 QA（跟进问题） | LLM 配合对话历史 + 每轮使用 RAG |
| 高度事实性、受监管领域 | 在权威语料库上进行抽取式 QA；绝不单独使用生成式 |

抽取式 QA 在 2026 年不那么流行，因为带 LLM 的 RAG 能处理更多场景。它仍在需要字面引用的场景中出货：法律研究、监管合规、审计工具。

## 交付物

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: 选择 QA 架构、检索策略和评估方案。
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

根据需求（语料库大小、问题类型、事实性约束、延迟预算）输出：

1. 架构。抽取式、带抽取式阅读器的 RAG、带生成式阅读器的 RAG，或闭卷 LLM。一句话说明理由。
2. 检索器。无、BM25、密集（指明编码器名称），或混合。
3. 阅读器。SQuAD 调优模型、指定名称的 LLM，或"领域微调的 DistilBERT"。
4. 评估。抽取式基准使用 EM + F1；生产环境使用答案准确性 + 引用准确性 + 拒绝校准。指明你正在测量什么以及如何测量。

对于监管或合规敏感的问题，拒绝闭卷 LLM 的答案。对于没有检索召回基线的任何 QA 系统，拒绝使用（不了解检索器是否检索到了正确段落就无法评估阅读器）。标记需要多跳推理的问题，指出需要专门的多跳检索器，如 HotpotQA 训练的系统。
```

## 练习题

1. **简单。** 在上面设置 SQuAD 抽取式管道，处理 10 个 Wikipedia 段落。手工编写 10 个问题。测量答案正确的频率。如果段落和问题都很清晰，你应该能看到 7-9 个正确。
2. **中等。** 添加一个拒绝分类器。当最佳检索分数低于阈值（例如 0.3 余弦相似度）时，返回"我不知道"而不是调用阅读器。在保留集上调整阈值。
3. **困难。** 在你选择的 10,000 文档语料库上构建 RAG 管道。实现混合检索（BM25 + 密集）并使用 RRF 融合（见第 14 课）。测量有无混合步骤时的答案准确性。记录哪些类型的问题获益最大。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 抽取式 QA | 找到答案片段 | 预测给定段落中答案的起始和结束索引。 |
| 开放域 QA | 在语料库上 QA | 没有给定段落；必须检索后再作答。 |
| RAG | 检索后生成 | 检索增强生成。检索器 + 阅读器管道。 |
| SQuAD | 标准基准 | Stanford Question Answering Dataset。EM + F1 指标。 |
| 幻觉 | 编造的答案 | 检索器输出的内容未被检索到的上下文支持。 |
| 拒绝校准 | 知道何时该闭嘴 | 系统无法回答时正确地说"我不知道"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 的标准密集检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
