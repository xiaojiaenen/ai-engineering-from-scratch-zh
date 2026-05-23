# 问答系统

> 三种系统塑造了现代问答技术。抽取式系统寻找文本片段，检索增强系统将答案锚定于文档，生成式系统直接生成答案。如今每个AI助手都是这三种技术的混合体。

**类型：** 构建项目  
**语言：** Python  
**前置课程：** 第五阶段·11（机器翻译）、第五阶段·10（注意力机制）  
**时间：** 约75分钟

## 问题描述

用户输入“初代iPhone何时发布？”，期望得到“2007年6月29日”。而非“苹果历史悠长多样”，也不是孤立存在的“2007”——用户需要的是直接、准确、有依据的答案。

过去十年，三种架构主导了问答领域。

- **抽取式问答。** 给定问题和已知包含答案的段落，定位答案在段落中的起止位置。SQuAD是该领域的标准基准。
- **开放域问答。** 不预先提供段落。先检索相关段落，再抽取或生成答案。这是当前所有RAG流程的基础。
- **生成式/闭卷问答。** 大型语言模型基于参数记忆作答，无需检索。推理速度最快，但事实可靠性最低。

2026年的趋势是混合模式：检索最相关的数个段落，然后提示生成式模型基于这些段落生成答案。这便是RAG（检索增强生成），第14课将深入讲解检索部分，本课程则构建问答部分。

## 核心概念

![问答架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 使用Transformer（BERT系列）联合编码问题与段落。训练两个预测头分别输出答案起止token索引。损失函数基于有效位置的交叉熵计算，输出为段落中的文本片段。按设计机制不会产生幻觉（构造限制），也无法处理段落无法回答的问题（构造限制）。

**检索增强式（RAG）。** 分两阶段执行。首先检索器从语料库中查找前`k`个相关段落；其次阅读器（抽取式或生成式）基于这些段落生成答案。检索器-阅读器分离的设计允许独立训练和评估。现代RAG常在此两者间增加重排序器。

**生成式。** 仅解码器的大语言模型（GPT、Claude、Llama）基于学习到的权重直接作答，无检索环节。对常识问题表现优异，对罕见或近期事实易出错。幻觉率与预训练数据中事实的出现频率呈负相关。

## 实现构建

### 步骤一：基于预训练模型的抽取式问答

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

`deepset/roberta-base-squad2`在SQuAD 2.0数据集上训练，该数据集包含不可回答问题。默认情况下，`question-answering`管线会返回最高分文本片段，即使模型的空值评分更高——它不会*自动*返回空答案。要实现明确的“无答案”行为，需在调用管线时传递`handle_impossible_answer=True`参数：此时仅当空值评分超过所有片段评分时才返回空答案。无论哪种情况，都应检查`score`字段。

### 步骤二：检索增强管线（框架示意）

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
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管线。稠密检索器（Sentence-BERT）通过语义相似度查找相关段落；抽取式阅读器（RoBERTa-SQuAD）从合并的顶部段落中提取答案片段。适用于小型语料库。若处理百万级文档库，需使用FAISS或向量数据库。

### 步骤三：基于RAG的生成式问答

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

提示模式至关重要。明确指示模型基于上下文作答，并在上下文不足时回复“我不知道”，相比简单提示可降低40-60%的幻觉率。更精细的提示模式还可添加引用来源、置信度评分和结构化提取。

### 步骤四：贴近现实的评估方法

SQuAD采用**完全匹配（EM）**和**token级F1**指标。EM在规范化处理（小写化、去除标点、删除冠词）后进行严格匹配——预测必须完全一致才得分，否则为零。F1基于预测与参考答案的token重叠度计算，给予部分分数。两者对同义表达评分偏低：“June 29, 2007”与“June 29th, 2007”通常EM为0（序数词导致规范化失败），但因token重叠仍能获得较高F1分数。

生产环境问答系统评估需关注：

- **答案准确性**（由LLM或人工判断，因指标无法捕捉语义等价性）
- **引用准确性。** 引用的段落是否实际支持答案？通过生成引用与检索段落的字符串匹配即可自动验证
- **拒答校准。** 当答案不在检索段落中时，系统能否正确回复“我不知道”？需测量误自信率
- **检索召回率。** 评估阅读器前，先检验检索器是否将正确段落置于前`k`位。若段落缺失，阅读器无法补救

### RAGAS：2026年生产环境评估框架

`RAGAS`专为RAG系统设计，是2026年的标准工具。无需标准参考答案即可评估四个维度：

- **忠实度。** 答案中的每个断言是否均来自检索上下文？通过基于NLI的蕴含关系衡量。这是核心幻觉评估指标
- **答案相关性。** 答案是否切题？通过从答案生成假设问题并与原问题比较来衡量
- **上下文精确度。** 检索到的文本块中，实际相关的比例是多少？低精确度意味着提示中存在噪音
- **上下文召回率。** 检索结果是否包含所有必要信息？低召回率将导致阅读器失败

无参考评分使系统能直接评估线上生产流量，无需构建标准答案集。对开放式问题可叠加LLM-as-judge，因为精确匹配指标在此场景无效。

`pip install ragas`。接入您的检索器+阅读器，即可为每个查询生成四个标量指标，并设置回归警报。

## 应用场景

2026年技术栈推荐方案：

| 用例 | 推荐方案 |
|---------|-------------|
| 给定段落，查找答案片段 | `deepset/roberta-base-squad2` |
| 固定语料库，不接受闭卷问答 | RAG：稠密检索器+LLM阅读器 |
| 实时文档库查询 | 混合检索（BM25+稠密）+重排序器的RAG（详见第14课） |
| 对话式问答（支持追问） | 带对话历史的LLM+每轮RAG检索 |
| 高事实性、受监管领域 | 基于权威语料库的抽取式问答；禁用纯生成式 |

抽取式问答在2026年非主流，因RAG+LLM能处理更多场景。但在需要逐字引用的场景（法律研究、合规审查、审计工具）中仍不可或缺。

## 部署实现

保存为`outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习任务

1. **初级。** 在10个维基百科段落上实现上述SQuAD抽取式管线。手写10个问题，统计答案正确率。若段落与问题质量过关，预期正确率应达7-9个。
2. **中级。** 增加拒答分类器。当检索最高分低于阈值（如余弦相似度0.3）时，返回“我不知道”而非调用阅读器。在保留集上调整阈值。
3. **高级。** 基于自选万级文档库构建RAG管线。实现混合检索（BM25+稠密）配合RRF融合（参见第14课），对比混合步骤前后的答案准确率，记录哪些问题类型受益最大。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答 | 查找答案片段 | 预测给定段落中答案的起止索引位置 |
| 开放域问答 | 语料库问答 | 不提供预设段落，需先检索后回答 |
| RAG | 检索后生成 | 检索增强生成，检索器+阅读器管线 |
| SQuAD | 标准基准 | 斯坦福问答数据集，采用EM+F1指标 |
| 幻觉 | 编造答案 | 阅读器输出未获检索上下文支持 |
| 拒答校准 | 知何时沉默 | 系统在无法回答时正确回复“我不知道” |

## 延伸阅读

- [Rajpurkar等 (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文
- [Karpukhin等 (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — 问答领域标准稠密检索器DPR
- [Lewis等 (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名RAG的开创性论文
- [Gao等 (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — RAG综合综述