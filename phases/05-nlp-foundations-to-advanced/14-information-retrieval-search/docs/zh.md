# 信息检索与搜索

> BM25 精确但脆弱。密集检索范围广但会遗漏关键词。混合检索是 2026 年的默认方案。其他一切皆为调优。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 5 · 02（词袋模型 + TF-IDF），阶段 5 · 04（GloVe、FastText、子词）  
**时间：** ~75 分钟

## 问题所在

用户输入“如果有人为了钱撒谎会怎样”，期望能找到实际涵盖此情况的法规：“刑法第 420 条”。关键词搜索完全找不到它（没有共享词汇）。如果嵌入未在法律文本上训练过，语义搜索也会错过它。真正的搜索必须能处理这两种情况。

信息检索是每个 RAG 系统、每个搜索栏、每个文档站模糊查询背后的管道。2026 年能在生产环境中工作的架构不是单一方法，而是一条互补方法链，每一种方法都前一种方法的失败之处进行补救。

本课将构建每个组件，并指明每个组件能捕获哪些失败。

## 概念

![混合检索：BM25 + 稠密检索 + RRF + 交叉编码器重排序](../assets/retrieval.svg)

四个层级。选择你需要的。

1.  **稀疏检索（BM25）。** 快速，精确匹配准确，语义理解能力差。运行在倒排索引上。在百万级文档库上，每次查询低于 10 毫秒。能正确找到法规引用、产品代码、错误消息、命名实体。
2.  **稠密检索。** 将查询和文档编码为向量。进行最近邻搜索。能捕获转述和语义相似性。会错过那些仅有一个字符不同的精确关键词匹配。使用 FAISS 或向量数据库，每次查询 50-200 毫秒。
3.  **融合。** 合并来自稀疏和稠密检索的排序列表。倒数排名融合（RRF）是简单的默认选择，因为它忽略了原始分数（这些分数处于不同尺度），只使用排名位置。当你知道在你的领域某个信号占主导时，加权融合是一个选项。
4.  **交叉编码器重排序。** 取融合后的前 30 名。运行交叉编码器（查询 + 文档一起输入，对每对进行评分）。保留前 5 名。交叉编码器比双编码器在每对上更慢，但准确度高得多。你通过只在前 30 名上运行它们来摊薄成本。

三路检索（BM25 + 稠密检索 + 学习型稀疏检索如 SPLADE）在 2026 年的基准测试中优于两路检索，但需要为学习型稀疏索引构建基础设施。对于大多数团队，两路检索加上交叉编码器重排序是最佳平衡点。

## 动手构建

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

有两个值得了解的参数。`k1=1.5` 控制词频饱和度；越高意味着词频重复的权重越大。`b=0.75` 控制长度归一化；0 表示忽略文档长度，1 表示完全归一化。默认值是原始论文中 Robertson 的建议，通常无需调优。

### 第 2 步：使用双编码器进行稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入进行 L2 归一化，使得点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，并且对大多数英语检索足够强大。对于多语言工作，使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高精度，使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。`k` 越高会拉平排名差异的贡献；`k` 越低则使排名靠前的位置主导结果。60 是论文中给出的默认值，通常无需调优。

### 第 4 步：混合搜索 + 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三阶段组合。BM25 找到词汇匹配。稠密检索找到语义匹配。RRF 合并两个排名，无需进行分数校准。交叉编码器使用查询-文档对对前 30 名重新评分，这能捕获双编码器遗漏的细粒度相关性。保留前 5 名。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档存在的查询中，它出现在前 k 名中的频率是多少？ |
| MRR（平均倒数排名） | 第一个相关文档的 1/排名的平均值。 |
| nDCG@k | 考虑了相关性等级，而不仅仅是二元的相关/不相关。 |

具体到 RAG，**检索器的 Recall@k** 是最重要的数字。如果正确的段落不在检索集中，你的阅读器就无法回答问题。

调试提示：对于失败的查询，比较稀疏和稠密检索的排名。如果其中一个找到了正确的文档而另一个没有，那么你遇到了词汇不匹配（修复：添加缺失的一半）或语义歧义（修复：使用更好的嵌入或重排序器）。

## 使用它

2026 年技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 文档 | 内存中的 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需单独的数据库。 |
| 100k-10M 文档 | 使用 FAISS 或 pgvector 进行稠密检索 + 使用 Elasticsearch / OpenSearch 进行 BM25 检索。并行运行。 |
| 10M+ 文档 | 支持混合检索的 Qdrant / Weaviate / Vespa / Milvus。对前 30 名进行交叉编码器重排序。 |
| 最高质量前沿 | 三路（BM25 + 稠密检索 + SPLADE）+ ColBERT 后期交互重排序 |

无论你选择什么，都要为评估做预算。在对端到端 RAG 准确率进行基准测试之前，先对检索召回率进行基准测试。阅读器无法修复检索器遗漏的内容。

### 2026 年生产环境 RAG 的宝贵经验

- **80% 的 RAG 失败源于数据摄取和分块，而非模型。** 团队们花数周时间更换 LLM 和调整提示词，而检索却默默地每隔三次查询就返回错误的上下文。首先修复分块问题。
- **分块策略比块大小更重要。** 固定大小的分割会破坏表格、代码和嵌套标题。基于句子的感知分块是默认选择；对于技术文档和产品手册，基于语义或 LLM 的分块效果更好。
- **父子文档模式。** 检索小的“子”块以提高精度。当来自同一父部分的多个子块出现时，换用父块以保留上下文。这无需重新训练即可持续提升答案质量。
- **k_rerank=3 通常最优。** 超过此数量的每个额外块都会增加 token 成本和生成延迟，而不会提升答案质量。如果对你来说 k=8 仍然比 k=3 好，那么重排序器性能不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设答案，嵌入该答案，然后进行检索。弥合了短问题与长文档之间的措辞差距。无需训练即可免费提升精度。
- **上下文预算低于 8K token。** 在该限制下持续命中意味着重排序器阈值过于宽松。
- **版本化一切。** 提示词、分块规则、嵌入模型、重排序器。任何漂移都会悄然破坏答案质量。在忠实度、上下文精度和未回答问题率上设置 CI 门控，可以在用户看到问题之前阻止回归。
- **三路检索（BM25 + 稠密检索 + 学习型稀疏检索如 SPLADE）在 2026 年基准测试中优于两路检索**，尤其是对于混合了专有名词和语义的查询。当基础设施支持 SPLADE 索引时部署它。

根据 2026 年的行业测量，合理的检索设计能将幻觉减少 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 部署

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1.  **简单。** 在一个包含 500 个文档的语料库上实现上述 `hybrid_search`。测试 20 个查询。比较仅 BM25、仅稠密检索和混合检索在 top-5 时的召回率。
2.  **中等。** 添加 MRR 计算。对于每个已知正确文档的测试查询，在 BM25、稠密检索和混合检索的排名中找到正确文档的排名。报告每种检索方式的 MRR。
3.  **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域对稠密编码器进行微调。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。根据词频、IDF 和长度对文档评分。 |
| 稠密检索 | 向量搜索 | 将查询 + 文档编码为向量，查找最近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时速度快。 |
| 交叉编码器 | 重排序模型 | 将查询 + 文档一起编码。慢但准确。 |
| RRF | 排名融合 | 通过对 `1/(k + rank)` 求和来组合两个排名。 |
| Recall@k | 检索指标 | 相关文档出现在前 k 名中的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，标准的双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小了与稠密检索差距的学习型稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后期交互检索。