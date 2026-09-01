# 信息检索与搜索

> BM25 精确但脆弱。向量检索覆盖面广但会遗漏关键词。混合检索是 2026 年的默认方案。其他都是调参。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 5 · 02（词袋 + TF-IDF），阶段 5 · 04（GloVe、FastText、子词）
**时间：** 约 75 分钟

## 问题

用户输入"有人撒谎骗钱会发生什么"，期望找到真正涵盖该行为的法条："印度刑法第 420 条"。关键词搜索会完全遗漏它（没有共享词汇）。如果嵌入模型没有在法律文本上训练过，语义搜索也会遗漏。真正的搜索必须同时处理这两种情况。

IR（信息检索）是每个 RAG 系统、每个搜索栏、每个文档站点的模糊查找背后的管线。2026 年在生产环境中可行的架构不是单一方法，而是一组互补方法的链式组合，每个方法捕获前一个方法的失败。

本课将构建每个组件，并命名每种方法能捕获的失败类型。

## 概念

![混合检索：BM25 + 向量 + RRF + 交叉编码器重排序](../assets/retrieval.svg)

四层。按需选取。

1. **稀疏检索（BM25）。** 快速、精确匹配时表现优异，对语义理解很差。在倒排索引上运行。百万级文档每次查询低于 10ms。能准确命中法条引用、产品编号、错误消息、命名实体。
2. **密集检索。** 将查询和文档编码为向量，进行最近邻搜索。捕捉 paraphrase 和语义相似性。会遗漏相差一个字符的精确关键词匹配。使用 FAISS 或向量数据库，每次查询 50-200ms。
3. **融合。** 合并稀疏和密集检索的排名列表。RRF（倒数排名融合）是简单默认选择，因为它忽略原始分数（它们处于不同尺度）而只使用排名位置。如果你知道某个信号在你的领域中占主导，也可以选择加权融合。
4. **交叉编码器重排序。** 取出融合后的 top-30。运行交叉编码器（查询 + 文档一起输入，对每对评分）。保留 top-5。交叉编码器比双编码器每对更慢，但准确率高得多。通过只对 top-30 运行来分摊开销。

三方检索（BM25 + 密集 + 学习式稀疏如 SPLADE）在 2026 年的基准测试中优于两方，但需要为学习式稀疏索引提供基础设施支持。对大多数团队来说，两方加交叉编码器重排序是最佳平衡点。

```figure
gx-hybrid-retrieval
```

## 构建它

### 步骤 1：从零实现 BM25

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

两个值得了解的参数。`k1=1.5` 控制词频饱和程度；越高意味着对词重复的权重越大。`b=0.75` 控制长度归一化；0 忽略文档长度，1 完全归一化。这些默认值来自原始论文中 Robertson 的建议，通常不需要调参。

### 步骤 2：使用双编码器进行密集检索

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

对嵌入进行 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维、快速且对大多数英语检索足够强。多语言工作请使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高精度请使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：倒数排名融合（RRF）

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。较高的 `k` 会使排名差异的贡献变平缓；较低的 `k` 会让顶部排名占主导。60 是已发布的默认值，通常不需要调参。

### 步骤 4：混合搜索 + 重排序

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

三个阶段组合而成。BM25 查找词汇匹配，密集检索查找语义匹配，RRF 在不需校准分数的情况下合并两份排名，交叉编码器利用查询 - 文档对重新评分 top-30，捕获双编码器遗漏的细粒度相关性。保留 top-5。

### 步骤 5：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 在正确答案存在的查询中，正确文档出现在 top-k 中的频率是多少？ |
| MRR（平均倒数排名） | 第一个相关文档排名的 1/rank 的平均值。 |
| nDCG@k | 不仅考虑二元的"相关/不相关"，还考虑相关性的程度。 |

对于 RAG 而言，检索器的 **Recall@k** 是最重要的数字。如果正确段落不在检索集合中，你的回答器无法回答。

调试技巧：对于失败的查询，对比稀疏和密集检索的排名。如果一方找到了正确文档而另一方没有，则存在词汇失配（修复方法：补充缺失的那半部分）或语义歧义（修复方法：更好的嵌入模型或重排序器）。

## 使用它

2026 年技术栈：

| 规模 | 技术栈 |
|------|--------|
| 1k-100k 文档 | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需单独的数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于密集检索 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus（支持混合检索）。对 top-30 进行交叉编码器重排序。 |
| 最高质量前沿 | 三方检索（BM25 + 密集 + SPLADE）+ ColBERT 晚期交互重排序 |

无论选择什么，都要为评估预留预算。在基准测试端到端 RAG 准确率之前，先基准测试检索 recall。回答器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 的血泪教训

- **80% 的 RAG 失败归因于文档摄入和分块策略，而非模型本身。** 团队花数周时间切换 LLM 和调整提示词，而检索却在每三次查询中就安静地返回错误上下文。先修复分块。
- **分块策略比分块大小更重要。** 固定大小的分块会破坏表格、代码和嵌套标题。句子感知的分块是默认选择；语义或基于 LLM 的分块对技术文档和产品手册更有价值。
- **父文档模式。** 检索小"子"分块以获得精度。当来自同一父章节的多个子分块出现时，切换回父块以保持上下文。这在不重新训练的情况下稳定提升答案质量。
- **k_rerank=3 通常是最佳值。** 超过此值后每增加一个分块都会增加 token 成本和生成延迟，但不会提升答案质量。如果对你来说 k=8 仍然比 k=3 更好，说明重排序器表现不佳。
- **HyDE / 查询扩展。** 根据查询生成假设性答案，对该答案进行嵌入并检索。弥合短查询与长文档之间的表述差距。免费提升精度，无需训练。
- **上下文预算不超过 8K token。** 在这个限制下频繁命中意味着重排序器阈值过于宽松。
- **版本化一切。** 提示词、分块规则、嵌入模型、重排序器。任何漂移都会悄无声息地破坏答案质量。在 CI 中设置忠实度、上下文精度和未回答问题率的门禁，在用户看到之前阻止回归。
- **三方检索（BM25 + 密集 + 学习式稀疏如 SPLADE）在 2026 基准上优于两方**，尤其是对于混合专有名词与语义的查询。当基础设施支持 SPLADE 索引时将其上线。

根据 2026 年行业测量，正确的检索设计可将幻觉减少 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 交付它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: 根据语料库和查询模式选择检索技术栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

根据需求（语料库规模、查询模式、延迟预算、质量要求、基础设施约束），输出：

1. 技术栈。仅 BM25、仅密集、混合（BM25 + 密集 + RRF）、混合 + 交叉编码器重排序，或三方检索（BM25 + 密集 + 学习式稀疏）。
2. 密集编码器。指定具体模型。匹配语言、领域和上下文长度。
3. 重排序器。如果使用的交叉编码器，指定具体模型。标注重排序会在 top-30 上增加 30-100ms 延迟。
4. 评估计划。Recall@10 是主要的检索器指标。MRR 用于多答案场景。先建立基线，再衡量增量改进。

拒绝在未提供证据证明密集检索能处理精确匹配的情况下，为包含命名实体、错误代码或产品 SKU 的语料库推荐仅密集检索。拒绝在高价值检索（法律、医疗）中跳过重排序，因为最终 top-5 决定了用户的整个答案。
```

## 练习

1. **简单。** 在 500 文档语料库上实现上述 `hybrid_search`。测试 20 个查询。比较 BM25-only、dense-only 和混合检索在 k=5 时的 recall。
2. **中等。** 添加 MRR 计算。对于每个有已知正确答案的测试查询，找出正确文档在 BM25、密集和混合排名中的位置。分别报告各方案的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域微调密集编码器。从 500 个查询 - 文档对构建训练集。比较微调前后的 recall。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BM25 | 关键词搜索 | Okapi BM25。根据词频、IDF 和长度对文档评分。 |
| 密集检索 | 向量搜索 | 将查询和文档编码为向量，查找最近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时快速。 |
| 交叉编码器 | 重排序模型 | 联合编码查询和文档。较慢但准确。 |
| RRF | 排名融合 | 通过求和 `1/(k + rank)` 合并两个排名。 |
| Recall@k | 检索指标 | 相关文档出现在 top-k 中的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典的双编码器方案。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 学习式稀疏检索器，缩小与密集检索的差距。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 原始论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 晚期交互检索。
