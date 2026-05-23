# 高级 RAG（分块策略、重排序、混合搜索）

> 基础 RAG 仅检索 top-k 最相似的文本块。这对简单问题有效。但对于多跳推理、模糊查询和大型语料库，它就力不从心了。高级 RAG 决定了系统是能处理 10 篇文档的演示，还是能处理 1000 万篇文档的实用系统。

**类型：** 构建
**语言：** Python
**前置课程：** 阶段 11，第 06 课（RAG）
**时间：** 约 90 分钟
**相关：** 阶段 5 · 23（RAG 的分块策略）涵盖了全部六种分块算法——递归、语义、句子、父子文档、延迟分块、上下文检索——并附有 Vectara/Anthropic 基准测试。本课在此基础上构建：混合搜索、重排序、查询转换。

## 学习目标

- 实现高级分块策略（语义、递归、父子），以保留文档结构和上下文
- 构建一个混合搜索管道，结合 BM25 关键词匹配、语义向量搜索和交叉编码器重排序
- 应用查询转换技术（HyDE、多查询、退后一步）来提高对模糊或复杂问题的检索效果
- 诊断并修复常见的 RAG 故障：检索到错误的文本块、答案不在上下文中、多跳推理失败

## 问题所在

你在第 06 课中构建了一个基础 RAG 管道。它在小型语料库上对直接的问题是有效的。现在试试这些问题：

**模糊查询**：“上个季度的收入是多少？”语义搜索返回了关于收入策略、收入预测以及首席财务官对收入增长的看法的文本块。它们都与“收入”这个词在语义上相似。但没有一个包含实际数字。正确的文本块写着“2025 年第三季度 4720 万美元”，但用的词是“收益”而不是“收入”。嵌入模型认为“收入策略”比“第三季度收益为 4720 万美元”更接近查询。

**多跳问题**：“哪个团队的客户满意度得分提升最高？”这需要找到每个团队的满意度得分，比较它们，并找出最大值。没有哪个单独的文本块包含答案。信息分散在各个团队的报告中。

**大型语料库问题**：你有 200 万个文本块。正确答案在 #1,847,293 号文本块中。你的 top-5 检索返回了 #14、#89,201、#1,200,000、#44 和 #901,333 号文本块。它们在嵌入空间中接近，但没有一个包含答案。在这个规模下，近似最近邻搜索引入的误差足以将相关结果挤出 top-k 之外。

基础 RAG 失败是因为向量相似度不等于相关性。一个文本块可能与查询在语义上相似，但对于回答它却毫无用处。高级 RAG 通过四种技术来解决这个问题：混合搜索（添加关键词匹配）、重排序（更仔细地为候选者评分）、查询转换（在搜索前修正查询）和更好的分块（以正确的粒度进行检索）。

## 核心概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似度）擅长理解含义。“如何取消我的订阅？”可以匹配“终止您的计划的步骤”，即使它们没有共享任何词语。但它会错过精确匹配。“错误代码 E-4021”可能无法匹配包含“E-4021”的文本块，如果嵌入模型将其视为噪声的话。

关键词搜索（BM25）则相反。它擅长精确匹配。“E-4021”可以完美匹配。但如果文档说的是“终止您的计划”，那么“取消我的订阅”将返回零结果。

混合搜索同时运行两者，然后合并结果。

**BM25**（最佳匹配 25）是标准的关键词搜索算法。自 1990 年代以来，它一直是搜索引擎的支柱。其公式为：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是术语 t 在文档 d 中的词频，IDF(t) 是逆文档频率，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和度（默认 1.2），b 控制长度归一化（默认 0.75）。

简单来说：当文档包含查询术语（尤其是罕见术语）时，BM25 会给文档更高的分数，但重复术语的收益会递减。一个包含 50 次“收入”一词的文档，其相关性并不是只包含一次的文档的 50 倍。

### 倒数排名融合

你有两个排序列表：一个来自向量搜索，一个来自 BM25。你如何将它们组合起来？倒数排名融合是标准方法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是一个常数（通常为 60），用于防止排名靠前的结果占据主导地位。

一个在向量搜索中排名 #1，在 BM25 中排名 #5 的文档得分为：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

一个在向量搜索中排名 #3，在 BM25 中排名 #2 的文档得分为：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 自然地平衡了两种信号。在两个列表中都排名靠前的文档会获得最佳分数。在一个列表中排名 #1 但在另一个列表中缺失的文档会获得中等分数。这种方法很稳健，因为它使用的是排名，而不是原始分数，所以两个系统之间分数分布的差异并不重要。

### 重排序

检索（无论是向量、关键词还是混合检索）速度快但精度低。它使用双编码器：查询和每个文档被独立地嵌入，然后进行比较。嵌入是一次性计算并缓存的。这可以扩展到数百万个文档。

重排序使用交叉编码器：查询和候选文档一起被输入到一个模型中，该模型输出一个相关性分数。模型可以同时看到两个文本，并能捕获它们之间的细粒度交互。交叉编码器可以理解“第三季度的收益是多少？”与包含“第三季度 4720 万美元”的文本块高度相关，即使双编码器错过了这种联系。

权衡在于：交叉编码器比双编码器慢 100-1000 倍，因为它们联合处理查询-文档对。你无法为一百万个文档预先计算交叉编码器分数。解决方案是：检索一个更大的候选集（从混合搜索中取 top-50），然后用交叉编码器重排以得到最终的 top-5。

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]
```

常见的重排序模型（2026 年版）：
- Cohere Rerank 3.5：托管 API，多语言支持，在混合语料库上召回率提升最佳
- Voyage rerank-2.5：托管 API，在托管选项中延迟最低
- Jina-Reranker-v2 Multilingual：开源权重，支持 100 多种语言
- bge-reranker-v2-m3：开源权重，强大的基线模型
- cross-encoder/ms-marco-MiniLM-L-6-v2：开源权重，可在 CPU 上运行用于原型设计
- ColBERTv2 / Jina-ColBERT-v2：后期交互多向量重排序器——在评分时间复杂度为 O(tokens) 而非 O(docs)

### 查询转换

有时问题不在检索，而在查询本身。“那个关于新政策变化的事情是什么？”是一个糟糕的搜索查询。它不包含任何具体术语。其嵌入表示很模糊。任何检索系统都无法从中找到正确的文档。

**查询改写**：将用户的查询重新表述为更好的搜索查询。LLM 可以做到这一点：

```
User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"
```

**HyDE（假设文档嵌入）**：不是用查询来搜索，而是生成一个假设的答案，将其嵌入，然后搜索与之相似的真实文档。

```
Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."
```

将假设的答案嵌入，并搜索与之相似的真实文档。其直觉是：假设的答案在嵌入空间中比原始问题更接近真实的答案。问题和答案具有不同的语言结构。通过生成假设答案，你弥合了嵌入中“问题空间”和“答案空间”之间的差距。

HyDE 在检索前增加了一次 LLM 调用。这会将延迟增加 500-2000 毫秒。当原始查询的检索质量较差时，这是值得的。

### 父子分块

标准分块迫使你做出权衡：小块用于精确检索，大块用于获取足够上下文。父子分块消除了这种权衡。

索引小文本块（128 个 token）用于检索。当检索到一个小文本块时，返回其父文本块（512 个 token）作为提示的一部分。小文本块与查询精确匹配。父文本块为 LLM 生成良好答案提供了足够的上下文。

```mermaid
graph TD
    P["Parent chunk (512 tokens)<br/>Full section about refund policy"]
    C1["Child chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child chunk (128 tokens)<br/>Enterprise: 60-day pro-rated"]
    C3["Child chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child chunk (128 tokens)<br/>How to submit a request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"matches child"| C2
    C2 -.->|"return parent"| P
```

查询“企业退款？”精确匹配子文本块 C2。但提示接收到的是完整的父文本块 P，其中包含了关于处理时间和提交流程的上下文。

### 元数据过滤

在运行向量搜索之前，按元数据过滤语料库：日期、来源、类别、作者、语言。这可以缩小搜索空间并防止不相关结果。

“上个月安全政策有什么变化？”应该只搜索过去 30 天内安全类别的文档。如果没有元数据过滤，你将搜索整个语料库，并可能检索到一份恰好语义相似的 2 年前的安全文档。

生产级 RAG 系统在每个文本块旁边存储元数据：源文档、创建日期、类别、作者、版本。向量数据库支持在相似性搜索之前按元数据进行预过滤，这对大规模性能至关重要。

### 评估

你构建了一个 RAG 系统。你怎么知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对于一组具有已知相关文档的测试问题，相关文档出现在 top-k 结果中的百分比是多少？如果一个问题的答案在 #47 号文本块中，那么 #47 号文本块是否出现在 top-5 中？

**忠实度**：生成的答案是否基于检索到的文档？如果检索到的文本块说“60 天退款窗口”，而模型说“90 天退款窗口”，那就是忠实度失败。尽管有正确的上下文，模型仍然产生了幻觉。

**答案正确性**：生成的答案是否与预期答案匹配？这是端到端指标。它结合了检索质量和生成质量。

一个简单的忠实度检查：获取生成答案中的每个断言，并验证它是否（实质性地）出现在检索到的文本块中。如果答案包含一个不在任何检索到的文本块中的事实，它很可能是幻觉。

```mermaid
graph TD
    subgraph "Evaluation Framework"
        Q["Test questions<br/>+ expected answers<br/>+ relevant doc IDs"]
        Q --> Ret["Retrieval evaluation<br/>Recall@k: are right<br/>docs retrieved?"]
        Q --> Faith["Faithfulness evaluation<br/>Is answer grounded<br/>in retrieved docs?"]
        Q --> Correct["Correctness evaluation<br/>Does answer match<br/>expected answer?"]
    end
```

## 动手构建

### 步骤 1：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 步骤 2：倒数排名融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 步骤 3：混合搜索管道

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 步骤 4：简单重排序器

在生产环境中，你会使用交叉编码器模型。这里我们构建一个重排序器，使用词重叠、术语重要性和短语匹配来为查询-文档相关性打分。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 步骤 5：HyDE（假设文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 步骤 6：父子分块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 步骤 7：忠实度评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 实际使用

使用真正的交叉编码器进行重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

使用真正的 LLM 实现 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

使用 Weaviate 进行生产级混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统使用的 alpha 值在 0.3 到 0.7 之间。

## 部署

本课产出：
- `outputs/prompt-advanced-rag-debugger.md` -- 一个用于诊断和修复 RAG 质量问题的提示词
- `outputs/skill-advanced-rag.md` -- 一项构建带混合搜索和重排序功能的生产级 RAG 的技能

## 练习

1.  在示例文档上比较 BM25、向量搜索和混合搜索。对于 5 个测试查询中的每一个，记录哪种方法在位置 #1 返回了最相关的文本块。混合搜索至少应该在 5 个查询中的 3 个中胜出。
2.  实现一个元数据过滤器。为每个文档添加一个“类别”字段（安全、计费、API、产品）。在运行向量搜索之前，将文本块过滤到相关类别。用“使用什么加密？”进行测试，并验证它只搜索安全类别的文本块。
3.  使用第 06 课中的简单生成函数构建一个完整的 HyDE 管道。比较所有 5 个测试查询中，直接查询搜索和 HyDE 搜索的检索质量（top-3 相关性）。HyDE 应能改善模糊查询的结果。
4.  在示例文档上实现父子分块策略。使用 child_size=30 和 parent_size=100。用子文本块搜索，但在提示中返回父文本块。将生成的答案与 chunk_size=50 的标准分块进行比较。
5.  创建一个评估数据集：10 个带有已知答案文本块的问题。测量 (a) 仅向量搜索、(b) 仅 BM25、(c) 混合搜索、(d) 混合 + 重排序的 Recall@3、Recall@5 和 Recall@10。绘制结果并确定重排序在哪些情况下帮助最大。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BM25 | “关键词搜索” | 一种概率性排序算法，根据词频、逆文档频率和文档长度归一化对文档进行评分 |
| 混合搜索 | “两全其美” | 并行运行语义（向量）和关键词（BM25）搜索，然后使用排名融合合并结果 |
| 倒数排名融合 | “合并排序列表” | 通过对所有列表中的每个文档求和 1/(k + rank) 来组合多个排序列表 |
| 重排序 | “第二轮评分” | 使用更昂贵的交叉编码器模型对初始检索的候选集进行重新评分 |
| 交叉编码器 | “联合查询-文档模型” | 一种将查询和文档作为单个输入的模型，输出相关性分数；比双编码器更准确，但速度太慢，无法用于全语料库搜索 |
| 双编码器 | “独立嵌入模型” | 一种独立嵌入查询和文档的模型；因为嵌入是预先计算的，所以速度快，但不如交叉编码器准确 |
| HyDE | “用假答案搜索” | 生成查询的假设答案，将其嵌入，然后搜索与之相似的真实文档 |
| 父子分块 | “小搜索，大上下文” | 索引小文本块以进行精确检索，但返回更大的父文本块以提供足够的上下文 |
| 元数据过滤 | “搜索前先缩小范围” | 在运行向量搜索之前，按属性（日期、来源、类别）过滤文档，以缩小搜索空间 |
| 忠实度 | “是否基于事实” | 生成的答案是否得到检索文档的支持，而不是来自模型训练数据的幻觉 |

## 扩展阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) -- BM25 的权威参考文献，解释了公式背后的概率基础
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) -- 原始的 RRF 论文，表明它优于更复杂的融合方法
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) -- HyDE 论文，证明假设文档嵌入可以在没有任何训练数据的情况下改进检索
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) -- 表明在 BM25 之上进行交叉编码器重排序能显著提高检索质量
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) -- 将提示构建和权重选择视为检索管道上的优化问题；如果想“编程 LLMs”而非“提示 LLMs”，请阅读此文。
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) -- GraphRAG 论文：实体-关系提取 + Leiden 社区检测用于查询聚焦的摘要；全局 vs 局部检索的区别。
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) -- 具有反思 token 的自评估 RAG；超越静态“检索-然后-生成”的智能体前沿。
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) -- 如何将自然语言查询翻译为结构化数据库查询（Text-to-SQL、Cypher）作为预检索步骤。