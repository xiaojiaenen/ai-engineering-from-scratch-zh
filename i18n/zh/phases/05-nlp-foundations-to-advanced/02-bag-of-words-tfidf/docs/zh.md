# 词袋、TF-IDF 与文本表示

> 先计数，再思考。到 2026 年，TF-IDF 在定义清晰的任务上仍胜过嵌入模型。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 01（文本处理），阶段 2 · 02（从零实现线性回归）
**时间：** 约 75 分钟

## 问题所在

模型需要数字，而你有的是字符串。

每条 NLP 流水线都必须回答同一个问题：如何将变长的 token 流转换为分类器可以消费的固定长度向量？该领域得出的第一个答案是那种"笨但管用"的方案。数单词。构造向量。

这个向量承载的生产级 NLP 任务多于任何嵌入模型。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排名（BM25 之前）、第一轮情感分析、学术 NLP 基准测试的第一个十年。2026 年的实践者在面向狭窄分类任务时，依然会首选它。它速度快、可解释，而且在"词出现与否才是关键"的任务上，往往与 4 亿参数的嵌入模型难以区分。

本课从零构建词袋和 TF-IDF，然后展示 scikit-learn 如何用三行代码完成同样的事情，最后指出那个让你转向嵌入模型的根本缺陷。

## 概念

**词袋（BoW）** 丢弃顺序信息。对每篇文档，统计每个词汇表中的词出现了多少次。向量长度等于词汇表大小。位置 `i` 就是词 `i` 的出现次数。

**TF-IDF** 对词袋进行重加权。一篇文档中出现频率过高的词信息量低，需要压低；在整个语料库中罕见但在某篇文档中频繁的词是信号，需要放大。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的词频，`df` 是文档频率（包含该词的文档数量），`N` 是文档总数。`log` 使得常见词的权重保持有界。

关键特性：两者都产生具有可解释轴线的稀疏向量。你可以查看训练好的分类器的权重，读出哪些词把文档推向哪个类别。但你做不到这一点——以 768 维的 BERT 嵌入而言。

```figure
bow-tfidf
```

## 动手构建

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任意单词级分词器均可；本课的 `code/main.py` 使用了简化的小写变体）。输出：`{词: 索引}` 字典。稳定的插入顺序意味着词索引 0 是首篇文档中首次见到的词。惯例各不相同；scikit-learn 按字母顺序排序。

### 步骤 2：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行对应文档，列对应词汇表索引。条目 `[i][j]` 是"词 `j` 在文档 `i` 中出现的次数"。文档 1 里 `cat` 出现两次，因为它确实出现了两次。文档 0 里 `ran` 出现零次，因为它根本没出现。

### 步骤 3：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    # 词频：用文档长度归一化词袋计数
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    # 文档频率：统计每个词出现在多少篇文档中
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    # 逆文档频率：使用平滑避免除零，并保证全出现词的 IDF 不为 0
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得提及的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保一篇在所有文档中都出现的词仍然拥有 IDF 1（而非 0），与 scikit-learn 的默认行为一致。其他实现使用原始 `log(N/df)`。两者都能工作；平滑版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        # 逐元素相乘得到 TF-IDF 向量
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词汇表词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三篇中，因此其 IDF 很低。`dog` 只出现一次，因此其 IDF 很高。向量是稀疏的（大部分条目很小），判别性词脱颖而出。

### 步骤 5：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        # 计算 L2 范数
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

不进行归一化的话，较长的文档会获得更大的向量并主导相似度分数。L2 归一化将所有文档投射到单位超球面上。此时行之间的余弦相似度等价于点积。

## 使用它

scikit-learn 提供了生产级实现。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用中完成分词、构建词汇表和词袋。`TfidfVectorizer` 在此基础上增加了 IDF 加权与 L2 归一化。两者都返回稀疏矩阵。对于 10 万篇文档，稠密版本无法装入内存；在分类器明确要求稠密格式之前，保持稀疏。

以下参数能彻底改变行为：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元组。通常提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 篇文档中的词。在噪声数据上精简词汇表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。近似停用词去除，无需硬编码列表。 |
| `stop_words="english"` | scikit-learn 内置停用词表。因任务而异——情感分析*不应*丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 替代原始 `tf`。当某个词在一篇文档中重复多次时有帮助。 |

### TF-IDF 仍占优的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词出现与否才是关键；语义细微差别无关紧要。
- 低数据场景（数百个标注样本）。TF-IDF 加上逻辑回归没有预训练成本。
- 任何对延迟敏感的系统。TF-IDF 加线性模型可在微秒内响应。让文档通过 Transformer 做嵌入则需要 10-100 毫秒。
- 必须解释预测结果的系统。检查分类器的系数。权重最高的正词就是原因。

### TF-IDF 失效的场景

语义盲区缺陷。考虑以下两篇文档：

- "这部电影根本不好看。"
- "这部电影非常棒。"

一篇是负面评价，一篇是正面评价。它们的 TF-IDF 交集恰好是 `{the, movie, was}`。词袋分类器必须记住：`not` 在 `good` 附近会翻转标签。在足够多的数据上它可以学会，但永远学不到理解句法模型那种优雅程度。

另一个缺陷：推理时的词表外（OOV）词。在 IMDb 评论上训练的 BoW 模型，遇到从未在训练中出现的 `Zoomer-approved` 时完全不知道如何处理。子词嵌入（课程 04）能处理这种情况，TF-IDF 不能。

### 混合方案：TF-IDF 加权嵌入

2026 年面向中等数据分类任务的务实默认方案：用 TF-IDF 权重对词嵌入做注意力加权。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入中获得语义容量，从 TF-IDF 中获得稀有词强调。分类器在池化向量上训练。在低于约 5 万标注样本的情感、主题和意图分类任务中，它的表现优于单独的 TF-IDF 或单独的均值池化嵌入。

## 交付使用

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定一个文本分类任务，推荐词袋、TF-IDF、嵌入或混合方案。
phase: 5
lesson: 02
---

你负责推荐文本向量化策略。根据任务描述，输出：

1. 表示方式（词袋、TF-IDF、Transformer 嵌入或混合方案）。用一句话说明理由。
2. 具体的向量化器配置。指明库名。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 上线前需要验证的一个失效场景。

当用户标注样本少于 500 且未展示 TF-IDF 基线存在语义失效时，拒绝推荐嵌入。当用于情感分析时，拒绝移除停用词（否定词携带信号）。将类别不平衡标记为仅靠更换向量化器无法解决的问题。

示例输入："将 3 万条客户支持工单分类到 12 个类别。大多数工单只有 2-3 句话。仅英文。需要可解释性以审计日志。"

示例输出：

- 表示方式：TF-IDF。3 万样本不算少；可解释性要求排除了稠密嵌入。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时本身就是停用词（"not working" vs "working"）。
- 待验证的失效：确认 `min_df=3` 不会丢弃稀有类别关键词。运行 `get_feature_names_out` 按类别过滤后人工检查。
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇表完全不重叠的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 针对 `n`-gram 进行计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上为 `["the cat", "cat sat"]` 产生二元组计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次，缓存）实现上述 TF-IDF 加权嵌入混合方案。在 20 Newsgroups 数据集上，将分类准确率与纯 TF-IDF 和纯均值池化嵌入进行对比。报告哪种方案在哪些场景下胜出。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------------------|
| BoW | 词频向量 | 单篇文档中词汇表词的计数。丢弃顺序信息。 |
| TF | 词频 | 词在文档中的出现次数，可选地按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数量。 |
| IDF | 逆文档频率 | 经过平滑的 `log(N / df)`。压低那些处处出现的词的权重。 |
| 稀疏向量 | 大部分为零 | 词汇表通常含 1 万-10 万个词；绝大多数词在任意给定文档中都不出现。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，涵盖每一个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 成为十年默认方案的那篇论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年对旧方法何时胜出及其原因的看法。
