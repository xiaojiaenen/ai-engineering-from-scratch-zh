# 词袋模型、TF-IDF 与文本表示

> 先计数，后思考。在 2026 年，对于定义明确的任务，TF-IDF 仍然击败嵌入。

**类型：** 构建
**语言：** Python
**前提知识：** 阶段 5 · 01（文本处理），阶段 2 · 02（从零实现线性回归）
**时间：** ~75 分钟

## 问题所在

模型需要数字。你拥有字符串。

每个 NLP 流程都必须回答同样的问题：如何将一个长度可变的 token 流转换成一个分类器可以使用的固定大小向量？该领域最初得出的答案是最简单但有效的那个：计算词频，生成一个向量。

这个向量支撑的生产环境 NLP 应用比任何嵌入模型都多。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排名（在 BM25 之前）、第一波情感分析、第一个十年的学术 NLP 基准测试。在 2026 年，从业者在狭窄的分类任务上仍然首先考虑它。它速度快、可解释性强，并且在那些词的出现才是关键的任务上，其表现常常与一个 4 亿参数的嵌入模型不相上下。

本节课将从零开始构建词袋模型，然后是 TF-IDF。接着展示 scikit-learn 如何用三行代码完成同样的事情。最后，点出那种会让你转向嵌入的失败模式。

## 核心概念

**词袋模型 (BoW)** 丢弃了词序。对于每个文档，统计词汇表中每个词出现了多少次。向量的长度等于词汇表的大小。位置 `i` 是单词 `i` 的计数。

**TF-IDF** 对词袋模型进行重新加权。一个在所有文档中都出现的词信息量低，因此将其权重降低。一个在整个语料库中罕见但在单个文档中频繁出现的词是重要信号，因此将其权重提高。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词项在文档中的频率，`df` 是文档频率（包含该词的文档数量），`N` 是文档总数。`log` 确保了常见词的权重保持有界。

关键特性：两者都生成具有可解释轴的稀疏向量。你可以查看一个训练好的分类器的权重，读出哪些词将文档推向各个类别。对于一个 768 维的 BERT 嵌入，你无法做到这一点。

## 动手构建

### 第 1 步：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何分词器都可以；本课的 `code/main.py` 使用了一个简化的纯小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着索引 0 对应第一个文档中出现的第一个词。惯例不同；scikit-learn 按字母顺序排序。

### 第 2 步：词袋

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

行代表文档。列代表词汇表索引。条目 `[i][j]` 表示“单词 `j` 在文档 `i` 中出现了多少次。”文档 1 有两次 `cat`，因为它确实出现了两次。文档 0 有零次 `ran`，因为它没出现。

### 第 3 步：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有两个值得提及的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保一个在所有文档中都出现的词，其 IDF 为 1（而非 0），这与 scikit-learn 的默认设置一致。其他实现使用原始的 `log(N/df)`。两种都可以；平滑版本更友好。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
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

三个文档，五个词汇词（`the`, `cat`, `sat`, `dog`, `ran`）。`the` 在所有三个文档中都出现，因此其 IDF 低。`dog` 只在一个文档中出现，因此其 IDF 高。向量是稀疏的（大部分条目很小），并且具有区分性的词凸显出来。

### 第 5 步：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不进行归一化，更长的文档会得到更大的向量，并主导相似度得分。L2 归一化将每个文档映射到单位超球面上。行之间的余弦相似度现在就变成了点积。

## 使用它

scikit-learn 提供了生产版本。

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

`CountVectorizer` 一次调用就完成了分词、构建词汇表和词袋转换。`TfidfVectorizer` 增加了 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万份文档，密集版本无法放入内存；请保持稀疏，直到分类器要求密集输入。

改变一切的参数：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元组。通常能提升分类效果。 |
| `min_df=2` | 去除在少于 2 个文档中出现的词。在有噪声数据上精简词汇表。 |
| `max_df=0.95` | 去除在超过 95% 文档中出现的词。近似于移除停用词，无需硬编码列表。 |
| `stop_words="english"` | scikit-learn 内置的停用词列表。取决于任务——情感分析*不应该*去除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始 `tf`。当一个词项在单个文档中重复出现多次时有帮助。 |

### TF-IDF 仍然获胜的场景（截至 2026 年）

- **垃圾邮件检测、主题标记、日志异常标记。** 词的出现才是关键；语义细微差别不重要。
- **数据量少的场景（数百个带标签样本）。** TF-IDF 加逻辑回归没有预训练成本。
- **对延迟要求高的场景。** TF-IDF 加线性模型在微秒内给出答案。通过 Transformer 嵌入一份文档需要 10-100 毫秒。
- **必须解释其预测的系统。** 检查分类器的系数。正系数最高的词就是原因。

### TF-IDF 失败的场景

语义盲失败。考虑以下两个文档：

- "这部电影一点都不好。"
- "这部电影很精彩。"

一条是差评，一条是好评。它们的 TF-IDF 交集恰好是 `{the, movie, was}`。词袋分类器必须记住单词 `not` 靠近 `good` 时会翻转标签。它可以在足够的数据上学会这一点，但永远不会像一个理解句法的模型那样优雅。

另一个失败：推理时出现未登录词。一个在 IMDb 评论上训练的词袋模型，如果 token `Zoomer-approved` 从未在训练中出现过，它就完全不知道如何处理。子词嵌入（第 4 课）可以处理这种情况。TF-IDF 则不能。

### 混合方法：TF-IDF 加权嵌入

2026 年对于中等数据量分类任务的务实默认方案：使用 TF-IDF 权重作为对词嵌入的注意力。

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

你既从嵌入中获得了语义能力，又从 TF-IDF 中获得了对罕见词的强调。分类器在池化后的向量上进行训练。在少于约 5 万个带标签样本的情感、主题和意图分类任务上，这优于单独使用其中任何一种方法。

## 部署

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同的文档得分为 1.0，词汇表不相交的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 生成 `n`-gram 的计数。测试 `n=2` 对 `["the", "cat", "sat"]` 是否生成 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次，缓存）构建上述 TF-IDF 加权嵌入的混合模型。在 20 Newsgroups 数据集上，与纯 TF-IDF 和纯均值池化嵌入的分类准确率进行比较。报告哪种方法在哪里获胜。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 单个文档中词汇表单词的计数。丢弃了词序。 |
| TF | 词项频率 | 一个词在文档中的计数，可选择按文档长度归一化。 |
| DF | 文档频率 | 包含该词至少一次的文档计数。 |
| IDF | 逆文档频率 | 经过 `log(N / df)` 平滑处理。降低在任何地方都出现的词的权重。 |
| 稀疏向量 | 大部分为零 | 词汇表通常有 1 万到 10 万个词；在任何给定文档中，大部分词都未出现。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 扩展阅读

- [scikit-learn — 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威的 API 参考，附有每个参数的说明。
- [Salton, G., & Buckley, C. (1988). 自动文本检索中的词项加权方法](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 这篇论文使 TF-IDF 成为十年间的默认选择。
- [“为什么 TF-IDF 仍然击败嵌入” — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年对旧方法何时获胜及其原因的探讨。