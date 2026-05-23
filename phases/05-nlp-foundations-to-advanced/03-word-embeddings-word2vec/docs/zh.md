# 词嵌入 — 从零实现 Word2Vec

> 一个词由其周围的词汇定义。基于这个思想训练一个浅层网络，几何结构便自然浮现。

**类型：** 构建
**语言：** Python
**前提知识：** 阶段 5 · 02 (词袋 + TF-IDF)，阶段 3 · 03 (从零实现反向传播)
**时间：** 约 75 分钟

## 问题所在

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。但它不知道它们几乎同义。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列举同义词来掩盖这个问题，但对于罕见术语、行业术语以及你未能预见的每种语言，这种方法都会失败。

你需要一种表示方式，使得 `dog` 和 `puppy` 在空间中彼此靠近，`king - man + woman` 靠近 `queen`，并且在 `dog` 上训练的模型能够免费地将一些信号迁移到 `puppy`。

Word2Vec 为我们提供了这样的空间。一个两层的神经网络，在万亿 token 上训练，于 2013 年发表。其架构简单得几乎令人尴尬。其结果重塑了 NLP 十年。

## 核心概念

**分布假说** (Firth, 1957)：“你将通过一个词的同伴来认识它。” 如果两个词出现在相似的语境中，它们很可能具有相似的含义。

Word2Vec 有两种形式，都利用了这一思想。

- **跳字模型。** 给定一个中心词，预测其周围的词。 `cat -> (the, sat, on)` 窗口大小为 2。
- **连续词袋模型。** 给定周围的词，预测中心词。 `(the, sat, on) -> cat`。

跳字模型训练速度较慢，但能更好地处理罕见词。它成为了默认方法。

网络只有一个隐藏层，没有非线性激活。输入是词表上的一个独热向量。输出是词表上的一个 softmax。训练完成后，你会丢弃输出层。隐藏层的权重就是嵌入向量。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

诀窍在于：对 10 万个词计算 softmax 的代价过于高昂。Word2Vec 使用**负采样**将其转化为一个二分类任务。预测“这个上下文词是否出现在这个中心词附近，是或否”。对于每个训练样本对，仅采样少量负例（非共现词）词汇，而不是在整个词表上计算 softmax。

## 开始构建

### 第一步：从语料库中生成训练样本对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每个 (中心词, 上下文词) 对都是一个正样本训练对。

### 第二步：嵌入表

两个矩阵。`W` 是中心词嵌入表（这是你最终保留的）。`W'` 是上下文词嵌入表（通常被丢弃，有时与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词表大小 1 万，维度 100 是比较现实的规模；用于教学，50 个词 × 16 维足够观察几何结构。

### 第三步：负采样目标函数

对于每个正样本对 `(center, context)`，从词表中随机采样 `k` 个词作为负样本。训练模型使得点积 `W[center] · W'[context]` 对于正样本尽可能高，对于负样本尽可能低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法公式：对正样本对的逻辑损失（希望 sigmoid 接近 1）加上对负样本对的逻辑损失（希望 sigmoid 接近 0）。梯度会同时流向两个嵌入表。完整的推导在原论文中；如果你想彻底掌握，最好用纸笔推导一遍。

### 第四步：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在足够大的语料库上训练足够的轮次后，共享上下文的词具有相似的中心词嵌入。在玩具语料库上，你会隐约看到这种效果。在数十亿 token 上，效果会非常明显。

### 第五步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 词向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这不是因为模型知道什么是皇室。而是因为向量 `(king - man)` 捕获了某种类似“皇室”的含义，将其加到 `woman` 上，就落在了皇室-女性区域附近。

## 实际应用

从零编写 Word2Vec 是教学目的。生产环境中的 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

在实际工作中，你几乎不需要自己训练 Word2Vec。你会下载预训练的词向量。

- **GloVe** — 斯坦福基于共现矩阵分解的方法。提供 50 维、100 维、200 维、300 维的检查点。覆盖范围广。第 04 课专门介绍 GloVe。
- **fastText** — Facebook 对 Word2Vec 的扩展，它嵌入字符 n-gram。通过组合子词来处理未登录词。第 04 课。
- **基于 Google News 的预训练 Word2Vec** — 300 维，300 万词词汇表，2013 年发布。至今仍被每日下载。

### Word2Vec 在 2026 年仍然胜出的场景

- 轻量级特定领域检索。在笔记本电脑上一小时内基于医学摘要训练，获得通用模型无法捕捉的专业向量。
- 类比式特征工程。 `gender_vector = mean(man - woman pairs)`。从其他词中减去它，得到一个性别中立轴。仍被用于公平性研究。
- 可解释性。100 维足够小，可以通过 PCA 或 t-SNE 绘图，实际观察到聚类的形成。
- 任何需要在设备端无 GPU 运行推理的场景。Word2Vec 查找只是一次行读取。

### Word2Vec 失败的场景

一词多义的壁垒。`bank` 只有一个向量。`river bank` 和 `financial bank` 共享它。`table` (电子表格 vs. 家具) 也共享它。下游分类器无法从向量中区分不同含义。

上下文嵌入 (ELMo, BERT, 此后的所有 Transformer) 通过为词的每个出现根据其上下文生成不同的向量解决了这个问题。这就是从 Word2Vec 到 BERT 的飞跃：从静态到上下文。第 7 阶段涵盖 Transformer 的部分。

未登录词问题是另一个失败点。如果训练数据中没有 `Zoomer-approved`，Word2Vec 就从未见过它。没有回退方案。fastText 通过子词组合修复了这个问题 (第 04 课)。

## 部署使用

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个微型语料库（关于猫和狗的 20 个句子）上运行训练循环。200 个轮次后，验证 `nearest(vocab, W, W[vocab["cat"]])` 返回 `dog` 在其前 3 名中。如果没有，增加轮次或扩大词表。
2. **中等。** 添加高频词子采样。频率高于 `10^-5` 的词，以与其频率成比例的概率从训练对中丢弃。测量这对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两个偏见轴：`he - she` 和 `doctor - nurse`。将职业词汇投影到这两个轴上。报告哪些职业的偏见差距最大。这是公平性研究人员使用的一种探测方法。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 词嵌入 | 词作为向量 | 一种从上下文中学习得到的稠密、低维（通常 100-300）表示。 |
| 跳字模型 | Word2Vec 的一个技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| 负采样 | 训练捷径 | 用对 `k` 个随机词进行二元分类来替代对整个词表的 softmax。 |
| 静态嵌入 | 每个词一个向量 | 不论上下文，同一个词对应相同向量。在处理一词多义时失败。 |
| 上下文嵌入 | 上下文敏感向量 | 根据周围词，为词的每次出现生成不同向量。Transformer 产生的就是这种。 |
| OOV | 未登录词 | 训练中未见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文。简短且可读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 最清晰的梯度推导，如果原论文的数学感觉过于密集。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html) — 真正有效的生产环境训练设置。