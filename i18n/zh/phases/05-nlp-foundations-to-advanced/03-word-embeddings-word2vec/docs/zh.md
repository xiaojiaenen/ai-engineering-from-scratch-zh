# 词嵌入（Word Embeddings）——从零实现 Word2Vec

> 看一个词与谁为伍，便知它是什么。用浅层网络训练这一思想，几何结构便自然显现。

**类型：** 构建
**语言：** Python
**先修：** 阶段 5 · 02（BoW + TF-IDF）、阶段 3 · 03（从零推导反向传播）
**时间：** 约 75 分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，但它不知道它们的意思是相近的。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以列出同义词来弥补这一缺陷，但这对罕见词、领域术语以及你未曾预料到的每种语言都无效。

你想要一种表示，使得 `dog` 和 `puppy` 在空间中彼此靠近。使得 `king - man + woman` 的结果落在 `queen` 附近。使得在 `dog` 上训练的模型能免费获得 `puppy` 的部分信号。

Word2Vec 给了我们这样的空间。两层神经网络，万亿 token 的训练规模，2013 年发表。架构几乎简单到令人尴尬。但其结果重塑了 NLP 长达十年。

## 概念

**分布假说**（Firth，1957）："You shall know a word by the company it keeps." 如果两个词出现在相似的上下文中，它们的意思很可能也相似。

Word2Vec 有两种变体，都利用了这一思想。

- **Skip-gram。** 给定中心词，预测周围的词。`cat -> (the, sat, on)`，窗口大小为 2。
- **CBOW（连续 bag of words）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词的处理更好。它成为了默认选择。

网络有一层没有非线性激活函数的隐藏层。输入是词汇表上的 one-hot 向量。输出是词汇表上的 softmax。训练完成后，丢弃输出层。隐藏层的权重就是嵌入向量。

```
one-hot(center) ── W ──▶ hidden (d 维) ── W' ──▶ softmax(vocab)
                          ^
                          这就是嵌入向量
```

关键技巧：对 10 万词做 softmax 代价过高。Word2Vec 使用**负采样**将其转化为二分类任务。预测"这个上下文词是否出现在这个中心词附近，是或否"。每个训练对采样少量负样本（不共现的词），而不是对整个词汇表计算 softmax。

```figure
word-vector-arithmetic
```

## 构建

### 步骤 1：从语料库生成训练对

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

窗口内的每个（中心词，上下文词）对都是一个正训练例。

### 步骤 2：嵌入表

两个矩阵。`W` 是中心词嵌入表（你要保留的那个）。`W'` 是上下文词表（通常丢弃，有时与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小的随机初始化。词表大小 1 万、维度 100 是合理的；用于教学时，50 个词 × 16 维就足以观察到几何结构。

### 步骤 3：负采样目标函数

对每个正对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负样本。训练模型使得点积 `W[center] · W'[context]` 对正样本高、对负样本低。

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

核心公式：正样本对的逻辑损失（希望 sigmoid 接近 1）加上负样本对的逻辑损失（希望 sigmoid 接近 0）。梯度同时流向两个表。完整推导见原始论文；如果你希望真正理解，用纸笔推导一遍。

### 步骤 4：在玩具语料库上训练

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

在大型语料库上经过足够多的 epoch 后，共享上下文的词会有相似的中心词嵌入。在玩具语料库上，你只能隐约看到效果。在数十亿 token 上，效果则非常明显。

### 步骤 5：类比技巧

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

在预训练的 Google News 300 维向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室，而是因为向量 `(king - man)` 捕获了类似"皇室"的含义，将其加到 `woman` 上就落在了皇室女性的区域附近。

## 使用

从零手写 Word2Vec 是一种学习过程。生产环境的 NLP 使用 `gensim`。

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

在实际工作中，你几乎不会自己训练 Word2Vec。你下载预训练向量即可。

- **GloVe** ——斯坦福的共现矩阵分解方法。50d、100d、200d、300d 等多个版本。通用覆盖效果好。课程 04 专门讲解 GloVe。
- **fastText** ——Facebook 的 Word2Vec 扩展，对字符 n-gram 进行嵌入。通过子词组合处理未登录词（OOV）。课程 04。
- **Google News 预训练 Word2Vec** ——300 维，300 万词词表，2013 年发布。至今仍每天被下载。

### 2026 年 Word2Vec 仍占优势的场景

- **轻量级领域专用检索。** 在笔记本电脑上用一小时训练医学摘要，获得通用模型无法捕获的专用向量。
- **类比式特征工程。** `gender_vector = mean(man - woman pairs)`。将其从其他词中减去，得到性别中立轴。仍在公平性研究中使用。
- **可解释性。** 100 维足够小，可以通过 PCA 或 t-SNE 可视化并实际看到簇的形成。
- **任何需要在无 GPU 的端设备上运行推理的场景。** Word2Vec 查询只是一次行读取。

### Word2Vec 的失败之处

**多义词墙。** `bank` 只有一个向量。`river bank` 和 `financial bank` 共享它。`table`（电子表格 vs. 家具）也共享它。下游的分类器无法从向量中区分这些含义。

上下文嵌入（ELMo、BERT 及之后的所有 transformer）通过在每个词出现时基于周围上下文生成不同的向量来解决此问题。这就是从 Word2Vec 到 BERT 的飞跃：从静态到上下文敏感。阶段 7 覆盖 transformer 部分。

**未登录词问题**是另一个失败点。如果 `Zoomer-approved` 不在训练数据中，Word2Vec 从未见过它，也没有回退机制。fastText 通过子词组合修复了这一点（课程 04）。

## 交付

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查 word2vec 模型。运行类比测试、查找邻居、诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

检查已训练的 Word 嵌入以验证其是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和词汇表，执行：

1. 三项经典类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦相似度。
2. 用户指定的五个领域专用词的最近邻测试。打印 top-5 邻居及余弦值。
3. 一项对称性检查。`similarity(a, b) == similarity(b, a)` 在浮点精度范围内成立。
4. 一项退化检查。如果任何嵌入的范数低于 0.01 或高于 100，则模型存在训练 bug。标记出来。

不要仅凭类比准确率就宣称模型优秀。类比基准测试容易被操纵，且无法迁移到下游任务。建议将内在评估与下游评估结合使用。
```

## 练习

1. **简单。** 在微型语料库（20 句关于猫和狗的句子）上运行训练循环。200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的 top-3 中包含 `dog`。如果没有，增加 epoch 数或扩大词表。
2. **中等。** 添加高频词下采样。频率高于 `10^-5` 的词按与其频率成正比的概率从训练对中丢弃。衡量对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两个偏差轴：`he - she` 和 `doctor - nurse`。将职业词汇投影到这两个轴上。报告偏差差距最大的职业。这是公平性研究者使用的探测方法。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------------|-----------------------|
| 词嵌入（Word embedding） | 词作为一个向量 | 从上下文学得的稠密低维（通常 100-300 维）表示。 |
| Skip-gram | Word2Vec 的技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| 负采样（Negative sampling） | 训练捷径 | 用与 `k` 个随机词的二分类替换对整个词汇表的 softmax。 |
| 静态嵌入（Static embedding） | 每词一个向量 | 不随上下文变化。在多义词上失败。 |
| 上下文嵌入（Contextual embedding） | 上下文敏感的向量 | 根据周围词，每次出现产生不同的向量。transformer 产出的就是这种。 |
| OOV | 未登录词 | 训练时未见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) —— 负采样论文。简短易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) —— 最清晰的梯度推导，如果原始论文的数学让人感到吃力，可以看这篇。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html) —— 生产环境中真正可用的训练设置。
