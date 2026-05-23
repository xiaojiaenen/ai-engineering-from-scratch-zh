# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词学习一个嵌入向量。GloVe 对共现矩阵进行分解。FastText 嵌入词的组成部分。BPE 则连接到了 Transformer 时代。

**类型：** 构建
**语言：** Python
**先决条件：** 第 5 阶段 · 03 (从零实现 Word2Vec)
**时间：** 约 45 分钟

## 问题背景

Word2Vec 留下了两个悬而未决的问题。

首先，有一条平行的研究路线是直接对共现矩阵进行分解（如 LSA、HAL），而不是像 Word2Vec 那样进行在线的跳字法更新。Word2Vec 的迭代方法是否本质上更好，还是说这种差异仅仅是由于两种方法处理词频计数方式不同所造成的？**GloVe** 给出了答案：使用经过精心设计的损失函数对矩阵进行分解，其效果可以匹配甚至超越 Word2Vec，并且训练成本更低。

其次，上述两种方法对于从未见过的词都无能为力。`Zoomer-approved`、`dogecoin`、任何上周才造出来的专有名词、每一个罕见词根的屈折变化形式。**FastText** 通过嵌入字符 n-gram 来解决这个问题：一个词是其组成部分（包括词素）的总和，因此即使是词表外的词也能得到一个合理的向量表示。

第三，当 Transformer 到来后，问题再次发生变化。单词级词汇表的规模上限大约在一百万条；而真实的语言比这要开放得多。**字节对编码**及其相关方法通过学习一个由常见子词单元组成的词汇表来解决这个问题，这种词汇表可以覆盖所有内容。每个现代大型语言模型所使用的分词器都是子词分词器。

本课将逐一讲解这三者，然后说明在何种情况下应选择哪一种。

## 核心概念

**GloVe (全局向量)。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 出现在词 `i` 上下文中的频率。训练词向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])` 成立。对损失函数进行加权，以避免高频词对主导训练。完成。

**FastText。** 一个词是其字符 n-gram 与该词本身的总和。`where` 变为 `<wh, whe, her, ere, re>, <where>`。词向量是这些分量向量的总和。训练方式与 Word2Vec 相同。好处是：未见过的词 (`whereupon`) 可以通过已知的 n-gram 组合得到表示。

**BPE (字节对编码)。** 从一个由单个字节（或字符）组成的词汇表开始。统计语料库中所有相邻的词对。将出现频率最高的词对合并为一个新 token。重复此过程 `k` 次。结果是得到一个包含 `k + 256` 个 token 的词汇表，其中高频序列（`ing`、`tion`、`the`）成为单个 token，而罕见词则被分解为熟悉的片段。任何句子都可以被分词。

## 实现细节

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个重要的组成部分需要说明。加权函数 `f(x) = (x/x_max)^alpha` 会降低出现频率极高的词对（如 `(the, and)`）的权重，以防止它们主导损失函数。最终的嵌入是 `W`（中心词）和 `W_tilde`（上下文词）两个表的向量之和。将两者相加是一个已被发表的技巧，其效果通常优于仅使用其中一个。

### FastText：基于子词的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合（通常为 3 到 6 个字符）表示。词嵌入是其 n-gram 嵌入的总和。在跳字法训练中，将此替换 Word2Vec 原本使用的单个向量。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于一个未见过的词，只要其部分 n-gram 是已知的，你仍然可以得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，因此这两个词向量在空间中会彼此靠近。

### BPE：学习得到的子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代会合并最常见的相邻词对。经过足够多次迭代后，高频子串（`low`、`est`、`tion`）会变成单个 token，而罕见词则被清晰地分解。

真实的 GPT/BERT/T5 分词器会学习 30k 到 100k 次合并。结果是：任何文本都可以被分词为一个长度有界的、由已知 ID 组成的序列，永远不会有词表外（OOV）的词。

## 实际应用

在实践中，你很少需要自己训练这些模型。你只需加载预训练的检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 Transformer 时代使用 BPE 风格的子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀用于标记词边界（这是 GPT-2 的惯例）。每个现代分词器都是 BPE 的变体、WordPiece（用于 BERT）或 SentencePiece（用于 T5、LLaMA）。

### 如何选择

| 场景 | 选择 |
|-----------|------|
| 需要预训练的通用词向量，且无需处理词表外词 | GloVe 300d |
| 需要预训练的通用词向量，且必须处理拼写错误/新词/形态丰富的语言 | FastText |
| 用于 Transformer 的任何任务（训练或推理） | 使用模型自带的分词器，切勿更换。 |
| 从头开始训练自己的语言模型 | 首先在你的语料库上训练一个 BPE 或 SentencePiece 分词器。 |
| 使用线性模型进行生产环境文本分类 | 仍然使用 TF-IDF，参考第 02 课。 |

## 部署

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1.  **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的杰卡德重叠度。你应该能看到大量共享的部分（`pla`、`lay`、`play`），这就是为什么 FastText 能在形态变体间很好地迁移。
2.  **中等。** 扩展 `learn_bpe` 以跟踪词汇表增长。将每个语料库字符对应的 token 数量绘制为合并次数的函数。你应该能看到起初压缩迅速，然后逐渐趋近于每个 token 约 2-3 个字符。
3.  **困难。** 在莎士比亚全集上训练一个 1k 次合并的 BPE。比较常见词与罕见专有名词的分词结果。测量分词前后每个词的平均 token 数。写下让你感到惊讶的发现。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 共现矩阵 | 词-词频率表 | `X[i][j]` = 词 `j` 出现在词 `i` 周围窗口中的频率。 |
| 子词 | 词的一部分 | 字符 n-gram (FastText) 或学习得到的 token (BPE/WordPiece/SentencePiece)。 |
| BPE | 字节对编码 | 迭代合并出现频率最高的相邻词对，直到词汇表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 无法处理。FastText 和 BPE 可以处理。 |
| 字节级 BPE | 基于原始字节的 BPE | GPT-2 使用的方案。词汇表以 256 个字节开始，因此没有任何词是 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，仅七页，仍是关于损失函数的最佳推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText 论文。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实践中的实际区别。