# Transformer之前的文本生成——N-gram语言模型

> 如果一个词令人意外，那么模型就很差。困惑度将意外变成了一个数字。平滑则使其保持有限。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段5 · 01（文本处理），阶段2 · 14（朴素贝叶斯）
**时间：** 约45分钟

## 问题所在

在Transformer、RNN、词嵌入出现之前，语言模型通过统计一个词在前`n-1`个词之后出现的频率来预测下一个词。统计"the cat" → "sat"出现47次，"the cat" → "jumped"出现12次，"the cat" → "refrigerator"出现0次。然后进行归一化以得到概率分布。

这就是n-gram语言模型。从1980年到2015年，它驱动着每一个语音识别器、每一个拼写检查器以及每一个基于短语的机器翻译系统。如今，当你需要廉价的设备端语言建模时，它仍然在运行。

有趣的问题是如何处理未见过的n-gram。一个原始的基于计数的模型会将任何它没见过的情况赋予零概率，这是灾难性的，因为句子很长，几乎每个长句子都至少包含一个未见过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney平滑就是其成果，而现代深度学习继承了其经验主义的传统。

## 核心概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定`n`（通常三元语法为3，四元语法为4）。根据计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练中未见过的n-gram概率为零。2007年对Brown语料库的一项研究发现，即使是四元语法模型，也有30%的保留测试四元语法在训练中未见过。没有平滑，你无法在任何真实文本上进行评估。

**平滑方法，按复杂程度排序：**

1.  **拉普拉斯（加一平滑）。** 给每个计数加1。简单，但对罕见事件效果很差。
2.  **Good-Turing。** 基于频率的频率，将概率质量从高频事件重新分配给未见过的事件。
3.  **插值。** 使用可调权重组合n-gram、(n-1)-gram等的估计值。
4.  **回退。** 如果n-gram的计数为零，则退回到(n-1)-gram。Katz回退对此进行了规范化。
5.  **绝对折扣。** 从所有计数中减去一个固定折扣`D`，将节省下来的质量重新分配给未见过的事件。
6.  **Kneser-Ney。** 绝对折扣加上为低阶模型选择的一个巧妙方法：使用*续接概率*（一个词出现在多少不同上下文中），而不是原始频率。

Kneser-Ney的见解很深刻。"San Francisco"是一个常见的二元语法。一元语法"Francisco"主要出现在"San"之后。朴素的绝对折扣会给予"Francisco"很高的一元语法概率（因为其计数高）。Kneser-Ney注意到"Francisco"只出现在一个上下文中，并相应地降低了其续接概率。结果：一个以"Francisco"结尾的新二元语法会得到适当的低概率。

**评估：困惑度。** 在保留测试集上，每个词平均负对数似然的指数。越低越好。困惑度为100意味着模型如同在100个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 开始构建

### 第1步：三元语法计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是n-gram计数和上下文计数。`<s>`和`</s>`是句子边界标记。

### 第2步：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加1。平滑了，但过度将质量分配给未见过的事件，同时也损害了已知的罕见事件。

### 第3步：Kneser-Ney（二元语法，插值形式）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

有三个活动部分。`continuation_prob`捕获"这个词出现在多少不同的上下文中？"（Kneser-Ney的创新点）。`lambda_prev`是通过折扣节省下来的质量，用于加权回退。最终概率是折扣后的主项加上加权的续接项。

### 第4步：使用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例进行采样。根据种子不同，每次输出都不同。若要获得类似集束搜索的输出，可在每一步选择argmax（贪心），并添加一个小的随机性旋钮（温度）。

### 第5步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于Brown语料库，一个调优良好的四元语法Kneser-Ney模型困惑度约为140。一个Transformer语言模型在相同测试集上达到15-30。差距大约是10倍。这就是该领域转向的原因。

## 应用它

- **经典NLP教学。** 这是了解平滑、最大似然估计和困惑度最清晰的途径。
- **KenLM。** 生产级n-gram库。在对延迟要求高的语音和机器翻译系统中用作重打分器。
- **设备端自动补全。** 键盘中的三元语法模型。至今仍在使用。
- **基线。** 在宣称你的神经语言模型很好之前，始终计算一个n-gram语言模型的困惑度。如果你的Transformer不能以显著优势击败Kneser-Ney，那说明有问题。

## 部署它

保存为`outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1.  **简单。** 在一个包含1,000句莎士比亚语料的语料库上训练一个三元语法语言模型。生成20个句子。它们会是局部合理但全局不连贯的。这是一个经典的演示。
2.  **中等。** 在莎士比亚语料的保留集上，为你的Kneser-Ney模型实现困惑度计算。与拉普拉斯平滑进行比较。你应该能看到Kneser-Ney的困惑度降低30-50%。
3.  **困难。** 构建一个三元语法拼写纠正器：给定一个拼写错误的词及其上下文，生成纠正选项，并根据语言模型的上下文概率进行排序。在Birkbeck拼写语料库（公开）上进行评估。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| N-gram | 词序列 | 由`n`个连续token组成的序列。 |
| 平滑 | 避免零概率 | 重新分配概率质量，使未见事件获得非零概率。 |
| 困惑度 | 语言模型质量指标 | `exp(-average log-prob)`在保留数据集上的值。越低越好。 |
| 回退 | 退回到更短上下文 | 如果三元语法计数为零，则使用二元语法。Katz回退对此进行了形式化。 |
| Kneser-Ney | n-gram的最佳平滑方法 | 绝对折扣 + 为低阶模型使用续接概率。 |
| 续接概率 | Kneser-Ney特有 | `P(w)`由其出现的不同上下文数量`w`加权，而非由原始计数加权。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — 关于n-gram语言模型和平滑的经典论述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立Kneser-Ney为最佳n-gram平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始的Kneser-Ney论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速的生产级n-gram语言模型，直至2026年仍在对延迟敏感的应用中使用。