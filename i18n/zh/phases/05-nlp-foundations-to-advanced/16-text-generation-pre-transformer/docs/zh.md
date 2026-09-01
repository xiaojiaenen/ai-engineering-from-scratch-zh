# Transformer之前的文本生成 —— N-gram语言模型

> 如果某个词出乎意料，说明模型很差。困惑度将意外程度量化为一个数值。平滑技术使其保持有限。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段5·01（文本处理）、阶段2·14（朴素贝叶斯）
**时间：** 约45分钟

## 问题背景

在transformers出现之前，在RNN之前，在词嵌入之前，语言模型通过统计某词跟随前`n-1`个词的频率来预测下一个词。统计"the cat"→"sat"出现47次，"the cat"→"jumped"出现12次，"the cat"→"refrigerator"出现0次。归一化后得到概率分布。

这就是n-gram语言模型。从1980年到2015年，它驱动了每一个语音识别器、每一个拼写检查器、每一个基于短语的机器翻译系统。当你需要低成本的设备端语言建模时，它仍在运行。

真正有趣的问题是：如何处理未见过（未训练到）的n-gram。一个原始的基于计数的模型会给所有未见过的事物分配零概率，而这会灾难性地导致问题——因为句子很长，几乎每个长句子都至少包含一个未出现的序列。五十年的平滑研究就是为了解决这个问题，Kneser-Ney平滑是其中成果，现代深度学习也继承了其经验主义传统。

## 概念

![N-gram模型：计数、平滑、生成](../assets/ngram.svg)

### 预测游戏

在所有这些技术出现之前，有一个实验定义了什么是语言模型。遮住英文句子的下一个字母。让人猜，一个一个猜，直到猜对。记下猜的次数。对几百个字母重复这个过程。

这些猜测次数不是 trivia。它们是对文本的无损重新编码：把猜测次数序列交给第二个相同的猜测者，他们就能重建每一个字母，因为在每个位置上，他们知道哪些猜测排在前面。你能用更少的符号重新编码的信息，每个符号携带的信息更少，所以猜测次数统计给英语的熵设定了一个上限。

Shannon在1951年做了这个实验，得到了一个至今仍支配该领域的数字。一个27符号的字母表（26个字母加空格）每字母可携带`log2(27) ≈ 4.75`比特。人类猜测者使用100个字母的上下文时，每字母落在0.6到1.3比特之间。英语大约是四分之三的强制移动。模型必须学习到的结构，在任何模型能够学习它之前就被测量出来了。

自此之后的每一个语言模型，都是这个游戏的一个机械玩家；本课中每一个评估数字，都是对游戏的评分：

- **交叉熵损失**是模型每个符号平均需要的比特数。训练LM就是最小化它在猜词游戏中的得分。
- **困惑度**是`2^比特数`（或`e^nats`）：在模型完成猜测后，仍面临的分支因子。均匀猜测27个符号对应困惑度27；每字母1比特的玩家困惑度为2。
- **上下文长度是玩家的记忆。** trigram模型用两个token的记忆玩这个游戏。transformer用100K token的记忆玩同一个游戏。规则从未改变；玩家变强了。

一个需要注意的单位切换：游戏评分每字母用比特（`log2`），而下面n-gram公式中每词token用nats（自然对数）——因为困惑度在nats下是`e^H`，在bits下是`2^H`，两者是同一测量、不同单位。

```figure
prediction-game
```

**N-gram概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定`n`（通常trigram取3，4-gram取4）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未出现的任何n-gram都会被分配概率零。一项针对Brown语料库的2007年研究发现，即使是一个4-gram模型，也有30%的 held-out 4-gram在训练中从未出现。没有平滑就无法对任何真实文本进行评估。

**平滑方法（按复杂程度递增）：**

1. **Laplace（加一平滑）。** 每个计数加1。简单，对稀有事件效果很差。
2. **Good-Turing。** 基于词频的频数，将概率质量从高频率事件重新分配到未见过事件。
3. **插值。** 用可训练权重组合n-gram、(n-1)-gram等估计。
4. **回退。** 若n-gram计数为零，则回退到(n-1)-gram。Katz回退对此进行了规范化。
5. **绝对discounting。** 从所有计数中减去固定折扣`D`，重新分配到未见过事件。
6. **Kneser-Ney。** 绝对discounting + 为低阶模型精心设计的选择：使用*延续概率*（一个词出现在多少个不同上下文中）而非原始频率。

Kneser-Ney的洞察很深刻。"San Francisco"是一个常见bigram。unigram "Francisco"几乎只出现在"San"之后。朴素绝对discounting会给予"Francisco"高unigram概率（因为计数高）。Kneser-Ney注意到"Francisco"只出现在一个上下文中，因此相应降低其延续概率。结果：以"Francisco"结尾的新bigram会被赋予适当低的概率。

**评估：困惑度。** held-out测试集上平均负对数似然的指数。越低越好。困惑度100意味着模型处于困惑状态，就像在100个词中均匀选择一样。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 构建

### 步骤1：trigram计数

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

输入是标记化句子的列表。输出是n-gram计数和上下文计数。`<s>`和`</s>`是句子边界。

### 步骤2：Laplace平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个计数加1。虽然实现了平滑，但会过度分配概率质量给未见过事件，同样伤害到已知的稀有事件。

### 步骤3：Kneser-Ney（bigram，插值）

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

三个关键部分。`continuation_prob`捕捉"这个词出现在多少个不同上下文中？"（Kneser-Ney的创新）。`lambda_prev`是折扣释放出的质量，用于加权回退。最终概率是discounted主项加上加权延续项。

### 步骤4：用采样生成文本

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

按概率比例采样。每次seed给出不同输出。若要类似beam-search的输出，每步选argmax（贪婪），并加入小随机性旋钮（温度）。

### 步骤5：困惑度

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

越低越好。对Brown语料库，一个调好的4-gram KN模型可达困惑度约140。transformer LM在相同测试集上可达15-30。差距约10倍。这就是该领域转向的原因。

## 使用场景

- **经典NLP教学。** 最清晰的平滑、MLE和困惑度入门。
- **KenLM。** 生产级n-gram库。在低延迟敏感的语音和MT系统中用作重排序器。
- **设备端自动补全。** 键盘中的trigram模型。至今仍在使用。
- **基线。** 在宣称你的神经LM表现良好之前，始终计算n-gram LM的困惑度。如果你的transformer没有大幅超越KN，那就是有问题。

## 交付

保存为`outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: 在训练神经LM之前，构建可复现的n-gram语言模型基线。
phase: 5
lesson: 16
---

给定语料库和目标用途（下一个词预测、重排序、困惑度基线），输出：

1. N-gram阶数。通用英语用trigram，语料库大时用4-gram，语音重排序用5-gram。
2. 平滑。Modified Kneser-Ney为默认；Laplace仅用于教学。
3. 库。生产用`kenlm`，教学用`nltk.lm`，除非是为了学习否则不要自己实现。
4. 评估。训练集和测试集使用一致分词下的held-out困惑度。

拒绝报告在不同分词系统之间比较的困惑度——困惑度数字仅在相同分词下才可比较。标记测试集中的OOV率；除非你在训练中预留特殊`<UNK>` token，否则KN处理OOV效果很差。
```

## 练习

1. **简单。** 在1000句莎士比亚语料库上训练trigram LM。生成20个句子。它们会在局部合理但全局不连贯。这是经典演示。
2. **中等。** 对KN模型在held-out莎士比亚子集上计算困惑度。与Laplace比较。你应该看到KN降低困惑度30-50%。
3. **困难。** 构建trigram拼写校正器：给定错词及其上下文，生成修正并按LM上下文概率排名。在Birkbeck拼写语料库（公开）上评估。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| N-gram | 词序列 | 连续`n`个token的序列。 |
| 平滑 | 避免零概率 | 重新分配概率质量，使未见过事件获得非零概率。 |
| 困惑度 | LM质量指标 | `exp(-平均对数概率)`在held-out数据上。越低越好。 |
| 回退 | 回退到较短上下文 | 若trigram计数为零，则使用bigram。Katz回退对此形式化了。 |
| Kneser-Ney | 最佳n-gram平滑 | 绝对discounting + 延续概率用于低阶模型。 |
| 延续概率 | KN特有 | `P(w)`按`w`出现的上下文数加权，而非原始计数。 |
| 文本熵 | 每符号信息量 | 给定上下文时编码下一个符号所需平均比特数。Shannon 1951年对印刷英语（最多100字母上下文）估计：0.6-1.3比特/字母，在任何模型存在之前就被测量出来。 |

## 延伸阅读

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — 定义目标的猜词游戏实验，每个语言模型仍在优化它。
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM和平滑的标准处理。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立Kneser-Ney为最佳n-gram平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — KN原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产级n-gram LM，2026年仍在用于对延迟敏感的应用。
