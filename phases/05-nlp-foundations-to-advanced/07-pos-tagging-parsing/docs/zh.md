# 词性标注与句法分析

> 语法分析曾一度不再流行。后来每个大语言模型流水线都需要验证结构化抽取，于是它又回来了。

**类型:** 构建项目  
**语言:** Python  
**前提课程:** 阶段 5 · 01 (文本处理), 阶段 2 · 14 (朴素贝叶斯)  
**时间:** ~45 分钟

## 问题背景

课程 01 曾承诺词形还原需要词性标签。若不知道 `running` 是动词，词形还原器无法将其还原为 `run`。若不知道 `better` 是形容词，则无法还原为 `good`。

这个承诺背后隐藏着一个完整的子领域。**词性标注**为每个词元分配语法类别。**句法分析**则复原句子的树形结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典自然语言处理花了二十年精进这两项任务。随后深度学习将它们合并为预训练转换器之上的词元分类任务，研究界便转向他处。

但应用界并非如此。每个结构化抽取流水线在底层仍在使用词性标注和依存树。大语言模型生成的 JSON 会根据语法约束进行验证。问答系统利用依存分析分解查询。机器翻译质量评估器则检查分析树的对齐情况。

值得了解。本课程介绍标签集、基线方法，以及何时应停止从头实现并直接调用 spaCy。

## 核心概念

**词性标注**为每个词元标注语法类别。**宾夕法尼亚树库 (PTB)** 标签集是英语的默认选择。包含 36 个标签，区分细致到普通读者会觉得繁琐：`NN` 单数名词，`NNS` 复数名词，`NNP` 专有名词单数，`VBD` 动词过去式，`VBZ` 动词第三人称单数现在时，等等。**通用依存关系 (UD)** 标签集更粗粒度（17 个标签）且与语言无关；已成为跨语言工作的默认标准。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**生成一棵树。两种主要风格：

- **成分句法分析。** 名词短语、动词短语、介词短语相互嵌套。输出是以词为叶节点的非终结类别树（NP, VP, PP）。
- **依存句法分析。** 每个词都有一个它所依存的中心词，并标注语法关系。输出是一棵每条边都是 (中心词, 依存词, 关系) 三元组的树。

依存句法分析在 2010 年代胜出，因为它能干净地泛化到各种语言，尤其是语序自由的语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 动手构建

### 步骤 1：最频繁标签基线

能工作的最简单词性标注器。对每个词，预测它在训练数据中最常出现的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，此基线准确率约 85%。不算高，但这是任何严肃模型不应低于的底线。

### 步骤 2：二元隐马尔可夫词性标注器

对序列的联合概率进行建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标签的标签概率）、发射概率（给定标签的词概率）。使用拉普拉斯平滑从计数估计两者。用维特比算法（在标签网格上动态规划）解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 语料库上的二元隐马尔可夫模型准确率约 93%。从 85% 到 93% 的飞跃主要来自转移概率——模型学到了 `DET NOUN` 很常见而 `NOUN DET` 很少见。

### 步骤 3：为何现代标注器更优

转移概率和发射概率是局部的。它们无法捕捉到 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie" 中是动词。使用任意特征（后缀、词形、前后词、词本身）的条件随机场可达约 97% 准确率。双向 LSTM-CRF 或转换器可达 98% 以上。

此任务的天花板由标注员间的分歧决定。在宾夕法尼亚树库上，人类标注员的一致性约为 97%。准确率超过 98% 的模型很可能是在测试集上过拟合了。

### 步骤 4：依存句法分析概述

从头实现完整的依存句法分析超出范围；标准教材处理在 Jurafsky 和 Martin 的著作中。需了解两大经典方法：

- **基于转移的**分析器（弧急切、弧标准）类似移进-归约解析器：读取词元，将其移入栈，并应用创建弧的归约操作。贪心解码速度快。经典实现是 MaltParser。现代神经版本：Chen 和 Manning 的基于转移的分析器。
- **基于图的**分析器（Eisner 算法、Dozat-Manning 双仿射）对每个可能的中心词-依存词边打分，并选取最大生成树。速度较慢但更准确。

对于大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上阅读 `dep` 列，句子的语法结构便一目了然。

## 实际应用

每个生产级自然语言处理库都将词性标注和依存句法分析器作为标准流水线的一部分发布。

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`)。快速、准确，与分词、命名实体识别、词形还原集成。`token.tag_` (PTB)，`token.pos_` (UD)，`token.dep_` (依存关系)。
- **Stanford NLP (stanza)**。斯坦福对 CoreNLP 的继任者。在 60 多种语言上达到最先进水平。
- **trankit**。基于转换器，UD 准确率高。
- **NLTK**。`pos_tag`。可用，速度慢，较旧。适合教学。

### 在 2026 年这仍然重要的场景

- **词形还原。** 课程 01 需要词性才能正确还原词形。始终如此。
- **从大语言模型输出中进行结构化抽取。** 验证生成的句子是否遵守语法约束（例如主谓一致、必需的修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "由韦斯·安德森执导、比尔·默里主演的电影" 通过分析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系与语言无关，可实现对新语言的零样本结构化分析。
- **低算力流水线。** 如果无法部署转换器，词性标注 + 依存分析 + 词汇表能带你走得相当远。

## 部署上线

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习题

1. **简单。** 在一个小型标注语料库（例如 NLTK 的 Brown 子集）上使用最频繁标签基线，测量其在留出句子上的准确率。验证约 85% 的结果。
2. **中等。** 训练上述二元隐马尔可夫模型，并报告每个标签的精确率/召回率。隐马尔可夫模型最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存分析从一个 1000 句样本中提取主谓宾三元组。在 50 个人工标注的三元组上进行评估。记录抽取失败的情况（通常是被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 人们怎么说 | 其实际含义 |
|------|------------|------------|
| 词性标签 | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| 宾夕法尼亚树库 | 标准标签集 | 英语专用。细分动词时态和名词数。 |
| 通用依存关系 | 多语言标签集 | 比 PTB 粗粒度；语言中立；跨语言工作的默认标准。 |
| 依存分析 | 句子树 | 每个词有一个中心词，每条边有一个语法关系。 |
| 维特比算法 | 动态规划 | 给定发射和转移概率，找到概率最高的标签序列。 |

## 扩展阅读

- [Jurafsky 和 Martin — 《语音与语言处理》第 8 和 18 章](https://web.stanford.edu/~jurafsky/slp3/) — 关于词性标注和分析的权威教材内容。
- [通用依存关系项目](https://universaldependencies.org/) — 每个多语言分析器使用的跨语言标签集和树库集合。
- [spaCy 语言特征指南](https://spacy.io/usage/linguistic-features) — `Token` 上每个属性的实用参考。
- [Chen 和 Manning (2014). 《使用神经网络的快速准确依存分析器》](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经分析器引入主流的论文。