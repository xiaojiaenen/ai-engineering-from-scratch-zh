# POS 标注与句法解析

> 语法分析一度不再流行。后来每条 LLM 流水线都需要验证结构化抽取，于是它回归了。

**类型：** Build
**语言：** Python
**前置条件：** 第 5 阶段 · 01（文本处理）、第 2 阶段 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题所在

第 01 课承诺过：词形还原需要词性标注。如果不清楚 `running` 是动词，词形还原器就无法将其还原为 `run`。如果不清楚 `better` 是形容词，就无法将其还原为 `good`。

那个承诺背后隐藏着一个完整的子领域。词性标注（POS tagging）为每个词分配语法类别；句法解析（syntactic parsing）恢复句子的树形结构：哪个词修饰哪个词、哪个动词支配哪些论元。经典 NLP 花了二十年时间完善这两个技术。随后深度学习将它们坍缩为预训练 Transformer 之上的一个词分类任务，研究社区随之转向。

应用社区没有。每条结构化抽取流水线底层仍在依赖 POS 和依存树。LLM 生成的 JSON 要接受语法约束的校验。问答系统通过依存解析分解查询。机器翻译质量评估器会检查 parse tree 的对齐情况。

值得了解。本课介绍标签集、基线模型，以及你该停止从零实现、转而调用 spaCy 的临界点。

## 概念

**词性标注** 为每个 token 打上语法类别标签。**Penn Treebank（PTB）标签集** 是英语默认标准。36 个标签，其中的细微区分让普通读者觉得繁琐：`NN` 单数名词，`NNS` 复数名词，`NNP` 专有名词单数，`VBD` 动词过去式，`VBZ` 动词第三人称单数现在时，等等。**Universal Dependencies（UD）标签集** 更粗粒度（17 个标签）且语言无关；它是跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法解析** 产出一棵树。两种主流风格：

- **成分解析（Constituency parsing）。** 名词短语、动词短语、介词短语互相嵌套。输出是一棵非终结符类别（NP、VP、PP）的树，词作为叶子节点。
- **依存解析（Dependency parsing）。** 每个词都有且仅有一个它依赖的中心词（head），并带有语法关系标签。输出是一棵树，每条边都是一个 (head, dependent, relation) 三元组。

依存解析在 2010 年代胜出，因为它能干净地跨语言泛化，尤其是自由语序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## 动手构建

### 第一步：最常见标签基线

最简单但能用的 POS 标注器。对每个词，预测它在训练数据中出现最频繁的标签。

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

在 Brown 语料上，该基线达到约 85% 准确率。不算好，但任何严肃模型都不应低于这个值。

### 第二步：二元 HMM 标注器

对序列的联合概率建模：

```
P(tags, words) = ∏ P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标签的当前标签）和发射概率（给定标签的词）。用带 Laplace 平滑的计数估计两者。用 Viterbi 解码（在标签格状结构上做动态规划）。

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

二元 HMM 在 Brown 语料上达到约 93% 准确率。从 85% 到 93% 的提升主要来自转移概率——模型学到了 `DET NOUN` 很常见而 `NOUN DET` 很少见。

### 第三步：为什么现代标注器更强

转移概率和发射概率都是局部的。它们无法捕捉 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie" 中是动词。带有任意特征的 CRF（后缀、词形、前后词、词本身）可达约 97%。BiLSTM-CRF 或 Transformer 可达 98%+。

此任务的上限由标注者分歧决定。人类标注者在 Penn Treebank 上的一致性约为 97%。超过 98% 的模型可能是在测试集上过拟合了。

### 第四步：依存解析概览

从零实现完整的依存解析超出本课范围；经典教材叙述见 Jurafsky 和 Martin。需要了解两类经典方法：

- **基于转换的解析器（Transition-based）**（arc-eager、arc-standard）类似于移进-归约解析器：读取 token，将其移入栈，然后执行创建弧的归约操作。贪心解码速度快。经典实现是 MaltParser。现代神经版本：Chen 和 Manning 的基于转换的解析器。
- **基于图灵的解析器（Graph-based）**（Eisner 算法、Dozat-Manning biaffine）为每条可能的 head-dependent 边打分，并选取最大生成树。更慢但更准确。

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

从下往上读 `dep` 列，句子的语法结构就清晰浮现出来了。

## 使用它

每条生产级 NLP 库都将 POS 标注和依存解析作为标准流水线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词 + NER + 词形还原集成。`token.tag_`（PTB）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford CoreNLP 的继任者。在 60+ 种语言上达到最先进水平。
- **trankit**。基于 Transformer，UD 准确率高。
- **NLTK**。`pos_tag`。可用，较慢，较旧。适合教学。

### 2026 年仍重要的场景

- **词形还原。** 第 01 课需要 POS 才能正确词形还原。始终需要。
- **LLM 输出的结构化抽取。** 校验生成句子是否符合语法约束（如主谓一致、必需修饰语）。
- **方面级情感分析。** 依存解析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过解析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系语言无关，使对新语言的零样本结构化分析成为可能。
- **低算力流水线。** 如果无法部署 Transformer，POS + 依存解析 + 词典已能取得令人惊讶的效果。

## 交付

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

## 练习

1. **简单。** 在小规模带标注语料（如 NLTK 的 Brown 子集）上使用最常见标签基线，测量在未见过句子上的准确率，验证约 85% 的结果。
2. **中等。** 训练上面的二元 HMM 并报告各标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存解析从 1000 句样本中提取主谓宾三元组。在 50 个人工标注的三元组上评估。记录抽取失败之处（通常是被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 人们常说的意思 | 实际含义 |
|------|----------------|---------|
| POS tag | 词的"类型" | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英语专用。细粒度的动词时态和名词数的区分。 |
| Universal Dependencies | 多语言标签集 | 比 PTB 更粗粒度；语言无关；跨语言工作默认选择。 |
| Dependency parse | 句子树 | 每个词有一个中心词，每条边对应一个语法关系。 |
| Viterbi | 动态规划 | 在给定发射概率和转移概率下，找出概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing，第 8 章和第 18 章](https://web.stanford.edu/~jurafsky/slp3/) — POS 和解析的经典教材叙述。
- [Universal Dependencies 项目](https://universaldependencies.org/) — 多语言解析器使用的跨语言标签集和树库集合。
- [spaCy 语言特征指南](https://spacy.io/usage/linguistic-features) — `Token` 上每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经解析器带入主流的那篇论文。
