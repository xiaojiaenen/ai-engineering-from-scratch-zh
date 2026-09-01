# 文本处理 —— 分词、词干提取、词形还原

> 语言是连续的，模型是离散的。预处理是连接两者的桥梁。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 2 · 14（朴素贝叶斯）
**预计时间：** 约 45 分钟

## 问题所在

模型无法"阅读" `"The cats were running."` 。它读取的是整数。

每个 NLP 系统的入口都面临着同样的三个问题：一个词从哪里开始？一个词的根本形式是什么？何时应当把 `"run"`、`"running"`、`"ran"` 视为同一个东西，何时又应当视为不同？

分词做错，模型就会从垃圾中学习。如果你的分词器把 `don't` 当作一个 token，却把 `do n't` 当作两个 token，训练分布就会被撕裂。如果你的词干提取器把 `organization` 和 `organ` 折叠成同一个词干，主题建模就会失效。如果你的词形还原器需要词性上下文但你没有传入，动词就会被当作名词处理。

本课从零实现这三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你看清各自的取舍。

## 概念

三个操作。每个都有各自的职责和失败的诱因。

**分词（Tokenization）** 将一段字符串拆分为 token。"Token" 这个词故意留有一定的模糊空间，因为合适的粒度取决于任务。经典 NLP 用词级别，Transformer 用子词级别，无空格语言（如中文）用字符级别。

**词干提取（Stemming）** 用规则切除词缀。快速、激进、笨拙。`running -> run`。`organization -> organ`。第二种情况就是失败模式的体现。

**词形还原（Lemmatization）** 借助语法知识将词归并到字典中的规范形式。较慢、准确、需要查找表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道这是比较级形式）。

经验法则：当速度重要且可容忍噪声时用词干提取（搜索索引、粗略分类）；当语义重要时用词形还原（问答、语义搜索、任何用户会看到的结果）。

```figure
edit-distance
```

## 动手实现

### 第 1 步：一个基于正则的词级分词器

最简单实用的分词器按非字母数字字符拆分，同时把标点单独作为 token 保留。不够完美，也不是最终方案，但一行就能跑通。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

按优先级排列的三个模式：带内部可选撇号的单词（`don't`、`it's`）、纯数字、单个非空白非字母数字字符作为独立 token（即标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要留意失败模式。`3pm` 会被拆成 `['3', 'pm']`，因为我们交替匹配字母段和数字段。大多数场景够用了。URL、邮箱、标签全部会出错。生产环境需要在通用规则之前添加专用模式。

### 第 2 步：一个 Porter 词干提取器（仅第 1a 阶段）

完整的 Porter 算法有五阶段规则。只实现第 1a 阶段就足以覆盖最常见的英语后缀并展现该模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上往下读这些规则。`ies -> i` 这条规则解释了为什么 `ponies -> poni` 而不是 `pony`。真正的 Porter 有第 1b 阶段可以修正这个问题。规则之间存在竞争关系，靠前的规则胜出。顺序比任何单条规则都更重要。

### 第 3 步：一个基于查表的词形还原器

真正的词形还原需要形态学知识。一个便于教学的简化版使用一个小查表 + 兜底逻辑。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是核心教学点。`watched` 不在我们的表中，而兜底逻辑只处理 `ing` 结尾。真正的词形还原本应覆盖 `ed` 结尾、不规则动词、形容词比较级、发生音变的复数（`children -> child`）。这正是生产系统使用 WordNet、spaCy 的词形分析器或完整形态分析器的原因。

### 第 4 步：把它们串起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的一环是词性标注器。阶段 5 · 07（词性标注）会构建一个。现在先把所有词默认标为 `NOUN`，并接受这个局限。

## 实际使用

NLTK 和 spaCy 提供的是生产级版本。各只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 能正确处理缩略形式、Unicode 以及你的正则表达式会漏掉的边界情况。`PorterStemmer` 执行全部五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 词性标签翻译成 WordNet 的缩写集合。上面那串翻译映射是多数教程跳过的那一步。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整个流水线隐藏在 `nlp(text)` 背后。分词、词性标注、词形还原全部运行。规模上比 NLTK 更快，开箱即用也更准确。代价是你难以单独替换其中的某个组件。

### 如何选择

| 场景 | 推荐 |
|------|------|
| 教学、研究、需要灵活替换组件 | NLTK |
| 生产环境、多语言、对速度敏感 | spaCy |
| Transformer 流水线（你反正会用模型的 tokenizer 做分词） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 两个几乎没人提醒你注意的失败模式

大多数教程只教算法然后就结束了。在真实预处理流水线上，有两件事会让你翻车，而且它们几乎从不被讨论。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原的行为。spaCy 2.x 中产生 `['do', "n't"]` 的逻辑，在 3.x 中可能产生 `["don't"]`。你的模型训练于一种分布，而推理却跑在另一种分布上。准确率会无声地退化，而没有人知道原因。在 `requirements.txt` 中锁定库的版本。编写一条预处理回归测试，对 20 个样本句子的分词结果进行冻结校验，每次升级时都运行它。

**训练 / 推理不匹配。** 训练时用激进的预处理（小写化、停用词去除、词干提取），部署时却在原始用户输入上运行，然后看着性能崩溃。这是生产 NLP 最常见的失败模式。如果训练时做了预处理，推理时必须运行完全相同的函数。把预处理作为一个函数随模型包一起交付，而不是让推理团队自行改写的一个 notebook 单元格。

## 交付物

一条可复用的提示词，帮助工程师在不读三本教材的情况下选择合适的预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取和词形还原的配置方案。
phase: 5
lesson: 01
---

你为经典 NLP 预处理提供建议。根据任务描述，输出：

1. 分词方案选择（正则、NLTK word_tokenize、spaCy 或 Transformer tokenizer）。说明理由。
2. 是否应该词干提取、词形还原、两者都做或都不做。说明理由。
3. 具体的库调用。给出函数名。如果涉及 NLTK，写出词性标签的翻译对照。
4. 一条用户应当测试的失败模式。

拒绝为用户可见文本推荐词干提取。拒绝在无词性标签时推荐词形还原。发现非英文输入时标注为需要不同的流水线。
```

## 练习

1. **简单。** 扩展 `tokenize` 使其将 URL 作为单个 token 保留。测试：`tokenize("Visit https://example.com today.")` 应产生一个 URL token。
2. **中等。** 实现 Porter 第 1b 阶段：如果单词包含元音且以 `ed` 或 `ing` 结尾，则移除之后缀。处理双辅音规则（`hopping -> hop`，不是 `hopp`）。
3. **困难。** 构建一个以 WordNet 为查表、在 WordNet 无条目时回退到你实现的 Porter 词干提取器的词形还原器。在与词性标注语料对比时，分别测量纯 WordNet 和纯 Porter 的准确率。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| Token | 一个词 | 模型实际消费的单位。可以是词、子词、字符或字节。 |
| Stem（词干） | 词根 | 基于规则的词缀剥离结果。不一定是真实存在的词。 |
| Lemma（词原） | 字典形式 | 你在词典中会查找到的那个形式。需要语法上下文才能正确计算。 |
| POS tag（词性标签） | 词类 | 如 NOUN、VERB、ADJ 等类别。词形还原需要它才能准确执行。 |
| Morphology（形态学） | 词的形态规则 | 词如何根据时态、数、格等改变形式。词形还原依赖于它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) —— 原始论文，五页篇幅，至今仍是解释最清晰的文献。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) —— 一个真实流水线是如何连接的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) —— 你可能尚未想到的分词边界情况。
