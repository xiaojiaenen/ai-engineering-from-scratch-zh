# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的，模型是离散的，预处理是桥梁。

**类型:** 实战构建
**语言:** Python
**前置要求:** 第二阶段 · 14 (朴素贝叶斯)
**时间:** 约45分钟

## 问题所在

模型无法阅读"The cats were running."，它只能读取整数。

每个NLP系统都始于相同的三个问题：一个词从哪里开始？词的根词是什么？我们如何在需要时将"run"、"running"、"ran"视为同一事物，在不需要时又视为不同事物？

分词错误，模型就会从垃圾中学习。如果你的分词器将 `don't` 视为一个token，但将 `do n't` 视为两个token，训练分布就会分裂。如果你的词干提取器将 `organization` 和 `organ` 折叠到相同的词干，主题建模就会失效。如果你的词形还原器需要词性上下文但你未提供，动词就会被当作名词处理。

本课从零构建这三个预处理原语，然后展示NLTK和spaCy如何实现相同的功能，让你了解其中的权衡。

## 核心概念

三种操作。每种都有其作用和失效模式。

**分词** 将字符串分割成token。"Token"故意定义模糊，因为合适的粒度取决于任务。经典NLP使用词级。Transformer使用子词。没有空格的语言使用字符级。

**词干提取** 用规则切除后缀。快速、激进、粗暴。`running -> run`。`organization -> organ`。第二个例子就是失效模式。

**词形还原** 使用语法知识将词还原为其字典形式。更慢、更准确，需要查找表或形态分析器。`ran -> run` (需要知道"ran"是"run"的过去式)。`better -> good` (需要知道比较级形式)。

经验法则：当速度重要且可以容忍噪声时使用词干提取（搜索索引、粗略分类）。当语义重要时使用词形还原（问答、语义搜索，任何用户会阅读的内容）。

## 动手构建

### 步骤1: 一个基于正则表达式的词分词器

最简单的实用分词器在非字母数字字符上分割，同时保留标点作为独立的token。不完美，不最终，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排序：包含可选内部撇号的词 (`don't`, `it's`)；纯数字；任何单个非空白非字母数字字符作为独立token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注意失效模式。`3pm` 被分割为 `['3', 'pm']`，因为我们在字母串和数字串之间交替。对大多数任务足够好。URL、电子邮件、标签都会损坏。在生产环境中，在通用模式之前添加特定模式。

### 步骤2: 一个Porter词干提取器 (仅步骤1a)

完整的Porter算法有五个阶段的规则。仅步骤1a就涵盖了最常见的英语后缀，并展示了模式。

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

自上而下阅读规则。`ies -> i` 规则是 `ponies -> poni` 而非 `pony` 的原因。真正的Porter有步骤1b会修正它。规则相互竞争。先定义的规则优先。顺序比任何单个规则都重要。

### 步骤3: 一个基于查找的词形还原器

真正的词形还原需要形态学知识。一个可教学的简化版本使用一个小型词形还原表和后备策略。

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

最后一个例子是关键的教学点。`watched` 不在我们的表中，我们的后备策略只能处理 `ing`。真正的词形还原涵盖 `ed`、不规则动词、比较级形容词、有音变的复数 (`children -> child`)。这就是为什么生产系统使用WordNet、spaCy的形态分析器或完整的形态分析工具。

### 步骤4: 将它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是一个词性标注器。第5阶段 · 07 (词性标注)会构建一个。目前，默认将所有词性设为 `NOUN`，并承认这一限制。

## 实际应用

NLTK和spaCy提供了生产版本。几行代码即可实现。

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

`word_tokenize` 处理缩写、Unicode、你的正则表达式遗漏的边界情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将NLTK的Penn Treebank词性标签转换为WordNet的缩写集。上面的转换连接是大多数教程跳过的部分。

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

spaCy将整个管道隐藏在 `nlp(text)` 后面。分词、词性标注和词形还原全部运行。大规模下比NLTK更快。开箱即用更准确。权衡在于你无法轻松替换单个组件。

### 如何选择

| 场景 | 选择 |
|------|------|
| 教学、研究、替换组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer管道（反正你会用模型自带的分词器） | 使用 `tokenizers` / `transformers` 并跳过经典预处理 |

### 两种没人警告你的失效模式

大多数教程讲完算法就停止了。有两件事会咬到真实的预处理管道，而且它们几乎从不被提及。

**可复现性漂移。** NLTK和spaCy会在版本之间改变分词和词形还原行为。spaCy 2.x中产生 `['do', "n't"]` 的结果在3.x中可能产生 `["don't"]`。你的模型在一个分布上训练。推理现在运行在另一个分布上。准确率悄然下降，无人知晓原因。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，冻结20个示例句子的预期分词结果。每次升级都运行它。

**训练/推理失配。** 训练时使用激进的预处理（小写化、去除停用词、词干提取），部署时使用原始用户输入，然后看着性能暴跌。这是最常见的生产NLP失败。如果你在训练期间进行了预处理，在推理期间必须运行完全相同的函数。将预处理作为模型包内的函数发布，而不是作为服务团队重写的笔记本单元格。

## 可交付成果

一个可复用的提示，帮助工程师选择预处理策略，而无需阅读三本教科书。

保存为 `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1.  **简单。** 扩展 `tokenize` 以将URL保留为单个token。测试：`tokenize("Visit https://example.com today.")` 应产生一个URL token。
2.  **中等。** 实现Porter步骤1b。如果一个词包含元音并以 `ed` 或 `ing` 结尾，则移除它。处理双辅音规则 (`hopping -> hop`, 而非 `hopp`)。
3.  **困难。** 构建一个词形还原器，使用WordNet作为查找表，当WordNet没有条目时回退到你的Porter词干提取器。在一个标注语料库上，对比纯WordNet和纯Porter测量准确率。

## 关键术语

| 术语 | 人们常说的含义 | 实际含义 |
|------|----------------|----------|
| Token | 一个词 | 模型消耗的任何单元。可以是词、子词、字符或字节。 |
| 词干 | 词的词根 | 基于规则去除后缀的结果。不总是一个真实的词。 |
| 词元 | 字典形式 | 你查找时看到的形式。需要语法上下文才能正确计算。 |
| 词性标签 | 词的类别 | 如名词、动词、形容词等类别。精确词形还原需要它。 |
| 形态学 | 词形变化规则 | 词基于时态、数量、格等如何改变形式。词形还原依赖于它。 |

## 扩展阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今仍是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实管道如何连接。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边界情况。