# 命名实体识别

> 把名字提取出来。听起来容易，直到你遇到边界模糊、实体嵌套和领域行话。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段5 · 02 (词袋 + TF-IDF)，阶段5 · 03 (词嵌入)
**时间：** ~75分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：苹果(ORG)、谷歌(ORG)、iPhone (PRODUCT)、search deal (也许是)、美国(GPE)。一个优秀的NER系统能正确地提取所有实体并标注其类型。一个差的系统会漏掉iPhone，将水果的苹果和苹果公司搞混，并把"US"标记为人名(PERSON)。

NER是每个结构化提取流水线背后的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人回复的实体链接、法律合同提取。你很少直接看到它；但你总是依赖它。

本课程将沿着经典路径（基于规则、HMM、CRF）走进现代方法（BiLSTM-CRF，然后是Transformer模型）。每一步都解决了前一步的特定局限性。这种模式本身就是一堂课。

## 核心概念

**BIO标注**（或BILOU）将实体提取转化为序列标注问题。给每个token标记为`B-TYPE`（实体开始）、`I-TYPE`（实体内部）或`O`（非实体）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多token实体通过链式标注实现：`New B-GPE`, `York I-GPE`, `City I-GPE`。理解BIO的模型可以提取任意跨度的实体。

架构演进：

- **基于规则。** 正则表达式 + 实体列表查询。对已知实体精度高，对新实体零覆盖。
- **HMM。** 隐马尔可夫模型。给定标签的token发射概率，标签到标签的转移概率。使用Viterbi解码。基于标注数据训练。
- **CRF。** 条件随机场。类似于HMM，但属于判别式模型，因此可以混合任意特征（词形、大小写、相邻词）。至今仍是2026年低资源部署场景下的经典生产主力。
- **BiLSTM-CRF。** 神经网络特征取代手工特征。LSTM双向读取句子，顶层的CRF层确保标签序列的一致性。
- **基于Transformer。** 使用token分类头微调BERT。精度最高，计算量最大。

## 动手构建

### 步骤1：BIO标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤2：手工设计的特征

对于经典（非神经网络）NER，特征是关键。一些有用的特征：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式是专有名词的高信号特征。

### 步骤3：一个简单的基于规则+词典的基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级的实体列表拥有从Wikipedia和DBpedia抓取的数百万条目。覆盖率很好。但歧义消解（`Apple` 公司 vs 水果）非常糟糕。这就是统计模型胜出的原因。

### 步骤4：CRF步骤（概要，非完整实现）

从头用50行代码实现完整的CRF，如果没有概率论基础，并无太大启发性。可以使用 `sklearn-crfsuite` 代替：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是L1和L2正则化。`all_possible_transitions=True` 让模型能够学习非法序列（例如，`I-ORG` 出现在 `O` 之后）是不太可能的，这就是CRF在不手动编写约束的情况下强制执行BIO一致性的原理。

### 步骤5：BiLSTM-CRF的改进

特征变为可学习的。输入：token嵌入（GloVe或fastText）。LSTM从左到右和从右到左读取。连接后的隐藏状态通过一个CRF输出层。CRF仍然强制标签序列的一致性；而LSTM用可学习的特征替代了手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于CRF层，使用 `torchcrf.CRF`（pip install pytorch-crf）。相对于手工特征的CRF，其提升是可测量的，但除非你有数万条标注句子，否则提升幅度可能比你预期的要小。

## 使用它

spaCy 开箱即用地提供了生产级的NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标记为 `ORG` 而非 `PRODUCT` —— spaCy的小模型对产品实体的覆盖较弱。大模型（`en_core_web_lg`）表现更好。Transformer模型（`en_core_web_trf`）则更胜一筹。

使用Hugging Face进行基于BERT的NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的B-X、I-X token合并成一个跨度。没有它，你只能得到token级别的标签，需要自己合并。

### 基于LLM的NER（2026年的选择）

在许多领域，基于零样本和少样本的LLM NER现在已经可以与微调模型相媲美，并且在标注数据稀缺时表现显著更好。

- **零样本提示。** 给LLM一个实体类型列表和一个示例schema。要求JSON输出。开箱即用；在新领域的精度中等。
- **ZeroTuneBio风格的提示。** 将任务分解为候选提取 → 含义解释 → 判断 → 复核。一个多阶段的提示（非单次）在生物医学NER上显著提升了精度。同样的模式适用于法律、金融和科学领域。
- **结合RAG的动态提示。** 对于每次推理调用，从一个小型的标注种子集中检索最相似的标注示例；动态构建少样本提示。在2026年的基准测试中，这比静态提示将GPT-4的生物医学NER F1值提升了11-12%。
- **按实体类型分解。** 对于长文档，一次调用提取所有实体类型会随着长度增加而导致召回率下降。针对每种实体类型运行一次提取。推理成本更高，但精度大幅提升。这是临床笔记和法律合同的标准模式。

2026年的生产建议：在收集训练数据之前，先从LLM零样本基线开始。通常其F1值已经足够好，以至于你根本不需要微调。

### 经典NER仍然胜出的场景

即使有LLM可用，在以下情况下经典NER仍然胜出：

- 延迟预算低于50毫秒。
- 你拥有数千个标注样本，并且需要98%+的F1值。
- 领域具有稳定的本体结构，预训练的CRF或BiLSTM能很好地迁移。
- 监管要求使用本地部署的、非生成式的模型。

### 它失效的场景

- **领域漂移。** 在CoNLL上训练的NER用于法律合同时，表现比实体列表还差。在你的领域上进行微调。
- **嵌套实体。** "Bank of America Tower" 既是组织机构(ORG)又是设施(FACILITY)。标准BIO无法表示重叠跨度。你需要嵌套NER（多轮或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." 基于token的模型有时会拆分它。使用 `aggregation_strategy` 或进行后处理。
- **稀疏类型。** 医学NER标签如DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型一无所知。Scispacy和BioBERT是那里的起点。

## 部署它

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在10个句子上验证往返一致性。
2. **中等。** 使用上面的sklearn-crfsuite CRF在CoNLL-2003英语NER数据集上训练。使用 `seqeval` 报告每个实体的F1值。典型结果：~84 F1。
3. **困难。** 在一个特定领域的NER数据集（医疗、法律或金融）上微调 `distilbert-base-cased`。与spaCy的小模型进行对比。记录数据泄露检查，并写下让你感到意外的地方。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|--------------|
| NER | 提取名字 | 用类型（PERSON, ORG, GPE, DATE, ...）标记token跨度。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 外部。 |
| BILOU | 更好的BIO | 增加了 `L-X`（最后），`U-X`（单元），用于更清晰的边界。 |
| CRF | 结构化分类器 | 建模标签间的转移，而不仅仅是发射。强制有效序列。 |
| 嵌套NER | 重叠的实体 | 一个跨度与其子跨度是不同的实体。BIO无法表达这一点。 |
| 实体级F1 | 正确的NER度量 | 预测的跨度必须与真实跨度完全匹配。Token级F1会高估准确率。 |

## 扩展阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF论文。经典之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 介绍了成为标准的token分类模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — 关于 `Doc.ents` 和 `Span` 每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的度量库。务必使用它。