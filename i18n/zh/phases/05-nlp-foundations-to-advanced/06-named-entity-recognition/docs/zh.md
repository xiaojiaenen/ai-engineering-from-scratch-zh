# 命名实体识别

> 抽取人名。听起来简单，直到你面对模糊边界、嵌套实体和领域术语时，就麻烦了。

**类型:** 构建
**语言:** Python
**前置条件:** 阶段 5 · 02（BoW + TF-IDF），阶段 5 · 03（词嵌入）
**时间:** ~75 分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（可能）、US（GPE）。一个优秀的 NER 系统会正确提取所有这些实体及其类型。一个糟糕的会漏掉 iPhone，把 Apple 水果和 Apple 公司混淆，还把 "US" 标成人物。

NER 是每个结构化抽取流水线背后的基石。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、对话机器人响应落地、法律合同抽取。你很少直接看到它，但总是依赖着它。

本课沿着经典路线（基于规则、HMM、CRF）走向现代路线（BiLSTM-CRF，然后是 Transformer）。每一步都解决前一步的一个特定局限。这个模式就是本课的核心。

## 概念

**BIO 标注**（或 BILOU）将实体抽取转化为序列标注问题。给每个 token 打上 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不在任何实体内）标签。

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

多 token 实体链条：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意跨度。

架构演进：

- **基于规则。** 正则表达式 +  gazetteer 查找。对已知实体精确度高，对新实体覆盖率为零。
- **HMM。** 隐马尔可夫模型。给定标签时 token 的发射概率，标签到标签的转移概率。维特比解码。在标注数据上训练。
- **CRF。** 条件随机场。类似于 HMM 但是判别式的，因此可以混合任意特征（词形、大写、相邻词）。到 2026 年仍是低资源部署的经典生产级工具。
- **BiLSTM-CRF。** 用神经特征代替手工特征。LSTM 双向读取句子，顶部的 CRF 层强制标签序列一致。
- **基于 Transformer。** 微调 BERT 并加上 token 分类头。准确率最高。计算开销最大。

```figure
ner-bio-tagging
```

## 构建

### 步骤 1：BIO 标注辅助函数

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

### 步骤 2：手工特征

对于经典（非神经网络）NER，特征是核心竞争力。有用的特征包括：

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

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大写模式对专有名词是高信号特征。

### 步骤 3：简单的基于规则 + 词典基线

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

生产级 gazetteer 有从 Wikipedia 和 DBpedia 抓取的数百万条目。覆盖率不错，但消歧（`Apple` 公司 vs 水果）很差。这正是统计模型胜出的原因。

### 步骤 4：CRF 步骤（概要，非完整实现）

在没有概率论基础的情况下从头用 50 行实现完整 CRF 没什么启发性。改用 `sklearn-crfsuite`：

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

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学会非法序列（如 `O` 后面的 `I-ORG`）概率很低，这就是 CRF 无需手动写约束就能强制 BIO 一致性的方式。

### 步骤 5：BiLSTM-CRF 增加了什么

特征变为可学习。输入：token 嵌入（GloVe 或 fastText）。LSTM 从左到右和从右到左读取。拼接后的隐状态经过 CRF 输出层。CRF 仍强制标签序列一致性；LSTM 用可学习特征代替手工特征。

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

CRF 层使用 `torchcrf.CRF`（pip install pytorch-crf）。相对于手工特征的 CRF 的提升可测量，但不如你想象的那么大——除非你有数万条标注句子。

## 使用

spaCy 开箱即用，自带生产级 NER。

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

注意 `iPhone` 被标为 `ORG` 而非 `PRODUCT`——spaCy 的小型模型对产品实体覆盖较弱。大型模型（`en_core_web_lg`）表现更好。Transformer 模型（`en_core_web_trf`）更好。

使用 Hugging Face 的 BERT-based NER：

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

`aggregation_strategy="simple"` 将连续的 B-X、I-X token 合并为一个跨度。如果不设此参数，你得到的是 token 级标签，需要自己合并。

### 基于 LLM 的 NER（2026 年方案）

零样本和少样本 LLM NER 在很多领域已与微调模型持平，且在标注数据稀缺时显著更好。

- **零样本提示。** 给 LLM 一个实体类型列表和一个示例 schema，要求 JSON 输出。开箱即用；在新领域准确率中等。
- **ZeroTuneBio 风格提示。** 将任务分解为候选抽取 → 含义解释 → 判断 → 复查。多阶段提示（非单次提示）大幅提升生物医学 NER 的准确率。同样模式适用于法律、金融和科学领域。
- **带 RAG 的动态提示。** 为每次推理调用从小型标注种子集中检索最相似的标注样例；动态构建少样本提示。2026 年基准测试中，这使 GPT-4 生物医学 NER F1 比静态提示提升 11-12%。
- **按实体类型分解。** 对长文档，一次性抽取所有实体类型的单调用随着长度增加会丢失召回率。每种实体类型运行一次抽取。推理成本更高，准确率显著提升。这是临床病历和法律合同的标准模式。

2026 年生产建议：在收集训练数据之前，先用 LLM 零样本做基线。很多时候 F1 已经足够好，你根本不需要微调。

### 经典 NER 仍占优的场景

即使 LLM 可用，经典 NER 在以下场景胜出：

- 延迟预算低于 50ms。
- 有数千条标注样例，需要 98%+ F1。
- 领域有稳定本体，预训练 CRF 或 BiLSTM 迁移效果好。
- 监管要求本地部署、非生成式模型。

### 哪里会失效

- **领域偏移。** CoNLL 训练的 NER 在法律合同上表现不如 gazetteer。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是 ORG 也是 FACILITY。标准 BIO 无法表示重叠跨度。你需要嵌套 NER（多轮或基于 span 的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." Token 级模型有时会把它们拆开。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医疗 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型对此一无所知。Scispacy 和 BioBERT 是那里的起点。

## 交付

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: 为给定抽取任务选择合适的 NER 方案。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

给定一个任务描述（领域、标签集、语言、延迟要求、数据量），输出：

1. 方案。基于规则 + gazetteer、CRF、BiLSTM-CRF，或 Transformer 微调。
2. 起始模型。点名（spaCy 模型 ID、Hugging Face checkpoint ID，或"从零开始自定义训练"）。
3. 标注策略。BIO、BILOU，或基于 span。一句话说明理由。
4. 评估。使用 `seqeval`。始终报告实体级 F1（非 token 级）。

除非用户已有预训练领域模型，否则拒绝在少于 500 条标注样例时推荐 Transformer 微调。标记嵌套实体需要基于 span 或多轮模型。如果用户提及"生产规模"且标签与 CoNLL-2003 相同，则要求 gazetteer 审计。
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英语 NER 数据集上训练上面的 sklearn-crfsuite CRF。使用 `seqeval` 报告各实体 F1。典型结果：~84 F1。
3. **困难。** 在领域专用 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小型模型对比。记录数据泄露检查并写下让你惊讶的内容。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| NER | 抽取人名 | 用类型标签 token 跨度（PERSON、ORG、GPE、DATE、...）。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 延续，`O` 在外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（末尾）、`U-X`（单元），边界更清晰。 |
| CRF | 结构化分类器 | 建模标签间转移而不仅是发射，强制合法序列。 |
| 嵌套 NER | 重叠实体 | 一个跨度是不同于其子跨度的另一实体。BIO 无法表达。 |
| 实体级 F1 | 正确的 NER 指标 | 预测跨度必须与真实跨度完全一致。Token 级 F1 会夸大准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。经典必读。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了成为标准的 token 分类模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — 关于 `Doc.ents` 和 `Span` 各属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的评估指标库。始终使用它。
