# 多语言NLP

> 一个模型，100+种语言，大多数语言零训练数据。跨语言迁移是2020年代的实用奇迹。

**类型：** 学习
**语言：** Python
**前置要求：** 第5阶段·04（GloVe、FastText、子词），第5阶段·11（机器翻译）
**时间：** 约45分钟

## 问题所在

英语有数十亿标注样本。乌尔都语只有数千。迈蒂利语几乎为零。任何服务全球受众的实用NLP系统都必须能在长尾语言上工作，而这些语言中并不存在任务特定的训练数据。

多语言模型通过在多种语言上同时训练一个模型来解决这个问题。共享表示让模型能够将高资源语言中学到的技能迁移到低资源语言。在英语情感分析上对模型进行微调，它就能开箱即用地对乌尔都语产生出人意料的出色情感预测。这就是零样本跨语言迁移，它重塑了NLP面向全球的交付方式。

本课讲述权衡点、经典模型，以及新手团队常踩坑的那个关键决策：选择迁移的源语言。

## 核心概念

![通过共享多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用 SentencePiece 或 WordPiece 分词器，在所有目标语言的文本上训练。词汇表是共享的：同一个子词单元在不同相关语言中表示相同的语素。英语和意大利语中的 `anti-` 获得同一个 token。

**共享表示。** 在多种语言上用掩码语言建模预训练的 Transformer 学到：不同语言中语义相近的句子会产生相近的隐藏状态。mBERT、XLM-R 和 NLLB 都体现这一点。英语中 "cat" 的嵌入与法语中 "chat"、西班牙语中 "gato" 聚在一起，整句嵌入也是如此。

**零样本迁移。** 在一种语言（通常英语）的标注数据上微调模型。在推理时，用它处理模型支持的任意其他语言。无需目标语言标签。对类型学相近的语言效果强劲，对相距较远的语言则较弱。

**少样本微调。** 添加 100-500 个目标语言标注样本。分类任务准确率可跃升至英语基线的 95-98%。这是多语言NLP中性价比最高的杠杆。

## 模型

| 模型 | 年份 | 覆盖 | 说明 |
|------|------|------|-----|
| mBERT | 2018 | 104种语言 | 在维基百科上训练。首个实用的多语言LM。低资源表现弱。 |
| XLM-R | 2019 | 100种语言 | 在 CommonCrawl 上训练（规模远大于维基百科）。确立跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100种语言 | XLM-R 配 1M token 词汇表（原为 250k）。低资源表现更好。 |
| mT5 | 2020 | 101种语言 | 用于多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200种语言 | Meta 的翻译模型；含 55 种低资源语言。 |
| BLOOM | 2022 | 46种语言 + 13种编程语言 | 开源 176B 多语言LLM。 |
| Aya-23 | 2024 | 23种语言 | Cohere 的多语言LLM。阿拉伯语、印地语、斯瓦希里语表现强劲。 |

按用例选择。分类任务用 XLM-R-base 作为合理默认即可。生成任务视翻译 vs 开放生成选用 mT5 或 NLLB。LLM 风格工作搭配 Aya-23 或使用显式多语言提示的 Claude。

## 源语言决策（2026研究）

大多数团队默认选用英语作为微调源。2026年的最新研究表明这往往并不正确。

语言相似性比原始语料库规模更能预测迁移质量。对于斯拉夫语目标语言，德语或俄语往往胜过英语。对于印度语目标语言，印地语往往胜过英语。**qWALS** 相似性度量（2026，基于《世界语言结构图谱》特征）可对此量化。**LANGRANK**（Lin等，ACL 2019）是另一种更早的方法，通过语言相似性、语料库规模和谱系亲缘度的组合对候选源语言排序。

实用规则：若目标语言存在类型学相近的高资源语言，先尝试用该语言微调，再与英语微调对比。

```figure
n5-crosslingual-bridge
```

## 动手实践

### 步骤1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一套 API。XLM-R 在 NLI 数据上训练，通过蕴含技巧迁移到分类效果良好。

### 步骤2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译文本在嵌入空间中彼此靠近。不同英语句子则距离更远。这正是跨语言检索、聚类和相似度任务能工作的原因。

### 步骤3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是稳妥默认值。过高的学习率会导致多语言对齐坍塌，模型退化为仅支持英语。

## 真正有用的评估

- **按语言划分的 held-out 集准确率。** 不要聚合。聚合会掩盖长尾语言问题。
- **与单语言基线对比。** 对数据充足的语言，从零训练的单语言模型有时反而优于多语言模型。需要实测验证。
- **实体级测试。** 目标语言命名实体。多语言模型对远离拉丁字母的文字系统分词往往较弱。
- **跨语言一致性。** 同义的两语言输入应产生相同预测。衡量差距。

## 投入使用

2026年技术栈：

| 任务 | 推荐方案 |
|-----|---------|
| 分类，100种语言 | 微调后的 XLM-R-base（~270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200种语言 | `facebook/nllb-200-distilled-600M`（见课程11） |
| 多语言生成 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言NLP | XLM-V 或在相关高资源语言上做领域微调 |

如果性能至关重要，务必为目标的源语言微调预留预算。零样本只是起点，而非终局。

### 分词税（低资源语言的常见问题）

多语言模型在所有语言间共享一个分词器。该词汇表在训练语料中以英语、法语、西班牙语、中文、德语为主导。对于任何非主导语言，三种税叠加悄无声息地发生：

- **丰度税。** 低资源语言文本每个词 tokenize 出来的 token 数远多于英语。一句印地语可能需要等价英语句子 3-5 倍的 token。这 3-5 倍消耗你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、重音变体、Unicode 规范化不匹配、大小写差异，在嵌入空间中都成为冷启动无关序列。模型无法学习到母语者视为理所当然的正字法对应关系。
- **容量溢出税。** 前两种税消耗上下文位置、层深度和嵌入维度。留给实际推理的容量比高资源语言从同一模型获得的容量系统性地更小。

实际症状：模型在印地语上训练正常，loss 曲线看起来正确，eval 困惑度尚可，但生产输出微妙地出错。形态学在句中崩溃，罕见屈折变化不可恢复。**修复坏掉的 tokenizer 无法靠数据规模弥补。**

缓解措施：为目标的源语言挑选覆盖良好的分词器（XLM-V 的 1M token 词汇表是直接解法）；训练前在 held-out 目标文本上验证分词丰度；对真正长尾的文字系统使用 byte-level 兜底（SentencePiece 的 `byte_fallback=True`，或 GPT-2 风格的 byte-level BPE），确保没有 OOV。

## 交付

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: 为多语言NLP任务选择源语言、目标模型和评估方案。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

根据需求（目标语言、任务类型、各语言可用标注数据），输出：

1. 微调源语言。默认英语；若目标语言存在类型学相近的高资源语言，检查 LANGRANK 或 qWALS。
2. 基座模型。XLM-R（分类）、mT5（生成）、NLLB（翻译）、Aya-23（生成式LLM）。
3. 少样本预算。如有可用，从100-500条目标语言样本起步。仅当标注不可行时使用零样本。
4. 评估方案。按语言报告准确率（而非聚合），跨语言一致性，非拉丁文字系统的实体级F1。

拒绝交付未做按语言评估的多语言模型——聚合指标会掩盖长尾失败。标记分词覆盖度低的文字系统（阿姆哈拉语、提格里尼亚语、众多非洲语言）需使用带 byte-fallback 的模型（SentencePiece 配 `byte_fallback=True`，或如 GPT-2 般的 byte-level 分词器）。
```

## 练习

1. **简单。** 在英语、法语、印地语、阿拉伯语各10句上运行零样本分类流水线，报告每种语言的准确率。预期法语强劲、印地语尚可、阿拉伯语波动。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在小规模混合语言语料上构建跨语言检索器。用英语查询，检索任意语言的文档。测量 recall@5。
3. **困难。** 对比英语源和印地语源微调在印地语分类任务上的效果。两种设置下均用 500 条目标语言样本做少样本微调。报告哪种源语言产出更高的印地语准确率以及差距幅度。这是 LANGRANK 论点的微缩复现。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------|---------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词汇表和参数。 |
| 跨语言迁移 | 一种语言训练，另一种语言推理 | 在源语言上微调，在无目标语言标签的情况下在目标语言上评估。 |
| 零样本 | 无目标语言标签 | 不在目标语言上微调即可迁移。 |
| 少样本 | 少量目标标签 | 用 100-500 条目标语言样本微调。 |
| mBERT | 首个多语言LM | 在维基百科上预训练的 104 语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言MT | No Language Left Behind。含 55 种低资源语言。 |

## 延伸阅读

- [Conneau 等（2019）。大规模无监督跨语言表示学习](https://arxiv.org/abs/1911.02116) —— XLM-R 论文。
- [Pires, Schlinger, Garrette（2019）。多语言BERT到底多跨语言？](https://arxiv.org/abs/1906.01502) —— 引发跨语言迁移研究线的分析论文。
- [Costa-jussà 等（2022）。没有语言被落下](https://arxiv.org/abs/2207.04672) —— NLLB-200 论文。
- [Üstün 等（2024）。Aya 模型：指令微调的开放获取多语言语言模型](https://arxiv.org/abs/2402.07827) —— Aya，Cohere 的多语言LLM。
- [语言相似性预测跨语言迁移学习性能（2026）](https://www.mdpi.com/2504-4990/8/3/65) —— qWALS / LANGRANK 源语言论文。
