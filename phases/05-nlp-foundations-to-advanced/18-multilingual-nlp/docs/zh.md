# 多语言自然语言处理

> 一个模型，支持100多种语言，且大多数语言无需训练数据。跨语言迁移是2020年代的实用奇迹。

**类型：** 学习  
**编程语言：** Python  
**先修知识：** 第5阶段·04（GloVe、FastText、子词分割），第5阶段·11（机器翻译）  
**所需时间：** 约45分钟

## 问题所在

英语拥有数十亿标注样本。乌尔都语仅有数千。迈蒂利语几乎没有。任何服务于全球用户的实用自然语言处理系统，都必须处理那些缺乏特定任务训练数据的“长尾”语言。

多语言模型通过同时训练多种语言来解决这个问题。共享表示使得模型能够将在高资源语言中学到的技能迁移到低资源语言。在英语情感分析数据上微调模型后，它便能直接在乌尔都语上产生出人意料地准确的情感预测。这就是零样本跨语言迁移，它重塑了自然语言处理技术走向世界的方式。

本课将说明其中的权衡、经典模型，以及阻碍多语言工作新手团队的那个关键决策：选择用于迁移的源语言。

## 核心概念

![通过共享的多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用基于所有目标语言文本训练的SentencePiece或WordPiece分词器。词汇表是共享的：同一个子词单元在相似的语言中表示相同的词素。英语和意大利语中的 `anti-` 会得到相同的token。

**共享表示。** 在多种语言上进行掩码语言建模预训练的Transformer模型，能学到不同语言中语义相似的句子会产生相似的隐藏状态。mBERT、XLM-R和NLLB都展现了这一点。英语中“cat”的嵌入会靠近法语的“chat”和西班牙语的“gato”，完整的句子嵌入也是如此。

**零样本迁移。** 在一种语言（通常是英语）的标注数据上微调模型。推理时，将其应用于模型支持的任何其他语言。无需目标语言的标签。对于类型学上相近的语言，效果很好；对于差异较大的语言，效果会减弱。

**少样本微调。** 添加100-500个目标语言的标注样本。在分类任务上，准确率能跃升至英语基线的95-98%。这是多语言自然语言处理中性价比最高的杠杆。

## 代表模型

| 模型 | 年份 | 覆盖语言数 | 备注 |
|-------|------|------------|-------|
| mBERT | 2018 | 104种语言 | 在维基百科上训练。首个实用的多语言语言模型。在低资源语言上表现较弱。 |
| XLM-R | 2019 | 100种语言 | 在CommonCrawl上训练（数据量远大于维基百科）。设定了跨语言基准。基础版2.7亿参数，大型版5.5亿参数。 |
| XLM-V | 2023 | 100种语言 | 拥有100万token词汇表的XLM-R（对比25万）。在低资源语言上表现更好。 |
| mT5 | 2020 | 101种语言 | 用于多语言生成的T5架构。 |
| NLLB-200 | 2022 | 200种语言 | Meta的翻译模型；包含55种低资源语言。 |
| BLOOM | 2022 | 46种语言 + 13种编程语言 | 多语言训练的开放1760亿参数大语言模型。 |
| Aya-23 | 2024 | 23种语言 | Cohere的多语言大语言模型。在阿拉伯语、印地语、斯瓦希里语上表现强劲。 |

根据用例选择。分类任务使用XLM-R-base作为稳妥的默认选择效果良好。生成任务则需要mT5或NLLB，取决于侧重翻译还是开放式生成。大语言模型风格的工作可与Aya-23或Claude配合使用，通过明确的多语言提示实现。

## 源语言决策（2026年研究）

大多数团队默认将英语作为微调源语言。最近的研究（2026年）表明这通常是错误的。

语言相似性比原始语料库大小更能预测迁移质量。对于斯拉夫语系目标语言，德语或俄语常常优于英语。对于印度语系目标语言，印地语常常优于英语。**qWALS** 相似性度量（2026年，基于世界语言结构地图集特征）对此进行了量化。**LANGRANK**（Lin等人，ACL 2019）是另一种更早的方法，它通过结合语言相似性、语料库大小和谱系关系来对候选源语言进行排序。

实用规则：如果你的目标语言在类型学上有一个相近的高资源亲属语言，尝试先用该语言进行微调，然后与英语微调的结果进行比较。

## 动手实践

### 步骤一：零样本跨语言分类

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

一个模型，三种语言，相同的API。在自然语言推理数据上训练的XLM-R通过蕴含关系技巧能很好地迁移到分类任务。

### 步骤二：多语言嵌入空间

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

翻译后的句子在嵌入空间中位置接近。一个不同的英语句子则距离较远。这使得跨语言检索、聚类和相似性计算成为可能。

### 步骤三：少样本微调策略

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

对于100-500个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认设置。学习率过高会导致多语言对齐崩溃，模型退化为仅适用于英语的模型。

## 有效的评估方法

- **按语言在预留测试集上评估准确率。** 不要聚合。聚合会掩盖长尾语言的问题。
- **与单语基线对比。** 对于数据足够的语言，有时从头训练的单语模型会优于多语言模型。需要进行测试。
- **实体级测试。** 测试目标语言中的命名实体。多语言模型对于远离拉丁字母的文字系统，分词效果通常较弱。
- **跨语言一致性。** 两种语言中相同含义的句子应产生相同的预测结果。测量其中的差距。

## 实际应用

2026年技术栈：

| 任务 | 推荐方案 |
|-----|-------------|
| 分类，100种语言 | 微调后的XLM-R-base（约2.7亿参数） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200种语言 | `facebook/nllb-200-distilled-600M`（参见第11课） |
| 生成式多语言任务 | Claude, GPT-4, Aya-23, mT5-XXL |
| 低资源语言自然语言处理 | XLM-V 或在相关的高资源语言上进行领域特定微调 |

如果性能很重要，始终要为目标语言的微调预留预算。零样本是一个起点，而非最终答案。

### 分词器税（低资源语言出了什么问题）

多语言模型在所有语言中共享一个分词器。该词汇表是在由英语、法语、西班牙语、中文、德语主导的语料库上训练的。对于主导集合之外的任何语言，三种“税”会悄无声息地叠加：

- **生育率税。** 低资源语言的文本每单词分词后得到的token数量远多于英语。一个印地语句子可能需要相当于英语句子3-5倍的token。这3-5倍的差异会消耗你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每一个拼写错误、变音符号变体、Unicode归一化不匹配或大小写变化，在嵌入空间中都变成一个冷启动的不相关序列。模型无法学习母语使用者视为理所当然的正字法对应关系。
- **容量溢出税。** 第1和第2种税消耗了上下文位置、层深度和嵌入维度。留给实际推理的部分，系统性地少于高资源语言从同一模型中获得的部分。

实际症状：你的模型在印地语上训练正常，损失曲线看起来正确，评估困惑度看起来合理，但生产环境的输出微妙地出错。词法形态在句子中途崩溃。罕见的词形变化无法恢复。**你无法通过增加数据量来修复一个有缺陷的分词器。**

缓解方法：为你的目标语言选择一个覆盖良好的分词器（XLM-V的100万token词汇表是一个直接的解决方案）；在训练前，在预留的目标文本上验证分词生育率；对于真正的长尾文字系统，使用字节级回退（SentencePiece `byte_fallback=True`，GPT-2风格的字节级BPE），确保永远不会出现词汇表外的词。

## 部署

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语上，每种语言运行10个句子的零样本分类流程。报告每种语言的准确率。你应该会看到法语表现强劲，印地语不错，阿拉伯语表现不一。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 构建一个跨语言检索器，用于一个小型混合语言语料库。用英语查询，检索任意语言的文档。测量recall@5。
3. **困难。** 比较英语源和印地语源的微调，用于印地语分类任务。在两种方案下，都使用500个目标语言样本进行少样本微调。报告哪种源语言产生了更好的印地语准确率，以及具体提高了多少。这是LANGRANK论文的一个小型复现。

## 关键术语

| 术语 | 人们常说的 | 其实际含义 |
|------|-----------------|-----------------------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词汇表和参数。 |
| 跨语言迁移 | 一种语言训练，另一种语言运行 | 在源语言上微调，在目标语言上评估，无需目标语言标签。 |
| 零样本 | 无需目标语言标签 | 不在目标语言上微调的迁移。 |
| 少样本 | 少量目标语言标签 | 用于微调的100-500个目标语言样本。 |
| mBERT | 首个多语言语言模型 | 在维基百科上预训练的104语言BERT模型。 |
| XLM-R | 标准的跨语言基线 | 在CommonCrawl上预训练的100语言RoBERTa模型。 |
| NLLB | Meta的200语言机器翻译模型 | “不让任何语言掉队”。包含55种低资源语言。 |

## 拓展阅读

- [Conneau 等人 (2019). 《大规模无监督跨语言表示学习》](https://arxiv.org/abs/1911.02116) —— XLM-R论文。
- [Pires, Schlinger, Garrette (2019). 《多语言BERT到底有多“多语言”？》](https://arxiv.org/abs/1906.01502) —— 开启跨语言迁移研究线的分析论文。
- [Costa-jussà 等人 (2022). 《不让任何语言掉队》](https://arxiv.org/abs/2207.04672) —— NLLB-200论文。
- [Üstün 等人 (2024). 《Aya模型：一个经过指令微调的开放获取多语言语言模型》](https://arxiv.org/abs/2402.07827) —— Aya，Cohere的多语言大语言模型。
- [《语言相似性预测跨语言迁移学习性能 (2026)》](https://www.mdpi.com/2504-4990/8/3/65) —— 关于qWALS / LANGRANK源语言的论文。