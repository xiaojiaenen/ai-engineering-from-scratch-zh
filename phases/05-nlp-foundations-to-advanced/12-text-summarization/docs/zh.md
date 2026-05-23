# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者想表达什么。这是不同的任务，有不同的陷阱。

**类型：** 构建
**语言：** Python
**先修课程：** 阶段5 · 02（词袋 + TF-IDF），阶段5 · 11（机器翻译）
**时间：** 约75分钟

## 问题描述

一篇2000字的新闻文章出现在你的信息流中。你需要用120个字概括它。你可以从文章中挑选三个最重要的句子（抽取式），或者用自己的话改写内容（生成式）。两者都被称为摘要。它们是完全不同的问题。

抽取式摘要本质上是一个排序问题。为每个句子打分，返回得分最高的`k`个句子。输出总是符合语法的，因为它是原文照搬。风险在于可能遗漏分布在文章各处的内容。

生成式摘要是一个生成问题。Transformer模型基于输入文本生成新的文本。输出流畅且能压缩内容，但可能会凭空捏造源文本中不存在的事实。风险在于自信地编造信息。

本课将同时构建这两种方法，并介绍各自固有的失败模式。

## 核心概念

![抽取式TextRank vs 生成式Transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，其中节点是句子，边是相似度。在图上运行PageRank（或类似算法），根据句子与其他所有句子的连接程度为其打分。得分最高的句子构成摘要。其经典实现是 **TextRank**（Mihalcea and Tarau, 2004）。

**生成式。** 在文档-摘要对上微调一个Transformer编码器-解码器模型（如BART、T5、Pegasus）。在推理阶段，模型读取文档，并通过交叉注意力机制逐个token地生成摘要。Pegasus尤其使用了一种间隙句子预训练目标，使其无需过多微调就能擅长摘要任务。

评估使用 **ROUGE**（面向召回的摘要评估）。ROUGE-1和ROUGE-2分别评估单词和二元组的重叠度。ROUGE-L评估最长公共子序列。分数越高越好，但ROUGE-L达到40算"良好"，50算"卓越"。每篇论文都会报告这三个指标。使用 `rouge-score` 包。

## 动手构建

### 步骤1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两个要点值得说明。相似度函数使用对数归一化的词重叠度，这是TextRank原始版本的方法。使用TF-IDF向量的余弦相似度也可以。阻尼系数0.85和迭代次数是PageRank的默认值。

### 步骤2：使用BART进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN已在CNN/DailyMail语料库上微调。它开箱即用地生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），请使用相应的Pegasus检查点或在您的目标数据上进行微调。

### 步骤3：ROUGE评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

务必使用词干提取。如果不使用，"running"和"run"会被算作不同的词，导致ROUGE得分偏低。

### 超越ROUGE（2026年的摘要评估）

ROUGE作为主导性的摘要指标已有二十年，但到2026年，仅靠它本身是不足的。一项对NLG论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）在2023年前逐步获得认可，现在大部分摘要论文都会同时报告它和ROUGE。
- **BARTScore** 将评估视为生成任务：根据源文本，评估一个预训练的BART模型分配给该摘要的概率。
- **MoverScore**（基于上下文嵌入的推土机距离）在2025年的摘要基准测试中跃居榜首，因为它比ROUGE更能捕捉语义重叠。
- **FactCC** 和 **基于问答的忠实度评估** 在2021-2023年间很常见，现在常被 **G-Eval** 所取代（这是一个GPT-4提示链，通过思维链推理对连贯性、一致性、流畅性、相关性进行评分）。
- **G-Eval** 及类似的LLM评判方法，在评估标准设计良好时，与人类判断的吻合度约为80%。

生产建议：报告ROUGE-L用于传统比较，BERTScore用于语义重叠，G-Eval用于连贯性和事实性。根据50-100份人工标注的摘要进行校准。

### 步骤4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为输出是原文照搬，尽管如果源句被断章取义、过时或引用顺序错误，仍然可能误导读者。这是生产系统在处理合规性相关内容时，仍然偏好抽取式方法的唯一最大原因。

需要明确的幻觉类型：

- **实体替换。** 源文是"张三"，摘要是"李四"。
- **数值漂移。** 源文是"25,000"，摘要是"2500万"。
- **极性反转。** 源文是"拒绝了报价"，摘要是"接受了报价"。
- **事实捏造。** 源文未提及CEO。摘要称CEO批准了。

有效的评估方法：

- **FactCC。** 一个在源句和摘要句之间的蕴涵关系上训练的二元分类器。预测为事实/非事实。
- **基于问答的事实性。** 询问QA模型答案存在于源文中的问题。如果摘要支持不同的答案，则标记出来。
- **实体级F1。** 比较源文与摘要中的命名实体。仅存在于摘要中的实体是可疑的。

对于任何面向用户且事实性至关重要的场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式摘要需要在流程中加入事实性检查。

## 应用指南

2026年的技术栈：

| 用例 | 推荐方案 |
|------|----------|
| 新闻，3-5句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优后的T5 |
| 多文档，长文本 | 任何具有32k+上下文的LLM，通过提示实现 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，因设计而具有低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

当计算资源不受限制时，具有长上下文的LLM在2026年通常能超越专门模型。代价是成本和可复现性；专门模型能提供更一致的输出。

## 部署上线

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1.  **简单。** 在5篇新闻文章上运行TextRank。将前3句与参考摘要进行比较。测量ROUGE-L。在CNN/DailyMail风格的文章上，你应该看到30-45的ROUGE-L。
2.  **中等。** 实现实体级事实性检查：从源文和摘要中提取命名实体（使用spaCy），计算源实体在摘要中的召回率，以及摘要实体相对于源文的精确率。高精确率和低召回率意味着安全但信息不足；低精确率意味着存在幻觉实体。
3.  **困难。** 在50篇CNN/DailyMail文章上，比较BART-large-CNN与一个LLM（Claude或GPT-4）。报告ROUGE-L、事实性（通过实体F1）以及每份摘要的成本。记录各自胜出的场景。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|----------|------------|
| 抽取式（Extractive） | 挑选句子 | 从源文中原文返回句子。永不产生幻觉。 |
| 生成式（Abstractive） | 改写 | 基于源文生成新的文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考文本之间的n-gram/最长公共子序列重叠度。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上运行PageRank。 |
| 事实性（Factuality） | 是否正确 | 摘要的断言是否得到源文支持。 |
| 幻觉（Hallucination） | 编造的内容 | 摘要中存在但源文不支持的内容。 |

## 扩展阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus与间隙句子目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 关于事实性的综述论文。