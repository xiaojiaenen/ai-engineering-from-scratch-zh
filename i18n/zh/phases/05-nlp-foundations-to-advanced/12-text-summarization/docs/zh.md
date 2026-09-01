# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者想表达什么。不同的任务，不同的陷阱。

**类型：** Build
**语言：** Python
**前置要求：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**时间：** ~75 分钟

## 问题所在

一篇 2000 词的新闻文章出现在你的信息流中。你需要用 120 词将其概括出来。你可以直接从文章里挑选最重要的三句话（抽取式），也可以用自己的话重新组织内容（生成式）。两者都叫摘要，但它们是完全不同的问题。

抽取式摘要是一个排序问题。给每个句子打分，返回最高的 `k` 个。输出总是符合语法的，因为它是逐字提取的。风险在于可能会遗漏分散在文章各处的内容。

生成式摘要是一个生成问题。Transformer 模型会根据输入生成新文本。输出流畅且高度压缩，但可能会幻觉出源文中不存在的事实。风险在于自信地编造内容。

本课程将实现这两种方法，并揭示它们各自特有的故障模式。

## 核心概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，节点是句子，边是相似度。在图上运行 PageRank（或类似算法）来根据句子与其他句子的关联程度进行打分。得分最高的句子即为摘要。经典实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要配对数据上微调 transformer 编码器-解码器模型（如 BART、T5、Pegasus）。在推理时，模型读取文档，并通过交叉注意力机制逐 token 生成摘要。Pegasus 尤其采用了 gap-sentence 预训练目标，使其无需大量微调就能出色地完成摘要任务。

使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）进行评估。ROUGE-1 和 ROUGE-2 计算单词和二元组的重叠度。ROUGE-L 计算最长公共子序列。分数越高越好，但 40 分 ROUGE-L 算“良好”，50 分才算“优秀”。每篇论文都会报告这三个指标。请使用 `rouge-score` 包。

```figure
summarize-collapse
```

## 动手实现

### 步骤 1：TextRank（抽取式）

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

有两点值得说明。相似度函数使用了经过对数归一化的词重叠，这是原版 TextRank 的变体。使用 TF-IDF 向量的余弦相似度也可以。阻尼因子 0.85 和迭代次数均为 PageRank 的默认值。

### 步骤 2：基于 BART 的生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 是在 CNN/DailyMail 语料库上微调的。它开箱即用即可生成新闻风格的摘要。对于其他领域（学术论文、对话、法律文本），请使用相应的 Pegasus checkpoint 或在目标数据上进行微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

务必使用词干提取。否则，“running” 和 “run” 会被视为不同的词，导致 ROUGE 评分偏低。

### ROUGE 之外（2026 年摘要评估）

ROUGE 在过去二十年中一直是主导性的摘要评估指标，但在 2026 年已显得不足。针对 NLG 论文的大规模元分析显示：

- **BERTScore**（基于上下文嵌入的相似度）在 2023 年之前不断普及，如今已大多与 ROUGE 一同出现在摘要论文中。
- **BARTScore** 将评估视为生成任务：通过预训练 BART 给定源文本时对摘要的生成概率来打分。
- **MoverScore**（基于上下文嵌入的地面距离）在 2025 年的摘要基准测试中登顶，因为它比 ROUGE 更好地捕捉了语义重叠。
- **FactCC** 和 **基于 QA 的可信度** 在 2021-2023 年间很常见，如今常被 **G-Eval** 取代（一种使用思维链推理对连贯性、一致性、流畅性、相关性进行打分的 GPT-4 提示链）。
- 当评分标准设计得当时，**G-Eval** 及类似的 LLM 裁判方法与人主观判断的一致性可达约 80%。

生产环境建议：为保持与传统指标的可比性，报告 ROUGE-L 和 BERTScore（语义重叠），并使用 G-Eval 评估连贯性与事实准确性。同时需用 50-100 条人工标注的摘要进行校准。

### 步骤 4：事实准确性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险低得多，因为其输出是逐字从源文中提取的，尽管若源句被脱离上下文、过时或顺序错乱地引用，仍可能产生误导。这是生产环境系统在处理合规相关内容由时仍首选抽取式方法的根本原因。

需要指出的幻觉类型：

- **实体替换。** 原文写的是“John Smith”，摘要却写成“John Brown”。
- **数字漂移。** 原文是“25,000”，摘要却成了“2500万”。
- **极性反转。** 原文说“拒绝了该提议”，摘要却写成“接受了该提议”。
- **事实捏造。** 原文未提及 CEO，摘要却声称 CEO 已批准。

行之有效的评估方法：

- **FactCC。** 在源句与摘要句之间的蕴含关系上训练的二级分类器，用于预测内容是否属实。
- **基于 QA 的事实核查。** 向 QA 模型提出答案位于源文中的问题。若摘要支持不同的答案，则标记为可疑。
- **实体级 F1 分数。** 对比源文与摘要中的命名实体。仅出现在摘要中的实体值得怀疑。

对于任何涉及用户且对事实准确性有要求的应用（新闻、医疗、法律、金融），抽取式是更安全的选择。生成式方法必须在流程中集成事实核查环节。

## 实际应用

2026 年技术栈：

| 使用场景 | 推荐方案 |
|---------|-------------|
| 新闻、3-5 句摘要、英文 | `facebook/bart-large-cnn` |
| 学术论文 | `google/pegasus-pubmed` 或微调后的 T5 |
| 多文档、长文本 | 任意支持 32k+ 上下文的 LLM，配合提示词 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式、从设计上降低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

2026 年，当算力不受限时，长上下文 LLM 通常能击败专用模型。代价是成本与可复现性；专用模型能提供更稳定的输出。

## 交付使用

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择抽取式或生成式，指定库，执行事实核查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

根据给定任务（文档类型、合规要求、长度、算力预算），输出：

1. 方法。抽取式或生成式。用一句话说明理由。
2. 基础模型/库。明确指出。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或 LLM 提示词。
3. 评估方案。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干提取的 rouge-score）。若为生成式，需增加事实核查。
4. 一个待排查的故障模式。实体替换在生成式新闻摘要中最常见；需标记那些源文实体未出现在摘要中的样本。

拒绝在无事实核查门禁的情况下对医疗、法律、金融或受监管内容进行生成式摘要。对于超出模型上下文窗口的输入，标记为需要分块 Map-Reduce 摘要（而非简单截断）。
```

## 练习

1. **简单。** 对 5 篇新闻文章运行 TextRank。将前 3 个句子与参考摘要进行对比。测量 ROUGE-L 分数。在 CNN/DailyMail 风格的文章上，预期得分为 30-45。
2. **中等。** 实现实体级事实核查：从源文和摘要中提取命名实体（使用 spaCy），计算摘要对源文实体的召回率以及摘要实体对源文的精确率。高精确率与低召回率意味着安全但简略；低精确率则意味着存在幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上对比 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实准确性（通过实体 F1 衡量）以及每条摘要的成本。记录各自的优势场景。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式 | 挑选句子 | 逐字返回源文中的句子。不会产生幻觉。 |
| 生成式 | 改写 | 基于源文生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考摘要之间的 N-gram / LCS 重叠度。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上运行 PageRank。 |
| 事实准确性 | 是否正确 | 摘要中的主张是否有源文支持。 |
| 幻觉 | 捏造内容 | 摘要中存在但源文未支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) —— 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-Training](https://arxiv.org/abs/1910.13461) —— BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) —— Pegasus 及其 gap-sentence 预训练目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) —— ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) —— 事实准确性领域的奠基论文。
