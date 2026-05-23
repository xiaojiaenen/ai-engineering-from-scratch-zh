# 自然语言推理 — 文本蕴含

> "t 蕴含 h" 意味着阅读 t 的人会得出结论 h 为真。NLI 的任务是预测蕴含 / 矛盾 / 中立。表面枯燥，实则在生产环境中不可或缺。

**类型:** 学习
**语言:** Python
**先决条件:** 第5阶段 · 05 (情感分析), 第5阶段 · 13 (问答)
**时间:** 约60分钟

## 问题描述

你构建了一个摘要生成器。它生成了一份摘要。你如何知道这份摘要没有包含幻觉？

你构建了一个聊天机器人。它回答了"是"。你如何知道这个答案得到了检索到的段落的支持？

你需要按主题分类10,000篇新闻文章。你没有训练标签。你能复用一个模型吗？

这三个问题都归结为自然语言推理（NLI）。NLI 问的是：给定一个前提 `t` 和一个假设 `h`，`h` 是被 `t` 蕴含、矛盾，还是中立（无关）？

- **幻觉检查：** `t` = 源文档，`h` = 摘要声明。非蕴含 = 幻觉。
- **有据可查的问答：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 表述化标签 ("这是关于体育的")。蕴含 = 预测的标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架底层都内置了一个 NLI 模型。

## 核心概念

![NLI: 三分类，前提 vs 假设](../assets/nli.svg)

**三种标签。**

- **蕴含。** `t` → `h`. "猫在垫子上" 蕴含 "有一只猫。"
- **矛盾。** `t` → ¬`h`. "猫在垫子上" 矛盾于 "没有猫。"
- **中立。** 两个方向都无法推断。"猫在垫子上" 对于 "猫饿了" 是中立的。

**不是逻辑蕴含。** NLI 是 *自然语言* 推理 — 一个典型的人类读者会做出的推断，而非严格的逻辑。"约翰遛狗" 在 NLI 中蕴含 "约翰有一条狗"，但严格的一阶逻辑只有在你将占有关系公理化时才接受这一点。

**数据集。**

- **SNLI** (2015). 57万个人工标注的句对，以图像描述作为前提。领域狭窄。
- **MultiNLI** (2017). 跨越10个语体的43.3万个句对。2026年的标准训练语料库。
- **ANLI** (2019). 对抗性 NLI。人类特意编写了旨在击溃现有模型的示例。难度更高。
- **DocNLI, ConTRoL** (2020–21). 文档长度的前提。测试多跳和长程推理。

**架构。** 一个 Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示输入一个三路 softmax。在 MNLI 上训练，在留出基准上评估，在分布内的句对上可获得 90%+ 的准确率。

**通过 NLI 实现零样本分类。** 给定一个文档和候选标签，将每个标签转换成一个假设 ("这段文本是关于体育的")。计算每个假设的蕴含概率。选择概率最大的。这是 Hugging Face 的 `zero-shot-classification` 管道背后的机制。

## 动手构建

### 步骤1：运行一个预训练的 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产环境的 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是默认的开放选择。DeBERTa-v3 在排行榜上名列前茅。

### 步骤2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}。"。可以通过 `hypothesis_template` 进行自定义。无需训练数据。无需微调。开箱即用。

### 步骤3：RAG 的忠实性检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实性的核心。将生成的答案拆分为原子声明。根据检索到的上下文检查每个声明。报告蕴含声明的比例。

### 步骤4：手工构建的 NLI 分类器（概念性）

参见 `code/main.py` 获取一个仅使用标准库实现的玩具模型：前提和假设通过词汇重叠 + 否定检测进行比较。其性能无法与 Transformer 模型竞争 — 但它展示了任务的基本形式：输入两段文本，输出三分类标签，损失 = 在 `{entail, contradict, neutral}` 上的交叉熵。

## 常见陷阱

- **仅基于假设的捷径。** 模型仅凭假设就能以约60%的准确率预测 SNLI 的标签，因为 "不"、"没人"、"从不" 与矛盾标签相关。这是检测标签泄露的强力基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI，但在 HANS/ANLI 上失败。请使用对抗性基准测试。
- **文档长度退化。** 单句 NLI 模型在文档长度前提上 F1 值下降20+。对于长上下文，请使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可导致准确率波动10+个百分点。请调整模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学和科学文本需要特定领域的 NLI 模型（例如，SciNLI, MedNLI）。

## 应用场景

2026年的技术栈：

| 使用场景 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘计算 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类 (轻量级) | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026年的元模式：NLI 是文本理解的胶带。每当你需要判断"A 支持 B 吗？"或"A 与 B 矛盾吗？"时 — 在调用另一个 LLM 之前，先试试 NLI。

## 部署上线

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习题

1. **简单。** 在20个手工构建的（前提、假设、标签）三元组上运行 `facebook/bart-large-mnli`，覆盖所有三个类别。测量准确率。加入对抗性的 "子序列启发式" 陷阱 ("我没有吃蛋糕" vs "我吃了蛋糕")，看看模型是否会失败。
2. **中等。** 在100条 AG News 标题上，比较零样本模板 `"This text is about {label}"` 与 `"The topic is {label}"` 和 `"{label}"`。报告准确率波动。
3. **困难。** 构建一个 RAG 忠实性检查器：原子声明分解 + 对每个声明进行 NLI。在50个带有黄金上下文的 RAG 生成答案上进行评估。与人工标签相比，测量假阳性和假阴性率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理 | 对前提-假设关系的三分类任务。 |
| RTE | 识别文本蕴含 | NLI 的旧名称；任务相同。 |
| 蕴含 | "t 意味着 h" | 典型读者在给定 t 的情况下会得出 h 为真。 |
| 矛盾 | "t 排除了 h" | 典型读者在给定 t 的情况下会得出 h 为假。 |
| 中立 | "未决定" | 从 t 到 h 没有任何方向的推断。 |
| 零样本分类 | NLI 作为分类器 | 将标签表述为假设，选择最大蕴含概率。 |
| 忠实性 | 答案是否有依据？ | 对（检索到的上下文，生成的答案）进行 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). 用于学习自然语言推理的大型标注语料库](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). 用于通过推理进行句子理解的广覆盖挑战语料库](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). 对抗性自然语言推理](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). 零样本文本分类基准测试](https://arxiv.org/abs/1909.00161) — NLI 作为分类器。
- [He et al. (2021). DeBERTa: 解码增强、注意力解耦的 BERT](https://arxiv.org/abs/2006.03654) — 2026年的 NLI 主力模型。