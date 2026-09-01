# 自然语言推理（NLI）——文本蕴含

> “t 蕴含 h”意味着，阅读 t 的人类会得出 h 为真的结论。NLI 的任务是预测蕴含/矛盾/中立关系。表面上看很枯燥，但在生产环境中却是核心基础设施。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 05（情感分析），Phase 5 · 13（问答）
**时间：** 约 60 分钟

## 问题背景

你构建了一个摘要生成器，它产出了一段摘要。你怎么知道这段摘要没有包含幻觉内容？

你构建了一个聊天机器人，它回答了“是”。你怎么知道这个答案得到了检索段落的支持？

你需要按主题对 10,000 篇新闻文章进行分类。你没有训练标签。能复用模型吗？

这三个问题最终都归结为自然语言推理（NLI）。NLI 的问题是：给定前提 `t` 和假设 `h`，`h` 是被 `t` 蕴含、矛盾，还是中立（无关）？

- **幻觉检查：** `t` = 源文档，`h` = 摘要中的声明。非蕴含 = 幻觉。
- **有依据的 QA：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 言语化的标签（如“这是一篇关于体育的文章”）。蕴含 = 预测标签。

同一个任务，三种生产用途。这就是为什么每个 RAG 评估框架底层都内置了 NLI 模型。

## 核心概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **蕴含（Entailment）。** `t` → `h`。“The cat is on the mat”蕴含“There is a cat。”
- **矛盾（Contradiction）。** `t` → ¬`h`。“The cat is on the mat”与“There is no cat.”矛盾。
- **中立（Neutral）。** 双方均无法推断。“The cat is on the mat”对“The cat is hungry.”是中立的。

**并非逻辑蕴含。** NLI 是*自然*语言推理——即典型人类读者会做出的推断，而非严格的形式逻辑。“John walked his dog”在 NLI 中蕴含“John has a dog”，但严格的谓词逻辑仅在你对“拥有”进行公理化后才能成立。

**数据集。**

- **SNLI**（2015）。57 万个人工标注的对，以图像字幕作为前提。领域较窄。
- **MultiNLI**（2017）。跨 10 种文体的 43.3 万对。2026 年的标准训练语料。
- **ANLI**（2019）。对抗性 NLI。人类专门编写了旨在击垮现有模型的示例。难度更高。
- **DocNLI、ConTRoL**（2020–21）。文档长度的前提。测试多跳推理和长距离推理。

**架构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表征输入一个三路 softmax。在 MNLI 上训练，在保留基准上评估，在分布内对上可获得 90%+ 的准确率。

**基于 NLI 的零样本分类。** 给定文档和候选标签，将每个标签转化为假设（如“This text is about sports”）。计算每个假设的蕴含概率，取最大值。这是 Hugging Face `zero-shot-classification` pipeline 的底层机制。

```figure
nli-router
```

## 动手实践

### 步骤 1：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 返回所有标签；替代已弃用的 return_all_scores=True

premise = "猫正在沙发上睡觉。"
hypothesis = "房间里有一只猫。"

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

生产环境中的 NLI 默认选择 `facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli`。DeBERTa-v3 在各大排行榜上位居榜首。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "中央银行降息后，股市强劲反弹。"
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板为“This example is about {label}.”。可通过 `hypothesis_template` 自定义。无需训练数据，无需微调，开箱即用。

### 步骤 3：RAG 的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度指标的核心。将生成的答案拆分为原子声明，逐一与检索到的上下文进行比对，并统计蕴含的比例。

### 步骤 4：手工实现的 NLI 分类器（概念性）

参考 `code/main.py` 中的仅用标准库的玩具示例：通过词法重叠 + 否定检测来比较前提和假设。竞争力不及 Transformer 模型，但它展示了任务的基本形态：输入两篇文本，输出三路标签，损失函数为 `{entail, contradict, neutral}` 上的交叉熵。

## 常见陷阱

- **仅依赖假设的捷径。** 模型仅凭假设就能在 SNLI 上达到约 60% 的准确率，因为“not”、“nobody”、“never”等词与矛盾标签相关。这为检测标签泄露提供了强有力的基线。
- **词法重叠启发式规则。** 子序列启发式规则（“每个子序列都是蕴含”）能通过 SNLI，但在 HANS/ANLI 上失败。请使用对抗性基准测试。
- **文档长度衰减。** 单句 NLI 模型在文档长度前提上的 F1 分数会下降 20 以上。长上下文请使用经 DocNLI 训练的模型。
- **零样本模板敏感性。** “This example is about {label}”、“{label}”和“The topic is {label}”三种写法可导致准确率波动 10 个百分点以上。需对模板进行调优。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医疗和科学文本需要领域特定的 NLI 模型（如 SciNLI、MedNLI）。

## 生产选型

2026 技术栈：

| 用途 | 模型 |
|------|------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速/边缘部署 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量级） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的元模式：NLI 是文本理解的万能胶。每当你需要判断“A 是否支持 B？”或“A 是否与 B 矛盾？”时，优先调用 NLI，而不是再发一次 LLM 请求。

## 交付配置

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

1. **简单。** 使用 `facebook/bart-large-mnli` 处理 20 条手工构造的（前提、假设、标签）三元组，覆盖全部三类。测量准确率。加入对抗性的“子序列启发式”陷阱（如“I did not eat the cake”与“I ate the cake”对比），观察模型是否会失效。
2. **中等。** 在 100 条 AG News 标题上，对比零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`。报告准确率的波动幅度。
3. **困难。** 构建 RAG 忠实度检查器：原子声明分解 + 逐条 NLI 判断。在 50 条带有金标准上下文的 RAG 生成答案上评估。与人工标注对比，统计假阳性和假阴性率。

## 核心术语

| 术语 | 通常的说法 | 实际含义 |
|------|------------|----------|
| NLI | 自然语言推理 | 对前提-假设关系的三分类。 |
| RTE | 文本蕴含识别 | NLI 的旧称，任务相同。 |
| Entailment | “t 蕴含 h” | 典型读者在给定 t 后会认为 h 为真。 |
| Contradiction | “t 排除 h” | 典型读者在给定 t 后会认为 h 为假。 |
| Neutral | “未定” | 从 t 无法向任一方向推断出 h。 |
| Zero-shot classification | 将 NLI 用作分类器 | 将标签言语化为假设，取蕴含概率最大者。 |
| Faithfulness | 答案是否得到支持？ | 对（检索上下文，生成答案）执行 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI 即分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 的主力模型。
