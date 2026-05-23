# 指代消解

> "她给他打了电话。他没接。医生在吃午餐。" 指涉了两个人的三个指称，但无人具名。指代消解能理清谁是谁。

**类型：** 学习
**语言：** Python
**先修知识：** 阶段5 · 06 (命名实体识别)，阶段5 · 07 (词性标注与解析)
**时间：** 约60分钟

## 问题描述

从一篇300词的文章中提取所有提及苹果公司的表达。当文中直接使用"苹果"时很容易。但当它说"该公司"、"他们"、"库比蒂诺的科技巨头"或"乔布斯的公司"时就难了。如果不能将这些提及解析到同一实体上，你的命名实体识别（NER）流程就会遗漏60%-80%的提及。

指代消解将所有指代同一现实世界实体的表达链接为一个集群。它是表层NLP（NER、解析）与下游语义（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

为什么在2026年这很重要：

- 摘要生成："首席执行官宣布了……" 对比 "蒂姆·库克宣布了……" — 摘要中应点明首席执行官的姓名。
- 问答系统："她给谁打了电话？" 需要先解析"她"。
- 信息抽取：知识图谱中同时存在"人物1创立了苹果"和"乔布斯创立了苹果"两个独立条目是错误的。
- 多文档信息抽取：合并关于同一事件的多篇文章中的提及，这属于跨文档指代消解。

## 核心概念

![指代消解聚类：提及 → 实体](../assets/coref.svg)

**任务定义。** 输入：一份文档。输出：提及（文本片段）的聚类，每个集群指向一个实体。

**提及类型。**

- **命名实体。** "蒂姆·库克"
- **名词性提及。** "首席执行官"、"这家公司"
- **代词性提及。** "他"、"她"、"他们"、"它"
- **同位语提及。** "蒂姆·库克，苹果公司的首席执行官，"

**技术架构。**

1.  **基于规则（Hobbs, 1978）。** 基于句法树，使用语法规则进行代词解析。是很好的基线方法。在代词消解上，其效果出人意料地难以超越。
2.  **提及对分类器。** 对每一对提及 (m_i, m_j)，预测它们是否共指。通过传递闭包进行聚类。2016年之前的标准方法。
3.  **提及排序。** 对每个提及，对候选先行词（包括"无先行词"）进行排序。选择得分最高者。
4.  **基于片段的端到端方法（Lee 等, 2017）。** 使用Transformer编码器。枚举所有长度上限内的候选片段。预测提及分数。为每个片段预测先行词概率。进行贪心聚类。现代默认方法。
5.  **生成式（2024+）。** 提示大语言模型："列出本文中每个代词及其先行词。" 对简单情况效果好，但在长文档和罕见指称上表现挣扎。

**评估指标。** 有五个标准指标（MUC， B³， CEAF， BLANC， LEA），因为没有单一指标能全面捕捉聚类质量。通常报告前三个指标的平均值作为CoNLL F1。2026年在CoNLL-2012数据集上的先进水平约为83 F1。

**已知难点。**

- 指代数页前引入实体的定指描述。
- 桥接回指（"轮子" → 先前提到的一辆车）。
- 中文、日语等语言中的零回指。
- 预指（代词出现在指称对象之前）："当**她**走进来时，玛丽笑了。"

## 动手实现

### 步骤 1：预训练神经网络指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档上，你会得到类似这样的结果：
- 集群 1: [Apple, The company, they]
- 集群 2: [new products]

### 步骤 2：基于规则的代词解析器（教学用）

参见 `code/main.py` 获取一个仅使用标准库的实现：

1.  提取提及：命名实体（大写片段）、代词（字典查找）、定指描述（"the X"）。
2.  对每个代词，查看前K个提及并按以下标准打分：
    - 性别/数的一致性（启发式规则）
    - 相近性（较近者优先）
    - 句法角色（主语优先）
3.  链接得分最高的先行词。

无法与神经网络模型竞争。但它展示了端到端模型必须做出的搜索空间和决策。

### 步骤 3：使用大语言模型进行指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需注意两种失败模式。第一，大语言模型会过度合并（"他"和"她"指代两个不同的人）。第二，大语言模型会在长文档中静默地丢弃提及。始终使用文本片段偏移量检查进行验证。

### 步骤 4：评估

标准的conll-2012脚本计算MUC、B³、CEAF-φ4并报告平均值。对于内部评估，首先在标注好的测试集上计算提及级别的精确率和召回率，然后添加提及链接的F1分数。

## 常见陷阱

- **单实体集群爆炸。** 有些系统将每个提及报告为其自己的集群。B³对此比较宽容，但MUC会对此进行惩罚。务必检查所有三个指标。
- **长上下文中的代词。** 在超过2000 token的文档上，性能下降约15 F1。需谨慎进行分块处理。
- **性别假设。** 硬编码的性别规则在涉及非二元指称、组织、动物时会失效。使用学习到的模型或中性评分。
- **大语言模型在长文档上的漂移。** 单次API调用无法可靠地对50+段落中的提及进行聚类。使用滑动窗口 + 合并策略。

## 应用指南

2026年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英文，单文档 | `en_coreference_web_trf` (spaCy-experimental) 或 AllenNLP 神经网络指代消解 |
| 多语言 | 基于OntoNotes或多语言CoNLL训练的SpanBERT / XLM-R |
| 跨文档事件共指 | 专用的端到端模型（2025-26年最先进） |
| 快速大语言模型基线 | GPT-4o / Claude 搭配结构化输出的指代消解提示 |
| 生产级对话系统 | 基于规则的后备方案 + 神经网络主模型 + 关键槽位人工审核 |

2026年中实际部署的集成模式：先运行NER，再运行指代消解，将指代消解集群合并到NER实体中。下游任务看到的是每个集群一个实体，而非每个提及一个实体。

## 部署上线

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1.  **简单。** 在5个精心构造的段落上运行 `code/main.py` 中基于规则的解析器。根据真实数据衡量提及链接准确率。
2.  **中等。** 在一篇新闻文章上使用预训练的神经网络指代消解模型。将聚类结果与你自己的手动标注进行比较。它在哪里失败了？
3.  **困难。** 构建一个指代消解增强的NER流程：先运行NER，然后通过指代消解集群进行合并。在100篇文章上衡量相比仅使用NER的实体覆盖率提升。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|-----------------|-----------------------|
| 提及 (Mention) | 一个指称 | 指代一个实体的文本片段（名称、代词、名词短语）。 |
| 先行词 (Antecedent) | "它"指代什么 | 后续提及与之共指的先前提及。 |
| 集群 (Cluster) | 该实体的所有提及 | 指代同一现实世界实体的所有提及的集合。 |
| 回指 (Anaphora) | 向后指代 | 后续提及指代先前提及（"他" → "约翰"）。 |
| 预指 (Cataphora) | 向前指代 | 先前提及指代后续提及（"当他到达时，约翰……"）。 |
| 桥接 (Bridging) | 隐式指代 | "我买了一辆车。轮子很糟糕。"（指的是那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1分数的平均值。 |

## 扩展阅读

- [Jurafsky & Martin, 《语音与语言处理》第3版 第26章 — 指代消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 权威教材章节。
- [Lee 等 (2017). 端到端神经网络指代消解](https://arxiv.org/abs/1707.07045) — 基于片段的端到端方法。
- [Joshi 等 (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 能提升指代消解的预训练方法。
- [Pradhan 等 (2012). CoNLL-2012 共享任务](https://aclanthology.org/W12-4501/) — 基准数据集。
- [Hobbs (1978). 解析代词指代](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典方法。