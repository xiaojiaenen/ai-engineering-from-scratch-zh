# 共指消解（Coreference Resolution）

> "她给他打了电话。他没有接听。医生正在吃午饭。"三处引用指向两个人，但没人有名字。共指消解就是搞清楚谁是谁。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段·06（NER），第5阶段·07（POS & Parsing）
**时间：** 约60分钟

## 问题所在

从一篇300词的文章中提取所有对苹果公司的提及。当文章说"Apple"时很简单，但当它说"the company"、"they"、"Cupertino's technology giant"或"Jobs's firm"时就困难了。如果不将这些提及解析为同一实体，你的NER管道会遗漏60-80%的提及。

共指消解将所有指向同一真实世界实体的表达式链接到一个簇中。它是表层NLP（NER、句法分析）与下游语义（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

2026年为何重要：

- 摘要："CEO宣布……"与"Tim Cook宣布……"——摘要应该点名CEO。
- 问答："她给谁打了电话？"需要解析"she"。
- 信息抽取：知识图谱中"PER1 founded Apple"和"Jobs founded Apple"作为独立条目是错误的。
- 多文档IE：跨文章合并有关同一事件的说法属于跨文档共指。

## 概念

![共指聚类：提及→实体](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：一个提及（跨度）的聚类，每个聚类指向一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **普通名词。** "the CEO", "the company"
- **代词。** "he", "she", "they", "it"
- **同位语。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs，1978）。** 使用语法规则基于句法树的代词解析。良好的基线。在代词上出乎意料地难以超越。
2. **提及对分类器。** 对于每对提及（m_i, m_j），预测它们是否共指。通过传递闭包聚类。2016年之前的标准方法。
3. **提及排序。** 对于每个提及，对候选先行词进行排序（包括"无先行词"）。选择得分最高者。
4. **端到端基于跨度（Lee等人，2017）。** Transformer编码器。枚举所有候选跨度直至长度上限。预测提及分数。预测每个跨度的先行词概率。贪心聚类。现代默认方案。
5. **生成式（2024+）。** 提示LLM："列出此文本中每个代词及其先行词。"在简单案例上效果良好，但在长文档和罕见指代上表现不佳。

**评估指标。** 五种标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能捕捉聚类质量。报告前三种的平均值作为CoNLL F1。2026年CoNLL-2012上的SOTA约为83 F1。

**已知的困难案例。**

- 指代提前数页引入实体的定指描述。
- 桥接性指称（"the wheels" → 之前提到的汽车）。
- 中文和日语等语言中的零形指称。
- 逆指（代词在所指之前）："When she walked in, Mary smiled."

```figure
coref-links
```

## 构建

### 步骤1：预训练神经共指（AllenNLP / spaCy实验性）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 实验性模型
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你可能会得到类似：
- 簇1：[Apple, The company, they]
- 簇2：[new products]

### 步骤2：基于规则的代词解析器（教学用）

参见`code/main.py`中的纯stdlib实现：

1. 提取提及：命名实体（大写跨度）、代词（字典查找）、定指描述（"the X"）。
2. 对于每个代词，查看前K个提及并按以下方式评分：
   - 性别/数一致性（启发式）
   - 时效性（越近越好）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

无法与神经模型竞争。但它展示了搜索空间和端到端模型必须做出的决策。

### 步骤3：使用LLM进行共指

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两种失败模式。首先，LLM会过度合并（"him"和"her"指代两个不同的人）。其次，LLM在长文档中会静默丢弃提及。始终通过跨度偏移检查进行验证。

### 步骤4：评估

标准的coNLL-2012脚本计算MUC、B³、CEAF-φ4并报告平均值。对于内部评估，从标注测试集上的跨度级精确率和召回率开始，然后添加提及链接F1。

## 陷阱

- **单例爆炸。** 某些系统报告每个提及作为独立簇。B³宽容，MUC惩罚这种做法。始终检查所有三个指标。
- **长上下文中的代词。** 在超过2,000词元的文档上性能下降约15 F1。仔细分块。
- **性别假设。** 硬编码的性别规则在非二元指代、组织、动物上会失效。使用学习到的模型或中性评分。
- **LLM在长文档上的漂移。** 单次API调用无法可靠地聚类超过50段的提及。使用滑动窗口+合并。

## 应用

2026年技术栈：

| 场景 | 选择 |
|------|------|
| 英语，单文档 | `en_coreference_web_trf`（spaCy实验性）或AllenNLP神经共指 |
| 多语言 | SpanBERT / XLM-R，在OntoNotes或多语言CoNLL上训练 |
| 跨文档事件共指 | 专用端到端模型（2025–26年SOTA） |
| 快速LLM基线 | GPT-4o / Claude配合结构化输出共指提示 |
| 生产对话系统 | 基于规则的回退 + 神经主要方法 + 关键槽位的手动审查 |

2026年投产的集成模式：先运行NER，再运行共指，将共指簇合并到NER实体中。下游任务看到的是一簇一个实体，而非一提及一个实体。

## 交付

保存为`outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: 选择合适的共指方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定用例（单文档/多文档、领域、语言），输出：

1. 方法。基于规则/神经基于跨度/LLM提示/混合。一句话说明理由。
2. 模型。如果是神经模型，提供命名检查点。
3. 集成。操作顺序：分词→NER→共指→下游任务。
4. 评估。在保留集上的CoNLL F1（MUC + B³ + CEAF-φ4平均值）+ 在20篇文档上的手动簇审查。

拒绝在超过2,000词元的文档上使用仅LLM的共指，除非采用滑动窗口合并。拒绝任何未附带提及级精确率-召回率报告的就指管道。对部署在人口统计学多样化文本上的性别启发式系统发出警告。
```

## 练习

1. **简单。** 在`code/main.py`中的基于规则的解析器上运行5段手工编写的段落。对照Ground Truth测量提及链接准确率。
2. **中等。** 在新闻文章上使用预训练的神经共指模型。将聚类与你的人工标注进行比较。它在何处失败？
3. **困难。** 构建共指增强的NER管道：先NER，然后通过共指簇合并。在100篇文章上测量实体覆盖率提升相对于仅NER的效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Mention（提及） | 一个引用 | 指向实体的文本跨度（名称、代词、名词短语）。 |
| Antecedent（先行词） | "it"所指的内容 | 后面一个提及与之共指的较早提及。 |
| Cluster（簇） | 实体的提及 | 所有指向同一真实世界实体的提及集合。 |
| Anaphora（回指） | 向后引用 | 较晚提及指代较早提及（"he" → "John"）。 |
| Cataphora（逆指） | 向前引用 | 较早提及指代较晚提及（"When he arrived, John..."）。 |
| Bridging（桥接） | 隐含引用 | "I bought a car. The wheels were bad."（那个车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 第26章 — 共指消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于跨度的端到端方案。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改进共指的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准测试。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典论文。
