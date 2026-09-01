# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接将它们锚定。关系抽取寻找它们之间的边。知识图谱是节点、边及其来源的总和。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段 5 · 06（NER），阶段 5 · 25（实体链接）  
**时间：** 约 60 分钟

## 问题描述

一位分析师读到："Tim Cook 于 2011 年成为 Apple 的首席执行官。" 包含四个事实：

- `(Tim Cook, 角色, 首席执行官)`
- `(Tim Cook, 雇主, Apple)`
- `(Tim Cook, 开始日期, 2011)`
- `(Apple, 类型, 组织)`

关系抽取（RE）将自由文本转换为结构化三元组 `(主体, 关系, 客体)`。对语料库进行聚合后，你就拥有了一个知识图谱。对其进行聚合和查询，就为 RAG、分析或合规审计提供了推理基础。

2026 年的问题：LLM 热情地抽取关系。过于热情。它们会 hallucinate 出源文本不支持的三元组。没有溯源信息，你无法区分真实三元组和看似合理但虚构的内容。2026 年的答案是类似 AEVS 的锚定与验证管道。

## 概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(主体实体, 关系类型, 客体实体)`。关系来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，无所不包）。

**三种抽取方法。**

1. **规则/模式匹配。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工编写的正则表达式。脆弱、精确、可解释。
2. **监督分类器。** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准方法。
3. **生成式 LLM。** 提示模型输出一组三元组。开箱即用。需要溯源，否则会 hallucinate 出看似合理但虚假的内容。

**AEVS（锚定 - 抽取 - 验证 - 补充，2026）。** 当前的幻觉缓解框架：

- **锚定（Anchor）。** 使用精确位置标识每个实体 span 和关系短语 span。
- **抽取（Extraction）。** 生成与锚定 span 关联的三元组。
- **验证（Verification）。** 将每个三元组元素匹配回源文本；拒绝任何无支持的项。
- **补充（Supplement）。** 覆盖检查确保没有锚定 span 被遗漏。

幻觉大幅下降。需要更多计算资源，但可审计。

**开放 vs 封闭的权衡。**

- **封闭本体。** 固定的属性列表（如 Wikidata 的 11,000+ 属性）。可预测、可查询、难以发明新内容。
- **Open IE。** 任何动词短语都成为关系。高召回率、低精确率、查询时混乱。

生产级知识图谱通常混合使用：开放 IE 用于发现，然后将关系规范化到封闭本体，再合并到主图中。

```figure
relation-triples
```

## 构建

### 第 1 步：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整玩具抽取器参见 `code/main.py`。Hearst 模式仍在特定领域的管道中使用，因为它们是可调试的。

### 第 2 步：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：输入文本，输出三元组，已转换为 Wikidata 属性 ID。在远程监督数据上微调。标准的开源权重基线。

### 第 3 步：带锚定的 LLM 提示抽取

```python
prompt = f"""从文本中提取 (subject, relation, object) 三元组。
对于每个三元组，包含源文本中的精确字符范围。

文本：{text}

输出 JSON：
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

仅包含完全由文本支持的三元组。不要超出文本明确陈述的范围进行推断。
"""
```

将每个返回的 span 与源文本进行验证。拒绝任何 `text[start:end] != triple_entity` 的项。这是 AEVS "验证" 步骤的最简形式。

### 第 4 步：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 丢弃未映射的开放关系或路由到人工审核
```

规范化通常占工程工作的 60-80%。为此预留预算。

### 第 5 步：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是所有 RAG-over-KG 系统的原子操作。使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展它。

## 陷阱

- **关系抽取前的共指消解。** "He founded Apple" — RE 需要知道 "he" 是谁。先运行共指消解（第 24 课）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析为同一节点。先进行实体链接（第 25 课）。
- **幻觉三元组。** LLM 会输出文本不支持的三元组。强制 span 验证。
- **关系规范化漂移。** Open IE 关系不一致（"was born in"、"came from"、"is a native of"）。折叠到规范 ID，否则图无法查询。
- **时间错误。** "Tim Cook is CEO of Apple" — 现在为真，2005 年为假。许多关系具有时间边界。使用限定符（Wikidata 中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在维基百科上训练。法律、医疗和科学文本通常需要领域微调的 RE 模型。

## 使用

2026 技术栈：

| 场景 | 选择 |
|------|------|
| 快速生产、通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 特定领域（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| LLM 提示、可审计输出 | AEVS 管道：锚定 → 抽取 → 验证 → 补充 |
| 大规模新闻 IE | 基于模式 + 监督混合 |
| 从零构建 KG | 开放 IE + 手动规范化流程 |
| 时序 KG | 提取带限定符（开始/结束时间、时间点） |

集成模式：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图加载。每个阶段都是潜在的质量门控。

## 交付

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: 设计具有溯源和规范化能力的关系抽取管道。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定语料库（领域、语言、规模）和下游用途（KG-RAG、分析、合规），输出：

1. 抽取器。基于模式/监督/LLM/AEVS 混合。根据精确率与召回率目标说明理由。
2. 本体。固定属性列表（Wikidata/领域）或带规范化流程的开放 IE。
3. 溯源。每个三元组携带源字符 span + 文档 ID。审计不可妥协。
4. 合并策略。规范实体 ID + 关系 ID + 时间限定符；去重策略。
5. 评估。在 200 个手工标注的三元组上评估精确率/召回率 + 在 LLM 抽取样本上测量幻觉率。

拒绝任何缺乏 span 验证（源溯源）的基于 LLM 的 RE 管道。拒绝开放 IE 输出未经规范化直接流入生产图的管道。标记在任何时间受限关系（雇主、配偶、职位）上没有时间限定符的管道。
```

## 练习

1. **简单。** 在 5 条新闻句子上的 `code/main.py` 中运行基于模式的抽取器。人工检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或小 LLM）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 管道：使用 LLM 抽取 + 与源文本对比验证 span。在 50 条维基百科风格句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 三元组 | 主体 - 关系 - 客体 | `(s, r, o)` 元组，是 KG 的原子单位。 |
| 开放 IE | 抽取一切 | 开放词汇的关系短语；高召回率、低精确率。 |
| 封闭本体 | 固定模式 | 有限的关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化 | 标准化一切 | 将表面名称/关系映射到规范 ID。 |
| AEVS | 有根据的抽取 | 锚定 - 抽取 - 验证 - 补充管道（2026）。 |
| 溯源 | 真实来源链接 | 每个三元组携带文档 ID + 字符 span 指向其来源。 |
| 远程监督 | 廉价标签 | 将文本与现有 KG 对齐以创建训练数据。 |

## 延伸阅读

- [Mintz 等人 (2009)。无需标注数据的关系抽取的远程监督](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021)。REBEL：端到端语言生成的关系抽取](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力模型。
- [Wadden 等人 (2019)。使用上下文化 span 表示的实体、关系和事件抽取 (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合 IE。
- [AEVS — 锚定 - 抽取 - 验证 - 补充框架](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉缓解设计。
- [Wikidata SPARQL 教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图查询。
