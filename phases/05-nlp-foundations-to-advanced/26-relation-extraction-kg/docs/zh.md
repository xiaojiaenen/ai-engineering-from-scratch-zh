# 关系抽取与知识图谱构建

> 命名实体识别（NER）负责找到实体，实体链接将其锚定，而关系抽取则负责发现它们之间的边。知识图谱正是由节点、边及其来源构成的整体。

**类型：** 构建
**语言：** Python
**先修课程：** 阶段5 · 06（NER），阶段5 · 25（实体链接）
**时间：** 约60分钟

## 问题描述

一位分析师读到："蒂姆·库克于2011年成为苹果公司CEO。" 四条事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）将自由文本转化为结构化三元组 `(subject, relation, object)`。在整个语料库上聚合，你就得到了一个知识图谱。聚合并查询，你就拥有了一个可用于RAG、分析或合规审计的推理基础。

2026年的难题：LLM们非常热衷于抽取关系。过于热衷了。它们会编造出原文本不支持的三元组。没有溯源信息，你无法区分真实三元组和看似合理的虚构内容。2026年的解决方案是采用AEVS风格的“锚定-验证”流水线。

## 核心概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭的本体（Wikidata属性、FIBO、UMLS）或开放集（OpenIE风格，不设限制）。

**三种抽取方法。**

1. **基于规则/模式。** Hearst模式："X such as Y" → `(Y, isA, X)`。加上手工编写的正则表达式。脆弱、精确、可解释。
2. **监督分类器。** 给定句子中的两个实体提及，从固定关系集中预测关系。在TACRED、ACE、KBP上训练。2015-2022年的标准方法。
3. **生成式LLM。** 提示模型输出三元组。开箱即用。需要溯源信息，否则会编造出看似合理的垃圾信息。

**AEVS（锚定-抽取-验证-补充，2026）。** 当前的幻觉缓解框架：

- **锚定。** 用精确位置标识每个实体跨度和关系短语跨度。
- **抽取。** 生成与锚定跨度关联的三元组。
- **验证。** 将每个三元组元素匹配回原文；拒绝任何无支持的内容。
- **补充。** 覆盖检查确保没有锚定跨度被遗漏。

幻觉率显著下降。计算成本更高，但可审计。

**开放与封闭的权衡。**

- **封闭本体。** 固定的属性列表（例如，Wikidata的11,000+个属性）。可预测。可查询。难以凭空捏造。
- **开放信息抽取（Open IE）。** 任何动词短语都可以成为关系。高召回率。低精确率。查询混乱。

生产环境的知识图谱通常混合使用：用开放IE进行发现，然后在合并到主图之前，将关系规范化到一个封闭本体上。

## 动手构建

### 步骤1：基于模式的关系抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见 `code/main.py` 获取完整的玩具抽取器。Hearst模式仍然出现在特定领域的流水线中，因为它们是可调试的。

### 步骤2：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL是一个序列到序列的关系抽取器：输入文本，输出三元组，并且已经转换为Wikidata属性ID。在远程监督数据上微调。标准的开权基线模型。

### 步骤3：带锚定的LLM提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

将每个返回的跨度与源文本进行验证。拒绝任何 `text[start:end] != triple_entity` 的内容。这是AEVS“验证”步骤的最小化形式。

### 步骤4：规范化到封闭本体

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
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占工程工作量的60-80%。务必预留预算。

### 步骤5：构建小型图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个“图谱RAG”系统的原子操作。使用RDF三元组存储（Blazegraph, Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展它。

## 常见陷阱

- **关系抽取前先做共指消解。** "He founded Apple"（他创立了苹果）—— 关系抽取需要知道“他”是谁。先运行共指消解（课程24）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须指向同一个节点。先进行实体链接（课程25）。
- **幻觉三元组。** LLM会生成文本不支持的三元组。强制执行跨度验证。
- **关系规范化漂移。** 开放IE的关系不一致（"was born in", "came from", "is a native of"）。折叠到规范ID，否则图谱无法查询。
- **时间错误。** "Tim Cook is CEO of Apple"（蒂姆·库克是苹果CEO）—— 现在正确，但在2005年是错误的。许多关系是有时间边界的。使用限定符（Wikidata中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL在维基百科上训练。法律、医学和科学文本通常需要针对领域微调的RE模型。

## 实际使用

2026年的技术栈：

| 场景 | 推荐方案 |
|------|----------|
| 快速生产，通用领域 | REBEL 或 LlamaPred + Wikidata规范化 |
| 特定领域（生物医学、法律） | SciREX风格的领域微调 + 自定义本体 |
| LLM提示，输出需审计 | AEVS流水线：锚定 → 抽取 → 验证 → 补充 |
| 大规模新闻信息抽取 | 基于模式 + 监督混合 |
| 从头构建知识图谱 | 开放IE + 手动规范化 |
| 时序知识图谱 | 使用限定符抽取（开始/结束时间、时间点） |

集成模式：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图加载。每个阶段都是一个潜在的质量关口。

## 保存代码

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 `code/main.py` 上对5个新闻句子运行模式抽取器。手动检查精确率。
2. **中等。** 对相同句子使用REBEL（或一个小型LLM）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建AEVS流水线：用LLM抽取 + 将跨度与源文本验证。在50个维基百科风格句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 三元组 | 主语-关系-宾语 | `(s, r, o)` 元组，是知识图谱的原子单位。 |
| 开放信息抽取 | 抽取任何内容 | 开放词汇关系短语；高召回率，低精确率。 |
| 封闭本体 | 固定模式 | 有限的关系类型集合（Wikidata, UMLS, FIBO）。 |
| 规范化 | 使一切标准化 | 将表面名称/关系映射到规范ID。 |
| AEVS | 基于事实的抽取 | 锚定-抽取-验证-补充流水线（2026）。 |
| 溯源 | 真相来源链接 | 每个三元组都带有文档ID和字符跨度指向其来源。 |
| 远程监督 | 廉价标签 | 将文本与现有知识图谱对齐以创建训练数据。 |

## 扩展阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — 序列到序列RE主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026年幻觉缓解设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图查询。