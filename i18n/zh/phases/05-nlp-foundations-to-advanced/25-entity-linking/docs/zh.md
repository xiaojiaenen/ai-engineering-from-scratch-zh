# 实体链接与消歧

> NER 找到了"Paris。"实体链接决定：巴黎，法国？帕丽斯·希尔顿？得克萨斯州巴黎？巴黎（特洛伊王子）？如果不做链接，你的知识图谱将始终模糊不清。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段 · 06（NER），第5阶段 · 24（共指消解）
**时间：** 约60分钟

## 问题

一句话写道："Jordan beat the press." 你的 NER 将"Jordan"标记为人物。很好。但**哪一个** Jordan？

- 迈克尔·乔丹（篮球）？
- 迈克尔·B·乔丹（演员）？
- 迈克尔·I·乔丹（伯克利机器学习教授——是的，这在 ML 论文中确实会混淆）？
- Jordan（约旦这个国家）？
- Jordan（希伯来语名字）？

实体链接（EL）将每个指代表达映射到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域知识库。两个子任务：

1. **候选生成。** 给定"Jordan"，哪些 KB 条目是合理的？
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都可学习，均有基准测试。整个管道已稳定运行了十年——变化的是消歧器的质量。

## 概念

![实体链接管道：指代表达 → 候选 → 消歧后实体](../assets/entity-linking.svg)

**候选生成。** 给定指代表达形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名词典涵盖大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引为每个指代表达返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten，2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、无需训练。
2. **基于嵌入（ESS / REL / Blink）。** 编码指代表达 + 上下文。编码每个候选的描述。取最大余弦相似度。2020-2024 年的默认方法。
3. **生成式（GENRE，2021；基于 LLM，2023+）。** 逐个 token 解码实体的规范名称。约束于有效实体名的 trie 树中，确保输出必然是有效的 KB ID。

**端到端 vs 管道模式。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）一次性完成 NER + 候选生成 + 消歧。生产系统中管道模式仍占主导，因为你可以灵活替换各个组件。

### 两个评估指标

- **指代表达召回率（候选生成）。** 在候选列表中，正确 KB 条目出现的比例。这是整个管道的下限。
- **消歧准确率 / F1。** 给定正确候选，top-1 预测正确的频率。

两个指标都要报告。一个在 80% 候选召回率下达到 99% 消歧准确率的系统，最终管道召回率只有 80%。

```figure
gx-entity-linking
```

## 构建

### 步骤1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万条（别名，实体）对。从 Wikidata 转储下载。存储为倒排索引。

### 步骤2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是一个示例。替换为基于嵌入的余弦相似度（见 `code/main.py` step-2 中的 transformer 版本）。

### 步骤3：基于嵌入的方法（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引阶段：预先嵌入每个 KB 实体一次。查询阶段：嵌入指代表达 + 上下文一次，与候选池做点积，取最大值。

### 步骤4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见课程20）确保只有合法标题才能被输出。与 KB 支持的 trie 树紧密集成。现代后继者是 REL-GEN 和基于 LLM 提示的 EL（带结构化输出）。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年最简单的可投产 EL 管道。

### 步骤5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1,393 篇路透社文章、3.4 万条指代表达、Wikipedia 实体。报告 KB 内准确率（`P@1`）和 KB 外 NIL 检测率。

## 陷阱

- **NIL 处理。** 某些指代表达不在 KB 中（新兴实体、冷门人物）。系统必须预测 NIL，而非猜测错误的实体。需单独评估。
- **指代表达边界错误。** 上游 NER 漏掉部分 span（"Bank of America" 仅被标记为"Bank"）。EL 召回率下降。
- **流行度偏差。** 训练过的系统过度预测频繁实体。ML 论文中提及"Michael I. Jordan"时常链接到篮球 Jordan。
- **跨语言 EL。** 将中文文本中的指代表达映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 陈旧。** 新公司、事件、人物不在去年的 Wikipedia 转储中。生产管道需要刷新循环。

## 应用

2026 技术栈：

| 场景 | 推荐方案 |
|------|----------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好，每日少量指代表达 | 用候选列表提示 Claude/GPT-4 + 约束 JSON |
| 领域特定 KB（医疗、法律） | 定制 BERT + KB 感知检索 + 在领域 AIDA 风格集上微调 |
| 极低延迟 | 仅精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年可投产的生产模式：NER → 共指消解 → 对每个指代表达做 EL → 将同一簇折叠为一个规范实体。输出：每份文档中每个实体对应一个 KB ID，而非每个指代表达一个。

## 交付

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: 设计实体链接管道——KB、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

根据用例（领域KB、语言、吞吐量、延迟预算）输出：

1. 知识库。Wikidata / Wikipedia / 定制KB。版本日期。刷新频率。
2. 候选生成器。别名索引、嵌入或混合方案。目标指代表达召回率 @ K。
3. 消歧器。先验+上下文、基于嵌入、生成式或LLM提示。
4. NIL 策略。top分阈值、分类器或显式NIL候选。
5. 评估。指代表达召回率 @ 30、top-1准确率、保留集的NIL检测F1。

拒绝缺少指代表达召回率基线的任何EL管道（不知道候选生成是否找到了正确实体就无法评估消歧器）。拒绝使用LLM提示EL但未约束到有效KB ID的任何管道。标记存在流行度偏差影响少数实体（如同名冲突）且未做领域微调的系统。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现先验+上下文消歧器，针对10个歧义指代表达（Paris、Jordan、Apple）。人工标注正确实体。测量准确率。
2. **中等。** 用句子 Transformer 编码50个歧义指代表达。嵌入每个候选的描述。比较基于嵌入的消歧与 Jaccard 上下文重叠。
3. **困难。** 构建一个1k实体的领域知识库（如公司中员工+产品）。实现 NER + EL 端到端。在100句保留集上测量准确率和召回率。

## 术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 实体链接（EL） | 链接到Wikipedia | 将指代表达映射到唯一KB条目。 |
| 候选生成 | 可能是什么？ | 为指代表达返回一小组合理的KB条目。 |
| 消歧 | 选正确的一个 | 用上下文给候选打分，选出最优。 |
| 别名索引 | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在KB中 | 明确预测无匹配KB条目。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia或领域KB。 |
| AIDA-CoNLL | 基准 | 1,393篇带有金标准实体链接的路透社文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基础先验+上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的骨干工作。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产栈。
