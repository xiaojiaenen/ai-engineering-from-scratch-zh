# 实体链接与消歧

> NER识别出"Paris"（巴黎）。实体链接需要确定：是巴黎（法国）？帕丽斯·希尔顿？德克萨斯州的帕里斯？特洛伊王子帕里斯？若不进行链接，你的知识图谱将始终存在歧义。

**类型：** 构建
**编程语言：** Python
**前置知识：** 阶段5·06（NER），阶段5·24（指代消解）
**时长：** 约60分钟

## 问题描述

句子写道："Jordan beat the press."（乔丹击败了媒体。）你的NER将"Jordan"标记为PERSON（人名）。很好。但*是哪个*乔丹？

- 迈克尔·乔丹（篮球运动员）？
- 迈克尔·B·乔丹（演员）？
- 迈克尔·I·乔丹（伯克利大学机器学习教授——是的，这种混淆在机器学习论文中真实存在）？
- 约旦（国家）？
- 约旦（希伯来名字）？

实体链接（EL）将每个提及解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia或你的领域知识库。包含两个子任务：

1.  **候选生成。** 对于"Jordan"，哪些知识库条目是可能的？
2.  **消歧。** 根据上下文，哪个候选是正确的？

两个步骤都可学习。两者都有基准测试。组合后的流水线已稳定运行十年——变化的是消歧器的质量。

## 概念理解

![实体链接流水线：提及 → 候选 → 消歧后的实体](../assets/entity-linking.svg)

**候选生成。** 根据提及的表面形式（"Jordan"），在别名索引中查找候选。Wikipedia别名词典覆盖了大多数命名实体："JFK" → 约翰·F·肯尼迪、杰奎琳·肯尼迪、肯尼迪机场、《JFK》（电影）。典型索引每个提及返回10-30个候选。

**消歧：三种方法。**

1.  **先验 + 上下文 (Milne & Witten, 2008)。** `P(entity | mention) × context-similarity(entity, text)`。效果好，速度快，无需训练。
2.  **基于嵌入 (ESS / REL / Blink)。** 编码提及和上下文。编码每个候选的描述。选择最大余弦相似度。2020-2024年的默认方法。
3.  **生成式 (GENRE, 2021; 基于LLM, 2023+)。** 逐token解码实体的规范名称。受限于有效实体名称的trie树，以确保输出是有效的知识库ID。

**端到端 vs 流水线。** 现代模型（ELQ, BLINK, ExtEnD, GENRE）在一次前向传播中完成NER + 候选生成 + 消歧。流水线系统在生产环境中仍占主导，因为你可以更换组件。

### 两个度量指标

-   **提及召回率（候选生成）。** 正确知识库条目出现在候选列表中的真值提及的比例。这是整个流水线的基准。
-   **消歧准确率 / F1分数。** 给定正确候选，top-1预测正确的比例。

务必报告两者。一个消歧准确率99%但候选召回率80%的系统，整体流水线准确率就是80%。

## 动手构建

### 第1步：从Wikipedia重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia别名数据：约1800万（别名，实体）对。从Wikidata数据包下载。存储为倒排索引。

### 第2步：基于上下文的消歧

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

Jaccard重叠是一个玩具示例。替换为基于嵌入的余弦相似度（见 `code/main.py` 步骤2的Transformer版本）。

### 第3步：基于嵌入（BLINK风格）

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

索引时，嵌入每个知识库实体一次。查询时，嵌入提及和上下文一次，与候选池进行点积，选择最大值。

### 第4步：生成式实体链接（概念）

GENRE逐字符解码实体的Wikipedia标题。受限解码（见第20课）确保只能输出有效标题。与基于知识库的trie树紧密集成。现代衍生版是REL-GEN和基于LLM提示的、带有结构化输出的实体链接。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是2026年最简单的可部署EL流水线。

### 第5步：在AIDA-CoNLL上评估

AIDA-CoNLL是标准EL基准：1,393篇路透社文章，34k个提及，Wikipedia实体。报告知识库内准确率（`P@1`）和知识库外NIL检测率。

## 陷阱提示

-   **NIL处理。** 一些提及不在知识库中（新兴实体、不知名人物）。系统必须预测NIL而不是猜测错误实体。需单独衡量。
-   **提及边界错误。** 上游NER遗漏了部分跨度（例如，"Bank of America"只被标记为"Bank"）。EL召回率下降。
-   **流行度偏差。** 训练过的系统过度预测常见实体。机器学习论文中提及"Michael I. Jordan"通常链接到篮球运动员乔丹。
-   **跨语言EL。** 将中文文本中的提及映射到英文Wikipedia实体。需要多语言编码器或翻译步骤。
-   **知识库过时。** 新公司、新事件、新人不在去年的Wikipedia数据包中。生产流水线需要刷新循环。

## 使用指南

2026年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM友好，每日提及少 | 使用候选列表 + 受限JSON提示 Claude/GPT-4 |
| 领域特定知识库（医疗、法律） | 定制BERT，带有知识库感知检索 + 在领域AIDA风格数据集上微调 |
| 极低延迟 | 仅使用先验精确匹配（Milne-Witten基线） |
| 研究SOTA | GENRE / ExtEnD / 生成式LLM-EL |

2026年可部署的生产模式：NER → 指代消解 → 对每个提及进行EL → 将每个指代簇合并为一个规范实体。输出：文档中每个实体一个知识库ID，而非每个提及一个。

## 部署实施

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习题

1.  **简单。** 在10个歧义提及（Paris, Jordan, Apple）上实现 `code/main.py` 中的先验+上下文消歧器。人工标注正确实体。测量准确率。
2.  **中等。** 使用句子Transformer编码50个歧义提及。嵌入每个候选的描述。比较基于嵌入的消歧与Jaccard上下文重叠。
3.  **困难。** 构建一个1000实体的领域知识库（例如你公司的员工 + 产品）。实现端到端的NER + EL。在100个保留句子上测量精确率和召回率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接 (EL) | 链接到Wikipedia | 将提及映射到知识库中的唯一条目。 |
| 候选生成 | 可能是谁？ | 为提及返回一个可能的知识库条目候选列表。 |
| 消歧 | 选出正确的 | 使用上下文对候选进行评分，选出胜者。 |
| 别名索引 | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在知识库中 | 明确预测没有知识库条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia或你的领域知识库。 |
| AIDA-CoNLL | 基准数据集 | 1,393篇带有真值实体链接的路透社文章。 |

## 延伸阅读

-   [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 先验+上下文方法的奠基之作。
-   [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力模型。
-   [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带有受限解码的生成式实体链接。
-   [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准论文。
-   [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产栈。