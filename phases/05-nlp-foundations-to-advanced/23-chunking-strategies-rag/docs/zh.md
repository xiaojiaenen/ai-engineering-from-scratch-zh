# RAG的分块策略

> 分块配置对检索质量的影响，不亚于嵌入模型的选择（Vectara NAACL 2025）。分块策略选错了，再好的重排序也救不回来。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段 · 14（信息检索），第5阶段 · 22（嵌入模型）
**时间：** 约60分钟

## 问题所在

你将一份50页的合同放入RAG系统。用户问：“终止条款是什么？”检索器返回的是封面页。为什么？因为模型是在512个token的块上训练的，而终止条款在20页之后，且跨页分割，局部关键词也与查询不相关。

解决办法不是“买个更好的嵌入模型”，而是分块。多大块？重叠多少？在哪里分割？带上下文吗？

2026年2月的基准测试结果令人意外：

- Vectara 2026年研究：递归512-token分块在69%的准确率上胜出，而语义分块为54%。
- SPLADE + Mistral-8B 在 Natural Questions 数据集上：重叠没有带来任何可衡量的收益。
- 上下文悬崖：当上下文达到约2,500个token时，响应质量急剧下降。

“显而易见”的答案（语义分块、20%重叠、1000个token）往往是错的。本课将帮你建立对六种策略的直觉，并告诉你何时该使用哪一种。

## 核心概念

![六种分块策略在同一段落上的可视化](../assets/chunking.svg)

**固定分块。** 每N个字符或token分割一次。最简单的基线。会在句子中间断开。压缩效果好，但连贯性差。

**递归分块。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 分割，然后是 `\n`、`.`，最后是空格。优雅地回退。2026年的默认选择。

**语义分块。** 嵌入每个句子。计算相邻句子间的余弦相似度。在相似度低于阈值处分割。保持主题连贯性。速度较慢；有时会产生非常小的40-token片段，影响检索。

**句子分块。** 按句子边界分割。每个块一个句子或包含N个句子的窗口。在约5k token以内，效果与语义分块相当，且成本只是后者的一小部分。

**父文档分块。** 为检索存储小子块，同时为上下文存储更大的父块。通过小子块检索；返回父块。优雅地降级：即使子块效果不佳，仍能返回合理的父块。

**晚期分块（2024）。** 先在token级别嵌入整个文档，然后将token嵌入池化为块嵌入。保留了跨块上下文。适用于长上下文嵌入器（BGE-M3, Jina v3）。计算量较高。

**上下文检索（Anthropic, 2024）。** 在每个块前添加由LLM生成的摘要，说明其在文档中的位置（例如“本块是终止条款的第3.2节...”）。在Anthropic自己的基准测试中，检索效果提升了35-50%。索引成本高。

### 击败所有默认策略的法则

根据查询类型匹配块大小：

| 查询类型 | 块大小 |
|----------|--------|
| 事实型（“CEO叫什么名字？”） | 256-512 tokens |
| 分析型 / 多跳推理 | 512-1024 tokens |
| 整节理解 | 1024-2048 tokens |

NVIDIA 2026年基准测试。块应足够大以包含答案及局部上下文，又足够小，使检索器的Top-K结果能聚焦于答案而非上下文噪音。

## 动手构建

### 第一步：固定与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第二步：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域数据上调整 `threshold`。太高 → 片段化。太低 → 产生一个巨块。

### 第三步：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：对父块去重。多个子块可能映射到同一个父块；全部返回会浪费上下文空间。

### 第四步：上下文检索（Anthropic模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引经过上下文增强的块。查询时，检索能从额外的上下文信号中受益。

### 第五步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

务必进行基准测试。适合你的语料库的“最佳”策略，可能不同于任何博客文章。

## 陷阱

- **仅针对事实型查询评估分块策略。** 多跳查询会揭示出截然不同的赢家。请使用按查询类型分层的评估集。
- **语义分块未设最小尺寸。** 会产生损害检索的40-token片段。始终强制执行 `min_tokens`。
- **重叠是迷信。** 2026年研究发现，重叠通常没有收益，且会使索引成本翻倍。实测，而非假设。
- **未设置最小/最大限制。** 5个token或5000个token的块都会破坏检索。请设置上下限。
- **跨文档分块。** 绝不要让一个块跨越两个文档。始终按文档分块，然后再合并。

## 使用指南

2026年的技术栈：

| 场景 | 策略 |
|------|------|
| 首次构建，语料库未知 | 递归分块，512 tokens，无重叠 |
| 事实型问答 | 递归分块，256-512 tokens |
| 分析型 / 多跳推理 | 递归分块，512-1024 tokens + 父文档分块 |
| 强交叉引用（合同、论文） | 晚期分块或上下文检索 |
| 对话/对话语料库 | 按轮次分块 + 说话者元数据 |
| 短文本（推文、评论） | 一个文档 = 一个块 |

从递归512开始。在一个50条查询的评估集上测量recall@5。从此处开始调优。

## 部署

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 分块一个20页文档。比较块数量和边界质量。
2. **中等。** 基于5个文档构建一个30条查询的评估集。测量递归、语义和父文档分块的recall@5。哪种胜出？是否与博客文章一致？
3. **困难。** 实现上下文检索。测量相对于基线递归分块的MRR改进。报告索引成本（LLM调用次数）与准确率提升。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 块 | 文档的一部分 | 被嵌入、索引和检索的子文档单元。 |
| 重叠 | 安全边际 | 相邻块之间共享的N个token；在2026年基准测试中通常无用。 |
| 语义分块 | 智能分块 | 在相邻句子嵌入相似度下降处分割。 |
| 父文档 | 两级检索 | 检索小子块，返回更大的父块。 |
| 晚期分块 | 先嵌入再分块 | 在token级别嵌入整个文档，然后池化为块向量。 |
| 上下文检索 | Anthropic的技巧 | 索引前，为每个块添加LLM生成的摘要。 |
| 上下文悬崖 | 2500-token墙 | 在RAG中观察到上下文约为2.5k token时质量下降（2026年1月）。 |

## 扩展阅读

- [Yepes 等人 / LangChain — 递归字符分割文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认选择。
- [Vectara (2024, NAACL 2025)。分块配置分析](https://arxiv.org/abs/2410.13070) — 分块与嵌入选择同样重要。
- [Jina AI — 长上下文嵌入模型中的晚期分块 (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 晚期分块论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — 使用LLM生成的上下文前缀，检索效果提升35-50%。
- [NVIDIA 2026年块大小基准测试 — Premai 总结](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型划分的块大小。