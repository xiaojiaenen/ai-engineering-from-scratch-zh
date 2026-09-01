# RAG 的分块策略

> 分块配置对检索质量的影响，不亚于嵌入模型的选择（Vectara NAACL 2025）。分块如果出错，再多的重排序也救不了你。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 14（信息检索）、阶段 5 · 22（嵌入模型）
**耗时：** 约 60 分钟

## 问题所在

你把一份 50 页的合同放入 RAG 系统。用户问："终止条款是什么？"检索器返回了封面页。为什么？因为模型是在 512 token 的分块上训练的，而终止条款在第 20 页，跨越了分页符，且没有任何局部关键词将其与查询联系起来。

解决方案不是"买更好的嵌入模型"。解决方案是分块。要多大？多大重叠？在哪里拆分？是否需要周围上下文？

2026 年 2 月的基准测试显示出令人惊讶的结果：

- Vectara 的 2026 年研究：递归 512 token 分块在准确率上击败了语义分块（69% → 54%）。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠未提供可测量的收益。
- 上下文悬崖：当上下文达到约 2,500 token 时，响应质量急剧下降。

"显而易见的"答案（语义分块、20% 重叠、1000 token）往往是不对的。本课将帮助你建立对六种策略的直觉，并告诉你何时选择哪种策略。

## 概念

![一个段落上展示的六种分块策略](../assets/chunking.svg)

**固定分块。** 每 N 个字符或 token 拆分一次。最简单的基线。会切断句子中间。压缩效果好，但连贯性差。

**递归分块。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 拆分，然后 `\n`，然后 `.`，然后空格。回退清晰。这是 2026 年的默认选择。

**语义分块。** 对每个句子进行嵌入。计算相邻句子之间的余弦相似度。在相似度低于阈值的地方拆分。能保持主题连贯性。速度较慢；有时会产生损害检索的 40 token 微小片段。

**句子分块。** 在句子边界处拆分。每个分块一个句子或 N 个句子的窗口。以极低的成本实现与语义分块相似的效果，直到约 5k token。

**父文档分块。** 存储较小的子分块用于检索，同时存储较大的父分块用于上下文。通过子分块检索；返回父分块。容错性好：糟糕的子分块仍能返回合理的父分块。

**延迟分块（2024 年）。** 先在 token 级别嵌入整个文档，然后将 token 嵌入聚合并入分块嵌入。保留跨分块的上下文。与长上下文嵌入器配合使用（BGE-M3、Jina v3）。计算成本更高。

**上下文检索（Anthropic，2024 年）。** 在每次索引前，用 LLM 生成的摘要前置到每个分块（"此分块是终止条款的第 3.2 节..."）。Anthropic 内部基准测试显示检索提升 35-50%。索引成本较高。

### 超越所有默认值的一条规则

将分块大小与查询类型匹配：

| 查询类型 | 分块大小 |
|------------|-----------|
| 事实型（"CEO 的名字是什么？"） | 256-512 token |
| 分析型 / 多跳 | 512-1024 token |
| 整节理解 | 1024-2048 token |

NVIDIA 的 2026 年基准测试。分块应足够大以包含答案及其局部上下文，又足够小以使检索器的 top-K 聚焦于答案而非上下文噪声。

```figure
n5-chunk-cuts
```

## 动手构建

### 步骤 1：固定分块与递归分块

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

### 步骤 2：语义分块

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

根据你的领域调整 `threshold`。太高 → 碎片化。太低 → 一个巨大的分块。

### 步骤 3：父文档分块

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

关键洞察：去重父分块。多个子分块可能映射到同一个父分块；全部返回会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
以下是需要定位的分块：<chunk>{c}</chunk>
请撰写 50-100 字的文字，将该分块置于文档的上下文中进行说明。"""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

对上下文化的分块建立索引。查询时，检索能从额外的周围信号中受益。

### 步骤 5：评估

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

始终进行基准测试。你的语料库上"最佳"的策略可能与任何博客文章不符。

## 常见陷阱

- **仅在事实型查询上评估分块。** 多跳查询会揭示截然不同的胜出者。使用按查询类型分层的评估集。
- **语义分块未设置最小尺寸。** 会产生 40 token 的碎片，损害检索。始终强制设置 `min_tokens`。
- **把重叠当作教条。** 2026 年的研究发现，重叠通常不提供任何收益，且会使索引成本翻倍。测量，不要假设。
- **没有最小/最大限制。** 5 token 或 5000 token 的分块都会破坏检索。进行钳制。
- **跨文档分块。** 永远不要让一个分块跨越两个文档。始终按文档分块，然后合并。

## 使用指南

2026 年的技术栈：

| 场景 | 策略 |
|-----------|----------|
| 初次构建，语料库未知 | 递归分块，512 token，无重叠 |
| 事实型问答 | 递归分块，256-512 token |
| 分析型 / 多跳 | 递归分块，512-1024 token + 父文档分块 |
| 大量交叉引用（合同、论文） | 延迟分块或上下文检索 |
| 对话 / 对话语料 | 话轮级分块 + 说话人元数据 |
| 短 utterances（推文、评论） | 一个文档 = 一个分块 |

从递归 512 开始。在 50 查询的评估集上测量 recall@5。然后在此基础上微调。

## 交付物

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: 针对给定语料库和查询分布，选择分块策略、大小和重叠率。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

给定语料库（文档类型、平均长度、领域）和查询分布（事实型 / 分析型 / 多跳），输出：

1. 策略。递归 / 句子 / 语义 / 父文档 / 延迟 / 上下文。说明原因。
2. 分块大小。token 数量。与查询类型关联的理由。
3. 重叠。默认为 0；如果大于 0，请说明理由。
4. 最小/最大限制。`min_tokens`、`max_tokens` 保护机制。
5. 评估方案。在 50 查询的分层评估集（事实型、分析型、多跳）上计算 recall@5。

拒绝任何未实施最小/最大分块尺寸限制的请求。拒绝在未经消融实验证明有效的情况下使用超过 20% 的重叠。标记未设置最小 token 下限的语义分块建议。
```

## 练习

1. **简单。** 使用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 对一份 20 页的文档进行分块。比较分块数量和边界质量。
2. **中等。** 在 5 份文档上构建 30 查询的评估集。测量 recursive、semantic 和 parent-document 的 recall@5。哪个胜出？是否与博客文章一致？
3. **困难。** 实现上下文检索。测量相对于基线 recursive 的 MRR 提升。报告索引成本（LLM 调用次数）与准确性提升。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------------|-----------------------|
| 分块 (Chunk) | 文档的一部分 | 被嵌入、索引和检索的子文档单元。 |
| 重叠 (Overlap) | 安全边际 | 相邻分块之间共享的 N 个 token；在 2026 年的基准测试中往往无用。 |
| 语义分块 | 智能分块 | 在相邻句子嵌入相似度下降的位置进行拆分。 |
| 父文档 | 两级检索 | 检索小分块，返回较大的父分块。 |
| 延迟分块 | 分块在嵌入之后 | 在 token 级别嵌入完整文档，再聚合并入分块向量。 |
| 上下文检索 | Anthropic 的技巧 | 在索引前将 LLM 生成的摘要前置到每个分块。 |
| 上下文悬崖 | 2500-token 墙 | 在 RAG 中观察到的质量下降，发生在约 2.5k 上下文 token 附近（2026 年 1 月）。 |

## 延伸阅读

- [Yepes 等人 / LangChain — 递归字符拆分文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认选择。
- [Vectara（2024，NAACL 2025）。分块配置分析](https://arxiv.org/abs/2410.13070) — 分块与嵌入选择一样重要。
- [Jina AI — 长上下文嵌入模型中的延迟分块（2024）](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 延迟分块论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — 通过 LLM 生成的上下文前缀实现 35-50% 的检索提升。
- [NVIDIA 2026 分块大小基准测试 — Premai 总结](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型的分块大小。
