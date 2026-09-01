# Embedding 模型 — 2026 年深度解析

> Word2Vec 给你一个词一个向量。现代 embedding 模型给你的是一整段文本一个向量，支持跨语言，并提供稀疏、稠密和多向量三种视角，大小可适配你的索引规模。选错了，你的 RAG 就会检索出错误的内容。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 5 · 03（Word2Vec），阶段 5 · 14（信息检索）
**时间：** 约 60 分钟

## 问题所在

你的 RAG 系统有 40% 的时间检索出了错误的段落。罪魁祸首很少是向量数据库或提示词，而是 embedding 模型本身。

2026 年选择 embedding 意味着在五个维度上做出权衡：

1. **稠密 vs 稀疏 vs 多向量。** 每段文本一个向量，还是每个 token 一个向量，还是带权重的稀疏词袋。
2. **语言覆盖范围。** 纯英语模型在纯英语任务上仍然更强；混合语料时多语言模型更优。
3. **上下文长度。** 512 token vs 8,192 token vs 32,768 token——而实际有效容量通常是标称最大值的 60%-70%。
4. **维度预算。** 3,072 个 float 全精度 = 每个向量 12 KB。1 亿个向量时存储成本约 1,300 美元/月。Matryoshka 截断可将此压缩 4 倍。
5. **开源 vs 托管。** 开源权重意味着你掌控栈和数据。托管意味着用控制权换取始终最新版本。

本课程命名这些权衡，让你基于证据做选择，而不是追上当季流行款。

## 概念

![稠密、稀疏和多向量 embeddings](../assets/embedding-modes.svg)

**稠密 embeddings。** 每段文本一个向量（通常为 384-3,072 维）。余弦相似度按语义相近程度对段落排序。代表作：OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认选择。

**稀疏 embeddings。** SPLADE 风格。Transformer 为每个词表 token 预测一个权重，然后将大部分置零。结果是一个大小为 \|vocab\| 的稀疏向量。捕获词汇匹配（类似 BM25），但带有学习得到的词项权重。对关键词密集查询效果强劲。

**多向量（晚期交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。评分使用 MaxSim：对每个查询 token，找到最相似的文档 token，将分数求和。存储和评分开销更高，但在长查询和领域特定语料上表现更优。

**BGE-M3：三者合一。** 单个模型同时输出稠密、稀疏和多向量表示。每种可以独立查询；分数通过加权求和融合。当你希望从单一 checkpoint 中获得灵活性的 2026 年首选。

**Matryoshka 表示学习。** 训练后，向量的前 N 个维度本身就能形成有用的独立 embedding。将 1,536 维向量截断到 256 维，仅损失约 1% 准确率，却获得 6 倍的存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 均支持。

### MTEB 排行榜只讲了一半的故事

Massive Text Embedding Benchmark——发布时涵盖 56 个任务、8 种任务类型，MTEB v2 已扩展至 100+ 任务。2026 年初，Gemini Embedding 2 在检索方面领先（MTEB-R 得分 67.71），Cohere embed-v4 在通用方面领先（MTEB 得分 65.2），BGE-M3 在开源多语言方面领先（63.0）。排行榜是必要但不充分的——务必在你的领域上自行基准测试。

### 三层模式

| 用例 | 模式 |
|------|------|
| 快速初筛 | 稠密双编码器（BGE-M3、text-3-small） |
| 召回提升 | 稀疏（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| 前 50 精确度 | 多向量（ColBERTv2）或 cross-encoder 重排序器 |

大多数生产环境栈同时使用这三种。

```figure
gx-matryoshka
```

## 动手实现

### 步骤 1：基线——用 Sentence-BERT 做稠密 embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度。务必设置它。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 的训练方式使得对前几个层级而言这是无损的。非 Matryoshka 模型（原始 Sentence-BERT）被截断时性能急剧下降。

### 步骤 3：BGE-M3 多函数能力

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。分数融合：

```python
dense_score = ... # dense_vecs 的余弦相似度
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

根据你的领域调整权重。

### 步骤 4：在自定义任务上运行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*代表性*子集上运行候选模型。不要仅信赖排行榜排名——你的领域才重要。

### 步骤 5：从零手推余弦相似度

见 `code/main.py`。平均哈希技巧 embeddings（仅 stdlib）。无法与 transformer embeddings 竞争，但展示了基本形状：分词 → 向量 → 归一化 → 点积。

## 常见陷阱

- **查询和文档使用同一模型。** 某些模型（Voyage、Jina-ColBERT）使用非对称编码——查询和文档通过不同的路径。务必检查模型说明卡。
- **缺少前缀。** `bge-*` 模型需要在查询前加上 `"Represent this sentence for searching relevant passages: "`。遗漏会导致 3-5 个百分点的召回率差距。
- **过度裁剪 Matryoshka。** 1,536 → 256 通常安全。1,536 → 64 则不行。在你的评估集上验证。
- **上下文截断。** 大多数模型会对超过最大长度的输入静默截断。长文档需要分块（见课程 23）。
- **忽略延迟尾部。** MTEB 分数会掩盖 p99 延迟。一个 600M 模型可能比 335M 模型高出 2 分，但每次查询成本高 3 倍。

## 应用场景

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 仅英语、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集查询 | 添加 SPLADE 稀疏，与稠密做 RRF 融合 |

2026 年推荐模式：从 BGE-M3 或 text-3-large 起步，在 MTEB 上按你的领域评估，若有领域专用模型领先超过 3 分则替换。

## 交付产物

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: 为给定语料和部署场景选择 embedding 模型、维度和检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定语料（规模、语言、领域、平均长度）、部署目标（云端 / 边缘 / 本地）、延迟预算和存储预算，输出：

1. 模型。指定 checkpoint 或 API，附一句话理由。
2. 维度。全维度 / Matryoshka 截断 / int8 量化。理由与存储预算挂钩。
3. 模式。稠密 / 稀疏 / 多向量 / 混合。理由。
4. 如模型说明卡要求，提供查询前缀 / 模板。
5. 评估计划。与领域相关的 MTEB 任务 + 保留领域评估（nDCG@10）。

拒绝建议将 Matryoshka 截断至 <64 维且未经领域验证的情况。拒绝在小于 1 万段的语料上使用 ColBERTv2（开销不合理）。对超 8k token 的长文档语料路由到 512 token 窗口的模型时发出警告。
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 以全维度（384）和 Matryoshka 128 两种模式对 100 个句子做编码。在 10 个查询上测量 MRR 下降幅度。
2. **中等。** 在你的领域 500 个段落上对比 BGE-M3 的稠密、稀疏和 colbert 模式。哪种在 recall@10 上胜出？RRF 融合是否胜过最优单一模式？
3. **困难。** 在你的 top-2 领域任务上对三个候选模型运行 MTEB。报告 MTEB 分数、100 查询批次的 p99 延迟和每百万查询成本。选出 Pareto 最优的那个。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| 稠密 embedding | 那个向量 | 每段文本一个固定大小的向量。用余弦相似度排序。 |
| 稀疏 embedding | 学习版 BM25 | 每个词表 token 一个权重；大部分为零；端到端训练。 |
| 多向量 | ColBERT 风格 | 每个 token 一个向量；MaxSim 评分；索引更大，召回更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身就是一个有效的小维度 embedding。 |
| MTEB | 那个基准 | Massive Text Embedding Benchmark——发布时 56 个任务，v2 中 100+。 |
| BEIR | 检索基准 | 18 个零样本检索任务；常被引用以证明跨领域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 生产环境中的晚期交互。
- [Hugging Face 上的 MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。
