# 嵌入模型 — 2026深度解析

> Word2Vec 为你提供每个词一个向量。现代嵌入模型为你提供每个段落一个向量、跨语言支持、稀疏、密集和多向量视图，并可调整尺寸以适配你的索引。选择错误会导致你的RAG检索到错误内容。

**类型：** 学习
**语言：** Python
**先决条件：** 阶段5 · 03 (Word2Vec)，阶段5 · 14 (信息检索)
**时间：** 约60分钟

## 问题所在

你的RAG系统有40%的时间检索到了错误的段落。罪魁祸首很少是向量数据库或提示词。问题出在嵌入模型上。

在2026年选择嵌入模型意味着要从五个维度进行权衡：

1. **密集 vs 稀疏 vs 多向量。** 每个段落一个向量，或每个token一个向量，还是一个稀疏的加权词袋。
2. **语言覆盖范围。** 单语英语模型在纯英语任务上仍然占优。多语言模型则在语料库混杂时胜出。
3. **上下文长度。** 512 tokens vs 8,192 vs 32,768 —— 而且实际有效容量通常是广告最大容量的60-70%。
4. **维度预算。** 全精度下的3,072个浮点数 = 每个向量12 KB。当有1亿个向量时，存储成本为每月1,300美元。Matryoshka截断可将此成本降低4倍。
5. **开源 vs 托管。** 开源权重意味着你控制整个技术栈和数据。托管服务则意味着你用控制权换取最新技术。

本课将阐明这些权衡，以便你能基于证据做出选择，而非仅凭上个季度的流行趋势。

## 核心概念

![密集、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**密集嵌入。** 每个段落一个向量（通常384-3,072维度）。余弦相似度根据语义接近度对段落进行排序。OpenAI `text-embedding-3-large`、BGE-M3密集模式、Voyage-3。默认选择。

**稀疏嵌入。** SPLADE风格。一个Transformer为词汇表中的每个token预测一个权重，然后将大部分置零。结果是一个大小为|词汇表|的稀疏向量。它捕获了词汇匹配（类似BM25），但使用学习到的词项权重。在关键词密集型查询上表现强劲。

**多向量（晚期交互）。** ColBERTv2、Jina-ColBERT。每个token一个向量。使用MaxSim进行评分：为每个查询token找到最相似的文档token，然后将分数相加。存储和评分成本更高，但在长查询和特定领域语料库上表现更优。

**BGE-M3：三者兼备。** 单个模型同时输出密集、稀疏和多向量表示。每种表示都可以独立查询；分数通过加权和进行融合。当你希望从一个检查点获得灵活性时，这是2026年的默认选择。

**Matryoshka表示学习。** 训练使得向量的前N个维度自身就构成一个有用的独立嵌入。将一个1,536维的向量截断为256维，可以以约1%的精度损失换取6倍的存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+都支持此特性。

### MTEB排行榜只能反映部分情况

大规模文本嵌入基准测试 —— 启动时（2022年）包含56个任务，横跨8种任务类型，在MTEB v2中扩展到100+个任务。在2026年初，Gemini Embedding 2在检索任务上领先（67.71 MTEB-R）。Cohere embed-v4在通用任务上领先（65.2 MTEB）。BGE-M3在开源权重多语言任务上领先（63.0）。排行榜是必要的，但并非充分条件 —— 务必在你自己的领域数据上进行基准测试。

### 三层模式

| 用例 | 模式 |
|----------|---------|
| 快速首轮筛选 | 密集双编码器 (BGE-M3, text-3-small) |
| 召回率提升 | 稀疏 (SPLADE, BGE-M3 sparse) + RRF融合 |
| 前50条结果精度优化 | 多向量 (ColBERTv2) 或 交叉编码器重排序器 |

大多数生产系统栈会同时使用这三种模式。

## 构建实现

### 步骤1：基线 —— 使用Sentence-BERT的密集嵌入

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

`normalize_embeddings=True` 使点积等于余弦相似度。务必进行设置。

### 步骤2：Matryoshka截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3和Voyage-4都经过训练，因此在前几个层级上截断是无损的。非Matryoshka模型（如原始的Sentence-BERT）在截断时性能会急剧下降。

### 步骤3：BGE-M3多功能性

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
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你自己的领域数据上调整权重。

### 步骤4：在自定义任务上进行MTEB评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在你的*代表性*子集上运行候选模型。不要只相信排行榜排名 —— 你的领域至关重要。

### 步骤5：手动实现余弦相似度

参见 `code/main.py`。使用平均哈希技巧嵌入（仅使用标准库）。其效果无法与Transformer嵌入相比，但它展示了基本流程：分词 → 向量化 → 归一化 → 点积。

## 陷阱与注意事项

- **对查询和文档使用相同模型。** 一些模型（Voyage、Jina-ColBERT）使用非对称编码 —— 查询和文档通过不同的路径处理。务必检查模型卡。
- **缺少前缀。** `bge-*` 模型需要在查询前添加 `"Represent this sentence for searching relevant passages: "`。如果忘记添加，召回率会有3-5个百分点的差距。
- **过度截断Matryoshka。** 1,536 → 256 通常是安全的。1,536 → 64 则不一定。请在你的评估集上进行验证。
- **上下文截断。** 大多数模型会默默截断超过其最大长度的输入。长文档需要进行分块（参见第23课）。
- **忽视延迟尾部。** MTEB分数掩盖了p99延迟。一个600M参数的模型可能比一个335M参数的模型高出2分，但每次查询的成本可能高出3倍。

## 使用建议

2026年的技术栈选择：

| 场景 | 选择 |
|-----------|------|
| 纯英语，要求快速，使用API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重，英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重，多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+ tokens） | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| 仅CPU部署 | Nomic Embed v2 (137M参数，MoE架构) |
| 存储受限 | Matryoshka截断 + int8量化 |
| 关键词密集型查询 | 添加SPLADE稀疏模式，并与密集模式进行RRF融合 |

2026年模式：从BGE-M3或text-3-large开始，用MTEB在你自己的领域数据上进行评估，如果某个领域特定模型优势超过3分，则进行替换。

## 部署上线

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 使用 `bge-small-en-v1.5` 将100个句子分别编码为全维度（384）和Matryoshka 128维度。在10个查询上衡量MRR的下降。
2. **中等。** 在你领域数据的500个段落上，比较BGE-M3的密集、稀疏和colbert模式。哪个在recall@10上胜出？RRF融合是否优于最佳单一模式？
3. **困难。** 在你领域前2个任务上，对三个候选模型运行MTEB评估。报告MTEB分数、在100个查询批次上的p99延迟，以及每百万次查询的成本（美元）。选择帕累托最优的那个模型。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|------|-----------------|-----------------------|
| 密集嵌入 | 那个向量 | 每个文本一个固定大小的向量。使用余弦相似度进行排序。 |
| 稀疏嵌入 | 学习到的BM25 | 每个词汇表token一个权重；大部分为零；端到端训练。 |
| 多向量 | ColBERT风格 | 每个token一个向量；MaxSim评分；索引更大，召回率更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前N个维度自身就是一个有效的、更小的嵌入。 |
| MTEB | 那个基准测试 | 大规模文本嵌入基准测试 —— 启动时56个任务，v2版本100+个任务。 |
| BEIR | 那个检索基准测试 | 18个零样本检索任务；常用于评估跨领域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 统一的三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 生产环境中的晚期交互。
- [Hugging Face 上的 MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。