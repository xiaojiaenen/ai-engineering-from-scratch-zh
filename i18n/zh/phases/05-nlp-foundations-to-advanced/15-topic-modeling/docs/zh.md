# 主题建模 — LDA 和 BERTopic

> LDA：文档是主题的混合物，主题是词的分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。目标相同，分解方式不同。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 02（BoW + TF-IDF）、Phase 5 · 03（Word2Vec）
**时间：** 约 45 分钟

## 问题所在

你手上有 10,000 条客户支持工单、50,000 篇新闻文章，或 200,000 条推文。你需要了解整个语料的主题，却不想逐篇阅读。没有标注好的类别，甚至不知道存在多少类别。

主题建模在无监督条件下解决这个问题。给它一个语料库，它会返回一组语义连贯的主题，以及每篇文档在这些主题上的分布。

两种算法家族占据主导地位。LDA（2003）把每个文档视为潜在主题的混合，每个主题视为词的分布。推断采用贝叶斯方法。当需要混合成员的主题分配和可解释的词级概率分布时，它仍在生产中广泛使用。

BERTopic（2020）用 BERT 编码文档，通过 UMAP 降维，用 HDBSCAN 聚类，再基于类 TF-IDF 提取主题词。它在短文本、社交媒体和语义相似度比词汇重叠更重要的场景下表现更优。每篇文档只分配一个主题，这对长文本而言是一种局限。

本课为两者建立直觉，并告诉你针对给定语料该选哪个。

## 概念

![LDA 混合模型 vs BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 生成故事。** 每个主题是词的分布，每个文档是主题的混合。要生成文档中的一个词：先从文档的主题混合中采样一个主题，再从该主题的分布中采样一个词。推断反过来做：根据观察到的词，推断每篇文档的主题分布和每个主题的词分布。Collapsed Gibbs 采样或变分贝叶斯完成这一数学推导。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合比例）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的词分布）。

**BERTopic 流程。**

1. 用句子变换器编码每篇文档（如 `all-MiniLM-L6-v2`），得到 384 维向量。
2. 用 UMAP 将维度降至约 5 维。BERT 嵌入维数太高，不适合直接聚类。
3. 用 HDBSCAN 聚类。基于密度，能生成大小不一的簇，并给出"异常值"标签。
4. 对每个簇，在簇内文档上计算类 TF-IDF，提取最高分的词作为主题词。

输出是每篇文档一个主题（外加 -1 异常值标签）。可选地，通过 HDBSCAN 的概率向量得到软分配。

```figure
topic-drift
```

## 动手构建

### 步骤 1：通过 scikit-learn 拟合 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"主题 {idx}: {' '.join(words)}")
```

注意：移除了停用词，`min_df` 和 `max_df` 过滤掉稀有和普遍出现的词，使用 CountVectorizer（而非 TfidfVectorizer）因为 LDA 期望的是原始词频计数。

### 步骤 2：BERTopic（生产级用法）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"主题 {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤用于排除 BERTopic 的异常值桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 库默认值为 10。本示例将其显式设为 15 以匹配本课规模。对于超过 10,000 篇文档的语料库，应将其提高到 50 或 100。

### 步骤 3：评估

两种方法都输出主题词。关键问题是这些词是否真正凝聚。

- **主题一致性（c_v）。** 在滑动窗口上下文中计算 Top 词对的 NPMI（归一化点互信息），将这些得分聚合为主题向量，再通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel` 并指定 `coherence="c_v"`。
- **主题多样性。** 所有主题 Top 词中唯一词的比例。越高越好（主题之间不重叠）。
- **定性检查。** 阅读每个主题的 Top 词，判断它们是否真正命名了某个有意义的概念。人工判断仍是最后一道防线。

## 如何选择

| 场景 | 推荐 |
|------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 包含主题混合的长文档 | LDA |
| 无 GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级的多主题分布 | LDA |
| 集成 LLM 进行主题命名 | BERTopic（原生支持） |
| 资源受限的边缘部署 | LDA |
| 追求最高语义一致性 | BERTopic |

实际选型最大的考虑因素是文档长度。BERT 嵌入有截断限制；LDA 词频计数不受长度影响。对于超出嵌入模型上下文窗口的长文档，要么分块后再聚合，要么改用 LDA。

## 如何使用

2026 年的工具栈：

- **BERTopic。** 短文本及语义重要的场景的默认选择。
- **`gensim.models.LdaModel`。** 经典 LDA，生产级成熟方案。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 便于实验的快速 LDA。
- **NMF。** 非负矩阵分解。LDA 的快速替代方案，短文本质量相当。
- **Top2Vec。** 设计与 BERTopic 类似。社区较小，但在某些基准上表现良好。
- **FASTopic。** 较新方法，在超大语料上比 BERTopic 更快。
- **LLM 主题命名。** 先用任意聚类方法得出簇，再让模型为每个簇命名。

## 成果交付

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: 为语料库选择 LDA 或 BERTopic。指定库、超参数和评估方法。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定语料库描述（文档数量、平均长度、领域、语言、计算预算），输出：

1. 算法。LDA / NMF / BERTopic / Top2Vec / FASTopic。一句话说明理由。
2. 配置。主题数量：`recommended = max(5, round(sqrt(n_docs)))`，40,000 篇以下的语料库上限为 200；仅在语料库真正大于 40,000 篇时才允许超过 200，并注明增加的计算开销。`min_df` / `max_df` 过滤条件和神经网络方法的嵌入模型也需列出。
3. 评估。通过 `gensim.models.CoherenceModel` 计算主题一致性（c_v）、主题多样性，以及 20 个样本的人工检查。
4. 需排查的故障模式。LDA：垃圾主题吸收停用词和频繁词。BERTopic：-1 异常值簇吞掉模棱两可的文档。

若文档长度超出嵌入模型的上下文窗口且无分块策略，则拒绝使用 BERTopic。若文本极短（推文、不足 10 词的评论），则拒绝使用 LDA，因为一致性会崩溃。将 n_topics 低于 5 的选择标记为很可能有误；将 40,000 篇以下语料库中超过 200 个主题标记为很可能过度细分。
```

## 练习

1. **入门。** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA，打印每个主题的 Top 10 词。手动为每个主题贴标签。算法是否找到了真实的类别？
2. **进阶。** 在同一份 20 Newsgroups 子集上拟合 BERTopic，比较找到的主题数量、Top 词和定性一致性，与 LDA 对照。哪种方法更清晰地还原了真实类别？
3. **挑战。** 在你的语料上分别计算 LDA 和 BERTopic 的 c_v 一致性。分别以 5、10、20、50 个主题运行，绘制一致性随主题数量变化的曲线。报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 主题（Topic） | 语料谈论的东西 | 词的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| 混合成员（Mixed membership） | 文档属于多个主题 | LDA 为每篇文档分配所有主题上的分布。 |
| UMAP | 降维 | 流形学习方法，保留局部结构；用于 BERTopic。 |
| HDBSCAN | 基于密度的聚类 | 发现大小不一的簇；对异常值输出"噪声"标签（-1）。 |
| c_v 一致性 | 主题质量度量 | 滑动窗口内主题 Top 词的平均点互信息。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 原始论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 原始论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 及其相关度量的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产环境参考文档，示例优秀。
