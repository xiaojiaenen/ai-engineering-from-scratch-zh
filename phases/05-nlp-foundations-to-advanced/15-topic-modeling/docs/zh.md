# 主题建模 — LDA 与 BERTopic

> LDA：文档是主题的混合体，主题是词上的概率分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。目标相同，基本原理不同。

**类型：** 学习  
**语言：** Python  
**前置要求：** 第 5 阶段 · 02（词袋 + TF-IDF），第 5 阶段 · 03（Word2Vec）  
**时间：** ~45 分钟

## 问题

你有一万张客服工单、五万篇新闻文章，或二十万条推文。你需要在不阅读的情况下了解这个集合的主题。你没有标注的类别标签，甚至不知道存在多少个类别。

主题建模可以在无监督条件下回答这个问题。给它一个语料库，它会返回一组数量不多的连贯主题，并给出每篇文档在这些主题上的分布。

两种算法家族占主导地位。LDA（2003年）将每篇文档视为潜在主题的混合体，每个主题是词上的概率分布。推理是贝叶斯式的。它仍然部署在需要混合主题分配和可解释词级概率分布的生产环境中。

BERTopic（2020年）使用BERT编码文档，用UMAP降维，用HDBSCAN聚类，并通过基于类的TF-IDF提取主题词。它在短文本、社交媒体以及语义相似性比词重叠更重要的任何场景中表现更佳。一篇文档对应一个主题，这对于长篇内容是一个限制。

本课程旨在建立对两者的直觉，并指导如何为给定语料库选择合适的方法。

## 核心概念

![LDA混合模型 vs BERTopic聚类](../assets/topic-modeling.svg)

**LDA生成故事。** 每个主题是词上的概率分布。每篇文档是主题的混合体。要生成文档中的一个词，先从文档的主题混合体中采样一个主题，然后从该主题的概率分布中采样一个词。推理则反向进行：根据观测到的词，推断每篇文档的主题分布和每个主题的词分布。坍缩吉布斯采样或变分贝叶斯方法执行相关计算。

LDA的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行之和为1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行之和为1（主题的词分布）。

**BERTopic流程。**

1. 使用句子转换器（如 `all-MiniLM-L6-v2`）编码每篇文档。生成384维向量。
2. 使用UMAP将维度降至约5维。BERT嵌入对于聚类而言维度过高。
3. 使用HDBSCAN进行聚类。基于密度，产生大小可变的聚类和一个"离群值"标签。
4. 对于每个聚类，计算其文档上的基于类的TF-IDF以提取顶级词。

输出是每篇文档一个主题（加上一个-1离群标签）。可选地，可以通过HDBSCAN的概率向量获得软成员资格。

## 动手构建

### 步骤1：通过scikit-learn实现LDA

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
        print(f"topic {idx}: {' '.join(words)}")
```

注意：移除了停用词，min_df和max_df过滤了罕见和过于普遍的词，使用CountVectorizer（而非TfidfVectorizer），因为LDA期望原始计数。

### 步骤2：BERTopic（生产环境）

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
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤丢弃了BERTopic的离群桶（HDBSCAN无法聚类的文档）。`min_topic_size` 控制HDBSCAN的最小聚类大小；BERTopic库的默认值是10。本示例为了课程规模显式设置为15。对于超过10,000篇文档的语料库，请增加到50或100。

### 步骤3：评估

两种方法都输出主题词。问题在于这些词是否连贯。

- **主题一致性（c_v）。** 组合滑动窗口上下文中的顶级词对之间的NPMI（归一化逐点互信息），将分数聚合为主题向量，并通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel` 和 `coherence="c_v"`。
- **主题多样性。** 所有主题顶级词中独特词的比例。越高越好（主题重叠度低）。
- **定性检查。** 阅读每个主题的顶级词。它们是否指代了真实事物？人工判断仍是最后一道防线。

## 何时选择哪种方法

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 包含主题混合的长文档 | LDA |
| 无GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| LLM集成进行主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大语义连贯性 | BERTopic |

最重要的实际考虑因素是文档长度。BERT嵌入会截断；LDA的词频统计适用于任何长度的文本。对于超过嵌入模型上下文长度的文档，要么分块聚合，要么使用LDA。

## 工具选择

2026年技术栈：

- **BERTopic。** 短文本和任何语义重要场景的默认选择。
- **`gensim.models.LdaModel`。** 经典LDA用于生产环境，成熟，久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 易于用于实验的LDA实现。
- **NMF。** 非负矩阵分解。LDA的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 设计与BERTopic类似。社区较小，但在某些基准测试中表现良好。
- **FASTopic。** 较新，在非常大的语料库上比BERTopic更快。
- **基于LLM的标注。** 运行任何聚类，然后提示模型为每个聚类命名。

## 部署

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **简单。** 在20 Newsgroups数据集上用5个主题拟合LDA。打印每个主题的前10个词。人工标注每个主题。算法是否找到了真实的类别？
2. **中等。** 在相同的20 Newsgroups子集上拟合BERTopic。比较发现的主题数量、顶级词和定性连贯性与LDA的结果。哪种方法更清晰地呈现了真实类别？
3. **困难。** 计算你的语料库上LDA和BERTopic的c_v一致性。分别用5、10、20、50个主题运行。绘制一致性与主题数量的关系图。报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 主题 | 语料库所关于的事物 | 词上的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| 混合成员资格 | 文档属于多个主题 | LDA为每篇文档分配一个涵盖所有主题的概率分布。 |
| UMAP | 降维 | 保持局部结构的流形学习；用于BERTopic。 |
| HDBSCAN | 密度聚类 | 发现大小可变的聚类；为离群点生成"噪声"标签（-1）。 |
| c_v一致性 | 主题质量度量 | 滑动窗口内顶级主题词的平均逐点互信息。 |

## 扩展阅读

- [Blei, Ng, Jordan (2003). 潜在狄利克雷分配](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA论文。
- [Grootendorst (2022). BERTopic：基于类TF-IDF的神经主题建模](https://arxiv.org/abs/2203.05794) — BERTopic论文。
- [Röder, Both, Hinneburg (2015). 探索主题一致性度量空间](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入了c_v及相关方法的论文。
- [BERTopic文档](https://maartengr.github.io/BERTopic/) — 生产环境参考。示例优秀。