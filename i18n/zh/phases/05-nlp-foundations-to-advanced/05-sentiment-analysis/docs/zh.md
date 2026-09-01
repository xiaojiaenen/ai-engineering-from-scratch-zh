# 情感分析

> 经典的 NLP 任务。古典文本分类中你需要了解的大部分内容都会在这里出现。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段 · 02（BoW + TF-IDF）、第2阶段 · 14（朴素贝叶斯）
**时长：** 约75分钟

## 问题描述

"The food was not great." 正面还是负面？

情感分析听起来很简单。评论者表达了喜欢或不喜欢某物，给句子打标签即可。但它成为经典 NLP 任务的原因在于，每个看似简单的案例背后都隐藏着困难。否定会翻转含义，讽刺会反转它，"not bad at all" 带有正面含义，尽管包含两个负向编码的词。表情符号携带的 signal 比周围文本更多。领域词汇很重要（`tight` 在音乐评论中与在时尚评论中的含义不同）。

情感分析是古典 NLP 的实战实验室。如果你理解为什么每个朴素基线都有特定的失败模式，你就理解了为什么需要发明更复杂的模型。本课将从零构建朴素贝叶斯基线，添加逻辑回归，并指出让生产级情感分析成为合规问题的陷阱。

## 概念

古典情感分析是两个步骤的流程。

1. **表示。** 将文本转为特征向量。BoW、TF-IDF 或 n-gram。
2. **分类。** 在有标注数据上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是"最笨"但有效的模型。假设给定标签后每个特征相互独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时，将概率相乘。"naive" 独立性假设荒谬得可笑，但结果却令人惊讶地强。原因在于：对于稀疏文本特征和中等规模数据，分类器更关心的是每个词偏向哪一边，而不是偏向多少。

逻辑回归修正了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为 bigram 特征会得到负权重。朴素贝叶斯无法对从未标注过的 bigrams 做到这一点。

```figure
sentiment-logits
```

## 构建

### 步骤1：一个真实的小数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意设置得很小。实际工作使用数万条数据（IMDb、SST-2、Yelp polarity）。数学原理相同。

### 步骤2：从零实现的多元朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）即拉普拉斯平滑。不加它会使得在某个类别中未出现的词概率为零，导致对数爆炸。`alpha=0.01` 在实践中很常见。`alpha=1.0` 是教学默认值。

### 步骤3：从零实现的逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在此很重要。文本特征是稀疏的；没有 L2 时模型会死记训练样本。从 `0.01` 开始并调优。

### 步骤4：处理否定（失败模式）

考虑 "not good" 和 "not bad"。BoW 分类器看到 `{not, good}` 和 `{not, bad}`，并从训练中较频繁出现的组合中学习。bigram 分类器看到 `not_good` 和 `not_bad` 并将它们学习为不同的特征。这通常就足够了。

另一种粗糙但有效的修复（当你没有 bigrams 时）：**否定范围界定**。在否定词之后的 token 前加 `NOT_` 前缀，直到下一个标点为止。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们赋予相反的权重。三行预处理代码，在情感基准测试上可带来可测量的准确率提升。

### 步骤5：有意义的评估指标

如果类别不平衡，仅报告准确率会具有误导性。真实的情感语料库通常是 70-80% 正面或 70-80% 负面；常量多数分类器即可获得 80% 准确率，但毫无用处。报告以下所有指标：

- **每类精确率和召回率。** 每个类别一对。宏平均得到尊重类别平衡的单一数值。
- **宏平均 F1（不平衡数据的主要指标）。** 每类 F1 分数的均值，等权重。在类别不平衡时使用此指标代替准确率。
- **加权 F1（备选）。** 与宏平均类似，但按类别频率加权。当不平衡本身具有业务含义时，与宏平均 F1 一起报告。
- **混淆矩阵。** 原始计数。在信任任何标量指标之前始终先检查它；它能揭示模型混淆了哪对类别。
- **每类错误样本。** 每个类别提取 5 个错误预测。阅读它们。没有什么比阅读实际错误更能发现问题。

对于严重不平衡的数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC** 代替准确率。AUPRC 对少数类更敏感，而这通常是你关心的（垃圾邮件、欺诈、罕见情感）。

**需要避免的常见 bug。** 在不平衡数据上报告微平均 F1 而非宏平均 F1，会得到一个看起来很髙的数字，因为它被多数类主导。宏平均 F1 强迫你看到少数类的性能。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用

scikit-learn 用六行代码即可正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三点。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加 bigrams 使 `not_good` 成为特征。`sublinear_tf=True` 抑制重复词。这三个标志是 75% 准确率的基线与 SST-2 上 85% 准确率的基线之间的差异。

### 何时转向 Transformer

- 讽刺检测。经典模型在此失败。就这样。
- 情感在中途转变的长评论。
- 基于方面的情感分析。"Camera was great but battery was terrible." 你需要将情感归因于特定方面。只有 Transformer 或结构化输出模型能做到。
- 非英语、低资源语言。Multilingual BERT 免费提供零样本基线。

如果你需要上述任何一项，跳到第7阶段（Transformer 深入）。否则，在 TF-IDF 加 bigrams 加否定处理上的朴素贝叶斯或逻辑回归是你的 2026 生产基线。

### 可复现性陷阱（再次出现）

重新训练情感模型是常规操作，但重新评估它们并非如此。论文中报告的准确率数字使用特定的划分、特定的预处理、特定的分词器。如果你不使用完全相同的流水线将自己的新模型与基线比较，你会得到误导性的差值。始终在你的流水线上重新生成基线，而不是引用论文中的数字。

## 交付

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单。** 在 scikit-learn 流水线中将 `apply_negation` 添加为预处理步骤，并在小规模情感数据集上测量 F1 的提升幅度。
2. **中等。** 实现类别加权逻辑回归（向 scikit-learn 传入 `class_weight="balanced"`，或自行推导梯度）。在合成 90-10 类别不平衡上测量其影响。
3. **困难。** 通过在情感模型的残差上训练第二个分类器来构建讽刺检测器。记录你的实验设置。当准确率低于随机水平时警告读者（2 类讽刺的随机水平约为 50%，大多数初次尝试会落在此处）。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------|---------|
| 极性 | 正面或负面 | 二元标签；有时扩展到中性或细粒度（5 星）。 |
| 基于方面的情感分析 | 每方面极性 | 将情感归因于文本中提到的特定实体或属性。 |
| 否定范围界定 | 反转附近的 token | 在标点之前将 "not" 之后的 token 前加 `NOT_`。 |
| 拉普拉斯平滑 | 计数加1 | 防止朴素贝叶斯中出现零概率特征。 |
| L2 正则化 | 缩小权重 | 在损失中加上 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础性综述。篇幅较长，但前四节涵盖了所有古典方法。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 证明了在短文本上 bigrams + 朴素贝叶斯难以被超越的论文。
- [scikit-learn 文本特征提取文档](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 和你将要调整的所有参数的参考。
