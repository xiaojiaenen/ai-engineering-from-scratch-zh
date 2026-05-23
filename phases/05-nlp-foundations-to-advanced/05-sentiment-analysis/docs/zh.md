# 情感分析

> 经典的 NLP 任务。关于传统文本分类需要了解的大部分知识都集中于此。

**类型：** 实战构建
**语言：** Python
**前置知识：** Phase 5 · 02 (词袋 + TF-IDF), Phase 2 · 14 (朴素贝叶斯)
**时间：** ~75 分钟

## 问题

“食物不怎么样。” 是正面还是负面？

情感分析听起来简单。评论者说喜欢或不喜欢某样东西。给句子贴标签。它成为经典 NLP 任务的原因在于，每一个看似简单的例子背后都隐藏着难题。否定会翻转含义。讽刺会反转含义。“Not bad at all”（一点也不差）尽管有两个消极编码的词，却是正面的。表情符号比周围的文本携带更多信号。领域词汇很重要（音乐评论中的 `tight` 对比时尚评论中的 `tight`）。

情感分析是传统 NLP 的实战演练场。如果你理解为什么每个朴素的基准模型都有特定的失败模式，你就能理解为什么每个更复杂的模型会被发明出来。本课从零开始构建一个朴素贝叶斯基准模型，添加逻辑回归，并指出那些使得生产环境中的情感分析成为合规级难题的陷阱。

## 概念

传统情感分析是一个两步配方。

1.  **表示。** 将文本转换为特征向量。词袋 (BoW)、TF-IDF 或 n-gram。
2.  **分类。** 在标记样本上拟合一个线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是能工作的最简单的模型。假设给定标签，每个特征都是独立的。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。在推理时，将概率相乘。这个“朴素”的独立性假设是荒谬地错误的，但其结果却出人意料地好。原因在于：对于稀疏的文本特征和适中的数据量，分类器更关心每个词倾向于哪一边，而不是倾斜多少。

逻辑回归修正了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为一个二元组（bigram）特征会得到一个负权重。朴素贝叶斯无法为它从未标记过的二元组做到这一点。

## 构建它

### 步骤 1：一个真实的迷你数据集

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

故意很小。真实工作会使用数万个样本（IMDb, SST-2, Yelp 极性）。数学原理是相同的。

### 步骤 2：从零开始实现多项式朴素贝叶斯

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

加法平滑（alpha=1.0）就是拉普拉斯平滑。没有它，一个在某个类别中未出现的词概率为零，取对数后会爆掉。`alpha=0.01` 在实践中很常见。`alpha=1.0` 是教学默认值。

### 步骤 3：从零开始实现逻辑回归

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

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2 正则化，模型会记忆训练样本。从 `0.01` 开始并进行调优。

### 步骤 4：处理否定（失败模式）

考虑“not good”（不好）和“not bad”（不坏）。一个词袋分类器看到 `{not, good}` 和 `{not, bad}`，并从训练中出现更多的那个中学习。一个二元组分类器看到 `not_good` 和 `not_bad`，并将它们作为不同的特征来学习。这通常就够了。

当没有二元组时，一个更粗糙但有效的修复方法是：**否定作用域（negation scoping）**。在否定词后面，直到下一个标点符号前，给 token 加上 `NOT_` 前缀。

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

现在 `good` 和 `NOT_good` 是不同的特征了。分类器可以赋予它们相反的权重。三行预处理代码，在情感分析基准上带来可测量的准确率提升。

### 步骤 5：重要的评估指标

如果类别不平衡，仅看准确率会产生误导。真实的情感语料库通常有 70-80% 是正面或 70-80% 是负面；一个恒定预测多数类的分类器也能达到 80% 准确率，但这毫无价值。报告以下每一项：

- **每类的精确率和召回率。** 每个类别一对。对它们进行宏平均（macro-average）得到一个尊重类别平衡的单一数字。
- **宏 F1（不平衡数据的主要指标）。** 每类 F1 分数的均值，权重相等。当类别不平衡时，使用此指标代替准确率。
- **加权 F1（替代指标）。** 与宏 F1 相同，但按类别频率加权。当不平衡本身具有业务含义时，与宏 F1 一起报告。
- **混淆矩阵。** 原始计数。在信任任何标量指标之前，务必检查它；它揭示了模型混淆了哪一对类别。
- **每类的错误样本。** 每个类别抽取 5 个错误预测。阅读它们。没有什么能替代阅读实际错误。

对于严重不平衡的数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC** 代替准确率。AUPRC 对少数类更敏感，而这通常是您关心的（垃圾邮件、欺诈、罕见情感）。

**要避免的常见错误。** 在不平衡数据上报告微 F1（micro-F1）而非宏 F1（macro-F1）会得到一个看起来很高的数字，因为它被多数类主导。宏 F1 强迫你看到少数类的表现。

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

## 使用它

scikit-learn 用六行代码就能正确实现。

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

注意三件事。`stop_words=None` 保留了否定。`ngram_range=(1, 2)` 添加了二元组，所以 `not_good` 成为一个特征。`sublinear_tf=True` 抑制了重复词。这三个标志是 SST-2 上 75% 准确率的基准和 85% 准确率的基准之间的区别。

### 何时使用 Transformer

- **讽刺检测。** 传统模型在这里会失败。毫无疑问。
- **长篇评论，其情感在文档中途发生转变。**
- **基于方面的情感分析。** “相机很棒，但电池很糟糕。”你需要将情感归因于具体方面。只能使用 Transformer 或结构化输出模型。
- **非英语、低资源语言。** 多语言 BERT 免费为您提供零样本基线。

如果您需要上述任何功能，请直接跳到第 7 阶段（Transformer 深入探讨）。否则，在 TF-IDF 加上二元组和否定处理上使用朴素贝叶斯或逻辑回归，就是您 2026 年的生产环境基准。

### 可复现性陷阱（再次）

重新训练情感模型是常规操作。重新评估它们则不是。论文中报告的准确率数字使用了特定的划分、特定的预处理、特定的分词器。如果您将新模型与基准进行比较，但没有使用完全相同的流程，您将得到具有误导性的差异。始终在您自己的流程上重新生成基准，而不是依赖论文中的数字。

## 部署它

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

1.  **简单。** 在 scikit-learn 管道中添加 `apply_negation` 作为预处理步骤，并在一个小型情感数据集上测量 F1 的变化。
2.  **中等。** 实现类别加权的逻辑回归（将 `class_weight="balanced"` 传递给 scikit-learn，或自己推导梯度）。测量其在合成 90-10 类别不平衡数据上的效果。
3.  **困难。** 通过在情感模型的残差上训练第二个分类器来构建一个讽刺检测器。记录你的实验设置。当你的准确率低于随机水平时，警告读者（两类讽刺的随机水平约为 50%，而大多数初次尝试都会落在这里）。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 极性 (Polarity) | 正面或负面 | 二元标签；有时扩展为中性或细粒度（5 星）。 |
| 基于方面的情感分析 (Aspect-based sentiment) | 每个方面的极性 | 将情感归因于文本中提到的特定实体或属性。 |
| 否定作用域 (Negation scoping) | 翻转附近 token | 在“not”后的 token 加上 `NOT_` 前缀，直到遇到标点。 |
| 拉普拉斯平滑 (Laplace smoothing) | 计数加 1 | 防止朴素贝叶斯中出现零概率特征。 |
| L2 正则化 (L2 regularization) | 收缩权重 | 在损失函数中添加 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础综述。很长，但前四节涵盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 这篇论文表明，在短文本上，二元组 + 朴素贝叶斯很难被超越。
- [scikit-learn 文本特征提取文档](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 以及您将要调整的每个旋钮的参考。