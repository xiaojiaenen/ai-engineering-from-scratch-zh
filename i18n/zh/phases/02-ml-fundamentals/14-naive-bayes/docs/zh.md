# Naive Bayes

> "朴素"假设是错误的，但它依然有效。这就是它的美妙之处。

**类型：** Build
**语言：** Python
**前置知识：** Phase 2，第 01-07 课（分类、贝叶斯定理）
**时间：** ~75 分钟

## 学习目标

- 从零开始实现带拉普拉斯平滑的多项式朴素贝叶斯，用于文本分类
- 解释为何朴素独立性假设在数学上是错误的，但在实践中能产生正确的类别排序
- 比较多项式、伯努利和高斯三种朴素贝叶斯变体，并根据特征类型选择合适的变体
- 在高维稀疏数据上将朴素贝叶斯与逻辑回归进行对比，并解释其中涉及的偏差-方差权衡

## 问题

你需要对文本进行分类。将电子邮件分为垃圾邮件或非垃圾邮件。将客户评论分为正面或负面。将客服工单分到各个类别。你有成千上万个特征（每个词一个），但训练数据有限。

大多数分类器在这里会受阻。逻辑回归需要足够的样本才能可靠地估计数千个权重。决策树一次只分裂一个词，会严重过拟合。在 10,000 维空间中的 KNN 毫无意义，因为每个点与其他所有点的距离都相等。

朴素贝叶斯能处理这种情况。它做了一个数学上错误的假设（给定类别后，每个特征都独立于其他所有特征），但在文本分类上仍然优于"更聪明"的模型，尤其是在训练集较小时。它只需单次遍历数据即可完成训练。它能扩展到数百万个特征。它能产生概率估计（尽管由于独立性假设，校准往往不佳）。

理解为何一个错误的假设能带来良好的预测结果，能教会你机器学习中的一个基本常识：最好的模型不是最正确的那个，而是针对你的数据具有最佳偏差-方差权衡的那个。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们需要的是 `P(class | features)` —— 给定文档中的词语，该文档属于某个类别的概率。我们可以从以下各项计算得到：
- `P(features | class)` —— 在该类别的文档中看到这些词的可能性
- `P(class)` —— 类别的先验概率（垃圾邮件总体有多常见？）
- `P(features)` —— 证据，对所有类别相同，因此在比较时可以忽略

具有最高 `P(class | features)` 的类别胜出。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征的共同概率。对于包含 10,000 个词的词汇表，你需要估计 2^10,000 种可能组合的分布。这不可能实现。

朴素假设：给定类别后，每个特征都是条件独立的。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

用一个不可能计算的整体分布来代替，你只需估计 n 个简单的单特征分布。每个只需要一个计数。

这个假设显然是错误的。"machine"和"learning"这两个词在任何文档中都不是独立的。但分类器不需要准确的概率估计。它需要的是正确的排序 —— 哪个类别的概率最高。独立性假设会引入系统性误差，但这些误差对所有类别的影响相似，因此排序仍然正确。

### 为何仍然有效

三个原因：

1. **排序优于校准。** 分类只需要排名最高的类别正确即可。即使 P(spam) = 0.99999 而真实概率是 0.7，分类器仍然能正确选择 spam。我们不需要准确的概率。我们需要正确的赢家。

2. **高偏差，低方差。** 独立性假设是一个强先验。它强烈约束模型，从而防止过拟合。在训练数据有限的情况下，一个略微错误但稳定的模型胜过理论上正确但极不稳定的模型。这就是偏差-方差权衡的体现。

3. **特征冗余相互抵消。** 相关特征提供冗余证据。分类器会双重计算这些证据，但会对正确的类别也双重计算。如果"machine"和"learning"总是同时出现，两者都为"tech"类提供证据。NB 将它们计数两次，但确实是对正确的类别计数两次。

第四个实际原因：朴素贝叶斯极其快速。训练只需单次遍历数据并计数频率。预测是一个矩阵乘法。你可以在几秒钟内训练百万份文档。这种速度意味着你可以更快地迭代，尝试更多的特征集，并进行更多的实验，而不是使用较慢的模型。

### 逐步推导数学过程

让我们通过一个具体的例子来追踪。假设有两个类别：spam 和 not-spam。词汇表包含三个词："free"、"money"、"meeting"。

训练数据：
- 垃圾邮件中提到"free"80 次，"money"60 次，"meeting"10 次（共 150 个词）
- 非垃圾邮件中提到"free"5 次，"money"10 次，"meeting"100 次（共 115 个词）
- 40% 的邮件是垃圾邮件，60% 是非垃圾邮件

使用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free"（2 次），"money"（1 次），"meeting"（0 次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

spam 以巨大优势胜出。"free"出现两次是支持 spam 的强证据。注意，"meeting"未出现对两个对数求和都贡献零值（0 * log(P)）—— 在多项式 NB 中，缺失的词没有影响。是伯努利 NB 显式建模词的缺失。

### 三种变体

朴素贝叶斯有三种形式。每种对 `P(feature | class)` 的建模方式不同。

#### 多项式朴素贝叶斯

将每个特征建模为计数。最适合特征为词频或 TF-IDF 值的文本数据。

```
P(word_i | class) = (word_i 在类别中的计数 + alpha) / (类别中的总词数 + alpha * 词汇表大小)
```

`alpha` 是拉普拉斯平滑（下面解释）。这种变体是文本分类的主力。

#### 高斯朴素贝叶斯

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别在每个特征上都有自己的均值和方差。当特征在每个类别内确实遵循钟形曲线时，这种方法效果很好。

#### 伯努利朴素贝叶斯

将每个特征建模为二元变量（存在或不存在）。最适合短文本或二元特征向量。

```
P(word_i | class) = (包含 word_i 的类别文档数 + alpha) / (类别中的总文档数 + 2 * alpha)
```

与多项式不同，伯努利显式惩罚词的缺失。如果"free"通常出现在垃圾邮件中但在这封邮件中缺失，伯努利会将此视为反对 spam 的证据。

### 何时使用每种变体

| 变体 | 特征类型 | 适用场景 | 示例 |
|------|---------|---------|------|
| 多项式 | 计数或频率 | 文本分类、词袋模型 | 邮件垃圾过滤、主题分类 |
| 高斯 | 连续值 | 具有近似正态分布特征的表格数据 | Iris 分类、传感器数据 |
| 伯努利 | 二元（0/1） | 短文本、二元特征向量 | SMS 垃圾过滤、存在/不存在特征 |

### 拉普拉斯平滑

当一个词出现在测试数据中，但从未出现在特定类别的训练数据中时，会发生什么？

不使用平滑：`P(word | class) = 0/N = 0`。一个零值乘以整个乘积会使 `P(class | features) = 0`，无论其他所有证据如何。一个未见过的词会摧毁整个预测，无论其他证据多么支持它。

拉普拉斯平滑为每个特征计数添加一个小计数 `alpha`（通常为 1）：

```
P(word_i | class) = (word_i 的计数 + alpha) / (类别中的总词数 + alpha * 词汇表大小)
```

使用 alpha=1，每个词至少获得一个微小的概率。测试邮件中出现的词"discombobulate"不再会摧毁 spam 概率。平滑具有贝叶斯解释：它等价于在词分布上放置一个均匀狄利克雷先验。

较高的 alpha 意味着更强的平滑（更均匀的分布）。较低的 alpha 意味着模型更信任数据。alpha 是一个你需要调优的超参数。

alpha 的影响：

| Alpha | 效果 | 使用场景 |
|-------|------|---------|
| 0.001 | 几乎无平滑，信任数据 | 非常大的训练集，预期无未见特征 |
| 0.1 | 轻度平滑 | 大型训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 重度平滑，展平分布 | 非常小的训练集，预期有很多未见特征 |

### 对数空间计算

将数百个概率相乘（每个都小于 1）会导致浮点数下溢。乘积在浮点数中表示为零，尽管真实值是一个非常小的正数。

解决方案：在对数空间中运算。与其乘以概率，不如加上它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这使预测变成点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是为什么朴素贝叶斯预测如此快速 —— 它与单层线性模型的运算相同。

### 朴素贝叶斯 vs 逻辑回归

两者都是用于文本的线性分类器。区别在于它们建模的内容。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|------|----------|---------|
| 类型 | 生成式（建模 P(X\|Y)） | 判别式（建模 P(Y|X)） |
| 训练 | 计数频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（不足以估计权重） |
| 大数据 | 更差（错误假设会损害性能） | 更好（灵活的边界） |
| 特征 | 假设独立性 | 处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率较差 | 概率更好 |

经验法则：先从朴素贝叶斯开始。如果你有足够的数据且 NB 性能趋于平稳，则切换到逻辑回归。

### 分类流程

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[统计词频]
    D --> E[应用平滑]
    E --> F[计算对数概率]
    F --> G[预测：argmax P(类别 | 词语)]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实际上，我们在对数空间中运算以避免浮点数下溢。与其乘以许多小概率，不如加上它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 实现

`code/naive_bayes.py` 中的代码从零开始实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零开始的实现：

1. **fit(X, y)**：对每个类别，计算每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本，对每个类别计算 log P(class) + 所有特征的 log P(feature_i | class) 之和。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回具有最高对数概率的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合后，预测仅仅是矩阵乘法加上偏置。这就是为什么朴素贝叶斯如此快速。

### GaussianNB

对于连续特征，我们为每个类别的每个特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯 PDF，跨特征相乘（在对数空间中相加）。

### 演示：文本分类

代码生成模拟两个类别（技术文章 vs 体育文章）的合成词袋数据。每个类别有不同的词频分布。MultinomialNB 使用词计数对它们进行分类。

合成数据的运作方式如下：我们创建 200 个"词"（特征列）。词 0-39 在技术文章中高频出现，在体育文章中低频出现。词 80-119 在体育文章中高频出现，在技术文章中低频出现。词 40-79 在两者中都是中等频率。这创造了一个现实场景，其中一些词是强类别指示器，而其他则是噪声。

### 演示：连续特征

代码生成类似 Iris 的数据（3 个类别，4 个特征，高斯簇）。GaussianNB 使用逐类别均值和方差进行分类。每个类别都有不同的中心（均值向量）和不同的分布（方差），模拟了测量值在不同类别之间有系统性差异的现实世界数据。

代码还演示了：
- **平滑对比：** 使用不同 alpha 值训练 MultinomialNB，以展示平滑强度对准确性的影响。
- **训练规模实验：** NB 准确性如何随着训练数据从 20 增加到 1600 个样本而提升。即使样本很少，NB 也能达到不错的准确性 —— 这是它的主要优势。
- **混淆矩阵：** 逐类别的精确率、召回率和 F1 分数，以显示 NB 在何处犯错。

### 预测速度

朴素贝叶斯预测是一个矩阵乘法。对于 n 个样本、d 个特征和 k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯 PDF 评估，每次涉及 d 个特征 = O(n * d * k)

两者在所有维度上都是线性的。与 KNN（需要计算到所有训练点的距离）或带 RBF 核的 SVM（需要计算与所有支持向量的核函数）相比。NB 在预测时的速度要快几个数量级。

## 使用它

使用 sklearn，两种变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用于文本分类的 sklearn：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码比较了从零开始实现的版本与 sklearn 在同一数据上的表现，以验证正确性。

### 与朴素贝叶斯配合使用 TF-IDF

原始词频赋予每次出现相同权重。但像"the"和"is"这样的常用词在每个类别中都频繁出现 —— 它们不携带任何信息。TF-IDF（词频-逆文档频率）降低常用词的权重，提升稀有、有区分性的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值是非负的，因此它们可以与 MultinomialNB 配合使用。TF-IDF + MultinomialNB 的组合是文本分类中最强的基线之一。它经常在少于 10,000 个训练样本的数据集上击败更复杂的模型。

### 用于短文本的 BernoulliNB

对于短文本（推文、短信、聊天记录），BernoulliNB 可能优于 MultinomialNB。短文本的词数较少，因此 MultinomialNB 依赖的频率信息噪音较大。BernoulliNB 只关心存在与否，这在短文本中更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 中的 `binary=True` 标志将所有计数转换为 0/1。如果不设置，BernoulliNB 仍然可以工作，但它看到的是它设计时未考虑的计数。

### 校准 NB 概率

NB 概率校准不佳。当 NB 说 P(spam) = 0.95 时，真实概率可能是 0.7。如果你需要可靠的概率估计（例如，设置阈值或与其它媒体模型结合），使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这使用交叉验证在 NB 的原始分数之上拟合一个逻辑回归。 resulting 概率更接近真实类别频率。

### 常见陷阱

1. **负特征值。** MultinomialNB 要求非负特征。如果你有负值（如某些设置下的 TF-IDF 或标准化特征），改用 GaussianNB，或将特征平移为正数。

2. **零方差特征。** GaussianNB 除以方差。如果某个特征在一个类别中方差为零（所有值相同），概率计算会出错。代码向所有方差添加一个小平滑项（1e-9）以防止此问题。

3. **类别不平衡。** 如果 99% 的邮件是非垃圾邮件，先验 P(not-spam) = 0.99 太强，会淹没似然证据。你可以手动设置类别先验或使用 sklearn 中的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它在计数上工作）。GaussianNB 也不需要缩放（它估计逐特征统计量）。这比逻辑回归和 SVM 有优势，后者对特征尺度敏感。

## 交付

本课产出：
- `outputs/skill-naive-bayes-chooser.md` —— 选择合适 NB 变体的决策技能
- `code/naive_bayes.py` —— 从零开始的 MultinomialNB 和 GaussianNB，以及与 sklearn 的对比

### 朴素贝叶斯何时失效

当独立性假设导致错误排序（而不仅是错误概率）时，NB 会失效。这发生在：

1. **强特征交互。** 如果类别取决于两个特征的组合而非各自单独（类似 XOR 的模式），NB 将完全错过它。每个特征单独不提供任何证据，且 NB 无法非线性地组合它们。

2. **高度相关的特征带有相反证据。** 如果特征 A 说"spam"而特征 B 说"not-spam"，但 A 和 B 完全相关（现实中它们始终一致），NB 将在不存在冲突的地方看到冲突证据。

3. **非常大的训练集。** 在有足够数据的情况下，逻辑回归等判别式模型学习真实决策边界并超越 NB。帮助小数据的独立性假设现在会拖累模型。

在实践中，这些失效模式在文本分类中很少见。文本特征数量众多、 individually 较弱，且独立性假设的误差倾向于相互抵消。对于具有少量强相关特征的表格数据，优先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 在不同 alpha 值（0.01、0.1、1.0、10.0、100.0）下在文本数据上训练 MultinomialNB。绘制准确率 vs alpha 图。性能在哪里达到峰值？为什么非常高的 alpha 会损害性能？

2. **特征独立性测试。** 获取一个真实文本数据集。选择两个明显相关的词（"machine"和"learning"）。计算 P(word1 | class) * P(word2 | class) 并与 P(word1 AND word2 | class) 比较。独立性假设错得多离谱？它是否影响分类准确性？

3. **伯努利实现。** 用 BernoulliNB 类扩展代码。将词袋转换为二元（存在/不存在），并在文本数据上将准确率与 MultinomialNB 进行比较。伯努利何时获胜？

4. **NB vs 逻辑回归。** 在文本数据上同时训练两者。从 100 个训练样本开始，增加到 10,000。为两者绘制准确率 vs 训练集大小图。逻辑回归在何时超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建完整的垃圾邮件分类器：分词原始邮件文本，构建词汇表，创建词袋特征，训练 MultinomialNB，使用精确率和召回率进行评估（而不仅是准确率 —— 为什么？）。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|----------|---------|
| 朴素贝叶斯 | "简单概率分类器" | 应用贝叶斯定理并假设特征在给定类别后条件独立的分类器 |
| 条件独立性 | "特征互不影响" | P(A, B \| C) = P(A \| C) * P(B \| C) —— 知道 C 后，了解 B 不会给你关于 A 的新信息 |
| 拉普拉斯平滑 | "加一平滑" | 向每个特征添加一个小计数，以防止零概率主导预测 |
| 先验 | "在看到数据之前你所相信的" | P(class) —— 在观察任何特征之前每个类别的概率 |
| 似然 | "数据拟合的程度" | P(features \| class) —— 如果类别已知，观察到这些特征的概率 |
| 后验 | "在看到数据之后你所相信的" | P(class \| features) —— 观察到特征后类别的更新概率 |
| 生成模型 | "建模数据如何生成" | 学习 P(X \| Y) 和 P(Y)，然后使用贝叶斯定理得到 P(Y \| X) 的模型 |
| 判别模型 | "建模决策边界" | 直接学习 P(Y \| X) 而不建模 X 如何生成的模型 |
| 对数概率 | "避免下溢" | 使用 log P 而非 P，以防止许多小数的乘积在浮点数中变为零 |

## 延伸阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 三种变体及数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) —— 经典的多项式 vs 伯努利文本比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) —— 针对文本的 NB 改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) —— 证明 NB 用更少数据时比 LR 收敛更快
