# 贝叶斯定理

> 概率是关于你的期望。贝叶斯定理是关于你所学到的。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 1, Lesson 06 (概率基础)
**时间：** 约 75 分钟

## 学习目标

- 应用贝叶斯定理，从先验、似然和证据计算后验概率
- 从零构建带有拉普拉斯平滑和对数空间计算的朴素贝叶斯文本分类器
- 比较 MLE 和 MAP 估计，并解释 MAP 如何对应于 L2 正则化
- 使用 Beta-二项共轭先验实现顺序贝叶斯更新，用于 A/B 测试

## 问题所在

一项医学检测的准确率是 99%。你检测呈阳性。你真正患病的概率是多少？

大多数人会说 99%。真实答案取决于这种疾病有多罕见。如果每 10,000 人中只有 1 人患病，那么阳性结果只给你大约 1% 的患病概率。另外 99% 的阳性结果是健康人的误报。

这不是脑筋急转弯。这是贝叶斯定理。每个垃圾邮件过滤器、每个医学诊断、每个量化不确定性的机器学习模型都在使用完全相同的推理。你从信念开始。你看到证据。你进行更新。

如果你在不理解这一点的情况下构建 ML 系统，你会误解模型输出、设置错误的阈值，并交付过度自信的预测。

## 概念解析

### 从联合概率到贝叶斯

你已经从 Lesson 06 知道条件概率是：

```
P(A|B) = P(A and B) / P(B)
```

并且对称地：

```
P(B|A) = P(A and B) / P(A)
```

两个表达式共享相同的分子：P(A and B)。令它们相等并重新排列：

```
P(A and B) = P(A|B) * P(B) = P(B|A) * P(A)

因此：

P(A|B) = P(B|A) * P(A) / P(B)
```

这就是贝叶斯定理。四个量，一个等式。

### 四个部分

| 部分 | 名称 | 含义 |
|------|------|------|
| P(A\|B) | 后验 | 在看到证据 B 之后你对 A 的更新信念 |
| P(B\|A) | 似然 | 如果 A 为真，证据 B 的概率有多大 |
| P(A) | 先验 | 在看到任何证据之前你对 A 的信念 |
| P(B) | 证据 | 在所有可能性下看到 B 的总概率 |

证据项 P(B) 充当归一化因子。你可以使用全概率定律将其展开：

```
P(B) = P(B|A) * P(A) + P(B|not A) * P(not A)
```

### 医学检测示例

一种疾病影响 1/10,000 的人。检测准确率是 99%（检出 99% 的患者，假阳性率为 1%）。

```
P(sick)              = 0.0001     (先验：疾病很罕见)
P(positive|sick)     = 0.99       (似然：检测能抓住它)
P(positive|healthy)  = 0.01       (假阳性率)

P(positive) = P(positive|sick) * P(sick) + P(positive|healthy) * P(healthy)
            = 0.99 * 0.0001 + 0.01 * 0.9999
            = 0.000099 + 0.009999
            = 0.010098

P(sick|positive) = P(positive|sick) * P(sick) / P(positive)
                 = 0.99 * 0.0001 / 0.010098
                 = 0.0098
                 = 0.98%
```

不到 1%。先验占主导。当一个条件很罕见时，即使是准确的检测也会产生大量的假阳性。这就是医生要求确认测试的原因。

### 垃圾邮件过滤器示例

你收到一封包含单词"lottery"的电子邮件。它是垃圾邮件吗？

```
P(spam)               = 0.3        (30% 的邮件是垃圾邮件)
P("lottery"|spam)     = 0.05       (5% 的垃圾邮件包含"lottery")
P("lottery"|not spam) = 0.001      (0.1% 的合法邮件包含"lottery")

P("lottery") = 0.05 * 0.3 + 0.001 * 0.7
             = 0.015 + 0.0007
             = 0.0157

P(spam|"lottery") = 0.05 * 0.3 / 0.0157
                  = 0.955
                  = 95.5%
```

一个词将概率从 30% 提高到 95.5%。真正的垃圾邮件过滤器同时应用贝叶斯 across 数百个单词。

### 朴素贝叶斯：独立性假设

朴素贝叶斯通过假设给定类别的所有特征条件独立，将此扩展到多个特征：

```
P(class | feature_1, feature_2, ..., feature_n)
  = P(class) * P(feature_1|class) * P(feature_2|class) * ... * P(feature_n|class)
    / P(feature_1, feature_2, ..., feature_n)
```

"朴素"部分是独立性假设。在文本中，单词的出现不是独立的（"New" 和 "York" 是相关的）。但该假设在实践中出奇地有效，因为分类器只需要对类别进行排名，而不需要产生校准的概率。

由于分母对所有类别都相同，你可以跳过它，只需比较分子：

```
score(class) = P(class) * product of P(feature_i | class)
```

选择得分最高的类别。

### 最大似然估计 (MLE)

如何从训练数据获取 P(feature|class)？计数。

```
P("free"|spam) = (包含"free"的垃圾邮件数量) / (垃圾邮件总数)
```

这就是 MLE：选择使观测数据最可能的参数值。你在最大化似然函数，对于离散计数来说，这简化为相对频率。

问题：如果一个单词在训练期间从未出现在垃圾邮件中，MLE 会给出概率为零。一个未见过的单词会摧毁整个乘积。用拉普拉斯平滑修复：

```
P(word|class) = (count(word, class) + 1) / (total_words_in_class + vocabulary_size)
```

给每个计数加 1 确保没有任何概率为零。

### 最大后验估计 (MAP)

MLE 问：什么参数最大化 P(data|parameters)？

MAP 问：什么参数最大化 P(parameters|data)？

根据贝叶斯定理：

```
P(parameters|data) 正比于 P(data|parameters) * P(parameters)
```

MAP 为参数本身添加了一个先验。如果你认为参数应该很小，你可以将这个偏好编码为对大值的惩罚先验。这与 ML 中的 L2 正则化完全相同。岭回归中的"脊"惩罚本质上就是权重的高斯先验。

| 估计方法 | 优化目标 | ML 对应物 |
|----------|----------|-----------|
| MLE | P(data\|params) | 无正则化的训练 |
| MAP | P(data\|params) * P(params) | L2 / L1 正则化 |

### 贝叶斯 vs 频率学派：实际差异

频率主义者将参数视为固定的未知量。他们问："如果我多次重复这个实验，会发生什么？"

贝叶斯主义者将参数视为分布。他们问："鉴于我已观察到的，我对参数有什么信念？"

对于构建 ML 系统的实际差异：

| 方面 | 频率学派 | 贝叶斯学派 |
|------|----------|------------|
| 输出 | 点估计 | 值分布 |
| 不确定性 | 置信区间（关于程序） | 可信区间（关于参数） |
| 小数据 | 可能过拟合 | 先验起到正则化作用 |
| 计算 | 通常更快 | 通常需要采样 (MCMC) |

大多数生产 ML 是频率学派的（SGD、点估计）。贝叶斯方法在你需要校准的不确定性（医疗决策、安全关键系统）或数据稀缺（少样本学习、冷启动）时表现出色。

### 为什么贝叶斯思维对 ML 很重要

联系比类比更深：

**先验就是正则化。** 权重上的高斯先验就是 L2 正则化。拉普拉斯先验就是 L1。每次你添加正则化项时，你都在对期望的参数值做出贝叶斯声明。

**后验就是不确定性。** 单个预测概率无法告诉你模型对该估计的信心程度。贝叶斯方法给你一个分布："我认为 P(spam) 在 0.8 到 0.95 之间。"

**贝叶斯更新就是在线学习。** 今天的后验成为明天的先验。当你的模型看到新数据时，它会增量更新信念，而不是从头重新训练。

**模型比较是贝叶斯的。** 贝叶斯信息准则 (BIC)、边缘似然和贝叶斯因子都使用贝叶斯推理在不拟合过度的情况下选择模型。

```figure
bayes-update
```

## 构建它

### 步骤 1：贝叶斯定理函数

```python
def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior

result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
print(f"P(sick|positive) = {result:.4f}")
```

### 步骤 2：朴素贝叶斯分类器

```python
import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class
```

对数概率防止下溢。乘很多小概率会产生小到浮点数无法表示的数字。对对数概率求和在数值上是稳定的，并且在数学上是等价的。

### 步骤 3：在垃圾邮件数据上训练

```python
train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]

classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]

for msg in test_messages:
    print(f"  '{msg}' -> {classifier.predict(msg)}")
```

### 步骤 4：检查学习到的概率

```python
def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab_size)
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")

print("\nTop spam words:")
show_top_words(classifier, "spam")
print("\nTop ham words:")
show_top_words(classifier, "ham")
```

## 使用它

scikit-learn 提供了生产就绪的朴素贝叶斯实现：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_docs)
clf = MultinomialNB()
clf.fit(X_train, train_labels)

X_test = vectorizer.transform(test_messages)
predictions = clf.predict(X_test)
for msg, pred in zip(test_messages, predictions):
    print(f"  '{msg}' -> {pred}")
```

相同的算法。CountVectorizer 处理分词和词汇表构建。MultinomialNB 内部处理平滑和对数概率。你自己的版本用 40 行代码做了同样的事情。

## 交付

这里构建的 NaiveBayes 类展示了完整的流水线：分词、带拉普拉斯平滑的概率估计、对数空间预测。`code/bayes.py` 中的代码可以端到端运行，无需 Python 标准库之外的依赖。

### 共轭先验

当先验和后验属于同一族分布时，该先验被称为"共轭"。这使贝叶斯更新在代数上非常干净——你得到闭式后验，而无需数值积分。

| 似然 | 共轭先验 | 后验 | 示例 |
|-----|---------|------|------|
| 伯努利 | Beta(a, b) | Beta(a + 成功次数, b + 失败次数) | 硬币翻转偏差估计 |
| 正态（方差已知） | Normal(mu_0, sigma_0) | Normal(加权均值, 较小方差) | 传感器校准 |
| 泊松 | Gamma(a, b) | Gamma(a + 计数之和, b + n) | 建模到达率 |
| 多项分布 | Dirichlet(alpha) | Dirichlet(alpha + 计数) | 主题建模、语言模型 |

为什么这很重要：没有共轭先验时，你需要蒙特卡洛采样或变分推理来近似后验。有了共轭先验，你只需更新两个数字。

Beta 分布是最常用的共轭先验。Beta(a, b) 代表你对概率参数的信念。均值是 a/(a+b)。a+b 越大，分布越集中（越自信）。

Beta 先验的特殊情况：
- Beta(1, 1) = 均匀分布。你对参数没有任何意见。
- Beta(10, 10) = 在 0.5 处尖锐。你强烈相信参数接近 0.5。
- Beta(1, 10) = 偏向 0。你认为参数很小。

更新规则非常简单：

```
先验:     Beta(a, b)
数据:      s 次成功, f 次失败
后验:     Beta(a + s, b + f)
```

没有积分。没有采样。只是加法。

### 顺序贝叶斯更新

贝叶斯推断本质上是顺序的。今天的后验成为明天的先验。这就是现实系统如何在无需重新处理所有历史数据的情况下增量学习。

具体示例：估计硬币是否公平。

**第 1 天：还没有数据。**
从 Beta(1, 1) —— 均匀先验开始。你没有偏好。
- 先验均值：0.5
- 先验在 [0, 1] 上是平坦的

**第 2 天：观察到 7 次正面，3 次反面。**
后验 = Beta(1 + 7, 1 + 3) = Beta(8, 4)
- 后验均值：8/12 = 0.667
- 证据表明硬币偏向正面

**第 3 天：再观察到 5 次正面，5 次反面。**
将昨天的后验用作今天的先验。
后验 = Beta(8 + 5, 4 + 5) = Beta(13, 9)
- 后验均值：13/22 = 0.591
- 平衡的新数据将估计值拉回 0.5 附近

```mermaid
graph LR
    A["先验<br/>Beta(1,1)<br/>均值 = 0.50"] -->|"7H, 3T"| B["后验 1<br/>Beta(8,4)<br/>均值 = 0.67"]
    B -->|"成为先验"| C["先验 2<br/>Beta(8,4)"]
    C -->|"5H, 5T"| D["后验 2<br/>Beta(13,9)<br/>均值 = 0.59"]
```

观察的顺序不影响结果。Beta(1,1) 一次性更新所有 12 次正面和 8 次反面，得到 Beta(13, 9) —— 相同的结果。顺序更新和批处理更新在数学上是等价的。但顺序更新允许你在每一步做出决策，而无需存储原始数据。

这是生产 ML 系统中在线学习的基础。Thompson 采样用于多臂老虎机、增量推荐系统和流式异常检测器都使用这种模式。

### 与 A/B 测试的联系

A/B 测试是伪装的贝叶斯推断。

设置：你在测试两种按钮颜色。变体 A（蓝色）和变体 B（绿色）。你想知道哪个获得更多点击。

贝叶斯 A/B 测试：

1. **先验。** 为两个变体从 Beta(1, 1) 开始。没有先验偏好。
2. **数据。** 变体 A：1000 次展示中 50 次点击。变体 B：1000 次展示中 65 次点击。
3. **后验。**
   - A: Beta(1 + 50, 1 + 950) = Beta(51, 951)。均值 = 0.051
   - B: Beta(1 + 65, 1 + 935) = Beta(66, 936)。均值 = 0.066
4. **决策。** 计算 P(B > A) —— B 的真实转化率高于 A 的概率。

解析计算 P(B > A) 很困难。但蒙特卡洛使其变得简单：

```
1. 从 Beta(51, 951) 抽取 100,000 个样本 -> samples_A
2. 从 Beta(66, 936) 抽取 100,000 个样本 -> samples_B
3. P(B > A) = B > A 的样本比例
```

如果 P(B > A) > 0.95，你发布变体 B。如果在 0.05 到 0.95 之间，你继续收集数据。如果 P(B > A) < 0.05，你发布变体 A。

相对于频率学派 A/B 测试的优势：
- 你得到一个直接的概率陈述："有 97% 的可能性 B 更好"
- 没有 p 值混淆。没有"未能拒绝原假设"的含糊其辞。
- 你可以在任何时候检查结果而不会膨胀假阳性率（没有"窥视问题"）
- 你可以纳入先验知识（例如，之前的测试表明转化率通常在 3-8% 之间）

| 方面 | 频率学派 A/B | 贝叶斯 A/B |
|------|-------------|------------|
| 输出 | p 值 | P(B > A) |
| 解释 | "如果 A=B，数据有多意外？" | "B 比 A 更好的可能性有多大？" |
| 提前停止 | 会膨胀假阳性 | 在任何时候都是安全的（给定合理选择的先验和正确指定的模型） |
| 先验知识 | 不使用 | 编码为 Beta 先验 |
| 决策规则 | p < 0.05 | P(B > A) > 阈值 |

## 练习

1. **多次测试。** 患者在两次独立测试中都呈阳性（两者准确率都是 99%，疾病患病率为 1/10,000）。两次测试后 P(sick) 是多少？使用第一次测试的后验作为第二次的先验。

2. **平滑的影响。** 用平滑值 0.01、0.1、1.0 和 10.0 运行垃圾邮件分类器。顶级单词概率如何变化？当 smoothing=0 且一个单词只出现在 ham 中时会发生什么？

3. **添加特征。** 扩展 NaiveBayes 类，使其同时使用消息长度（短/长）作为与词频并列的特征。从训练数据估计 P(short|spam) 和 P(short|ham)，并将其合并到预测得分中。

4. **手动 MAP。** 给定观测数据（10 次硬币翻转中有 7 次正面），使用 Beta(2,2) 先验计算偏差的 MAP 估计。将其与 MLE 估计（7/10）进行比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 先验 (Prior) | "我的初始猜测" | 观察到证据之前的 P(hypothesis)。在 ML 中：正则化项。 |
| 似然 (Likelihood) | "数据拟合得好坏" | P(evidence\|hypothesis)。在特定假设下观测数据有多可能。 |
| 后验 (Posterior) | "我的更新信念" | P(hypothesis\|evidence)。先验乘以似然，然后归一化。 |
| 证据 (Evidence) | "归一化常数" | 所有假设下的 P(data)。确保后验之和为 1。 |
| 朴素贝叶斯 (Naive Bayes) | "那个简单的文本分类器" | 假设给定类别的特征相互独立的分类器。尽管假设错误，但效果很好。 |
| 拉普拉斯平滑 (Laplace smoothing) | "加一平滑" | 向每个特征添加一个小计数，以防止未见数据的零概率。 |
| MLE | "直接用频率" | 选择最大化 P(data\|parameters) 的参数。没有先验。小数据可能过拟合。 |
| MAP | "带先验的 MLE" | 选择最大化 P(data\|parameters) * P(parameters) 的参数。等同于正则化的 MLE。 |
| 对数概率 (Log-probability) | "在对数空间工作" | 使用 log(P) 而非 P，以避免乘以许多小数时出现浮点下溢。 |
| 假阳性 (False positive) | "错误的警报" | 检测呈阳性，但真实状态为阴性。驱动基率谬误。 |

## 延伸阅读

- [3Blue1Brown: Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) - 使用医学检测示例的可视化解释
- [Stanford CS229: Generative Learning Algorithms](https://cs229.stanford.edu/notes2022fall/cs229-notes2.pdf) - 朴素贝叶斯及其与判别模型的关联
- [Think Bayes](https://greenteapress.com/wp/think-bayes/) - 免费书籍，使用 Python 代码的贝叶斯统计学
- [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) - 生产实现及何时使用每种变体
