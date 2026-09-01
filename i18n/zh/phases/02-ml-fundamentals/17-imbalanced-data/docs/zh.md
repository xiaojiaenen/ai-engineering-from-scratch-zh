# 处理不平衡数据

> 当你的数据 99% 都是"正常"时，准确率是一个谎言。

**类型：** 构建
**语言：** Python
**前置条件：** 第 2 阶段，课程 01-09（尤其是评估指标）
**时间：** 约 90 分钟

## 学习目标

- 从零实现 SMOTE，并解释合成过采样与随机重复的区别
- 使用 F1、AUPRC 和 Matthews 相关系数而非准确率来评估不平衡分类器
- 比较类别权重、阈值调整和重采样策略，并为给定的不平衡比例选择正确的方法
- 构建一个完整的非平衡数据管道，结合 SMOTE、类别权重和阈值优化

## 问题所在

你构建了一个欺诈检测模型。它达到了 99.9% 的准确率。你庆祝着。然后你发现它对每一笔交易都预测为"非欺诈"。

这不是 bug。当只有 0.1% 的交易是欺诈时，这是理性的做法。模型学会了始终猜测多数类可以最小化总体误差。它在技术上是正确的，但完全无用。

这种情况在一切真正的分类场景中都普遍存在。疾病诊断：阳性率 1%。网络入侵：攻击率 0.01%。制造缺陷：缺陷率 0.5%。垃圾邮件过滤：垃圾邮件占比 20%。流失预测：流失用户占比 5%。少数类越重要，它往往越稀有。

准确率失败是因为它将所有正确预测一视同仁。正确标记合法交易和正确捕捉欺诈都只计为一个准确点。但捕捉欺诈才是模型存在的理由。我们需要强迫模型关注稀有但重要类别的指标、技术和训练策略。

## 概念讲解

### 为什么准确率会失效

考虑一个包含 1000 个样本的数据集：990 个负样本，10 个正样本。一个始终预测负的模型：

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0 (TP) | 10 (FN) |
| 实际为负 | 0 (FP) | 990 (TN) |

准确率 = (0 + 990) / 1000 = 99.0%

该模型一个欺诈都没抓到。零疾病。零缺陷。但准确率说是 99%。这就是准确率在不平衡问题上的危险性所在。

### 更好的指标

**精确率 (Precision)** = TP / (TP + FP)。在所有被标记为正样的中，有多少是真的？高精确率意味着误报少。

**召回率 (Recall)** = TP / (TP + FN)。在所有实际为正样的中，我们抓住了多少？高召回率意味着漏报少。

**F1 分数** = 2 * precision * recall / (precision + recall)。调和平均数。比算术平均数更严厉地惩罚精确率和召回率之间的极端不平衡。

**F-beta 分数** = (1 + beta²) * precision * recall / (beta² * precision + recall)。当 beta > 1 时，召回率更重要。当 beta < 1 时，精确率更重要。在欺诈检测中常用 F2（遗漏欺诈比误报更严重）。

**AUPRC**（精确率-召回率曲线下面积）。类似于 AUC-ROC，但对不平衡数据提供更多信息。随机分类器的 AUPRC 等于正类比例（不像 ROC 那样是 0.5）。这使得改进更容易被观察到。

**Matthews 相关系数** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。范围从 -1 到 +1。只有当模型在两类上都表现良好时才会给出高分。即使在类别大小差异很大时也能保持平衡。

对于上面那个"始终预测负"的模型：精确率 = 0/0（未定义，通常设为 0），召回率 = 0/10 = 0，F1 = 0，MCC = 0。这些指标正确地识别出该模型毫无价值。

### 不平衡数据管道

```mermaid
flowchart TD
    A[不平衡数据集] --> B{不平衡比例？}
    B -->|轻度: 80/20| C[类别权重]
    B -->|中度: 95/5| D[SMOTE + 阈值调整]
    B -->|严重: 99/1| E[SMOTE + 类别权重 + 阈值]
    C --> F[训练模型]
    D --> F
    E --> F
    F --> G[用 F1 / AUPRC / MCC 评估]
    G --> H{足够好吗？}
    H -->|否| I[尝试不同策略]
    H -->|是| J[部署并监控]
    I --> B
```

### SMOTE：合成少数类过采样技术

随机过采样会复制现有的少数类样本。这有效，但风险是过拟合，因为模型会反复看到相同的点。

SMOTE 创建新的合成少数类样本，这些样本是合理的，但不是副本。算法如下：

1. 对于每个少数类样本 x，在其其他少数类样本中找到其 k 个最近邻
2. 随机选择一个邻居
3. 在 x 和该邻居之间的线段上创建一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这在真实少数类点之间进行插值，在特征空间的相同区域创建样本，而不仅仅是复制现有数据。

```mermaid
flowchart LR
    subgraph 原始["原始少数类点"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE 生成"]
        direction TB
        S1["选择 x1，邻居 x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    原始 --> SMOTE
    subgraph 结果["增强集合"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["合成点 (1.2, 2.2)"]
    end
    SMOTE --> 结果
```

### 采样策略对比

**随机过采样**：复制少数类样本来匹配多数类数量。
- 优点：简单，无信息损失
- 缺点：完全重复会导致过拟合，增加训练时间

**随机欠采样**：删除多数类样本来匹配少数类数量。
- 优点：训练速度快，简单
- 缺点：丢弃了潜在的有用多数类数据，方差更高

**SMOTE**：通过插值创建合成少数类样本。
- 优点：生成新数据点，与随机过采样相比减少过拟合
- 缺点：可能在决策边界附近创建噪声样本，不考虑多数类分布

| 策略 | 更改的数据 | 风险 | 适用场景 |
|----------|-------------|------|-------------|
| 过采样 | 少数类被复制 | 过拟合 | 小数据集，中度不平衡 |
| 欠采样 | 多数类被删除 | 信息丢失 | 大数据集，需要快速训练 |
| SMOTE | 添加合成少数类 | 边界噪声 | 中度不平衡，有足够少数类样本用于 k-NN |

### 类别权重

不改变数据，而是改变模型对待错误的方式。对错误分类少数类赋予更高的权重。

对于一个包含 950 个负样本和 50 个正样本的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类获得 19 倍的权重。错误分类一个正样本的成本等同于错误分类 19 个负样本。这迫使模型关注少数类。

在逻辑回归中，这会修改损失函数：

```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于样本 i 的类别。

类别权重在期望意义上等价于过采样，但不会创建新数据点。这使它们更快，并避免了重复样本的过拟合风险。

### 阈值调整

大多数分类器输出概率。默认阈值是 0.5：如果 P(positive) >= 0.5，则预测为正类。但 0.5 是任意的。当类别不平衡时，最优阈值通常要低得多。

流程：
1. 训练一个模型
2. 在验证集上获取预测概率
3. 扫描 0.0 到 1.0 的阈值
4. 计算每个阈值的 F1（或其他所选指标）
5. 选择最大化你指标的阈值

```mermaid
flowchart LR
    A[模型] --> B[预测概率]
    B --> C[扫描阈值 0.0 到 1.0]
    C --> D[计算每个阈值的 F1]
    D --> E[选择最佳阈值]
    E --> F[在生产中使用]
```

一个模型可能为欺诈交易输出 P(fraud) = 0.15。在阈值 0.5 下，这被分类为非欺诈。在阈值 0.10 下，它被正确抓住。概率校准不如排序重要——只要欺诈获得的概率高于非欺诈，就存在一个能将它们分开的阈值。

### 成本敏感学习

类别权重的泛化。不是使用均匀成本，而是分配特定的错误分类成本：

| | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0（正确） | C_FN = 100 |
| 实际为负 | C_FP = 1 | 0（正确） |

遗漏一笔欺诈交易（FN）的成本是误报（FP）的 100 倍。模型优化总成本，而非总错误数。

这是最原则性的方法，当你能够估计现实世界的成本时。漏诊癌症的成本与导致额外活检的误报成本截然不同。明确这些成本会迫使做出正确的权衡。

### 决策流程图

```mermaid
flowchart TD
    A[开始：不平衡数据集] --> B{有多不平衡？}
    B -->|"< 70/30"| C["轻度：先尝试类别权重"]
    B -->|"70/30 到 95/5"| D["中度：SMOTE + 类别权重"]
    B -->|"> 95/5"| E["严重：组合多种策略"]
    C --> F{数据量足够吗？}
    D --> F
    E --> F
    F -->|"< 1000 个样本"| G["过采样或 SMOTE，避免欠采样"]
    F -->|"1000-10000"| H["SMOTE + 阈值调整"]
    F -->|"> 10000"| I["欠采样可以，或使用类别权重"]
    G --> J[训练 + 用 F1/AUPRC 评估]
    H --> J
    I --> J
    J --> K{召回率足够高吗？}
    K -->|否| L[降低阈值]
    K -->|是| M{精确率可接受吗？}
    M -->|否| N[提高阈值或添加特征]
    M -->|是| O[交付]
```

```figure
class-imbalance
```

## 构建实现

### 步骤 1：生成了一个不平衡数据集

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 步骤 2：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 步骤 3：随机过采样和欠采样

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 步骤 4：带类别权重的逻辑回归

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 步骤 5：阈值调整

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 步骤 6：评估函数

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 步骤 7：比较所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 基线：无处理
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# 过采样
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# 类别权重
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# 阈值调整（在保留的验证集上调整，而非测试集）
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件在一个脚本中运行所有这些并打印结果。

## 使用

使用 scikit-learn 和 imbalanced-learn，这些技术都是一行代码：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零实现的代码清楚地展示了每种技术的原理。SMOTE 只是对少数类进行 k-NN 插值。类别权重乘以损失。阈值调整是对截止值的 for 循环。没有什么神奇的。

## 交付

本课程产生：
- `outputs/skill-imbalanced-data.md` -- 处理不平衡分类问题的决策清单

## 练习

1. **Borderline-SMOTE**：修改 SMOTE 实现，只为靠近决策边界的少数类点生成合成样本（这些点的 k 近邻中包含多数类样本）。在类别重叠的数据集上与标准 SMOTE 比较结果。

2. **代价矩阵优化**：实现代价敏感学习，其中代价矩阵是一个参数。创建一个函数，接收代价矩阵并返回最小化期望代价的最优预测。使用不同的代价比率（1:10、1:100、1:1000）进行测试，并绘制精确率-召回率权衡如何变化。

3. **阈值校准**：实现 Platt scaling（在模型的原始输出上拟合逻辑回归以生成校准概率）。比较校准前后的精确率-召回率曲线。展示校准不改变排序（AUC 保持不变）但使概率更有意义。

4. **平衡 Bagging 集成**：训练多个模型，每个模型都在平衡的 bootstrap 样本上训练（全部少数类 + 多数类的随机子集）。对它们的预测取平均。将此方法与使用 SMOTE 的单个模型进行比较。测量性能与跨运行的方差。

5. **不平衡比例实验**：使用一个平衡的数据集，逐步增加不平衡比例（50/50、70/30、90/10、95/5、99/1）。对每个比例，分别用和不用 SMOTE 训练。绘制两种方法的 F1 vs 不平衡比例图。SMOTE 在什么比例开始产生有意义的差异？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 类别不平衡 | "一个类别的样本远远多于另一个" | 数据集中的类别分布显著偏斜，导致模型偏向多数类 |
| SMOTE | "合成过采样" | 通过在现有少数类样本及其 k 个最近少数类邻居之间插值来创建新的少数类样本 |
| 类别权重 | "让稀有类别的错误代价更高" | 将损失函数乘以类别特定权重，使模型更严厉地惩罚少数类误分类 |
| 阈值调整 | "移动决策边界" | 将分类的概率截止值从默认的 0.5 改为优化所需指标的某个值 |
| 精确率-召回率权衡 | "你无法两全" | 降低阈值会捕捉更多正例（更高召回率）但也标记更多假正例（更低精确率），反之亦然 |
| AUPRC | "PR 曲线下面积" | 将精确率-召回率曲线汇总为单个数字；当类别严重不平衡时比 AUC-ROC 更具信息性 |
| Matthews 相关系数 | "平衡指标" | 预测标签与实际标签之间的相关性，只有当模型在两个类别上都表现良好时才会给出高分 |
| 代价敏感学习 | "不同错误代价不同" | 将现实世界的错误分类代价纳入训练目标，使模型优化总成本而非错误计数 |
| 随机过采样 | "复制少数类" | 重复少数类样本以平衡类别数量；简单但容易因重复点而过拟合 |

## 延伸阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) -- 原始 SMOTE 论文，仍然是引用最多的不平衡学习工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 全面综述，涵盖采样、代价敏感和算法方法
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/) -- Python 库，包含 SMOTE 变体、欠采样策略和管道集成
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 何时以及为何在不平衡问题中优先选择 PR 曲线而非 ROC 曲线
