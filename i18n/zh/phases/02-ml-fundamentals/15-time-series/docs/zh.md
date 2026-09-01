# 时间序列基础

> 过去的表现确实能预测未来的结果——只要你先检查平稳性。

**类型：** Build
**语言：** Python
**前置要求：** Phase 2，课程 01-09
**时间：** 约 90 分钟

## 学习目标

- 将时间序列分解为趋势、季节性和残差分量，并检验平稳性
- 实现滞后特征和滚动统计量，将时间序列转化为监督学习问题
- 构建前向验证框架，防止未来数据泄漏到训练中
- 解释为什么随机训练/测试划分对时间序列无效，并演示其相对于正确时序划分的性能差距

## 问题所在

你拥有按时间排序的数据。日销售额、小时温度、每分钟 CPU 使用率、周股价。你想预测下一个值、下一周、下一季度。

你伸出手中的标准 ML 工具箱：随机训练/测试划分、交叉验证、特征矩阵输入，预测结果输出。每一步都是错的。

时间序列破坏了标准 ML 所依赖的假设。样本不是独立的——今天的温度取决于昨天。在回测中表现良好的特征在生产环境中失效，因为它们依赖于随时间变化的模式。

一个用随机交叉验证获得 95% 准确率的模型，用正确的基于时间的方法评估可能只有 55%。这个差距不是技术细节。它是模型仅适用于纸面与实际生产环境的区别。

本课程内容涵盖：时间数据的独特之处、如何诚实地评估模型、如何将时间序列转化为标准 ML 模型可以处理的特征。

## 概念解析

### 时间序列的独特之处

标准 ML 假设 i.i.d.——独立同分布。每个样本从同一分布中独立抽取。时间序列同时违反这两个条件：

- **不独立。** 今天的股价取决于昨天。本周的销售与上周相关。
- **不同分布。** 分布随时间漂移。12 月的销售与 3 月的销售看起来不同。

这些违反不是小事。它们改变了你构建特征、评估模型以及选择算法的方式。

```mermaid
flowchart LR
    subgraph IID["标准 ML (i.i.d.)"]
        direction TB
        S1[样本 1] ~~~ S2[样本 2]
        S2 ~~~ S3[样本 3]
    end
    subgraph TS["时间序列 (非 i.i.d.)"]
        direction LR
        T1[t=1] --> T2[t=2]
        T2 --> T3[t=3]
        T3 --> T4[t=4]
    end

    style S1 fill:#dfd
    style S2 fill:#dfd
    style S3 fill:#dfd
    style T1 fill:#ffd
    style T2 fill:#ffd
    style T3 fill:#ffd
    style T4 fill:#ffd
```

在标准 ML 中，样本是可互换的。打乱它们的顺序什么都不会改变。在时间序列中，顺序就是一切。打乱顺序会破坏信号。

### 时间序列的组成

每个时间序列都是一个组合：

```mermaid
flowchart TD
    A[观测时间序列] --> B[趋势]
    A --> C[季节性]
    A --> D[残差/噪声]

    B --> E[长期方向：上升、下降、平稳]
    C --> F[重复模式：日、周、年]
    D --> G[移除趋势和季节性后的随机变化]
```

- **趋势：** 长期方向。收入每年增长 10%。全球气温上升。
- **季节性：** 固定间隔的重复模式。零售销售额在 12 月激增。空调使用量在 7 月达到峰值。
- **残差：** 移除趋势和季节性后剩余的部分。如果残差看起来像白噪声，说明分解捕捉到了信号。

### 平稳性

如果一个时间序列的统计特性（均值、方差、自相关）不随时间变化，则称该序列是平稳的。大多数预测方法假设平稳性。

**为什么重要：** 非平稳序列的均值会漂移。在 1 月数据上训练的模型学到的均值与 2 月将显示的均值不同。它会有系统性偏差。

**如何检查：** 计算窗口内的滚动均值和滚动标准差。如果它们漂移，则序列是非平稳的。

**如何解决：** 差分。不对原始值建模，而是对连续值之间的变化建模：

```
diff[t] = value[t] - value[t-1]
```

如果一次差分不能使序列平稳，再次应用（二阶差分）。大多数现实世界的序列最多需要两轮差分。

**示例：**

原始序列：[100, 102, 106, 112, 120]
一阶差分：  [2, 4, 6, 8]（仍然呈上升趋势）
二阶差分：   [2, 2, 2]（恒定——平稳）

原始序列具有二次趋势。一阶差分将其变为线性趋势。二阶差分使其变为水平。在实践中，你几乎不需要超过两轮。

**形式化检验：** 增广迪基-富勒（ADF）检验是平稳性的标准统计检验。原假设是"序列是非平稳的"。p 值低于 0.05 意味着你可以拒绝原假设并得出平稳性的结论。我们不会从头实现 ADF（它需要渐近分布表），但代码中的滚动统计方法提供了实用的视觉检查。

### 自相关

自相关衡量时间 t 的值与时间 t-k（k 步之前）的值之间的相关程度。自相关函数（ACF）绘制了每个滞后 k 的相关系数。

**ACF 告诉你：**
- 序列的记忆深度。如果 ACF 在滞后 5 之后降为零，则 5 步之前的值无关紧要。
- 是否存在季节性。如果 ACF 在滞后 12（月度数据）出现峰值，则存在年度季节性。
- 要创建多少个滞后特征。使用直到 ACF 变得可忽略不计的滞后。

**偏自相关函数（PACF）** 消除间接相关。如果今天与 3 天前相关只是因为它与昨天都相关，那么滞后 3 的 PACF 将为零，而滞后 3 的 ACF 不为零。

### 滞后特征：将时间序列转化为监督学习

标准 ML 模型需要一个特征矩阵 X 和一个目标 y。时间序列给你一个单列值。桥梁就是滞后特征。

取序列 [10, 12, 14, 13, 15] 并创建滞后 1 和滞后 2 特征：

| lag_2 | lag_1 | target |
|-------|-------|--------|
| 10    | 12    | 14     |
| 12    | 14    | 13     |
| 14    | 13    | 15     |

现在你有了一个标准的回归问题。任何 ML 模型（线性回归、随机森林、梯度提升）都可以从滞后特征预测目标值。

可以创建的其他特征：
- **滚动统计量：** 最后 k 个值的均值、标准差、最小值、最大值
- **日历特征：** 星期几、月份、是否节假日、是否周末
- **差分后的值：** 与前一步的变化
- **扩展统计量：** 累积均值、累积求和
- **比率特征：** 当前值 / 滚动均值（距离近期平均有多远）
- **交互特征：** lag_1 * day_of_week（工作日对动量的影响）

**滞后多少个？** 使用自相关函数。如果 ACF 在滞后 10 之前显著，则使用至少 10 个滞后。如果有周季节性，包括滞后 7（以及可能的 14）。更多的滞后给模型更多历史，但也增加了过拟合的风险。

**目标对齐陷阱。** 在创建滞后特征时，目标必须是时间 t 的值，所有特征必须使用时间 t-1 或更早的值。如果你不小心将时间 t 的值作为特征包含在内，你就有了一个完美的预测器——以及一个完全无用的模型。这是时间序列特征工程中常见的 bug。

### 前向验证

这是本课程最重要的概念。标准的 k 折交叉验证随机将样本分配给训练集和测试集。对于时间序列，这会泄漏未来信息。

```mermaid
flowchart TD
    subgraph WRONG["随机划分（错误）"]
        direction LR
        W1[1 月] --> W2[3 月]
        W2 --> W3[2 月]
        W3 --> W4[5 月]
        W4 --> W5[4 月]
        style W1 fill:#fdd
        style W3 fill:#fdd
        style W5 fill:#fdd
        style W2 fill:#dfd
        style W4 fill:#dfd
    end

    subgraph RIGHT["前向验证（正确）"]
        direction LR
        R1["训练：1-3 月"] --> R2["测试：4 月"]
        R3["训练：1-4 月"] --> R4["测试：5 月"]
        R5["训练：1-5 月"] --> R6["测试：6 月"]
        style R1 fill:#dfd
        style R2 fill:#fdd
        style R3 fill:#dfd
        style R4 fill:#fdd
        style R5 fill:#dfd
        style R6 fill:#fdd
    end
```

前向验证：
1. 在截至时间 t 的数据上训练
2. 预测时间 t+1（或多步预测的 t+1 到 t+k）
3. 滑动窗口向前
4. 重复

每个测试折叠只包含在训练数据之后的数据。没有未来泄漏。这给了你一个诚实的模型部署性能估计。

**扩展窗口** 使用所有历史数据进行训练（窗口增长）。**滑动窗口** 使用固定大小的训练窗口（窗口滑动）。如果你认为旧数据仍然相关，使用扩展窗口。当世界发生变化且旧数据有害时，使用滑动窗口。

### ARIMA 直觉

ARIMA 是经典的时间序列模型。它有三个组件：

- **AR（自回归）：** 从过去值预测。AR(p) 使用最后 p 个值。
- **I（差分整合）：** 差分以实现平稳性。I(d) 应用 d 轮差分。
- **MA（移动平均）：** 从过去的预测误差预测。MA(q) 使用最后 q 个误差。

ARIMA(p, d, q) 组合了所有三个。你根据 ACF/PACF 分析或自动搜索（auto-ARIMA）选择 p、d、q。

我们不会从头实现 ARIMA——它需要超出本课程范围的数值优化。关键洞察是理解每个组件的作用，以便你能够解释 ARIMA 结果并知道何时使用它。

### 何时使用什么

| 方法 | 最佳适用 | 处理季节性 | 处理外部特征 |
|----------|---------|-------------------|------------------------|
| 滞后特征 + ML | 具有许多外部特征的表格数据 | 通过日历特征 | 是 |
| ARIMA | 单变量序列，短期预测 | SARIMA 变体 | 否（ARIMAX 有限） |
| 指数平滑 | 简单趋势 + 季节性 | 是（Holt-Winters） | 否 |
| Prophet | 商业预测、节假日 | 是（傅里叶项） | 有限 |
| 神经网络（LSTM、Transformer） | 长序列、多序列 | 学习得到 | 是 |

对于大多数实际问题，滞后特征 + 梯度提升是最强的起点。它自然地处理外部特征，不需要平稳性，且易于调试。

### 预测时域与策略

单步预测预测一个时间步。多步预测预测多个步。有三种策略：

**递归（迭代）：** 预测一步，将预测值作为下一步的输入。简单但误差累积——每个预测使用上一个预测，所以错误会累积。

**直接：** 为每个时域训练一个单独的模型。模型 1 预测 t+1，模型 5 预测 t+5。没有误差累积，但每个模型的训练样本更少，且它们不共享信息。

**多输出：** 训练一个同时输出所有时域的模型。跨时域共享信息，但需要支持多输出的模型（或自定义损失函数）。

对于大多数实际问题，短时域（1-5 步）从递归开始，长时域使用直接方法。

### 时间序列中的常见错误

| 错误 | 原因 | 解决方法 |
|---------|---------------|-----------|
| 随机训练/测试划分 | 来自标准 ML 的习惯 | 使用前向验证或时序划分 |
| 使用未来特征 | 不小心包含了时间 t 的特征 | 审查每个特征的时序对齐 |
| 对季节性过拟合 | 模型记住了日历模式 | 在测试集中保留完整的季节性周期 |
| 忽略尺度变化 | 收入翻倍但模式不变 | 对百分比变化建模而非绝对值 |
| 过多滞后特征 | "更多历史更好" | 使用 ACF 确定相关滞后 |
| 不做差分 | "模型会自己搞定" | 树模型处理趋势；线性模型需要平稳性 |

```figure
f3-series-decompose
```

## 动手实现

`code/time_series.py` 中的代码从头实现了核心构建模块。

### 滞后特征生成器

```python
def make_lag_features(series, n_lags):
    n = len(series)
    X = np.full((n, n_lags), np.nan)
    for lag in range(1, n_lags + 1):
        X[lag:, lag - 1] = series[:-lag]
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], series[valid]
```

这将一维序列转换为特征矩阵，其中每行包含最后 `n_lags` 个值作为特征，当前值作为目标。

### 前向交叉验证

```python
def walk_forward_split(n_samples, n_splits=5, min_train=50):
    assert min_train < n_samples, "min_train must be less than n_samples"
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)
```

每个划分确保训练数据严格在测试数据之前。训练窗口随着每个折叠而扩展。

### 简单自回归模型

纯 AR 模型就是对滞后特征的线性回归：

```python
class SimpleAR:
    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.weights = None
        self.bias = None

    def fit(self, series):
        X, y = make_lag_features(series, self.n_lags)
        # 通过正态方程求解
        X_b = np.column_stack([np.ones(len(X)), X])
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.bias = theta[0]
        self.weights = theta[1:]
        return self
```

这在概念上与课程 02 中的线性回归相同，但应用于同一变量的时间滞后版本。

### 平稳性检验

代码计算滚动统计量以从视觉和数值上评估平稳性：

```python
def check_stationarity(series, window=50):
    rolling_mean = np.array([
        series[max(0, i - window):i].mean()
        for i in range(1, len(series) + 1)
    ])
    rolling_std = np.array([
        series[max(0, i - window):i].std()
        for i in range(1, len(series) + 1)
    ])
    return rolling_mean, rolling_std
```

如果滚动均值漂移或滚动标准差变化，则序列是非平稳的。应用差分并再次检查。

代码还通过比较序列的前半部分和后半部分来检查平稳性。如果均值差异超过半个标准差或方差比超过 2 倍，则将序列标记为非平稳。

### 自相关

```python
def autocorrelation(series, max_lag=20):
    n = len(series)
    mean = series.mean()
    var = series.var()
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        cov = np.mean((series[:n-k] - mean) * (series[k:] - mean))
        acf[k] = cov / var if var > 0 else 0
    return acf
```

## 使用指南

使用 sklearn，你可以直接将滞后特征用于任何回归器：

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_lag_features(series, n_lags=10)

for train_idx, test_idx in walk_forward_split(len(X)):
    model = Ridge(alpha=1.0)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

对于 ARIMA，使用 statsmodels：

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(5, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
```

`time_series.py` 中的代码演示了两种方法，并使用前向验证进行比较。

### sklearn TimeSeriesSplit

sklearn 提供了 `TimeSeriesSplit`，它实现了前向验证：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

这等效于我们从头实现的 `walk_forward_split`，但集成到 sklearn 的交叉验证框架中。你可以将其与 `cross_val_score` 一起使用：

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
print(f"平均分数: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### 评估指标

时间序列预测使用回归指标，但带有时间感知上下文：

- **MAE（平均绝对误差）：** \|y_true - y_pred\| 的平均值。易于理解原始单位。"平均而言，预测偏差 3.2 度。"
- **RMSE（均方根误差）：** 均方误差的平方根。对大误差的惩罚比 MAE 更重。当大误差比许多小误差更糟时使用。
- **MAPE（平均绝对百分比误差）：** \|error / true_value\| * 100 的平均值。与尺度无关，适用于跨不同序列的比较。但当真实值为零时未定义。
- **朴素基线比较：** 始终与简单基线进行比较。季节性朴素基线预测一个周期前的值（昨天、上周）。如果你的模型不能胜过朴素预测，那就有问题。

### 滚动特征

代码演示了在滞后特征中添加滚动统计量（7 天和 14 天窗口的均值、标准差、最小值、最大值）。这些给模型提供了关于近期趋势和波动性的信息，而滞后特征本身无法捕捉这些信息。

例如，如果滚动均值在上升，表明存在上升趋势。如果滚动标准差在增加，表明波动性增大。这些是树模型可以学习但线性模型无法学习的模式类型。

## 部署准备

本课程产出：
- `outputs/prompt-time-series-advisor.md` -- 用于框定时间序列问题的提示词
- `code/time_series.py` -- 滞后特征、前向验证、AR 模型、平稳性检验

### 必须胜过的基线

在构建任何模型之前，建立基线：

1. **最后值（持续性）。** 预测明天的值与今天相同。对于许多序列，这出奇地难以超越。
2. **季节性朴素。** 预测今天的值与上周同一天（或去年同一天）相同。如果你的模型不能胜过这个，它没有学到任何超越季节性的有用模式。
3. **移动平均。** 预测最后 k 个值的平均值。平滑噪声但无法捕捉突然变化。

如果你的复杂 ML 模型输给了季节性朴素基线，你有一个 bug。最常见原因：特征中的未来泄漏、错误的评估方法，或者序列本身就是随机且不可预测的。

### 实用建议

1. **从绘图开始。** 在任何建模之前，绘制原始序列。寻找趋势、季节性、异常值、结构性断点（行为的突然变化）。30 秒的视觉检查通常比一小时的自动分析更有用。

2. **先差分，再建模。** 如果序列有明显的趋势，在创建滞后特征之前对其进行差分。树模型可以处理趋势，但线性模型不能，而且差分从不hurt。

3. **保留至少一个完整的季节性周期。** 如果你有周季节性，你的测试集需要至少一个完整的星期。如果是月度，至少一个完整的月份。否则你无法评估模型是否捕捉到了季节性模式。

4. **在生产中监控。** 时间序列模型随着世界变化而退化。跟踪滚动预测误差。当误差开始增加时，在最新数据上重新训练模型。

5. **警惕制度变化。** 在疫情期间训练的数据不会预测疫情后的行为。将已知制度变化的指示器作为特征包含，或使用会遗忘旧数据的滑动窗口。

6. **对偏斜序列进行对数变换。** 收入、价格和计数通常右偏。取对数稳定方差并使乘法模式变为加法，线性模型可以处理。在对数空间预测，然后取指数回到原始单位。

## 练习

1. **平稳性实验。** 生成具有线性趋势的序列。用滚动统计量检查平稳性。应用一阶差分。再次检查。二次趋势需要多少轮差分才能平稳？

2. **滞后选择。** 在季节性序列（周期=7）上计算 ACF。哪些滞后具有最高的自相关？仅使用这些滞后创建滞后特征（不是连续滞后）。与使用 1 到 7 滞后相比，准确率是否提高？

3. **前向验证 vs 随机划分。** 在滞后特征上训练 Ridge 回归。用随机 80/20 划分和前向验证进行评估。随机划分高估了多少性能？

4. **特征工程。** 在滞后特征中添加滚动均值（窗口=7）、滚动标准差（窗口=7）和星期几特征。使用前向验证比较有和无这些额外特征的准确率。

5. **多步预测。** 修改 AR 模型以预测 5 步 ahead 而非 1 步。比较两种策略：(a) 预测一步，将预测值作为下一步的输入（递归），(b) 为每个时域训练单独的模型（直接）。哪种更准确？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------------|----------------------|
| 平稳性 | "统计特性不随时间变化" | 均值、方差和自相关结构随时间保持恒定的序列 |
| 差分 | "减去连续值" | 计算 y[t] - y[t-1] 以消除趋势并实现平稳性 |
| 自相关（ACF） | "序列与自身的关联程度" | 时间序列与其滞后副本之间的相关性，作为滞后的函数 |
| 偏自相关（PACF） | "仅直接相关" | 移除所有更短滞后影响后的滞后 k 自相关 |
| 滞后特征 | "过去的值作为输入" | 使用 y[t-1]、y[t-2]、...、y[t-k] 作为特征预测 y[t] |
| 前向验证 | "尊重时间顺序的交叉验证" | 训练数据在时间上始终先于测试数据的评估方法 |
| ARIMA | "经典时间序列模型" | 自回归积分滑动平均：组合过去值（AR）、差分（I）和过去误差（MA） |
| 季节性 | "重复的日历模式" | 与日历周期（日、周、年）相关的规律可预测周期 |
| 趋势 | "长期方向" | 序列水平随时间的持续增加或减少 |
| 扩展窗口 | "使用所有历史" | 训练集随每个折叠增长的验证方法 |
| 滑动窗口 | "固定大小历史" | 训练集为固定长度窗口向前滑动的验证方法 |

## 延伸阅读

- [Hyndman 和 Athanasopoulos，《Forecasting: Principles and Practice（第 3 版）》](https://otexts.com/fpp3/) -- 最佳免费时间序列预测教材
- [scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) -- sklearn 的前向验证分割器
- [statsmodels ARIMA 文档](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) -- 带有诊断的 ARIMA 实现
- [Makridakis 等，《The M5 Competition（2022）》](https://www.sciencedirect.com/science/article/pii/S0169207021001874) -- 大规模预测竞赛，展示 ML 方法与统计方法的对比
