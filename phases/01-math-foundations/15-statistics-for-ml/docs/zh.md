# 机器学习中的统计学

> 统计学是判断你的模型是真正有效还是仅仅侥幸成功的方法。

**类型：** 构建
**语言：** Python
**先决条件：** 第一阶段，第06课（概率与分布）、第07课（贝叶斯定理）
**时间：** 约120分钟

## 学习目标

- 从零开始计算描述性统计量、皮尔逊/斯皮尔曼相关性以及协方差矩阵
- 执行假设检验（t检验、卡方检验），并正确解释p值和置信区间
- 使用bootstrap重采样构建任意指标的置信区间，无需分布假设
- 使用效应量度量区分统计显著性与实际显著性

## 问题所在

你训练了两个模型。模型A在测试集上得分0.87。模型B得分0.89。你部署了模型B。三周后，生产环境指标比以前更差了。发生了什么？

模型B实际上并没有优于模型A。0.02的差异是噪声。你的测试集太小，或者方差太高，或者两者兼而有之。你发布的其实是伪装成改进的随机性。

这种情况经常发生。Kaggle排行榜的变动。无法复现的论文。基于几百个样本就宣布胜出者的A/B测试。根本原因总是一样的：有人跳过了统计学。

统计学为你提供了区分信号与噪声的工具。它告诉你一个差异何时是真实的，你应该有多大的信心，以及在信任一个结果之前你需要多少数据。每个机器学习管道、每次模型比较、每个实验都需要统计学。没有它，你就是在猜测。

## 概念详解

### 描述性统计：总结你的数据

在你建模之前，需要了解数据的模样。描述性统计将数据集压缩为几个数字，以捕捉其形状。

**集中趋势度量** 回答“中心在哪里？”的问题。

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点。中位数是中间点。当它们出现分歧时，你的分布是偏斜的。收入分布通常具有 均值 >> 中位数（由于亿万富翁导致的右偏）。训练过程中的损失分布通常具有 均值 << 中位数（由于简单样本导致的左偏）。

**离散度度量** 回答“数据有多分散？”的问题。

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数** 将排序后的数据分成100等份。第25百分位数（Q1）意味着25%的值低于此点。第50百分位数是中位数。第75百分位数是Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在机器学习中，你关心百分位数是为了推理延迟、预测置信度分布以及理解误差分布。一个平均误差很低但第99百分位数（P99）误差很糟糕的模型，对于安全关键型应用可能毫无用处。

**样本统计量 vs 总体统计量。** 从样本计算方差时，应除以 (n-1) 而非 n。这是贝塞尔校正。它补偿了你的样本均值并非真实总体均值这一事实。分母为n时，你会系统性地低估真实方差。使用(n-1)，估计量是无偏的。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实践中：如果n很大（数千个样本），差异可以忽略不计。如果n很小（几十个样本），这就很重要了。

### 相关性：变量如何共同变化

相关性衡量两个变量之间线性关系的强度和方向。

**皮尔逊相关系数** 衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

皮尔逊假设关系是线性的，且两个变量都大致服从正态分布。它对异常值敏感。一个极端点可以将r从0.1拖到0.9。

**斯皮尔曼等级相关性** 衡量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何时使用哪个：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：** 相关性不等于因果关系。冰淇淋销量和溺水死亡人数相关，因为两者都在夏天增加。你的模型的准确率和参数数量相关，但增加参数并不自动提高准确率（参见：过拟合）。

### 协方差矩阵

两个变量之间的协方差衡量它们如何共同变化：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于d个特征，协方差矩阵C是一个 d x d 矩阵，其中 C[i][j] = Cov(特征_i, 特征_j)。对角线元素 C[i][i] 是每个特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与PCA的联系。** PCA对协方差矩阵进行特征分解。特征向量是主成分（最大方差方向）。特征值告诉你每个成分捕获了多少方差。这正是第10课所涵盖的内容，但现在你明白了为什么协方差矩阵是正确要分解的东西：它编码了你数据中所有成对的线性关系。

**与相关性的联系。** 相关矩阵是标准化变量（每个变量除以其标准差）的协方差矩阵。相关性对协方差进行归一化，使所有值落在[-1, 1]区间内。

### 假设检验

假设检验是一个在不确定性下做决策的框架。你从一个声明开始，收集数据，然后判断数据是否与该声明一致。

**设置：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p值** 是在假设H0为真的情况下，观察到与你所观察到的数据一样极端（或更极端）的概率。它不是H0为真的概率。这是统计学中最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间** 为一个参数提供了可能值的范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度告诉你精度。宽区间意味着不确定性高。窄区间意味着你的估计是精确的（但未必准确，如果你的数据有偏的话）。

### t检验

t检验比较均值。有几种类型。

**单样本t检验：** 总体均值是否不同于一个假设值？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**双样本t检验（独立）：** 两个组的均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对t检验：** 当测量值成对出现时（在同一数据划分上评估相同模型）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在机器学习中，配对t检验很常见：你在相同的10个交叉验证折上运行两个模型，并逐对比较它们的分数。

### 卡方检验

卡方检验检查观察频率是否与期望频率匹配。对分类数据很有用。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### 机器学习模型的A/B测试

机器学习中的A/B测试与网页A/B测试不同。模型比较有特定的挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性 vs 实际显著性

一个结果可能具有统计显著性，但在实际意义上毫无价值。只要有足够的数据，即使微不足道的差异也会变得统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量** 量化差异有多大，与样本量无关：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终同时报告p值和效应量。p值告诉你差异是否真实。效应量告诉你它是否重要。

### 多重比较问题

当你检验多个假设时，有些会“显著”纯属偶然。如果你在 alpha = 0.05 下检验20件事，即使没有任何真实效应，你也预期会有1个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**邦费罗尼校正：** 将alpha除以检验次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在机器学习中，当你在多个指标上比较一个模型、测试许多超参数配置或在多个数据集上进行评估时，这很重要。

### Bootstrap方法

Bootstrap通过有放回地重采样你的数据来估计一个统计量的抽样分布。无需对底层分布做任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap置信区间（百分位数法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么bootstrap对机器学习很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap用于模型比较：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对t检验更稳健，因为它不假设任何分布。

### 参数检验 vs 非参数检验

**参数检验** 假设一个特定的分布（通常是正态分布）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验** 不假设任何分布：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数检验：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在机器学习实验中，你通常有小的n（5或10个交叉验证折），因此像Wilcoxon符号秩检验这样的非参数检验通常比t检验更合适。

### 中心极限定理：实际含义

中心极限定理指出，随着n增大，样本均值的分布趋近于正态分布，无论底层总体分布如何。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**为什么这对机器学习很重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**中心极限定理做不到的事：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### 机器学习论文中常见的统计错误

1.  **在训练集上测试。** 这保证会导致过拟合。务必留出模型在训练期间从未见过的数据。
2.  **没有置信区间。** 只报告单个准确率数字而不报告不确定性，会使结果不可复现、无法验证。
3.  **忽略多重比较。** 测试50种配置，并在不校正的情况下报告最佳结果，会夸大假阳性率。
4.  **混淆统计显著性与实际显著性。** 对一个0.01%的准确率提升报告p值为0.001是没有意义的。
5.  **在不平衡数据上使用准确率。** 在一个99%都是负类的数据集上达到99%准确率，意味着模型什么都没学到。使用精确率、召回率、F1分数或AUC。
6.  **挑选指标。** 只报告你的模型胜出的指标。诚实的评估应报告所有相关指标。
7.  **在训练/测试划分间泄露信息。** 在划分前进行归一化，或使用未来数据预测过去。
8.  **小测试集且无方差估计。** 在100个样本上进行评估并声称有2%的提升，这是噪声，不是信号。
9.  **假设数据独立但实际不独立。** 来自同一患者的医学图像，来自同一文档的多个句子。组内的观测值是相关的。
10. **P-hacking（P值操纵）。** 尝试不同的检验、子集或排除标准，直到得到 p < 0.05。结果只是搜索过程的产物。

## 动手实现

你将实现：

1.  **从零实现描述性统计**（均值、中位数、众数、标准差、百分位数、四分位距）
2.  **相关性函数**（皮尔逊和斯皮尔曼，以及协方差矩阵）
3.  **假设检验**（单样本t检验、双样本t检验、卡方检验）
4.  **Bootstrap置信区间**（用于任何统计量，无需假设）
5.  **A/B测试模拟器**（生成数据、检验、检查第一类和第二类错误）
6.  **统计显著性 vs 实际显著性演示**（展示大n如何使一切都变得“显著”）

全部从零开始，仅使用 `math` 和 `random`。不使用numpy，不使用scipy。

## 关键术语

| 术语 | 定义 |
|---|---|
| 均值 (Mean) | 值的总和除以计数。对异常值敏感。 |
| 中位数 (Median) | 排序后数据的中间值。对异常值稳健。 |
| 标准差 (Standard deviation) | 方差的平方根。以原始单位衡量离散度。 |
| 百分位数 (Percentile) | 给定百分比数据落于其下的值。 |
| 四分位距 (IQR) | 四分位距。Q3减去Q1。中间50%数据的离散度。 |
| 皮尔逊相关系数 (Pearson correlation) | 衡量两个变量之间的线性关联。范围[-1, 1]。 |
| 斯皮尔曼相关系数 (Spearman correlation) | 使用秩次衡量单调关联。 |
| 协方差矩阵 (Covariance matrix) | 所有特征之间成对协方差的矩阵。 |
| 零假设 (Null hypothesis) | 无效应或无差异的默认假设。 |
| p值 (p-value) | 在零假设为真时，观察到如此极端数据的概率。 |
| 置信区间 (Confidence interval) | 在给定置信水平下，一个参数的可能值范围。 |
| t检验 (t-test) | 检验均值是否显著不同。使用t分布。 |
| 卡方检验 (Chi-squared test) | 检验观察频率是否不同于期望频率。 |
| 效应量 (Effect size) | 差异的大小，与样本量无关。科恩d值是常用的。 |
| 邦费罗尼校正 (Bonferroni correction) | 将显著性阈值除以检验次数以控制假阳性。 |
| Bootstrap | 有放回重采样以估计抽样分布。 |
| 第一类错误 (Type I error) | 假阳性。当H0为真时拒绝H0。 |
| 第二类错误 (Type II error) | 假阴性。当H0为假时未能拒绝H0。 |
| 统计功效 (Statistical power) | 正确拒绝一个假的H0的概率。功效 = 1 减去 第二类错误率。 |
| 中心极限定理 (Central limit theorem) | 随着样本量增大，样本均值收敛于正态分布。 |
| 参数检验 (Parametric test) | 假设数据服从特定分布（通常是正态分布）。 |
| 非参数检验 (Non-parametric test) | 不假设任何分布。基于秩次或符号进行检验。 |