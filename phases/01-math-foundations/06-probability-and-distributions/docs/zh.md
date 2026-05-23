# 概率与分布

> 概率是AI表达不确定性的语言。

**类型：** 学习
**语言：** Python
**先修课程：** 第一阶段，课程01-04
**时间：** ~75分钟

## 学习目标

- 从头实现伯努利、分类、泊松、均匀和正态分布的概率质量函数（PMF）和概率密度函数（PDF）
- 计算期望值和方差，并运用中心极限定理解释为何高斯分布无处不在
- 使用数值稳定性技巧（减去最大logit）构建softmax和log-softmax函数
- 从logits计算交叉熵损失，并将其与负对数似然联系起来

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从50,000个候选词中选择下一个词。一个扩散模型通过从学习到的分布中采样来生成图像。这些都是概率在起作用。

模型做出的每一个预测都是一个概率分布。每一个损失函数都衡量预测分布与真实分布之间的差距。每一步训练都在调整参数，使得一个分布看起来更像另一个分布。没有概率，你将无法读懂任何一篇机器学习论文，调试任何一个模型，或理解为什么你的训练损失是NaN。

## 概念

### 事件、样本空间和概率

样本空间 S 是所有可能结果的集合。事件是样本空间的一个子集。概率将事件映射到0到1之间的数字。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

概率的全部内容由三条公理定义：
1. 对于任何事件A，P(A) >= 0
2. P(S) = 1（总会发生某些事情）
3. 当A和B不能同时发生时，P(A或B) = P(A) + P(B)

其他所有内容（贝叶斯定理、期望、分布）都源自这三条规则。

### 条件概率与独立性

P(A|B) 是在B发生的条件下A发生的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

当知道一个事件不会告诉你关于另一个事件的任何信息时，这两个事件是独立的：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币是独立的。不放回地抽牌则不是。

### 概率质量函数与概率密度函数

离散随机变量具有概率质量函数（PMF）。每个结果都有一个可以直接读取的具体概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量具有概率密度函数（PDF）。单点处的密度不是概率。概率来自于对密度在某个区间上的积分。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这种区别在机器学习中很重要。分类输出是PMF（离散选择）。VAE的潜在空间使用PDF（连续的）。

### 常见分布

**伯努利分布：** 一次试验，两个结果。模拟二元分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**分类分布：** 一次试验，k个结果。模拟多类分类（softmax输出）。

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**均匀分布：** 所有结果可能性相同。用于随机初始化。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**正态分布（高斯分布）：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**泊松分布：** 固定区间内罕见事件的计数。模拟事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望值与方差

期望值是加权平均结果。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差衡量围绕均值的分散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在机器学习中，期望值表现为损失函数（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度的高方差意味着训练是嘈杂的。

### 联合分布与边际分布

联合分布 P(X, Y) 描述两个随机变量。

联合PMF示例（X = 天气，Y = 雨伞）：

| | Y=0 (无雨伞) | Y=1 (有雨伞) | 边际分布 P(X) |
|---|---|---|---|
| X=0 (晴天) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (雨天) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边际分布 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边际分布通过对另一个变量求和得到：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表中的行和列总和就是边际分布。

### 为何正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均值）收敛于正态分布，而与原始分布无关。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这就是为什么：
- 测量误差近似正态（许多微小独立源）
- 神经网络中的权重初始化使用正态分布
- SGD中的梯度噪声近似正态（许多样本梯度的和）
- 正态分布是给定均值和方差的最大熵分布

### 对数概率

原始概率会导致数值问题。将许多小概率相乘很快会下溢为零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率解决了这个问题。乘法变成了加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0（因为 0 < P <= 1）
- 越负 = 越不可能
- 交叉熵损失是正确类的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax将其转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax技巧：在指数运算前减去最大logit以防止溢出。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 结合了 softmax 和 log 以实现数值稳定性。PyTorch 在交叉熵损失内部使用了它。

### 采样

采样意味着从一个分布中抽取随机值。在机器学习中：
- Dropout 随机采样哪些神经元置零
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要诸如逆变换采样、拒绝采样或重参数化技巧（用于VAE）等技术。

## 动手构建

### 步骤1：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 步骤2：从头实现PMF和PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步骤3：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步骤4：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步骤5：Softmax与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步骤6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含所有可视化的完整实现请见 `code/probability.py`。

## 实际使用

使用 NumPy 和 SciPy，以上所有内容都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从头构建了这些。现在你知道库调用在背后做了什么。

## 练习题

1.  为指数分布实现逆变换采样。通过采样10,000个值并比较直方图与真实PDF来验证。
2.  为两个灌铅骰子构建一个联合分布表。计算边际分布，并检查这两个骰子是否独立。
3.  计算一个5类分类器的交叉熵损失，当它输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]` 且正确类别索引为3时。然后使用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。
4.  编写一个函数，该函数接受一个对数概率列表，并返回最可能的序列、总对数概率以及等效的原始概率。用一个50个词的句子测试它，其中每个词的概率为0.01。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|----------------|----------------------|
| 样本空间 | “所有可能性” | 实验每个可能结果的集合 S |
| PMF | “概率函数” | 给出每个离散结果的确切概率的函数，总和为1 |
| PDF | “概率曲线” | 连续变量的密度函数。对其在一个区间上积分得到概率 |
| 条件概率 | “给定某条件的概率” | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维和贝叶斯定理的基础 |
| 独立性 | “它们互不影响” | P(A 且 B) = P(A) * P(B)。知道一个事件不会告诉你关于另一个事件的任何信息 |
| 期望值 | “平均值” | 所有结果的概率加权和。损失函数就是一个期望值 |
| 方差 | “有多分散” | 偏离均值的期望平方差。高方差 = 噪杂、不稳定的估计 |
| 正态分布 | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于中心极限定理而无处不在 |
| 中心极限定理 | “平均值变得正态” | 许多独立样本的均值收敛于正态分布，而与来源无关 |
| 联合分布 | “两个变量一起” | P(X, Y) 描述了 X 和 Y 结果每种组合的概率 |
| 边际分布 | “对另一个变量求和” | P(X) = sum_y P(X, Y)。从联合分布中恢复一个变量的分布 |
| 对数概率 | “概率的对数” | log P(x)。将乘积转换为和，防止长序列中的数值下溢 |
| Softmax | “将分数转换为概率” | softmax(z_i) = exp(z_j) / sum(exp(z_j))。将实值 logits 映射到有效的概率分布 |
| 交叉熵 | “损失函数” | -sum(p_true * log(p_predicted))。衡量两个分布有多不同。越低越好 |
| Logits | “模型原始输出” | Softmax 之前的未归一化分数。得名于逻辑函数 |
| 采样 | “抽取随机值” | 根据概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown: 但是中心极限定理是什么？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 平均值为何变得正态的直观证明
- [斯坦福 CS229 概率回顾](https://cs229.stanford.edu/section/cs229-prob.pdf) - 简明参考，涵盖此处所有内容及更多
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为何数值稳定性很重要以及如何实现