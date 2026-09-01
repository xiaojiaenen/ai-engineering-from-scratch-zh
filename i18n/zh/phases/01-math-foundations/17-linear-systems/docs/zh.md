```markdown
# 线性系统

> 求解 Ax = b 是数学中最古老的问题，但仍然在驱动你的神经网络。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段，课程 01（线性代数直觉），02（向量与矩阵），03（矩阵变换）
**时间：** 约 120 分钟

## 学习目标

- 使用带部分主元的高斯消元法和回代求解 Ax = b
- 用 LU、QR 和 Cholesky 分解对矩阵进行因式分解，并解释每种方法的适用场景
- 推导最小二乘的正则方程，并将其与线性回归和岭回归联系起来
- 利用条件数诊断病态系统，并应用正则化来稳定求解

## 问题背景

每次你训练线性回归时，都在求解一个线性系统。每次你计算最小二乘拟合时，都在求解一个线性系统。每次神经网络层计算 `y = Wx + b` 时，都在求解线性系统的一边。当你添加正则化项时，你修改了该系统。当你使用高斯过程时，你在对一个矩阵进行因式分解。当你为了 Mahalanobis 距离求逆协方差矩阵时，你正在求解一个线性系统。

方程 Ax = b 无处不在。A 是一个已知系数矩阵，b 是一个已知输出向量，x 是你想要求解的未知向量。在线性回归中，A 是数据矩阵，b 是目标向量，x 是权重向量。整个模型简化为：找到 x，使得 Ax 尽可能接近 b。

本课程从头开始构建求解该方程的所有主要方法。你将理解为什么某些方法更快而另一些更稳定，为什么某些方法仅适用于方阵系统而另一些能处理超定系统，以及为什么矩阵的条件数决定了你的答案是否有意义。

## 概念解析

### Ax = b 的几何含义

线性方程组有几何解释。每个方程定义一个超平面。解就是所有超平面的交点（或点集）。

```
2x + y = 5          二维空间中的两条线。
x - y  = 1          它们相交于 x=2, y=1。
```

```mermaid
graph LR
    A["2x + y = 5"] --- S["解：(2, 1)"]
    B["x - y = 1"] --- S
```

三种可能发生的情况：

```mermaid
graph TD
    subgraph "唯一解"
        A1["两条线相交于一个点"]
    end
    subgraph "无解"
        A2["两条线平行——无交点"]
    end
    subgraph "无穷多解"
        A3["两条线重合——每一点都是解"]
    end
```

在矩阵形式中，"唯一解"意味着 A 可逆。"无解"意味着系统不一致。"无穷多解"意味着 A 有空空间（null space）。大多数机器学习问题属于"无精确解"类别，因为你有更多的方程（数据点）比未知量（参数）。这就是最小二乘法出现的原因。

### 列视角与行视角

有两种阅读 Ax = b 的方式。

**行视角。** A 的每一行定义一个方程。每个方程是一个超平面。解是所有这些超平面的交点。

**列视角。** A 的每一列是一个向量。问题变成了：A 的列的什么线性组合能产生 b？

```
A = | 2  1 |    b = | 5 |
    | 1 -1 |        | 1 |

行视角：同时求解 2x + y = 5 和 x - y = 1。

列视角：找到 x1, x2 使得：
  x1 * [2, 1] + x2 * [1, -1] = [5, 1]
  2 * [2, 1] + 1 * [1, -1] = [4+1, 2-1] = [5, 1]   验证。
```

列视角更为基础。如果 b 在 A 的列空间中，则系统有解。如果 b 不在列空间中，则找到列空间中的最近点。这个最近点就是最小二乘解。

### 高斯消元法

高斯消元法将 Ax = b 转化为上三角系统 Ux = c，然后通过回代求解。这是最直接的方法。

算法如下：

```
1. 对于每一列 k（主元列）：
   a. 在列 k 中、第 k 行及以下的元素中找到最大值（部分主元法）。
   b. 将该行与第 k 行交换。
   c. 对于 k 下方的每一行 i：
      - 计算乘数 m = A[i][k] / A[k][k]
      - 从行 i 中减去 m 倍的行 k。
2. 回代：从最后一个方程向上求解。
```

示例：

```
原始：
| 2  1  1 | 8 |       R2 = R2 - (2)R1     | 2  1   1 |  8 |
| 4  3  3 |20 |  -->  R3 = R3 - (1)R1 --> | 0  1   1 |  4 |
| 2  3  1 |12 |                            | 0  2   0 |  4 |

                       R3 = R3 - (2)R2     | 2  1   1 |  8 |
                                       --> | 0  1   1 |  4 |
                                           | 0  0  -2 | -4 |

回代：
  -2 * x3 = -4    -->  x3 = 2
  x2 + 2  = 4     -->  x2 = 2
  2*x1 + 2 + 2 = 8 --> x1 = 2
```

高斯消元法的计算量为 O(n^3) 次运算。对于一个 1000×1000 的系统，约需十亿次浮点运算。这很快，但如果需要用相同的 A 求解多个系统，你可以做得更好。

### 部分主元法：为什么重要

如果不使用主元法，高斯消元法可能会失败或产生错误结果。如果主元元素为零，你会除以零。如果它很小，会放大舍入误差。

```
糟糕的主元：                       使用部分主元法：
| 0.001  1 | 1.001 |            先交换行：
| 1      1 | 2     |            | 1      1 | 2     |
                                 | 0.001  1 | 1.001 |
m = 1/0.001 = 1000              m = 0.001/1 = 0.001
R2 = R2 - 1000*R1               R2 = R2 - 0.001*R1
| 0.001  1     | 1.001   |      | 1      1     | 2     |
| 0     -999   | -999.0  |      | 0      0.999 | 0.999 |

x2 = 1.000（正确）             x2 = 1.000（正确）
x1 = (1.001 - 1)/0.001          x1 = (2 - 1)/1 = 1.000（正确）
   = 0.001/0.001 = 1.000        因为乘数很小，所以稳定。
```

在有限精度的浮点算术中，未使用主元法的版本可能会丢失有效数字。部分主元法总是选择可用的最大主元来最小化误差放大。

### LU 分解

LU 分解将 A 分解为下三角矩阵 L 和上三角矩阵 U：A = LU。L 矩阵存储了高斯消元法中的乘数。U 矩阵是消元的结果。

```
A = L @ U

| 2  1  1 |   | 1  0  0 |   | 2  1   1 |
| 4  3  3 | = | 2  1  0 | @ | 0  1   1 |
| 2  3  1 |   | 1  2  1 |   | 0  0  -2 |
```

为什么要因式分解而不仅仅是消元？因为一旦有了 L 和 U，对于任意新的 b，求解 Ax = b 只需 O(n^2) 次运算：

```
Ax = b
LUx = b
令 y = Ux：
  Ly = b    （前向代入，O(n^2)）
  Ux = y    （回代，O(n^2)）
```

O(n^3) 的代价在分解阶段只支付一次。之后每次求解只需 O(n^2)。如果你需要用相同的 A 但不同的 b 向量求解 1000 个系统，LU 可以节省约 1000/3 倍的工作量。

使用部分主元法，得到 PA = LU，其中 P 是记录行交换的置换矩阵。

### QR 分解

QR 分解将 A 分解为正交矩阵 Q 和上三角矩阵 R：A = QR。

正交矩阵具有性质 Q^T Q = I。其列是标准正交向量。乘以 Q 保持长度和角度不变。

```
A = Q @ R

Q 有标准正交列：Q^T Q = I
R 是上三角矩阵

求解 Ax = b：
  QRx = b
  Rx = Q^T b    （只需乘以 Q^T，无需求逆）
  回代得到 x。
```

QR 在求解最小二乘问题时比 LU 数值上更稳定。Gram-Schmidt 过程逐列构建 Q：

```
给定 A 的列 a1, a2, ...：

q1 = a1 / ||a1||

q2 = a2 - (a2 · q1) * q1        （减去在 q1 上的投影）
q2 = q2 / ||q2||                （归一化）

q3 = a3 - (a3 · q1) * q1 - (a3 · q2) * q2
q3 = q3 / ||q3||

R[i][j] = qi · aj    对于 i <= j
```

每一步都去掉在所有先前 q 向量上的分量，只留下新的正交方向。

### Cholesky 分解

当 A 是对称的（A = A^T）且正定时（所有特征值为正），可以将其分解为 A = L L^T，其中 L 是下三角矩阵。这就是 Cholesky 分解。

```
A = L @ L^T

| 4  2 |   | 2  0 |   | 2  1 |
| 2  5 | = | 1  2 | @ | 0  2 |

L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]    对于 i > j
```

Cholesky 比 LU 快两倍且只需一半的存储空间。它仅适用于对称正定矩阵，但这些矩阵经常出现：

- 协方差矩阵是对称半正定的（带正则化时为正定）。
- 高斯过程中的核矩阵是对称正定的。
- 凸函数在最小值处的 Hessian 矩阵是对称正定的。
- A^T A 总是对称半正定的。

在高斯过程中，你用 Cholesky 分解核矩阵 K，然后求解 K alpha = y 以得到预测均值。Cholesky 因子还给出了对数边际似然：log det(K) = 2 * sum(log(diag(L)))。

### 最小二乘：当 Ax = b 无精确解时

如果 A 是 m×n 矩阵且 m > n（方程数多于未知数），则系统是超定的。不存在精确解。相反，你最小化平方误差：

```
最小化 ||Ax - b||^2

这是残差平方和：
  sum((A[i,:] @ x - b[i])^2 for i in range(m))
```

极小值满足正则方程：

```
A^T A x = A^T b
```

推导：展开 ||Ax - b||^2 = (Ax - b)^T (Ax - b) = x^T A^T A x - 2 x^T A^T b + b^T b。对 x 求梯度并令其为零：2 A^T A x - 2 A^T b = 0。

```
原始系统（超定，4 个方程，2 个未知数）：
| 1  1 |         | 3 |
| 1  2 | x     = | 5 |       没有精确的 x 能满足所有 4 个方程。
| 1  3 |         | 6 |
| 1  4 |         | 8 |

正则方程：
A^T A = | 4  10 |    A^T b = | 22 |
        | 10 30 |            | 63 |

求解：x = [1.5, 1.7]

这就是线性回归。x[0] 是截距，x[1] 是斜率。
```

### 正则方程 = 线性回归

二者之间存在精确的对应关系。在线性回归中，数据矩阵 X 的每一行对应一个样本，每一列对应一个特征。目标向量 y 的每个元素对应一个样本。权重向量 w 满足：

```
X^T X w = X^T y
w = (X^T X)^(-1) X^T y
```

这是线性回归的闭式解。每次调用 `sklearn.linear_model.LinearRegression.fit()` 都会计算这个解（或通过 QR 或 SVD 计算等价形式）。

在矩阵中添加正则化项 lambda * I，就得到岭回归：

```
(X^T X + lambda * I) w = X^T y
w = (X^T X + lambda * I)^(-1) X^T y
```

正则化使矩阵条件更好（更容易准确求逆），并通过将权重向零收缩来防止过拟合。矩阵 X^T X + lambda * I 在 lambda > 0 时总是对称正定的，因此你可以用 Cholesky 分解来求解它。

### 伪逆（Moore-Penrose）

伪逆 A+ 将矩阵求逆推广到了非方阵和奇异矩阵。对于任意矩阵 A：

```
x = A+ b

其中 A+ = V Sigma+ U^T    （通过 SVD 计算）
```

Sigma+ 通过取每个非零奇异值的倒数并转置结果形成。如果 A = U Sigma V^T，则 A+ = V Sigma+ U^T。

```
A = U Sigma V^T        （SVD）

Sigma = | 5  0 |       Sigma+ = | 1/5  0  0 |
        | 0  2 |                | 0  1/2  0 |
        | 0  0 |

A+ = V Sigma+ U^T
```

伪逆给出最小范数最小二乘解。如果系统：
- 有唯一解：A+ b 给出该解。
- 无解：A+ b 给出最小二乘解。
- 有无穷多解：A+ b 给出具有最小 ||x|| 的那个解。

NumPy 的 `np.linalg.lstsq` 和 `np.linalg.pinv` 内部都使用 SVD。

### 条件数

条件数衡量解对输入微小变化的敏感程度。对于矩阵 A，条件数为：

```
kappa(A) = ||A|| * ||A^(-1)|| = sigma_max / sigma_min
```

其中 sigma_max 和 sigma_min 是最大和最小奇异值。

```
良态（kappa ~ 1）：                  病态（kappa ~ 10^15）：
b 的微小变化 -->                     b 的微小变化 -->
x 的微小变化                         x 的巨大变化

| 2  0 |   kappa = 2/1 = 2          | 1   1          |   kappa ~ 10^15
| 0  1 |   可安全求解               | 1   1+10^(-15) |   解是垃圾
```

经验法则：
- kappa < 100：安全，解是准确的。
- kappa ~ 10^k：从浮点算术中损失约 k 位精度。
- kappa ~ 10^16（对于 float64）：解毫无意义。矩阵实际上是奇异的。

在机器学习中，特征近共线会导致病态。正则化（添加 lambda * I）将条件数从 sigma_max / sigma_min 改善为 (sigma_max + lambda) / (sigma_min + lambda)。

### 迭代方法：共轭梯度

对于非常大的稀疏系统（数百万个未知数），直接方法如 LU 或 Cholesky 太昂贵了。迭代方法通过多次迭代改进猜测来逼近解。

共轭梯度（CG）用于求解对称正定矩阵的 Ax = b。它在至多 n 次迭代内找到精确解（在精确算术下），但如果 A 的特征值聚集，通常收敛快得多。

```
算法简述：
  x0 = 初始猜测（通常为零）
  r0 = b - A x0           （残差）
  p0 = r0                 （搜索方向）

  对于 k = 0, 1, 2, ...：
    alpha = (rk · rk) / (pk · A pk)
    x_{k+1} = xk + alpha * pk
    r_{k+1} = rk - alpha * A pk
    beta = (r_{k+1} · r_{k+1}) / (rk · rk)
    p_{k+1} = r_{k+1} + beta * pk
    if ||r_{k+1}|| < 容差：停止
```

CG 应用于：
- 大规模优化（Newton-CG 方法）
- 偏微分方程离散化
- 核矩阵太大无法分解的核方法
- 其他迭代求解器的预处理

收敛速率取决于条件数。条件越好的系统收敛越快，这也是正则化有益的原因之一。

### 完整图景：何时使用何种方法

| 方法 | 要求 | 计算量 | 适用场景 |
|------|------|--------|----------|
| 高斯消元法 | 方阵、非奇异 A | O(n^3) | 一次性求解方阵系统 |
| LU 分解 | 方阵、非奇异 A | O(n^3) 分解 + O(n^2) 求解 | 同一 A 的多次求解 |
| QR 分解 | 任意 A（m >= n） | O(mn^2) | 最小二乘、数值稳定 |
| Cholesky 分解 | 对称正定 A | O(n^3/3) | 协方差矩阵、高斯过程、岭回归 |
| 正则方程 | 超定（m > n） | O(mn^2 + n^3) | 线性回归（小 n） |
| SVD / 伪逆 | 任意 A | O(mn^2) | 秩亏系统、最小范数解 |
| 共轭梯度 | 对称正定、稀疏 A | O(n * k * nnz) | 大型稀疏系统，k = 迭代次数 |

### 与机器学习的联系

本课程中的所有方法都出现在生产级 ML 中：

**线性回归。** 闭式解求解正则方程 X^T X w = X^T y。这通过 Cholesky（如果 n 较小）或 QR（如果数值稳定性重要）或 SVD（如果矩阵可能秩亏）完成。

**岭回归。** 在 X^T X 中添加 lambda * I。正则化系统 (X^T X + lambda * I) w = X^T y 总是可通过 Cholesky 求解，因为 X^T X + lambda * I 在 lambda > 0 时是对称正定的。

**高斯过程。** 预测均值需要求解 K alpha = y，其中 K 是核矩阵。K 的 Cholesky 分解是标准方法。对数边际似然使用 log det(K) = 2 sum(log(diag(L)))。

**神经网络初始化。** 正交初始化使用 QR 分解创建列向量标准正交的权重矩阵。这防止深层网络中的信号坍缩。

**预处理。** 大规模优化器使用不完全 Cholesky 或不完全 LU 作为共轭梯度求解器的预处理子。

**特征工程。** X^T X 的条件数告诉你特征是否共线。如果 kappa 很大，删除特征或添加正则化。

```figure
linear-system-conditioning
```

## 动手实现

### 步骤 1：带部分主元的高斯消元法

```python
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"矩阵在主元 {k} 处奇异或接近奇异")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x
```

### 步骤 2：LU 分解

```python
def lu_decompose(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U

def lu_solve(P, L, U, b):
    n = len(b)
    Pb = P @ b.astype(float)

    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x
```

### 步骤 3：Cholesky 分解

```python
def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("矩阵不是正定矩阵")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L
```

### 步骤 4：通过正则方程的最小二乘

```python
def least_squares_normal(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return gaussian_elimination(AtA, Atb)

def ridge_regression(A, b, lam):
    n = A.shape[1]
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b
    L = cholesky(AtA)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (Atb[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L.T[i, i]
    return x
```

### 步骤 5：条件数

```python
def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]
```

## 实际应用

将所有部分整合在一起，对真实数据进行线性回归和岭回归：

```python
np.random.seed(42)
X_raw = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X_raw @ w_true + np.random.randn(100) * 0.1

X = np.column_stack([np.ones(100), X_raw])

w_ols = least_squares_normal(X, y)
print(f"OLS 权重（我们的实现）：    {w_ols}")

w_np = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS 权重（NumPy）：   {w_np}")
print(f"最大差异：{np.max(np.abs(w_ols - w_np)):.2e}")

w_ridge = ridge_regression(X, y, lam=1.0)
print(f"岭回归权重（我们的实现）：  {w_ridge}")

from sklearn.linear_model import Ridge
ridge_sk = Ridge(alpha=1.0, fit_intercept=False)
ridge_sk.fit(X, y)
print(f"岭回归权重（sklearn）： {ridge_sk.coef_}")
```

## 交付成果

本课程产出：
- `code/linear_systems.py` 包含从头实现的高斯消元法、LU 分解、Cholesky 分解、最小二乘和岭回归
- 一个工作演示，证明正则方程和 sklearn 的 LinearRegression 产生相同的权重

## 练习

1. 使用你的高斯消元法、LU 求解器和 `np.linalg.solve` 求解系统 `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]`。验证三者给出的答案在浮点容差内一致。

2. 生成一个 50×5 随机矩阵 X 和目标 y = X @ w_true + noise。使用正则方程、QR（通过 `np.linalg.qr`）、SVD（通过 `np.linalg.svd`）和 `np.linalg.lstsq` 求解 w。比较这四种解。测量 X^T X 的条件数，并解释它如何影响你对方法的信任程度。

3. 通过使两列几乎相同来创建一个近奇异矩阵（例如，第 2 列 = 第 1 列 + 1e-10 * noise）。计算其条件数。在有和没有正则化的情况下求解 Ax = b（添加 0.01 * I）。比较解和残差。解释为什么正则化有帮助。

4. 对一个 100×100 随机对称正定矩阵实现共轭梯度算法。统计收敛到容差 1e-8 所需的迭代次数。与理论最大值 n 次迭代进行比较。

5. 在大小为 10、50、200、500 的对称正定矩阵上，比较你的 Cholesky 求解器、LU 求解器和 `np.linalg.solve` 的运行时间。绘制结果。验证 Cholesky 大约比 LU 快 2 倍。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 线性系统 | "求解 x" | 一组线性方程 Ax = b。找到 x 意味着找到在变换 A 下产生输出 b 的输入。 |
| 高斯消元法 | "行约简" | 使用行运算系统地消去对角线以下的条目，产生可通过回代求解的上三角系统。O(n^3)。 |
| 部分主元法 | "为稳定性交换行" | 在对第 k 列进行消元之前，将第 k 列中绝对值最大的行交换到主元位置。防止除以小数。 |
| LU 分解 | "分解为三角形" | 将 A 写成 A = LU，其中 L 是下三角矩阵（存储乘数），U 是上三角矩阵（消元后的矩阵）。将 O(n^3) 的代价分摊到多次求解中。 |
| QR 分解 | "正交分解" | 将 A 写成 A = QR，其中 Q 有标准正交列，R 是上三角矩阵。对于最小二乘比 LU 更稳定。 |
| Cholesky 分解 | "矩阵的平方根" | 对于对称正定矩阵 A，将 A 写成 A = LL^T。成本是 LU 的一半。用于协方差矩阵、核矩阵和岭回归。 |
| 最小二乘 | "精确解不可得时的最佳拟合" | 当系统超定（方程数多于未知数）时，最小化残差平方和 ||Ax - b||^2。 |
| 正则方程 | "微积分捷径" | A^T A x = A^T b。将 ||Ax - b||^2 的梯度设为零。这就是线性回归的闭式解。 |
| 伪逆 | "非方阵的求逆" | 通过 SVD 的 A+ = V Sigma+ U^T。对任意矩阵（无论方阵还是矩形、奇异或非奇异）给出最小范数最小二乘解。 |
| 条件数 | "答案有多可信" | kappa = sigma_max / sigma_min。衡量对输入扰动的敏感性。损失约 log10(kappa) 位精度。 |
| 岭回归 | "正则化最小二乘" | 求解 (X^T X + lambda I) w = X^T y。添加 lambda I 改善条件数并将权重向零收缩。防止过拟合。 |
| 共轭梯度 | "大矩阵的迭代 Ax=b" | 对称正定系统的迭代求解器。至多 n 步收敛。适用于因式分解过于昂贵的大型稀疏系统。 |
| 超定系统 | "数据多于参数" | m × n 系统中 m > n。不存在精确解。最小二乘找到最佳近似。这就是每个回归问题。 |
| 回代 | "从底向上求解" | 给定上三角系统，从最后一个方程开始求解，然后向后代入。O(n^2)。 |
| 前向代入 | "从顶向下求解" | 给定下三角系统，从第一个方程开始求解，然后向前代入。O(n^2)。用于 LU 求解的 L 步骤。 |

## 延伸阅读

- [MIT 18.06：线性代数](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)（Gilbert Strang）——关于线性系统和矩阵因式分解的决定性课程
- [数值线性代数](https://people.maths.ox.ac.uk/trefethen/text.html)（Trefethen & Bau）——理解数值稳定性、条件数和算法失效原因的权威参考
- [矩阵计算](https://www.cs.cornell.edu/cv/GolubVanLoan4/golubandvanloan.htm)（Golub & Van Loan）——涵盖所有矩阵算法的百科全书式参考
- [3Blue1Brown：逆矩阵](https://www.3blue1brown.com/lessons/inverse-matrices) ——求解 Ax = b 的几何直观
```
