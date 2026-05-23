# 线性方程组

> 求解 Ax = b 是数学中最古老的问题，而今它仍在驱动你的神经网络。

**类型:** 构建
**语言:** Python
**前置知识:** 阶段1，课程01（线性代数直觉），02（向量与矩阵），03（矩阵变换）
**时间:** 约120分钟

## 学习目标

- 使用带部分主元的高斯消元法和回代法求解 Ax = b
- 运用 LU、QR 和 Cholesky 分解对矩阵进行因式分解，并解释每种方法的适用场景
- 推导最小二乘的正规方程，并将其与线性回归和岭回归联系起来
- 使用条件数诊断病态系统，并应用正则化来稳定它们

## 问题

每次训练线性回归模型时，你都在求解一个线性方程组。每次计算最小二乘拟合时，你也在求解一个线性方程组。每当一个神经网络层计算 `y = Wx + b` 时，它实际上是在评估一个线性方程组的一边。当你添加正则化时，你在修改该系统。当你使用高斯过程时，你在对一个矩阵进行因式分解。当你为了计算马氏距离而对协方差矩阵求逆时，你同样在求解一个线性方程组。

方程 Ax = b 无处不在。A 是已知系数的矩阵。b 是已知输出的向量。x 是你想要找到的未知向量。在线性回归中，A 是你的数据矩阵，b 是你的目标向量，x 是权重向量。整个模型可以简化为：找到 x，使得 Ax 尽可能接近 b。

本课程将从头开始构建求解该方程的所有主要方法。你将理解为什么有些方法速度快而有些更稳定，为什么有些方法仅适用于方阵系统而其他方法能处理超定系统，以及为什么矩阵的条件数决定了你的答案是否有意义。

## 概念

### Ax = b 的几何意义

线性方程组有几何解释。每个方程定义一个超平面。解就是所有超平面相交的点（或点集）。

```
2x + y = 5          Two lines in 2D.
x - y  = 1          They intersect at x=2, y=1.
```

```mermaid
graph LR
    A["2x + y = 5"] --- S["Solution: (2, 1)"]
    B["x - y = 1"] --- S
```

可能出现三种情况：

```mermaid
graph TD
    subgraph "One Solution"
        A1["Lines intersect at a single point"]
    end
    subgraph "No Solution"
        A2["Lines are parallel — no intersection"]
    end
    subgraph "Infinite Solutions"
        A3["Lines are identical — every point is a solution"]
    end
```

矩阵形式下，“一个解”意味着 A 可逆。“无解”意味着系统不一致。“无穷多解”意味着 A 有零空间。大多数机器学习问题属于“无精确解”类别，因为你拥有的方程（数据点）多于未知数（参数）。这正是最小二乘法发挥作用的地方。

### 列图 vs 行图

有两种方式来理解 Ax = b。

**行图。** A 的每一行定义一个方程。每个方程是一个超平面。解是它们相交的地方。

**列图。** A 的每一列是一个向量。问题变成：A 的列的什么线性组合能产生 b？

```
A = | 2  1 |    b = | 5 |
    | 1 -1 |        | 1 |

Row picture: solve 2x + y = 5 and x - y = 1 simultaneously.

Column picture: find x1, x2 such that:
  x1 * [2, 1] + x2 * [1, -1] = [5, 1]
  2 * [2, 1] + 1 * [1, -1] = [4+1, 2-1] = [5, 1]   check.
```

列图更为根本。如果 b 位于 A 的列空间内，则系统有解。如果 b 不在列空间内，则你需要找到列空间中最近的点。那个最近点就是最小二乘解。

### 高斯消元法

高斯消元法将 Ax = b 转换为一个上三角系统 Ux = c，然后通过回代法求解。这是最直接的方法。

算法如下：

```
1. For each column k (the pivot column):
   a. Find the largest entry in column k at or below row k (partial pivoting).
   b. Swap that row with row k.
   c. For each row i below k:
      - Compute multiplier m = A[i][k] / A[k][k]
      - Subtract m times row k from row i.
2. Back substitute: solve from the last equation upward.
```

示例：

```
Original:
| 2  1  1 | 8 |       R2 = R2 - (2)R1     | 2  1   1 |  8 |
| 4  3  3 |20 |  -->  R3 = R3 - (1)R1 --> | 0  1   1 |  4 |
| 2  3  1 |12 |                            | 0  2   0 |  4 |

                       R3 = R3 - (2)R2     | 2  1   1 |  8 |
                                       --> | 0  1   1 |  4 |
                                           | 0  0  -2 | -4 |

Back substitute:
  -2 * x3 = -4    -->  x3 = 2
  x2 + 2  = 4     -->  x2 = 2
  2*x1 + 2 + 2 = 8 --> x1 = 2
```

高斯消元法的时间复杂度为 O(n^3)。对于一个 1000x1000 的系统，大约需要十亿次浮点运算。速度很快，但如果你需要用相同的 A 求解多个系统，还可以做得更好。

### 部分主元：为什么它很重要

没有主元选择，高斯消元法可能会失败或产生垃圾结果。如果主元元素为零，你会遇到除以零错误。如果主元很小，你会放大舍入误差。

```
Bad pivot:                       With partial pivoting:
| 0.001  1 | 1.001 |            Swap rows first:
| 1      1 | 2     |            | 1      1 | 2     |
                                 | 0.001  1 | 1.001 |
m = 1/0.001 = 1000              m = 0.001/1 = 0.001
R2 = R2 - 1000*R1               R2 = R2 - 0.001*R1
| 0.001  1     | 1.001   |      | 1      1     | 2     |
| 0     -999   | -999.0  |      | 0      0.999 | 0.999 |

x2 = 1.000 (correct)            x2 = 1.000 (correct)
x1 = (1.001 - 1)/0.001          x1 = (2 - 1)/1 = 1.000 (correct)
   = 0.001/0.001 = 1.000        Stable because the multiplier is small.
```

在精度有限的浮点运算中，不选主元的版本可能会损失有效数字。部分主元总是选择最大的可用主元，以最小化误差放大。

### LU 分解

LU 分解将 A 分解为一个下三角矩阵 L 和一个上三角矩阵 U：A = LU。L 矩阵存储高斯消元过程中的乘数。U 矩阵是消元的结果。

```
A = L @ U

| 2  1  1 |   | 1  0  0 |   | 2  1   1 |
| 4  3  3 | = | 2  1  0 | @ | 0  1   1 |
| 2  3  1 |   | 1  2  1 |   | 0  0  -2 |
```

为什么要因式分解而不是直接消元？因为一旦你有了 L 和 U，对任意新的 b 求解 Ax = b 只需 O(n^2) 的时间：

```
Ax = b
LUx = b
Let y = Ux:
  Ly = b    (forward substitution, O(n^2))
  Ux = y    (back substitution, O(n^2))
```

O(n^3) 的计算成本在分解过程中只支付一次。随后的每次求解都是 O(n^2)。如果你需要用相同的 A 但不同的 b 向量求解 1000 个系统，LU 分解在总工作量上节省了大约 1000/3 的因子。

结合部分主元，你得到 PA = LU，其中 P 是记录行交换的置换矩阵。

### QR 分解

QR 分解将 A 分解为一个正交矩阵 Q 和一个上三角矩阵 R：A = QR。

正交矩阵具有性质 Q^T Q = I。它的列是标准正交向量。乘以 Q 会保持长度和角度不变。

```
A = Q @ R

Q has orthonormal columns: Q^T Q = I
R is upper triangular

To solve Ax = b:
  QRx = b
  Rx = Q^T b    (just multiply by Q^T, no inversion needed)
  Back substitute to get x.
```

在数值稳定性方面，QR 比 LU 更适合求解最小二乘问题。Gram-Schmidt 过程逐列构建 Q：

```
Given columns a1, a2, ... of A:

q1 = a1 / ||a1||

q2 = a2 - (a2 . q1) * q1        (subtract projection onto q1)
q2 = q2 / ||q2||                (normalize)

q3 = a3 - (a3 . q1) * q1 - (a3 . q2) * q2
q3 = q3 / ||q3||

R[i][j] = qi . aj    for i <= j
```

每一步都移除沿所有先前 q 向量的分量，只留下新的正交方向。

### Cholesky 分解

当 A 是对称矩阵（A = A^T）且正定（所有特征值均为正）时，你可以将其分解为 A = L L^T，其中 L 是下三角矩阵。这就是 Cholesky 分解。

```
A = L @ L^T

| 4  2 |   | 2  0 |   | 2  1 |
| 2  5 | = | 1  2 | @ | 0  2 |

L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]    for i > j
```

Cholesky 分解速度是 LU 的两倍，并且所需存储空间减半。它仅适用于对称正定矩阵，但这些矩阵经常出现：

- 协方差矩阵是对称半正定的（通过正则化可变为正定）。
- 高斯过程中的核矩阵是对称正定的。
- 凸函数在最小值处的海森矩阵是对称正定的。
- A^T A 总是对称半正定的。

在高斯过程中，你通过 Cholesky 对核矩阵 K 进行分解，然后求解 K alpha = y 以获得预测均值。Cholesky 因子还为你提供边缘似然的对数行列式：log det(K) = 2 * sum(log(diag(L)))。

### 最小二乘法：当 Ax = b 无精确解时

如果 A 是 m x n 矩阵且 m > n（方程多于未知数），则系统是超定的。不存在精确解。相反，你要最小化平方误差：

```
minimize ||Ax - b||^2

This is the sum of squared residuals:
  sum((A[i,:] @ x - b[i])^2 for i in range(m))
```

最小化解满足正规方程：

```
A^T A x = A^T b
```

推导：展开 ||Ax - b||^2 = (Ax - b)^T (Ax - b) = x^T A^T A x - 2 x^T A^T b + b^T b。对 x 求梯度并设为零：2 A^T A x - 2 A^T b = 0。

```
Original system (overdetermined, 4 equations, 2 unknowns):
| 1  1 |         | 3 |
| 1  2 | x     = | 5 |       No exact x satisfies all 4 equations.
| 1  3 |         | 6 |
| 1  4 |         | 8 |

Normal equations:
A^T A = | 4  10 |    A^T b = | 22 |
        | 10 30 |            | 63 |

Solve: x = [1.5, 1.7]

This is linear regression. x[0] is the intercept, x[1] is the slope.
```

### 正规方程 = 线性回归

这种联系是精确的。在线性回归中，你的数据矩阵 X 每行对应一个样本，每列对应一个特征。目标向量 y 每个条目对应一个样本。权重向量 w 满足：

```
X^T X w = X^T y
w = (X^T X)^(-1) X^T y
```

这是线性回归的闭式解。每次调用 `sklearn.linear_model.LinearRegression.fit()` 都在计算这个（或通过 QR 或 SVD 的等价形式）。

在矩阵中添加正则化项 lambda * I 就得到了岭回归：

```
(X^T X + lambda * I) w = X^T y
w = (X^T X + lambda * I)^(-1) X^T y
```

正则化改善了矩阵的条件数（使其更易于精确求逆），并通过将权重向零收缩来防止过拟合。当 lambda > 0 时，矩阵 X^T X + lambda * I 总是对称正定的，因此你可以使用 Cholesky 来求解它。

### 伪逆（Moore-Penrose）

伪逆 A+ 将矩阵求逆推广到非方阵和奇异矩阵。对于任意矩阵 A：

```
x = A+ b

where A+ = V Sigma+ U^T    (computed via SVD)
```

Sigma+ 是通过取每个非零奇异值的倒数并转置结果形成的。如果 A = U Sigma V^T，则 A+ = V Sigma+ U^T。

```
A = U Sigma V^T        (SVD)

Sigma = | 5  0 |       Sigma+ = | 1/5  0  0 |
        | 0  2 |                | 0  1/2  0 |
        | 0  0 |

A+ = V Sigma+ U^T
```

伪逆给出最小范数最小二乘解。如果系统有：
- 一个解：A+ b 给出该解。
- 无解：A+ b 给出最小二乘解。
- 无穷多解：A+ b 给出 ||x|| 最小的那个解。

NumPy 的 `np.linalg.lstsq` 和 `np.linalg.pinv` 内部都使用 SVD。

### 条件数

条件数衡量解对输入微小变化的敏感程度。对于矩阵 A，条件数定义为：

```
kappa(A) = ||A|| * ||A^(-1)|| = sigma_max / sigma_min
```

其中 sigma_max 和 sigma_min 分别是最大和最小的奇异值。

```
Well-conditioned (kappa ~ 1):        Ill-conditioned (kappa ~ 10^15):
Small change in b -->                Small change in b -->
small change in x                    huge change in x

| 2  0 |   kappa = 2/1 = 2          | 1   1          |   kappa ~ 10^15
| 0  1 |   safe to solve            | 1   1+10^(-15) |   solution is garbage
```

经验法则：
- kappa < 100：安全，解是精确的。
- kappa ~ 10^k：你的浮点运算会损失大约 k 位有效数字的精度。
- kappa ~ 10^16（对于 float64）：解毫无意义。矩阵实际上是奇异的。

在机器学习中，病态条件发生在特征几乎共线时。正则化（添加 lambda * I）可将条件数从 sigma_max / sigma_min 改善为 (sigma_max + lambda) / (sigma_min + lambda)。

### 迭代方法：共轭梯度法

对于非常大的稀疏系统（数百万未知数），像 LU 或 Cholesky 这样的直接方法计算成本太高。迭代方法通过多次迭代改进猜测值来逼近解。

共轭梯度法（CG）在 A 是对称正定时求解 Ax = b。它最多在 n 步内（精确算术下）找到精确解，但如果 A 的特征值聚集，通常收敛得更快。

```
Algorithm sketch:
  x0 = initial guess (often zero)
  r0 = b - A x0           (residual)
  p0 = r0                 (search direction)

  For k = 0, 1, 2, ...:
    alpha = (rk . rk) / (pk . A pk)
    x_{k+1} = xk + alpha * pk
    r_{k+1} = rk - alpha * A pk
    beta = (r_{k+1} . r_{k+1}) / (rk . rk)
    p_{k+1} = r_{k+1} + beta * pk
    if ||r_{k+1}|| < tolerance: stop
```

共轭梯度法应用于：
- 大规模优化（牛顿-CG 方法）
- 求解偏微分方程离散化
- 核矩阵过大无法进行因式分解的核方法
- 其他迭代求解器的预条件处理

收敛速度取决于条件数。条件更好的系统收敛更快，这是正则化有帮助的另一个原因。

### 全景图：何时使用哪种方法

| 方法 | 要求 | 复杂度 | 使用场景 |
|------|------|--------|----------|
| 高斯消元法 | 方阵，非奇异 A | O(n^3) | 方阵系统的一次性求解 |
| LU 分解 | 方阵，非奇异 A | O(n^3) 分解 + O(n^2) 求解 | 用相同 A 进行多次求解 |
| QR 分解 | 任意 A (m >= n) | O(mn^2) | 最小二乘，数值稳定 |
| Cholesky | 对称正定 A | O(n^3/3) | 协方差矩阵，高斯过程，岭回归 |
| 正规方程 | 超定 (m > n) | O(mn^2 + n^3) | 线性回归（小 n） |
| SVD / 伪逆 | 任意 A | O(mn^2) | 秩亏系统，最小范数解 |
| 共轭梯度法 | 对称正定，稀疏 A | O(n * k * nnz) | 大型稀疏系统，k = 迭代次数 |

### 与机器学习的联系

本课程中的每种方法都应用于生产环境中的机器学习：

**线性回归。** 闭式解求解正规方程 X^T X w = X^T y。可通过 Cholesky（如果 n 小）或 QR（如果数值稳定性重要）或 SVD（如果矩阵可能秩亏）完成。

**岭回归。** 在 X^T X 上添加 lambda * I。正则化系统 (X^T X + lambda * I) w = X^T y 总是可以通过 Cholesky 求解，因为当 lambda > 0 时，X^T X + lambda * I 是对称正定的。

**高斯过程。** 预测均值需要求解 K alpha = y，其中 K 是核矩阵。对 K 进行 Cholesky 分解是标准方法。对数边缘似然使用 log det(K) = 2 sum(log(diag(L)))。

**神经网络初始化。** 正交初始化使用 QR 分解来创建列是标准正交的权重矩阵。这可以防止深度网络中的信号崩塌。

**预条件处理。** 大规模优化器使用不完全 Cholesky 或不完全 LU 作为共轭梯度求解器的预条件子。

**特征工程。** X^T X 的条件数告诉你特征是否共线。如果 kappa 很大，则删除特征或添加正则化。

## 构建它

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
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

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
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L
```

### 步骤 4：通过正规方程求解最小二乘

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

## 使用它

将各部分组合起来，用于真实数据上的线性回归和岭回归：

```python
np.random.seed(42)
X_raw = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X_raw @ w_true + np.random.randn(100) * 0.1

X = np.column_stack([np.ones(100), X_raw])

w_ols = least_squares_normal(X, y)
print(f"OLS weights (ours):    {w_ols}")

w_np = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS weights (numpy):   {w_np}")
print(f"Max difference: {np.max(np.abs(w_ols - w_np)):.2e}")

w_ridge = ridge_regression(X, y, lam=1.0)
print(f"Ridge weights (ours):  {w_ridge}")

from sklearn.linear_model import Ridge
ridge_sk = Ridge(alpha=1.0, fit_intercept=False)
ridge_sk.fit(X, y)
print(f"Ridge weights (sklearn): {ridge_sk.coef_}")
```

## 交付

本课程产出：
- `code/linear_systems.py`，包含从头开始实现的高斯消元法、LU 分解、Cholesky 分解、最小二乘法和岭回归
- 一个工作演示，展示正规方程和 sklearn 的 LinearRegression 产生相同的权重

## 练习

1.  使用你的高斯消元法、你的 LU 求解器和 `np.linalg.solve` 求解系统 `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]`。验证三者都在浮点容差范围内给出相同答案。

2.  生成一个 50x5 的随机矩阵 X 和目标 y = X @ w_true + noise。使用正规方程、QR（通过 `np.linalg.qr`）、SVD（通过 `np.linalg.svd`）和 `np.linalg.lstsq` 求解 w。比较所有四个解。测量 X^T X 的条件数并解释它如何影响你信任哪种方法。

3.  通过使两列几乎相同（例如，第 2 列 = 第 1 列 + 1e-10 * noise）来创建一个几乎奇异的矩阵。计算其条件数。分别使用有无正则化（添加 0.01 * I）求解 Ax = b。比较解和残差。解释正则化为何有帮助。

4.  为一个 100x100 的随机对称正定矩阵实现共轭梯度算法。计算其收敛到容差 1e-8 需要多少次迭代。与理论最大值 n 次迭代进行比较。

5.  在大小为 10、50、200、500 的对称正定矩阵上，对你的 Cholesky 求解器、你的 LU 求解器和 `np.linalg.solve` 进行计时。绘制结果。验证 Cholesky 大约比 LU 快 2 倍。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|--------------|
| 线性方程组 | “求解 x” | 一组线性方程 Ax = b。找到 x 意味着找到在变换 A 下产生输出 b 的输入。 |
| 高斯消元法 | “行化简” | 使用行操作系统地将对角线以下的元素清零，生成一个可通过回代法求解的上三角系统。O(n^3)。 |
| 部分主元 | “为稳定性交换行” | 在消去第 k 列之前，将该列中绝对值最大的行交换到主元位置。防止除以小数。 |
| LU 分解 | “分解成三角矩阵” | 将 A = LU，其中 L 是下三角矩阵（存储乘数），U 是上三角矩阵（消元后的矩阵）。将 O(n^3) 成本分摊到多次求解中。 |
| QR 分解 | “正交分解” | 将 A = QR，其中 Q 具有标准正交列，R 是上三角矩阵。对于最小二乘问题比 LU 更稳定。 |
| Cholesky 分解 | “矩阵的平方根” | 对于对称正定 A，将 A = LL^T。成本是 LU 的一半。用于协方差矩阵、核矩阵和岭回归。 |
| 最小二乘法 | “当精确解不可能时的最佳拟合” | 当系统超定（方程多于未知数）时，最小化平方残差 ||Ax - b||^2 的和。 |
| 正规方程 | “微积分捷径” | A^T A x = A^T b。将 ||Ax - b||^2 的梯度设为零。这就是线性回归的闭式解。 |
| 伪逆 | “非方阵的求逆” | 通过 SVD 得到 A+ = V Sigma+ U^T。为任何矩阵（方阵或矩形、奇异或非奇异）提供最小范数最小二乘解。 |
| 条件数 | “这个答案有多可信” | kappa = sigma_max / sigma_min。衡量对输入扰动的敏感性。损失大约 log10(kappa) 位有效数字精度。 |
| 岭回归 | “正则化最小二乘” | 求解 (X^T X + lambda I) w = X^T y。添加 lambda I 改善条件数并将权重向零收缩。防止过拟合。 |
| 共轭梯度法 | “大矩阵的迭代 Ax=b 求解器” | 用于对称正定系统的迭代求解器。最多在 n 步内收敛。适用于因式分解成本过高的大型稀疏系统。 |
| 超定系统 | “数据多于参数” | m x n 系统中 m > n。不存在精确解。最小二乘法找到最佳近似。这是每个回归问题。 |
| 回代法 | “从下往上求解” | 给定一个上三角系统，先求解最后一个方程，然后向后代入。O(n^2)。 |
| 前代法 | “从上往下求解” | 给定一个下三角系统，先求解第一个方程，然后向前代入。O(n^2)。用于 LU 求解中的 L 步。 |

## 延伸阅读

- [MIT 18.06: 线性代数](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (Gilbert Strang) -- 关于线性方程组和矩阵分解的权威课程
- [数值线性代数](https://people.maths.ox.ac.uk/trefethen/text.html) (Trefethen & Bau) -- 理解数值稳定性、条件数和算法失败原因的标准参考书
- [矩阵计算](https://www.cs.cornell.edu/cv/GolubVanLoan4/golubandvanloan.htm) (Golub & Van Loan) -- 涵盖所有矩阵算法的百科全书式参考
- [3Blue1Brown: 逆矩阵](https://www.3blue1brown.com/lessons/inverse-matrices) -- 关于求解 Ax = b 几何意义的视觉直觉