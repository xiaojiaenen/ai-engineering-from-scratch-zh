# 线性代数直觉

> 每一个 AI 模型，本质上只是一堆披着华丽外衣的矩阵运算。

**类型：** 学习
**语言：** Python、Julia
**前置知识：** Phase 0
**预计时间：** ~60 分钟

## 学习目标

- 用 Python 从零实现向量与矩阵运算（加法、点积、矩阵乘法）
- 从几何角度解释点积、投影和 Gram-Schmidt 过程的含义
- 通过行化简判断一组向量的线性无关性、秩和基底
- 将线性代数概念与其在 AI 中的应用联系起来：嵌入（embeddings）、注意力分数和 LoRA

## 问题背景

打开任意一份机器学习论文。在第一页内，你就能看到向量、矩阵、点积和变换。缺乏线性代数直觉时，它们只是抽象符号；有了直觉，你能看清神经网络到底在做什么——在空间里移动点。

你不需要成为数学家。你需要从几何上理解这些操作的含义，然后亲手把它们写出来。

## 核心概念

### 向量是点（也是方向）

向量只是一串数字。但这些数字有意义——它们是空间中的坐标。

**二维向量 [3, 2]：**

| x | y | 点 |
|---|---|-------|
| 3 | 2 | 向量从原点 (0,0) 指向平面上 (3, 2) 的位置 |

该向量的模为 sqrt(3² + 2²) = sqrt(13)，指向右上方。

在 AI 中，向量代表一切：
- 一个词 → 768 个数字组成的向量（它在嵌入空间中的"含义"）
- 一张图像 → 数百万个像素值组成的向量
- 一个用户 → 一组偏好组成的向量

### 矩阵是变换

矩阵将一个向量变换为另一个向量。它可以旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before
        A["点 A"]
        B["点 B"]
    end
    subgraph Matrix["矩阵乘法"]
        M["M（变换）"]
    end
    subgraph After
        A2["点 A'"]
        B2["点 B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在 AI 中，矩阵就是模型本身：
- 神经网络权重 → 将输入映射为输出的矩阵
- 注意力分数 → 决定关注什么的矩阵
- 嵌入 → 将词映射为向量的矩阵

### 点积衡量相似度

两个向量的点积告诉你它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

同向：          a · b > 0  （相似）
垂直：          a · b = 0  （无关）
反向：          a · b < 0  （不相似）
```

搜索引擎、推荐系统和 RAG 正是这样工作的——找到点积高的向量。

### 线性无关

一组向量线性无关，当且仅当集合中没有任何一个向量可以表示为其他向量的线性组合。如果 v1、v2、v3 相互独立，它们张成三维空间。如果其中一个可由其他向量组合而来，它们只张成一个平面。

为什么它对 AI 很重要：你的特征矩阵应当具有线性无关的列。如果两个特征完全相关（线性相关），模型无法区分它们各自的影响。这会导致回归中的多重共线性问题——权重矩阵变得不稳定，微小的输入变化会产生剧烈的输出波动。

**具体例子：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 和 v2 是线性无关的——没有任何一个是另一个的标量倍数或线性组合。但 v3 = 2*v1 + v2，因此 {v1, v2, v3} 构成线性相关集合。这三个向量全部落在 xy 平面上。无论怎样组合它们，都无法到达 [0, 0, 1]。你拥有三个向量，却只有两个自由维度。

在一个数据集中：如果 feature_3 = 2*feature_1 + feature_2，加入 feature_3 不会给模型带来任何新信息。更糟的是，它使正规方程奇异——权重没有唯一解。

### 基底与秩

基底是张成整个空间的一组最小线性无关向量。基底向量的个数就是空间的维度。

三维空间的标准基底是 {[1,0,0], [0,1,0], [0,0,1]}。但三维空间中任意三个独立向量都构成合法基底。选择基底就是选择坐标系。

矩阵的秩 = 线性无关列的数量 = 线性无关行的数量。若秩 < min(行数, 列数)，则矩阵是秩亏的。这意味着：
- 方程组有无穷多解（或无解）
- 变换过程中信息丢失
- 矩阵不可逆

| 情况 | 秩 | 对机器学习的含义 |
|------|------|------------------|
| 满秩（秩 = min(m, n)）| 最大可能值 | 存在唯一的最小二乘解。模型条件良好。 |
| 秩亏（秩 < min(m, n)）| 低于最大值 | 特征冗余。权重解不唯一。需要正则化。 |
| 秩为 1 | 1 | 每一列都是某一个向量的缩放副本。所有数据落在一条直线上。 |
| 近秩亏（极小奇异值）| 数值上偏低 | 矩阵病态。微小输入噪声会导致巨大的输出变化。使用 SVD 截断或岭回归。 |

### 投影

将向量 **a** 投影到向量 **b** 上，得到 **a** 在 **b** 方向上的分量：

```
proj_b(a) = (a · b / b · b) × b
```

残差 (a - proj_b(a)) 与 b 垂直。这种正交分解是最小二乘拟合的基础。

投影无处不在：
- 线性回归最小化观测值与列空间之间的距离——解本身就是投影
- PCA 将数据投影到方差最大的方向上
- Transformer 中的注意力计算 query 对 key 的投影

```mermaid
graph LR
    subgraph Projection["a 在 b 上的投影"]
        direction TB
        O["原点"] --> |"b（方向）"| B["b"]
        O --> |"a（原始）"| A["a"]
        O --> |"proj_b(a)"| P["投影点"]
        A -.-> |"残差（垂直）"| P
    end
```

**示例：** a = [3, 4]，b = [1, 0]

proj_b(a) = (3×1 + 4×0) / (1×1 + 0×0) × [1, 0] = 3 × [1, 0] = [3, 0]

投影去掉了 y 分量。这是最简单形式的降维——扔掉你不在意的那些方向。

### Gram-Schmidt 过程

将任意一组独立向量转换为一组标准正交基底。标准正交意味着每个向量长度为 1，且任意两个向量互相垂直。

算法步骤：
1. 取第一个向量，归一化
2. 取第二个向量，减去它在第一个向量上的投影，再归一化
3. 取第三个向量，减去它在所有先前向量上的投影，再归一化
4. 对剩余向量重复上述步骤

```
输入：  v1, v2, v3, ...（线性无关）

u1 = v1 / |v1|

w2 = v2 - (v2 · u1) × u1
u2 = w2 / |w2|

w3 = v3 - (v3 · u1) × u1 - (v3 · u2) × u2
u3 = w3 / |w3|

输出： u1, u2, u3, ...（标准正交基底）
```

这正是 QR 分解的内部工作原理。Q 是标准正交基底，R 记录投影系数。QR 分解用于：
- 求解线性方程组（比高斯消元更稳定）
- 计算特征值（QR 算法）
- 最小二乘回归（标准数值方法）

```figure
eigen-directions
```

## 动手实现

### 步骤 1：从零实现向量（Python）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### 步骤 2：从零实现矩阵（Python）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### 步骤 3：为什么这对 AI 很重要

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("这就是神经网络层所做的——矩阵乘法。")
```

### 步骤 4：Julia 版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia 支持 Unicode 运算符
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# 矩阵-向量乘法
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("这是一个神经网络层。")
```

### 步骤 5：从零实现线性无关性检测和投影（Python）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 实战用法

现在用 NumPy 做同样的事情——这才是实践中你会用到的：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 用 NumPy 计算秩、投影和 QR 分解

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch —— 带自动求导的张量就是向量

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积对 x 的梯度就是 y 本身。PyTorch 自动完成了计算。神经网络中的每一个操作都由这类操作构建——矩阵乘法、点积、投影——而自动微分追踪所有这些操作的梯度。

你从零实现了 NumPy 一行代码就能完成的事。现在你知道了底层发生了什么。

## 交付成果

本课产出：
- `outputs/prompt-linear-algebra-tutor.md` —— 用于引导 AI 助手通过几何直觉教授线性代数的提示词

## 知识关联

本课所有内容均与当代 AI 的具体环节相连：

| 概念 | 出现位置 |
|------|----------|
| 点积 | Transformer 中的注意力分数、RAG 中的余弦相似度 |
| 矩阵乘法 | 每一个神经网络层、每一个线性变换 |
| 线性无关 | 特征选择、避免多重共线性 |
| 秩 | 判断方程组是否可解、LoRA（低秩自适应） |
| 投影 | 线性回归（投影到列空间）、PCA |
| Gram-Schmidt / QR | 数值求解器、特征值计算 |
| 标准正交基底 | 稳定的数值计算、白化变换 |

LoRA 值得特别提及。它通过将权重更新分解为低秩矩阵来微调大语言模型。与其更新一个 4096×4096 的权重矩阵（1600 万参数），LoRA 只更新两个尺寸为 4096×16 和 16×4096 的矩阵（13.1 万参数）。秩为 16 的约束意味着 LoRA 假设权重更新存在于完整的 4096 维空间中的某个 16 维子空间内。这就是线性代数在实际发挥作用。

## 练习

1. 实现 `Vector.angle_between(other)`，返回两个向量之间以度数为单位的夹角
2. 创建一个 2D 缩放矩阵，将 x 坐标翻倍、y 坐标变为三倍，然后将其作用于向量 [1, 1]
3. 给定 5 个随机生成的类词向量（维度 50），使用余弦相似度找出最相似的一对
4. 验证 Gram-Schmidt 的输出确实是标准正交的：检查每一对向量的点积是否为 0，每个向量的模是否为 1
5. 创建一个秩为 2 的 3×3 矩阵。用 `rank()` 方法验证。然后解释这些列向量张成什么几何对象
6. 将向量 [1, 2, 3] 投影到 [1, 1, 1] 上。结果在几何上代表什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 向量 | "一支箭" | 一串数字，表示 n 维空间中的一个点或方向 |
| 矩阵 | "数字表格" | 将向量从一个空间映射到另一个空间的变换 |
| 点积 | "逐元素相乘再求和" | 衡量两个向量对齐程度的指标——相似度搜索的核心 |
| 嵌入 | "AI 的黑魔法" | 表示某事物（词、图像、用户）含义的向量 |
| 线性无关 | "互不重叠" | 集合中没有任何一个向量能写成其他向量的线性组合 |
| 秩 | "有多少维度" | 矩阵中线性无关列（或行）的数量 |
| 投影 | "影子" | 一个向量在另一个向量方向上的分量 |
| 基底 | "坐标轴" | 张成空间的一组最小线性无关向量 |
| 标准正交 | "垂直的单位向量" | 两两垂直且长度均为 1 的向量 |
