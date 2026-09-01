# 向量、矩阵与运算

> 每一个神经网络本质上就是带有多余步骤的矩阵乘法。

**类型：** 实践练习
**语言：** Python, Julia
**前置知识：** 第一阶段，第 01 课（线性代数直觉）
**时间：** 约 60 分钟

## 学习目标

- 构建一个包含逐元素运算、矩阵乘法、转置、行列式和逆矩阵的 Matrix 类
- 区分逐元素乘法和矩阵乘法，并解释各自的适用场景
- 仅使用从头实现的 Matrix 类实现单个全连接神经网络层 (`relu(W @ x + b)`)
- 解释广播机制规则以及偏置加法在神经网络框架中的工作原理

## 问题背景

你想构建一个神经网络。你阅读代码并看到这样的内容：

```
output = activation(weights @ input + bias)
```

这里的 `@` 是矩阵乘法。`weights` 是一个矩阵，`input` 是一个向量。如果你不知道这些运算的含义，这行代码就像是魔法。如果你了解，它就是三层神经网络中前向传播的全部核心，仅由三个运算组成。

你的模型处理的每张图像都是一个像素值矩阵。每个词嵌入都是一个向量。每个神经网络的每一层都是一次矩阵变换。就像不懂变量就无法写代码一样，不熟练掌握矩阵运算就无法构建 AI 系统。

本课程从零开始构建这种熟练度。

## 概念讲解

### 向量：有序数字列表

向量是有方向和大小的有序数字列表。在 AI 中，向量代表数据点、特征或参数。

```
v = [3, 4]        # 一个二维向量
w = [1, 0, -2]    # 一个三维向量
```

二维向量 `[3, 4]` 指向平面上坐标 (3, 4) 的位置。其长度（模）为 5（3-4-5 直角三角形）。

### 矩阵：数字网格

矩阵是一个二维网格，由行和列组成。一个 m x n 矩阵有 m 行和 n 列。

```
A = | 1  2  3 |     # 2x3 矩阵（2 行，3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵将输入向量转换为输出向量。一个有 784 个输入和 128 个输出的层使用 128x784 的权重矩阵。

### 维度为何重要

矩阵乘法有一条严格的规则：`(m x n) @ (n x p) = (m x p)`。内部维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

内部维度: 784 = 784  -- 合法
```

如果你在 PyTorch 中遇到维度不匹配错误，原因就在这里。

### 运算速查表

| 运算 | 作用 | 神经网络中的用途 |
|------|------|----------------|
| 加法 | 逐元素合并 | 向输出添加偏置 |
| 标量乘法 | 缩放每个元素 | 学习率 * 梯度 |
| 矩阵乘法 | 变换向量 | 层的前向传播 |
| 转置 | 行列互换 | 反向传播 |
| 行列式 | 单个数值摘要 | 判断可逆性 |
| 逆矩阵 | 逆转变换 | 求解线性方程组 |
| 单位矩阵 | 不做任何变换的矩阵 | 初始化、残差连接 |

### 逐元素乘法 vs 矩阵乘法

这个区别让初学者经常困惑。

逐元素乘法：对应位置相乘。两个矩阵必须是相同形状。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：行与列的点积。内部维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算，不同的结果，不同的规则。

### 广播机制

当你将一个偏置向量添加到输出矩阵时，形状不匹配。广播机制会将较小的数组拉伸以适配。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播将向量沿行方向展开：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都会自动完成这一操作。理解它能避免当形状看起来不对但代码能正常运行时的困惑。

```figure
vector-projection
```

## 动手实现

### 步骤 1：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 步骤 2：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("矩阵奇异，不存在逆矩阵")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 步骤 3：观察运行效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 步骤 4：与神经网络连接

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"输入维度: {inputs.shape}")
print(f"权重维度: {weights.shape}")
print(f"输出维度: {output.shape}")
print(f"输出: {output.data}")
```

这就是单个全连接层：`output = relu(W @ x + b)`。每个神经网络中的每个全连接层都完全执行这个操作。

## 实际使用

NumPy 可以用更少的代码行数实现上述所有功能，且速度提升几个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (逐元素) =\n", A * B)
print("A @ B (矩阵乘法) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\n神经网络层: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"输出:\n{output}")
```

Python 中的 `@` 运算符调用 `__matmul__`。NumPy 使用用 C 和 Fortran 编写的优化 BLAS 例程来实现它。相同的数学运算，快 100 倍。

NumPy 中的广播机制：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动将一维偏置沿两个行方向广播。这就是每个神经网络框架中偏置加法的工作原理。

## 交付成果

本课程产出一个用于通过几何直觉教授矩阵运算的提示词。参见 `outputs/prompt-matrix-operations.md`。

此处构建的 Matrix 类是我们在第三阶段第 10 课构建的迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()` 并确认得到单位矩阵。用三个不同的 2x2 矩阵进行测试。当行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix 类，使用伴随矩阵法计算 3x3 矩阵的逆。用 NumPy 的 `np.linalg.inv` 测试验证。

3. **构建两层网络。** 仅使用你的 Matrix 类（不使用 NumPy），创建一个两层神经网络：输入 (3) -> 隐藏层 (4) -> 输出 (2)。初始化随机权重，执行前向传播，并验证所有维度正确。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 向量 | "一根箭头" | 有序数字列表。在 AI 中：高维空间中的一个点。 |
| 矩阵 | "数字表格" | 线性变换。它将向量从一个空间映射到另一个空间。 |
| 矩阵乘法 | "就是把数字相乘" | 第一个矩阵的每一行与第二个矩阵的每一列的点积。顺序很重要。 |
| 转置 | "翻转一下" | 行列互换。将 m x n 矩阵变为 n x m 矩阵。在反向传播中至关重要。 |
| 行列式 | "从矩阵中算出的某个数" | 衡量矩阵缩放面积（2D）或体积（3D）的程度。零意味着变换压缩了某个维度。 |
| 逆矩阵 | "反转矩阵的操作" | 逆转变换的矩阵。仅当行列式不为零时才存在。 |
| 单位矩阵 | "无聊的矩阵" | 矩阵中的乘法 1。用于残差连接（ResNets）。 |
| 广播 | "形状修正魔法" | 通过沿缺失维度重复，将较小数组拉伸以匹配较大数组。 |
| 逐元素 | "普通乘法" | 对应位置相乘。两个数组必须有相同的形状（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: 线性代数本质](https://www.3blue1brown.com/topics/linear-algebra) - 对本文涵盖的所有运算提供可视化直觉
- [NumPy 广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的确切规则
- [斯坦福 CS229 线性代数回顾](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习的线性代数简洁参考
