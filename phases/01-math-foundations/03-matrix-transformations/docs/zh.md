# 矩阵变换

> 矩阵是重塑空间的机器。理解它对每个点的作用，你就能理解整个变换。

**类型：** 构建
**语言：** Python, Julia
**前置要求：** 阶段1，课程01-02（线性代数直觉，向量与矩阵运算）
**时间：** 约75分钟

## 学习目标

- 构建旋转、缩放、剪切和反射矩阵，并将其应用于二维和三维点
- 通过矩阵乘法组合多个变换，并验证顺序的重要性
- 从特征方程计算2x2矩阵的特征值和特征向量
- 解释特征值为何决定PCA方向、RNN稳定性和谱聚类行为

## 问题所在

当你阅读PCA时，看到“找到协方差矩阵的特征向量”。当你研究模型稳定性时，看到“检查所有特征值的模是否小于1”。当你了解数据增强时，看到“应用随机旋转”。在你理解矩阵如何从几何上作用于空间之前，这一切都没有意义。

矩阵不仅仅是数字的网格。它们是空间机器。旋转矩阵旋转点。缩放矩阵拉伸它们。剪切矩阵倾斜它们。神经网络应用于数据的每一个变换都是这些操作之一或它们的组合。本课程使这些操作具体化。

## 概念解释

### 变换即矩阵

二维中的每个线性变换都可以写成一个2x2矩阵。该矩阵准确地告诉你基向量 [1, 0] 和 [0, 1] 最终到达何处。其他一切都由此决定。

```mermaid
graph LR
    subgraph Before["Standard Basis"]
        e1["e1 = [1, 0] (along x)"]
        e2["e2 = [0, 1] (along y)"]
    end
    subgraph Transform["Matrix M"]
        M["M = columns are new basis vectors"]
    end
    subgraph After["After Transformation M"]
        e1p["e1' = new x-basis"]
        e2p["e2' = new y-basis"]
    end
    e1 --> M --> e1p
    e2 --> M --> e2p
```

### 旋转

二维旋转角度θ保持距离和角度不变。它沿着圆弧移动每个点。

```mermaid
graph LR
    subgraph Before["Before Rotation"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Rot["Rotate 45 degrees"]
        R["R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]"]
    end
    subgraph After["After Rotation"]
        Ap["A'(0.71, 2.12)"]
        Bp["B'(-1.41, 1.41)"]
    end
    A --> R --> Ap
    B --> R --> Bp
```

在三维中，你绕一个轴旋转。每个轴都有自己的旋转矩阵：

```
Rz(theta) = | cos  -sin  0 |     Rotate around z-axis
            | sin   cos  0 |     (x-y plane spins, z stays)
            |  0     0   1 |

Rx(theta) = | 1   0     0    |   Rotate around x-axis
            | 0  cos  -sin   |   (y-z plane spins, x stays)
            | 0  sin   cos   |

Ry(theta) = |  cos  0  sin |     Rotate around y-axis
            |   0   1   0  |     (x-z plane spins, y stays)
            | -sin  0  cos |
```

### 缩放

缩放沿每个轴独立地拉伸或压缩。

```mermaid
graph LR
    subgraph Before["Before Scaling"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Scale["Scale sx=2, sy=0.5"]
        S["S = [[2, 0], [0, 0.5]]"]
    end
    subgraph After["After Scaling"]
        Ap["A'(4, 0.5)"]
        Bp["B'(0, 1)"]
    end
    A --> S --> Ap
    B --> S --> Bp
```

### 剪切

剪切在保持另一个轴固定的同时倾斜一个轴。它将矩形变成平行四边形。

```mermaid
graph LR
    subgraph Before["Before Shear"]
        A["A(1, 0)"]
        B["B(0, 1)"]
    end
    subgraph Shear["Shear in x, k=1"]
        Sh["Shx = [[1, k], [0, 1]]"]
    end
    subgraph After["After Shear"]
        Ap["A(1, 0) unchanged"]
        Bp["B'(1, 1) shifted"]
    end
    A --> Sh --> Ap
    B --> Sh --> Bp
```

剪切矩阵：
- `Shx = [[1, k], [0, 1]]` 将x移动 k * y
- `Shy = [[1, 0], [k, 1]]` 将y移动 k * x

### 反射

反射将点跨越一个轴或直线进行镜像。

```mermaid
graph LR
    subgraph Before["Before Reflection"]
        A["A(2, 1)"]
    end
    subgraph Reflect["Reflect across y-axis"]
        R["[[-1, 0], [0, 1]]"]
    end
    subgraph After["After Reflection"]
        Ap["A'(-2, 1)"]
    end
    A --> R --> Ap
```

反射矩阵：
- 关于y轴反射：`[[-1, 0], [0, 1]]`
- 关于x轴反射：`[[1, 0], [0, -1]]`

### 组合：链接变换

应用变换A然后B，等同于将它们的矩阵相乘：`result = B @ A @ point`。顺序很重要。先旋转再缩放与先缩放再旋转会产生不同的结果。

```mermaid
graph LR
    subgraph Path1["Rotate 90 then Scale (2, 0.5)"]
        P1["(1, 0)"] -->|"Rotate 90"| P2["(0, 1)"] -->|"Scale"| P3["(0, 0.5)"]
    end
```

组合后：`S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["Scale (2, 0.5) then Rotate 90"]
        Q1["(1, 0)"] -->|"Scale"| Q2["(2, 0)"] -->|"Rotate 90"| Q3["(0, 2)"]
    end
```

组合后：`R @ S = [[0, -0.5], [2, 0]]`

结果不同。矩阵乘法不满足交换律。

### 特征值和特征向量

当矩阵作用于大多数向量时，它们的方向会改变。特征向量很特殊：矩阵只对其进行缩放，从不旋转它们。缩放因子就是特征值。

```
A @ v = lambda * v

v is the eigenvector (direction that survives)
lambda is the eigenvalue (how much it stretches)

Example: A = | 2  1 |
             | 1  2 |

Eigenvector [1, 1] with eigenvalue 3:
  A @ [1,1] = [3, 3] = 3 * [1, 1]     (same direction, scaled by 3)

Eigenvector [1, -1] with eigenvalue 1:
  A @ [1,-1] = [1, -1] = 1 * [1, -1]  (same direction, unchanged)
```

该矩阵沿 [1, 1] 方向拉伸空间3倍，并保持 [1, -1] 方向不变。其他所有方向都是这两个方向的混合。

### 特征分解

如果一个矩阵有n个线性无关的特征向量，它可以被分解为：

```
A = V @ D @ V^(-1)

V = matrix whose columns are eigenvectors
D = diagonal matrix of eigenvalues
V^(-1) = inverse of V

This says: rotate into eigenvector coordinates, scale along each axis, rotate back.
```

### 特征值的重要性

**PCA。** 协方差矩阵的特征向量是主成分。特征值告诉你每个成分捕获了多少方差。按特征值排序，保留前k个，你就实现了降维。

**稳定性。** 在循环网络和动力系统中，模大于1的特征值会导致输出爆炸。模小于1会导致输出消失。这就是梯度消失/爆炸问题的一句话表述。

**谱方法。** 图神经网络使用邻接矩阵的特征值。谱聚类使用拉普拉斯矩阵的特征值。特征向量揭示了图的结构。

### 行列式作为体积缩放因子

变换矩阵的行列式告诉你它缩放面积（二维）或体积（三维）的程度。

```
det = 1:   area preserved (rotation)
det = 2:   area doubled
det = 0:   space crushed to lower dimension (singular)
det = -1:  area preserved but orientation flipped (reflection)

| det(Rotation) | = 1        (always)
| det(Scale sx, sy) | = sx * sy
| det(Shear) | = 1           (area preserved)
| det(Reflection) | = -1     (orientation flipped)
```

## 动手构建

### 步骤1：从零开始构建变换矩阵（Python）

```python
import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

point = [1.0, 0.0]
angle = math.pi / 4

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")
```

### 步骤2：变换的组合

```python
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Same? {result1 == result2}")
```

### 步骤3：从零开始计算特征值（2x2矩阵）

对于2x2矩阵 `[[a, b], [c, d]]`，特征值满足特征方程：`lambda^2 - (a+d)*lambda + (ad - bc) = 0`。

```python
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Matrix: {A}")
print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
```

### 步骤4：行列式作为体积缩放因子

```python
def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(singular)     = {det_2x2(singular):.1f}")
print("Singular: columns are proportional, space collapses to a line.")
```

## 实际应用

NumPy使用优化的例程处理所有这些操作。

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
print(f"Rotate (1,0) by 45 deg: {R @ point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")
```

### 使用NumPy进行三维旋转

```python
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D point: {point_3d}")
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")
```

## 总结

本课程为PCA（阶段2）和神经网络权重分析奠定了几何基础。这里构建的特征值/特征向量代码，与在生产ML系统中驱动降维、谱聚类和稳定性分析的算法相同。

## 练习

1. 对单位正方形（角点在[0,0], [1,0], [1,1], [0,1]）应用旋转、缩放和剪切变换。打印每种变换后的角点。验证旋转是否保持角点之间的距离。
2. 使用特征方程手动计算矩阵 [[4, 2], [1, 3]] 的特征值。然后使用你的从零开始的函数和NumPy进行验证。
3. 创建三个变换的组合（旋转30度，缩放[1.5, 0.8]，剪切kx=0.3），并将其应用于排列成圆形的8个点。打印变换前后的坐标。计算组合矩阵的行列式，并验证它等于各个行列式的乘积。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 旋转矩阵 | "旋转物体" | 一个正交矩阵，它在保持距离和角度的同时，沿着圆弧移动点。行列式恒为1。 |
| 缩放矩阵 | "放大物体" | 一个对角矩阵，它沿每个轴独立地拉伸或压缩。行列式是缩放因子的乘积。 |
| 剪切矩阵 | "倾斜物体" | 一个矩阵，它将一个坐标按比例移动到另一个坐标，将矩形变成平行四边形。行列式为1。 |
| 反射 | "镜像物体" | 一个跨越轴或平面翻转空间的矩阵。行列式为-1。 |
| 组合 | "做两件事" | 将变换矩阵相乘以链接操作。顺序很重要：B @ A 意味着先应用A，然后应用B。 |
| 特征向量 | "特殊方向" | 矩阵只缩放而不旋转的方向。变换的“指纹”。 |
| 特征值 | "拉伸程度" | 矩阵缩放其特征向量的标量因子。可以是负数（翻转）或复数（旋转）。 |
| 特征分解 | "拆解矩阵" | 将矩阵写成 V @ D @ V^(-1)，将其分离为基本的缩放方向和幅度。 |
| 行列式 | "矩阵中的一个数" | 变换缩放面积（二维）或体积（三维）的因子。为零表示变换不可逆。 |
| 特征方程 | "特征值的来源" | det(A - λ * I) = 0。其根即为特征值的多项式。 |

## 延伸阅读

- [3Blue1Brown: 线性变换](https://www.3blue1brown.com/lessons/linear-transformations) -- 矩阵如何重塑空间的视觉直觉
- [3Blue1Brown: 特征向量和特征值](https://www.3blue1brown.com/lessons/eigenvalues) -- 关于特征向量几何含义的最佳视觉解释
- [MIT 18.06 课程第21讲：特征值和特征向量](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- Gilbert Strang的经典讲解