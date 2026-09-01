# 奇异值分解

> SVD 是线性代数的瑞士军刀。每个矩阵都有它。每个数据科学家都需要它。

**类型：** Build
**语言：** Python, Julia
**前置条件：** Phase 1, Lessons 01 (线性代数直觉), 02 (向量与矩阵运算), 03 (矩阵变换)
**耗时：** ~120 分钟

## 学习目标

- 通过幂迭代实现 SVD，并解释 U、Sigma 和 V^T 的几何含义
- 应用截断 SVD 进行图像压缩，并衡量压缩比与重建误差
- 通过 SVD 计算 Moore-Penrose 伪逆以求解超定最小二乘系统
- 将 SVD 与 PCA、推荐系统（潜在因子）以及 NLP 中的潜在语义分析联系起来

## 问题所在

你有一个 1000x2000 的矩阵。可能是用户-电影评分矩阵，可能是文档-词频表，也可能是图像的像素值矩阵。你需要对它进行压缩、去噪、发现隐藏结构，或者用它求解最小二乘系统。特征分解只对方阵有效，即使如此，它还要求矩阵拥有完全线性无关的特征向量集合。

SVD 适用于任意矩阵。任意形状。任意秩。无需任何条件。它将矩阵分解为三个因子，揭示矩阵对空间所做的几何变换。它是整个线性代数中最通用、最实用的分解方法。

## 核心概念

### SVD 的几何含义

任意矩阵，无论形状如何，都会按顺序执行三个操作：旋转、缩放、旋转。SVD 使这一分解显式化。

```
A = U * Sigma * V^T

      m x n     m x m    m x n    n x n
     (任意)    (旋转)   (缩放)   (旋转)
```

给定任意矩阵 A，SVD 将其分解为：
- V^T 在输入空间（n 维）中旋转向量
- Sigma 沿每个轴进行缩放（拉伸或压缩）
- U 将结果旋转到输出空间（m 维）

```mermaid
graph LR
    A["输入空间（n 维）\n数据云\n（任意方向）"] -->|"V^T\n(旋转)"| B["缩放空间\n与坐标轴对齐\n然后由 Sigma 缩放"]
    B -->|"U\n(旋转)"| C["输出空间（m 维）\n旋转至输出方向"]
```

可以这样理解。你将一个矩阵交给 SVD，它会告诉你："这个矩阵先将输入球体由 V^T 旋转，再由 Sigma 拉伸为椭球，最后由 U 旋转该椭球。"奇异值就是椭球各轴的长度。

### 完整分解

对于形状为 m x n 的矩阵 A：

```
A = U * Sigma * V^T

其中：
  U     是 m x m，正交矩阵（U^T U = I）
  Sigma 是 m x n，对角矩阵（对角线上为奇异值）
  V     是 n x n，正交矩阵（V^T V = I）

奇异值 sigma_1 >= sigma_2 >= ... >= sigma_r > 0
其中 r = rank(A)
```

U 的列称为左奇异向量。V 的列称为右奇异向量。Sigma 的对角线元素称为奇异值。它们始终非负，并按降序排列。

### 左奇异向量、奇异值、右奇异向量

SVD 的每个分量都有独特的几何含义。

**右奇异向量（V 的列）：** 它们构成输入空间（R^n）的标准正交基。它们是输入空间中矩阵映射到输出空间正交方向的向量。可以理解为定义域的"自然坐标系"。

**奇异值（Sigma 的对角线）：** 它们是缩放因子。第 i 个奇异值告诉你矩阵沿第 i 个右奇异向量方向拉伸了多少。奇异值为零意味着矩阵完全压缩了该方向。

**左奇异向量（U 的列）：** 它们构成输出空间（R^m）的标准正交基。第 i 个左奇异向量是第 i 个右奇异向量（缩放后）在输出空间中落到的方向。

它们之间的关系：

```
A * v_i = sigma_i * u_i

矩阵 A 将第 i 个右奇异向量 v_i
缩放 sigma_i 倍，映射到第 i 个左奇异向量 u_i。
```

这让你能逐坐标地理解任意矩阵的作用。

### 外积形式

SVD 可以写为秩-1 矩阵之和：

```
A = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T

每一项 sigma_i * u_i * v_i^T 都是一个秩-1 矩阵（外积）。
整个矩阵是 r 个这样的矩阵之和，其中 r 是秩。
```

这种形式是低秩近似的基础。每一项增加一层结构信息。第一项捕捉最重要的单一模式，第二项捕捉次重要的模式，依此类推。截断该和式可在任意指定秩下得到最优近似。

```
秩-1 近似：    A_1 = sigma_1 * u_1 * v_1^T
              （捕捉主导模式）

秩-2 近似：    A_2 = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T
              （捕捉两个最重要的模式）

秩-k 近似：    A_k = 前 k 项之和
              （由 Eckart-Young 定理保证最优）
```

### 与特征分解的关系

SVD 与特征分解密切相关。A 的奇异值和奇异向量直接来自 A^T A 和 A A^T 的特征值和特征向量。

```
A^T A = V * Sigma^T * U^T * U * Sigma * V^T
      = V * Sigma^T * Sigma * V^T
      = V * D * V^T

其中 D = Sigma^T * Sigma 是对角矩阵，对角线元素为 sigma_i^2。

因此：
- 右奇异向量（V）是 A^T A 的特征向量
- 奇异值的平方（sigma_i^2）是 A^T A 的特征值

类似地：
A A^T = U * Sigma * V^T * V * Sigma^T * U^T
      = U * Sigma * Sigma^T * U^T

因此：
- 左奇异向量（U）是 A A^T 的特征向量
- A A^T 的特征值也是 sigma_i^2
```

这一联系告诉我们三件事：
1. 奇异值始终是实数且非负（它们是半正定矩阵特征值的平方根）。
2. 你可以通过对 A^T A 做特征分解来计算 SVD，但这会对方差条件数平方，导致数值精度损失。专门的 SVD 算法避免了这一问题。
3. 当 A 为方阵且对称半正定时，SVD 与特征分解是相同的。

### 截断 SVD：低秩近似

Eckart-Young-Mirsky 定理表明，A 的最佳秩-k 近似（在 Frobenius 范数和谱范数下）通过仅保留前 k 个奇异值及其对应向量获得：

```
A_k = U_k * Sigma_k * V_k^T

其中：
  U_k     是 m x k  （U 的前 k 列）
  Sigma_k 是 k x k  （Sigma 的左上 k x k 块）
  V_k     是 n x k  （V 的前 k 列）

近似误差 = sigma_{k+1}  （谱范数下）
         = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)  （Frobenius 范数下）
```

这不仅是"不错的"近似，它是可证明的、秩-k 矩阵中最好的近似。没有其他秩-k 矩阵比它更接近 A。

| 分量 | 相对大小 | 是否保留在秩-3 近似中？ |
|------|----------|------------------------|
| sigma_1 | 最大 | 是 |
| sigma_2 | 大 | 是 |
| sigma_3 | 中等偏大 | 是 |
| sigma_4 | 中等 | 否（误差） |
| sigma_5 | 中等偏小 | 否（误差） |
| sigma_6 | 小 | 否（误差） |
| sigma_7 | 非常小 | 否（误差） |
| sigma_8 | 极小 | 否（误差） |

保留前 3 个：A_3 捕捉最大的三个奇异值。误差 = 剩余值（sigma_4 到 sigma_8）。

如果奇异值衰减得快，小 k 就能捕捉矩阵的大部分信息。如果衰减慢，说明矩阵没有低秩结构。

### SVD 图像压缩

灰度图像是一个像素强度矩阵。一张 800x600 的图像有 480,000 个值。SVD 可以用远更少的值来近似它。

```
原始图像：800 x 600 = 480,000 个值

秩-k SVD：
  U_k:      800 x k 个值
  Sigma_k:  k 个值
  V_k:      600 x k 个值
  合计：    k * (800 + 600 + 1) = k * 1401 个值

  k=10:   14,010 个值   （原始数据的 2.9%）
  k=50:   70,050 个值   （原始数据的 14.6%）
  k=100: 140,100 个值   （原始数据的 29.2%）

  k 越小，压缩比越好，
  但视觉质量会下降。
```

关键洞察：自然图像的奇异值快速衰减。前几个奇异值捕捉宏观结构（形状、渐变），后面的值捕捉细节和噪声。在秩 50 处截断通常能产生与原始图像几乎相同视觉效果，同时减少 85% 的存储量。

### SVD 用于推荐系统

Netflix 大奖赛让这一应用闻名。你有一个用户-电影评分矩阵，其中大多数条目缺失。

```
             电影1   电影2   电影3   电影4   电影5
  用户1      [  5      ?       3       ?       1  ]
  用户2      [  ?      4       ?       2       ?  ]
  用户3      [  3      ?       5       ?       ?  ]
  用户4      [  ?      ?       ?       4       3  ]

  ? = 未知评分
```

核心思想：这个评分矩阵是低秩的。用户的品味并非完全独立。存在少数几个潜在因子（动作片 vs 剧情片、老片 vs 新片、烧脑 vs 感官刺激）来解释大部分偏好。

对（填充后的）评分矩阵做 SVD 分解为：
- U：潜在因子空间中的人用户画像
- Sigma：每个潜在因子的重要性
- V^T：潜在因子空间中的电影画像

用户对某电影的预测评分是用户画像与电影画像的点积（加权奇异值）。低秩近似填充了缺失条目。

实际应用中，你会使用 Simon Funk 的增量 SVD 或 ALS（交替最小二乘法）等变体，直接处理缺失数据。但核心理念相同：通过 SVD 进行潜在因子分解。

### SVD 在 NLP 中的应用：潜在语义分析

潜在语义分析（LSA），也称为潜在语义索引（LSI），将 SVD 应用于词-文档矩阵。

```
             文档1   文档2   文档3   文档4
  "cat"      [  3      0      1      0  ]
  "dog"      [  2      0      0      1  ]
  "fish"     [  0      4      1      0  ]
  "pet"      [  1      1      1      1  ]
  "ocean"    [  0      3      0      0  ]

经过秩 k=2 的 SVD 处理后：

  每个文档变为 2D "概念空间"中的一个点。
  每个词变为同一 2D 空间中的一个点。
  主题相似的文档聚类在一起。
  含义相近的词聚类在一起。

  "cat" 和 "dog" 彼此靠近（宠物类）。
  "fish" 和 "ocean" 彼此靠近（水生概念）。
  如果文档1 和文档3 主题相似，它们会聚类在一起。
```

LSA 是最早成功从原始文本中提取语义相似性的方法之一。它有效是因为同义词倾向于出现在相似的文档中，所以 SVD 将它们归入同一潜在维度。现代词嵌入（Word2Vec、GloVe）可视为这一思想的延续。

### SVD 去噪

含噪数据的信号集中在顶部奇异值上，噪声分散在所有奇异值上。截断可以去除噪声基底。

**干净信号奇异值：**

| 分量 | 大小 | 类型 |
|------|------|------|
| sigma_1 | 非常大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 接近零 | 可忽略 |
| sigma_5 | 接近零 | 可忽略 |

**含噪信号奇异值（噪声叠加到所有值上）：**

| 分量 | 大小 | 类型 |
|------|------|------|
| sigma_1 | 非常大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 小 | 噪声 |
| sigma_5 | 小 | 噪声 |
| sigma_6 | 小 | 噪声 |
| sigma_7 | 小 | 噪声 |

```mermaid
graph TD
    A["所有奇异值"] --> B{"有明显间隙？"}
    B -->|"高于间隙"| C["信号：保留这些（前 k 个）"]
    B -->|"低于间隙"| D["噪声：丢弃这些"]
    C --> E["用 A_k 重建以获得去噪版本"]
```

这在信号处理、科学测量和数据清洗中使用。任何你拥有被加性噪声污染的矩阵的场景，截断 SVD 都是分离信号与噪声的严谨方法。

### 通过 SVD 计算伪逆

Moore-Penrose 伪逆 A+ 将矩阵求逆推广到非方阵和奇异矩阵。SVD 使其计算变得简单。

```
如果 A = U * Sigma * V^T，那么：

A+ = V * Sigma+ * U^T

其中 Sigma+ 的构造方式：
  1. 转置 Sigma（交换行和列）
  2. 将每个非零对角线元素 sigma_i 替换为 1/sigma_i
  3. 保持零不变

对于 A（m x n）：      A+ 是（n x m）
对于 Sigma（m x n）：  Sigma+ 是（n x m）
```

伪逆用于求解最小二乘问题。如果 Ax = b 没有精确解（超定系统），那么 x = A+ b 就是最小二乘解（最小化 ||Ax - b||）。

```
超定系统（方程数多于未知数）：

  [1  1]         [3]
  [2  1] x   =   [5]       无精确解。
  [3  1]         [6]

  x_ls = A+ b = V * Sigma+ * U^T * b

  给出使残差平方和最小的 x。
  与正规方程 (A^T A)^(-1) A^T b 结果相同，
  但数值上更稳定。
```

### 数值稳定性优势

对 A^T A 做特征分解会将奇异值平方（A^T A 的特征值是 sigma_i^2）。这会使条件数平方，放大数值误差。

```
示例：
  A 的奇异值为 [1000, 1, 0.001]
  A 的条件数：1000 / 0.001 = 10^6

  A^T A 的特征值为 [10^6, 1, 10^{-6}]
  A^T A 的条件数：10^6 / 10^{-6} = 10^{12}

  直接计算 SVD：处理条件数 10^6
  通过 A^T A 计算：处理条件数 10^{12}
                   （损失 6 位额外精度）
```

现代 SVD 算法（Golub-Kahan 双对角化）直接在 A 上工作，从不构造 A^T A。这就是为什么你应该始终优先使用 `np.linalg.svd(A)` 而非 `np.linalg.eig(A.T @ A)`。

### 与 PCA 的关系

PCA 就是对中心化数据做 SVD。这不是类比。它就是完全相同的计算。

```
给定数据矩阵 X（n_samples x n_features），已中心化（已减去均值）：

协方差矩阵：C = (1/(n-1)) * X^T X

PCA 寻找 C 的特征向量。但是：

  X = U * Sigma * V^T    （X 的 SVD）

  X^T X = V * Sigma^2 * V^T

  C = (1/(n-1)) * V * Sigma^2 * V^T

因此主成分恰好是右奇异向量 V。
每个分量的解释方差为 sigma_i^2 / (n-1)。

在 sklearn 中，PCA 使用 SVD 实现，而非特征分解。
它更快且数值更稳定。
```

这意味着你在 Lesson 10 学到的所有关于降维的知识，其底层就是 SVD。PCA 是机器学习中最常见的 SVD 应用。

```figure
svd-rank-reconstruction
```

## 动手构建

### 步骤 1：通过幂迭代从零实现 SVD

思路：为了找到最大奇异值及其对应向量，对 A^T A（或 A A^T）使用幂迭代。然后对矩阵进行消去（deflate），重复上述过程以找到下一个奇异值。

```python
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V
```

### 步骤 2：测试并与 NumPy 比较

```python
np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("我们的奇异值:", np.round(S_ours, 4))
print("NumPy 奇异值:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
print(f"重建误差: {np.linalg.norm(A - A_reconstructed):.8f}")
```

### 步骤 3：图像压缩演示

```python
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")
```

### 步骤 4：去噪

```python
np.random.seed(42)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"含噪误差:    {np.linalg.norm(noisy - clean):.4f}")
print(f"去噪误差: {np.linalg.norm(denoised - clean):.4f}")
print(f"改进幅度:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")
```

### 步骤 5：伪逆

```python
A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD 伪逆解:       {x_svd}")
print(f"np.linalg.lstsq 解:   {x_lstsq}")
print(f"np.linalg.pinv 解:    {x_pinv}")
```

## 实际应用

完整的演示代码在 `code/svd.py` 中。运行它以查看 SVD 在图像压缩、推荐系统、潜在语义分析和去噪中的应用。

```bash
python svd.py
```

Julia 版本 `code/svd.jl` 展示了使用 Julia 原生 `svd()` 函数和 `LinearAlgebra` 包实现相同概念的方法。

```bash
julia svd.jl
```

## 交付成果

本课产出：
- `outputs/skill-svd.md` — 一份技能说明，阐述在实际项目中何时以及如何应用 SVD

## 练习题

1. 在不使用幂迭代的情况下从零实现完整 SVD。改为对 A^T A 做特征分解以获得 V 和奇异值，然后计算 U = A V Sigma^{-1}。比较你的幂迭代版本与 NumPy 版本的数值精度。

2. 加载一张真实的灰度图像（或将其转换为灰度）。分别以秩 1、5、10、25、50、100 进行压缩。对每个秩计算压缩比和相对误差。找出图像视觉上可接受的最低秩。

3. 构建一个微型推荐系统。创建一个 10x8 的用户-电影评分矩阵，包含一些已知条目。用行均值填充缺失条目。计算 SVD 并重建秩-3 近似。使用重建矩阵预测缺失评分。验证预测是否合理。

4. 创建一个 100x50 的文档-词矩阵，包含 3 个合成主题。每个主题有 5 个关联词。添加噪声。应用 SVD 并验证前 3 个奇异值远大于其余值。将文档投影到 3D 潜在空间，检查同一主题的文档是否聚类在一起。

5. 生成一个干净的低秩矩阵（秩 3，尺寸 50x40），并以不同水平添加高斯噪声（sigma = 0.1, 0.5, 1.0, 2.0）。对每个噪声水平，通过遍历 k 从 1 到 40 并测量对干净矩阵的重建误差，找到最优截断秩。绘制最优 k 随噪声水平的变化曲线。

## 关键术语

| 术语 | 人们常这样说 | 实际含义 |
|------|-------------|---------|
| SVD | "分解任意矩阵" | 将 A 分解为 U Sigma V^T，其中 U 和 V 是正交矩阵，Sigma 是对角线上为非负元素的对角矩阵。适用于任意形状的任何矩阵。 |
| 奇异值 | "这个分量有多重要" | Sigma 的第 i 个对角线元素。衡量矩阵沿第 i 个主方向拉伸的程度。始终非负，按降序排列。 |
| 左奇异向量 | "输出方向" | U 的一列。输出空间中第 i 个右奇异向量（缩放 sigma_i 后）所映射到的方向。 |
| 右奇异向量 | "输入方向" | V 的一列。输入空间中矩阵映射到第 i 个左奇异向量（缩放 sigma_i 后）的方向。 |
| 截断 SVD | "低秩近似" | 仅保留前 k 个奇异值及其向量。产生可证明的最优秩-k 近似（Eckart-Young 定理）。 |
| 秩 | "真实维度" | 非零奇异值的数量。告诉你矩阵实际使用了多少个独立方向。 |
| 伪逆 | "广义逆" | V Sigma+ U^T。对非零奇异值取倒数，保持零为零。用于求解非方阵或奇异矩阵的最小二乘问题。 |
| 条件数 | "对误差的敏感程度" | sigma_max / sigma_min。条件数大意味着微小的输入变化会引起巨大的输出变化。SVD 可直接揭示这一点。 |
| 潜在因子 | "隐藏变量" | SVD 发现的低秩空间中的一个维度。在推荐系统中，潜在因子可能对应类型偏好。在 NLP 中，可能对应主题。 |
| Frobenius 范数 | "矩阵总大小" | 所有元素平方和的平方根。等于所有奇异值平方和的平方根。用于度量近似误差。 |
| Eckart-Young 定理 | "SVD 给出最优压缩" | 对于任意目标秩 k，截断 SVD 在所有可能的秩-k 矩阵中最小化近似误差。 |
| 幂迭代 | "找到最大特征向量" | 反复将随机向量与矩阵相乘并归一化。收敛到具有最大特征值的特征向量。是许多 SVD 算法的基础组件。 |

## 延伸阅读

- [Gilbert Strang: Linear Algebra and Its Applications, Chapter 7](https://math.mit.edu/~gs/linearalgebra/) — 关于 SVD 及其应用的全方位讲解
- [3Blue1Brown: But what is the SVD?](https://www.youtube.com/watch?v=vSczTbgc8Rc) — SVD 的几何直觉
- [We Recommend a Singular Value Decomposition](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) — 美国数学学会提供的通俗易懂概述
- [Netflix Prize and Matrix Factorization](https://sifter.org/~simon/journal/20061211.html) — Simon Funk 关于 SVD 用于推荐的原始博客文章
- [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) — SVD 在 NLP 中的原始应用
- [Numerical Linear Algebra by Trefethen and Bau](https://people.maths.ox.ac.uk/trefethen/text.html) — 理解 SVD 算法及其数值性质的权威指南
