# 范数与距离

> 距离函数定义了“相似”的含义。选择错误，下游一切都会崩坏。

**类型：** 实践
**语言：** Python
**前置知识：** 阶段1，课程01（线性代数直觉）、02（向量、矩阵与运算）
**时间：** 约90分钟

## 学习目标

- 从零开始实现 L1、L2、余弦、马氏、杰卡德和编辑距离函数
- 为给定的ML任务选择合适的距离度量，并解释替代方案为何失效
- 将L1和L2范数与LASSO和Ridge正则化及其几何约束区域联系起来
- 演示同一数据集在不同度量下如何产生不同的最近邻

## 问题描述

你有两个向量。也许是词嵌入。也许是用户画像。也许是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择的距离函数。两个数据点在一种度量下可能是最近邻，在另一种度量下可能相距甚远。你的KNN分类器、推荐引擎、向量数据库、聚类算法、损失函数——都依赖于这个选择。选错了，你的模型优化的就是错误的目标。

没有普遍最优的距离。L2适用于空间数据。余弦相似度主导NLP。杰卡德处理集合。编辑距离处理字符串。马氏距离考虑相关性。Wasserstein距离移动概率质量。每一种都编码了关于“相似”含义的不同假设。

本课从头构建每个主要距离函数，展示何时该使用哪种工具，并演示相同的数据根据使用的度量不同如何产生完全不同的最近邻。

## 核心概念

### 范数：衡量向量大小

范数衡量向量的“大小”。任意两个向量之间的距离函数都可以写成它们之差的范数：d(a, b) = ||a - b||。因此，理解范数就是理解距离。

### L1范数（曼哈顿距离）

L1范数对所有分量的绝对值求和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离，因为它衡量的是在只能在轴上移动（不能走对角线）的城市网格中你需要走多远。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用L1：
- 高维稀疏数据（文本特征、独热编码）
- 当你需要对离群点稳健时（单个巨大的差异不会主导）
- 特征选择问题（L1正则化促进稀疏性）

与L1正则化（Lasso）的联系：将 ||w||_1 加到损失函数中会惩罚权重绝对值的总和。这会将小权重推向恰好为零，从而执行自动特征选择。L1惩罚在权重空间中创建菱形约束区域，菱形的角位于某些权重为零的轴上。

与损失函数的联系：平均绝对误差（MAE）是预测值与目标值之间L1距离的平均值。它线性惩罚所有误差，与MSE相比对离群点更稳健。

### L2范数（欧几里得距离）

L2范数是直线距离。分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是你在几何课上学到的距离。n维空间中的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用L2：
- 中低维连续数据
- 当特征尺度具有可比性时
- 物理距离（空间数据、传感器读数）
- 像素级图像相似度

与L2正则化（Ridge）的联系：将 ||w||_2^2 加到损失函数中会惩罚大的权重。与L1不同，它不会将权重推向零。它按比例将所有权重向零收缩。L2惩罚创建圆形约束区域，因此在轴上没有角。权重会变小，但很少恰好为零。

与损失函数的联系：均方误差（MSE）是L2距离平方的平均值。平方操作对大的误差比小的误差惩罚更重。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp范数：通用家族

L1和L2是Lp范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的p值产生不同形状的“单位球”（到原点距离为1的所有点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity范数（切比雪夫距离）

当p趋近于无穷大时，Lp范数收敛于最大绝对分量值。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的那个维度决定。其他所有维度都被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用L-infinity：
- 当任何单个维度的最坏情况偏差很重要时
- 游戏棋盘（国际象棋中的国王按L-infinity移动：朝任意方向移动一步代价为1）
- 制造公差（每个维度都必须在规格范围内）

### 余弦相似度与余弦距离

余弦相似度衡量两个向量之间的夹角，忽略它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

其范围从-1（方向相反）到+1（方向相同）。垂直向量的余弦相似度为0。

余弦距离将其转换为距离：余弦距离 = 1 - 余弦相似度。范围从0（方向相同）到2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦相似度主导NLP和嵌入：在文本中，文档长度不应影响相似度。一篇关于猫的文档长度是另一篇关于猫文档的两倍，它应该仍然是“相似的”。余弦相似度忽略大小（长度），只关心方向。两篇词分布相同但长度不同的文档指向同一方向，获得1.0的余弦相似度。

何时使用余弦相似度：
- 文本相似度（TF-IDF向量、词嵌入、句子嵌入）
- 任何大小是噪声、方向是信号的领域
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎总是使用余弦或点积）

### 点积相似度与余弦相似度

两个向量的点积是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是点积经两个向量的大小归一化后的结果。当两个向量都是单位向量（大小=1）时，点积和余弦相似度是相同的。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

何时它们不同：点积包含大小信息。大小更大的向量会获得更高的点积分数。这在一些你希望“受欢迎”项目排名更高的检索系统中很重要。大小充当了隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：
- 当你想要纯粹的方向相似度时，使用余弦相似度
- 当大小携带有意义的信息时，使用点积
- 许多向量数据库（Pinecone, Weaviate, Qdrant）允许你在这两者之间选择
- 如果你的嵌入已经做了L2归一化，那么选择哪个就不重要了

### 马氏距离

欧几里得距离平等对待所有维度。但如果你的特征相关或尺度不同，L2会给出误导性的结果。

马氏距离考虑了数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直觉上：马氏距离首先对数据进行去相关和归一化（白化），然后在变换后的空间中计算L2距离。如果S是单位矩阵（特征不相关且方差为单位方差），则马氏距离退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：
- 离群点检测（距离均值马氏距离大的点是离群点）
- 当特征具有不同尺度和相关性时的分类
- 当你有足够的数据来估计可靠的协方差矩阵时
- 制造业的质量控制（多元过程监控）

### 杰卡德相似度（用于集合）

杰卡德相似度衡量两个集合的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

范围从0（无重叠）到1（集合完全相同）。杰卡德距离 = 1 - 杰卡德相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用杰卡德：
- 比较标签、类别或特征的集合
- 基于词汇出现（而非频率）的文档相似度
- 近似重复检测（杰卡德的MinHash近似）
- 比较二值特征向量（存在/缺失数据）
- 评估分割模型（交并比 = 杰卡德）

### 编辑距离（莱文斯坦距离）

编辑距离计算将一个字符串转换为另一个字符串所需的最少单字符操作次数。操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

使用动态规划计算。填充一个矩阵，其中条目(i, j)是字符串A的前i个字符和字符串B的前j个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查与纠正
- DNA序列比对（带权重的操作）
- 模糊字符串匹配
- 混乱文本数据的去重

### KL散度（不是距离，但被用作距离）

KL散度衡量一个概率分布与另一个概率分布的差异程度。它将在课程09中介绍，但它属于这里的讨论，因为人们将其用作“距离”，尽管它并不是。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键属性：KL散度**不**是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求。它也不满足三角不等式。它是一种散度，而不是距离。

前向KL (D_KL(P || Q)) 是“均值寻求”的：Q试图覆盖P的所有模式。
反向KL (D_KL(Q || P)) 是“模式寻求”的：Q专注于P的单一模式。

当你看到KL散度时：
- VAE（ELBO中的KL项将潜在分布推向先验分布）
- 知识蒸馏（学生试图匹配教师的分布）
- RLHF（KL惩罚使微调后的模型保持接近基础模型）
- 策略梯度方法（约束策略更新）

### Wasserstein距离（推土机距离）

Wasserstein距离衡量将一个概率分布转换为另一个所需的最小“工作量”。把它想象成：如果一个分布是一堆土，另一个是一个坑，你需要移动多少土以及移动多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它简化为累积分布函数绝对差值的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么Wasserstein重要：
- 它是一个真正的度量（对称，满足三角不等式）
- 即使分布不重叠，它也能提供梯度（KL散度会趋向无穷大）
- 这一特性使其成为Wasserstein GAN（WGAN）的核心，解决了原始GAN的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用Wasserstein：
- GAN训练（WGAN, WGAN-GP）
- 比较可能不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离

| 任务 | 最佳距离 | 原因 |
|------|----------|------|
| 文本相似度 | 余弦相似度 | 大小是噪声，方向是意义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | 稳健，不会放大罕见的巨大差异 |
| 集合重叠（标签、类别） | 杰卡德 | 数据本质上是集合值的，不是向量 |
| 字符串匹配 | 编辑距离 | 操作映射到人类编辑的直觉 |
| 离群点检测 | 马氏距离 | 考虑特征相关性和尺度 |
| 比较分布 | KL散度 | 衡量使用Q而非P编码信息所损失的信息 |
| GAN训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| 嵌入（向量数据库） | 余弦或点积 | 嵌入被训练为在方向中编码意义 |
| 推荐 | 点积 | 大小可以编码受欢迎程度或置信度 |
| DNA序列 | 加权编辑距离 | 替换代价因核苷酸对而异 |
| 制造业质量控制 | L-infinity | 任何维度的最坏情况偏差都很重要 |

### 与损失函数的联系

损失函数是应用于预测值与目标值的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化在损失函数中加入了对权重的范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么L1产生稀疏性而L2不会：想象二维权重空间中的约束区域。L1是菱形，L2是圆形。损失函数的等高线（椭圆）最有可能在菱形的角处接触，那里一个权重为零。它们在圆上的光滑点处接触，那里两个权重都非零。

### 最近邻搜索

每个距离函数都意味着一个最近邻搜索问题：给定一个查询点，在数据集中找到最近的点。

在具有n个d维点的数据集中，精确的最近邻搜索每个查询的时间复杂度是O(n * d)。对于大型数据集，这太慢了。

近似最近邻（ANN）算法以微小的准确性损失换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（分层可导航小世界图）是现代向量数据库中的主导算法。它构建了一个多层图，其中每个节点连接到其近似的最近邻。搜索从顶层（稀疏，长跳）开始，逐层下降到底层（密集，短跳）。

## 动手实践

### 步骤1：所有范数和距离函数

完整的实现参见 `code/distances.py`。每个函数都仅使用基本的Python数学从头构建。

### 步骤2：相同数据，不同距离，不同邻居

`distances.py` 中的演示创建了一个数据集，选择一个查询点，并展示最近邻如何根据距离度量的不同而改变。在L1下“最近”的点，在L2或余弦距离下可能不是最近的。

### 步骤3：嵌入相似度搜索

代码包含一个模拟嵌入相似度搜索，使用余弦相似度与L2距离查找与查询最相似的“文档”，展示排名可能不同。

## 实际应用

最常见的实际应用：在向量数据库中查找相似项目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索向量数据库时，这就是底层发生的事情。嵌入模型将文本映射到向量。向量数据库使用ANN算法计算你的查询向量与每个存储向量之间的余弦相似度（或点积），以避免检查所有向量。

## 练习

1. 计算点(1, 2, 3)和(4, 0, 6)之间的L1、L2和L-infinity距离。验证对于任意点对，L-inf <= L2 <= L1总是成立。证明为什么这个顺序是有保证的。

2. 创建两个余弦相似度高（> 0.9）但L2距离大（> 10）的向量。从几何角度解释正在发生什么。然后创建两个余弦相似度低（< 0.3）但L2距离小（< 0.5）的向量。

3. 实现一个函数，该函数接受一个数据集和一个查询点，并返回在L1、L2、余弦和马氏距离下的最近邻。找到一个数据集，使得这四种度量对最近点的判断都不一致。

4. 使用CDF方法手动计算[0.5, 0.5, 0, 0]和[0, 0, 0.5, 0.5]之间的Wasserstein距离。然后计算[0.25, 0.25, 0.25, 0.25]和[0, 0, 0.5, 0.5]之间的距离。哪个更大，为什么？

5. 实现用于近似杰卡德相似度的MinHash。生成100个随机集合，计算所有点对的精确杰卡德相似度，并与使用50、100和200个哈希函数的MinHash近似结果进行比较。绘制近似误差图。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 范数 | “向量的大小” | 将向量映射到非负标量的函数，满足三角不等式、绝对齐次性，且仅对零向量为零 |
| L1范数 | “曼哈顿距离” | 分量绝对值的和。在优化中产生稀疏性。对离群点稳健 |
| L2范数 | “欧几里得距离” | 分量平方和的平方根。欧几里得空间中的直线距离 |
| Lp范数 | “广义范数” | 分量绝对值p次幂和的p次方根。L1和L2是特例 |
| L-infinity范数 | “最大范数”或“切比雪夫距离” | 最大绝对分量值。p趋近于无穷时Lp的极限 |
| 余弦相似度 | “向量夹角” | 点积经两个向量的大小归一化后的结果。范围从-1到+1。忽略向量长度 |
| 余弦距离 | “1减去余弦相似度” | 将余弦相似度转换为距离。范围从0到2 |
| 点积 | “未归一化的余弦” | 分量逐项乘积的和。等于余弦相似度乘以两个向量的大小 |
| 马氏距离 | “考虑相关性的距离” | 在使用数据协方差矩阵进行白化（去相关和归一化）的空间中的L2距离 |
| 杰卡德相似度 | “集合重叠度” | 交集大小除以并集大小。用于集合，而非向量 |
| 编辑距离 | “莱文斯坦距离” | 将一个字符串转换为另一个字符串所需的最小插入、删除和替换操作数 |
| KL散度 | “分布间的距离” | 不是真正的距离（不对称）。衡量使用Q而非P编码P所额外需要的比特数 |
| Wasserstein距离 | “推土机距离” | 将一个分布的质量传输到另一个分布所需的最小工作量。是真正的度量 |
| 近似最近邻 | “ANN搜索” | 比精确搜索快得多地找到近似最近点的算法（HNSW, LSH, IVF） |
| HNSW | “向量数据库算法” | 分层可导航小世界图。用于快速近似最近邻搜索的多层图 |
| L1正则化 | “Lasso” | 将权重的L1范数加到损失中。将权重推向零（稀疏性） |
| L2正则化 | “Ridge”或“权重衰减” | 将权重的L2范数平方加到损失中。将权重向零收缩但不产生稀疏性 |
| 弹性网络 | “L1 + L2” | 结合L1和L2正则化。比单独使用任一种更好地处理相关的特征组 |

## 扩展阅读

- [FAISS: 用于高效相似性搜索的库](https://github.com/facebookresearch/faiss) - Meta用于十亿规模ANN搜索的库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将推土机距离引入GAN的论文
- [局部敏感哈希 (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基础的ANN算法
- [词表示的有效估计 (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，其中余弦相似度成为嵌入的默认度量
- [scikit-learn邻居模块文档](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn中距离度量和邻居算法的实用指南