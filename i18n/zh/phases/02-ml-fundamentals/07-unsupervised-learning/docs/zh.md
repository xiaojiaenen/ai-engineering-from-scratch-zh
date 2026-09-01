# 无监督学习

> 无需标签，无需教师。算法自主发现结构。

**类型：** 构建
**语言：** Python
**前置条件：** 第一阶段（范数与距离、概率与分布），第二阶段第 1-6 课
**时间：** 约 90 分钟

## 学习目标

- 从零实现 K-Means、DBSCAN 和高斯混合模型，并比较它们的聚类行为
- 使用轮廓系数和手肘法评估聚类质量，选择最优 K 值
- 解释 DBSCAN 在哪些场景下优于 K-Means，并识别哪个算法能处理非球形簇和离群点
- 使用聚类方法构建异常检测流水线，标记偏离正常模式的点

## 问题所在

迄今为止的每一节 ML 课程都假设数据带有标签："这是输入，这是正确答案。" 在现实世界中，标签成本高昂。一家医院拥有数百万条患者记录，但没有人手工为每条记录标注疾病类别。一个电商网站拥有数百万条用户会话记录，但没有人手工标记客户群体。一个安全团队拥有网络日志，但没有人标记每一条异常。

无监督学习在不被告知要寻找什么的情况下发现模式。它可以将相似的数据点分组，发现隐藏的结构，并凸显异常值。如果说监督学习是从一本带有答案key的教科书中学习，那么无监督学习就是对着原始数据凝视，直到模式自行显现。

但有一个问题：没有标签，你就无法直接衡量"对"或"错"。你需要不同的工具来评估算法发现的结构是否有意义。

## 概念解析

### 聚类：将相似事物分组

聚类将每个数据点分配到一组（簇），使得同一组内的点比与其他组的点更相似。核心问题永远是："相似"是什么意思？

```mermaid
flowchart LR
    A[原始数据] --> B{选择方法}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[层次聚类]
    B --> F[GMM]
    C --> G[平面、球形簇]
    D --> H[任意形状、噪声检测]
    E --> I[嵌套簇的树]
    F --> J[软分配、椭圆簇]
```

### K-Means：主力算法

K-Means 将数据精确划分为 K 个簇。每个簇有一个质心（质心即重心），每个点归属于最近的质心。

Lloyd 算法：

1. 随机选取 K 个点作为初始质心
2. 将每个数据点分配到最近的质心
3. 重新计算每个质心为其所有分配点的均值
4. 重复步骤 2-3，直到分配不再发生变化

目标函数（惯性）衡量每个点到其分配质心的平方距离之和。K-Means 最小化该值，但只能找到局部最优解。不同的初始化可能产生不同的结果。

### 选择 K

两种标准方法：

**手肘法：** 对 K = 1, 2, 3, ..., n 运行 K-Means。绘制惯性 vs K 的曲线。寻找"手肘"点，即继续增加簇数不再显著降低惯性的位置。

**轮廓系数：** 对每个点，衡量它与自身簇的相似度（a）与最近的其他簇的相似度（b）。轮廓系数为 (b - a) / max(a, b)，取值范围为 -1（错误聚类）到 +1（良好聚类）。对所有点取平均得到全局分数。

### DBSCAN：基于密度的聚类

K-Means 假设簇是球形的，且需要提前指定 K。DBSCAN 不做这两个假设。它将簇找为被稀疏区域分隔的密集区域。

两个参数：
- **eps**：邻域的半径
- **min_samples**：形成密集区域所需的最小点数

三种类型的点：
- **核心点**：在 eps 距离内至少有 min_samples 个点
- **边界点**：在某个核心点的 eps 距离内，但自身不是核心点
- **噪声点**：既非核心也非边界。这些即离群点。

DBSCAN 将通过 eps 相连的核心点连接到同一个簇中。边界点加入附近核心点所在的簇。噪声点不属于任何簇。

优点：可以发现任意形状的簇，自动确定簇的数量，识别离群点。缺点：难以处理密度变化的簇。

### 层次聚类

构建嵌套簇的树（树状图）。

凝聚法（自底向上）：
1. 将每个点初始化为独立的簇
2. 合并最近的两个簇
3. 重复直到只剩一个簇
4. 在所需层级截断树状图，得到 K 个簇

簇之间的"接近程度"可以用以下方式度量：
- **单连接**：两个簇中任意两点间的最小距离
- **全连接**：两个簇中任意两点间的最大距离
- **平均连接**：所有点对之间距离的平均值
- **Ward 方法**：使簇内总方差增加最小的合并方式

### 高斯混合模型（GMM）

K-Means 给出硬分配：每个点恰好属于一个簇。GMM 给出软分配：每个点属于每个簇都有一个概率。

GMM 假设数据由 K 个高斯分布混合生成，每个高斯分布有其自身的均值和协方差。期望最大化（EM）算法在以下两步之间交替：

- **E步**：计算每个点属于每个高斯分布的概率
- **M步**：更新每个高斯分布的均值、协方差和混合权重，以最大化数据的似然

GMM 可以建模椭圆簇（而不仅仅是 K-Means 的球形簇），并能自然地处理重叠簇。

### 何时使用哪种方法

| 方法 | 适用场景 | 避免使用场景 |
|------|----------|-------------|
| K-Means | 大数据集、球形簇、已知 K | 不规则形状、存在离群点 |
| DBSCAN | K 未知、任意形状、离群点检测 | 密度变化大、维度极高 |
| 层次聚类 | 小数据集、需要树状图、K 未知 | 大数据集（O(n²) 内存） |
| GMM | 重叠簇、需要软分配 | 超大数据集、维度太多 |

### 利用聚类进行异常检测

聚类天然支持异常检测：
- **K-Means**：远离所有质心的点即为异常点
- **DBSCAN**：噪声点按定义就是异常点
- **GMM**：在所有高斯分布下概率均很低的点即为异常点

```figure
kmeans-step
```

## 动手构建

### 步骤 1：从零实现 K-Means

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  在第 {iteration + 1} 次迭代收敛")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 步骤 2：手肘法与轮廓系数

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("手肘法：")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: 惯性={inertia:.2f}")

    print("\n轮廓系数：")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: 轮廓系数={score:.4f}")

    return inertias
```

### 步骤 3：从零实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 步骤 4：高斯混合模型（EM 算法）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM 在第 {iteration + 1} 次迭代收敛")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 步骤 5：生成测试数据并运行全部代码

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== 3 个 blob 上的 K-Means ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  质心: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  轮廓系数: {sil:.4f}")

    print("\n=== 手肘法 ===")
    find_best_k(data, max_k=6)

    print("\n=== 3 个 blob 上的 DBSCAN ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  找到 {n_clusters} 个簇，{n_noise} 个噪声点")

    print("\n=== 3 个 blob 上的 GMM ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  均值: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  权重: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  轮廓系数: {gmm_sil:.4f}")

    print("\n=== 月牙形数据上的 DBSCAN（非球形簇） ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  找到 {n_moon_clusters} 个簇，{n_moon_noise} 个噪声点")

    print("\n=== 月牙形数据上的 K-Means（将无法正确分离） ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  轮廓系数: {moon_sil:.4f}")
    print("  K-Means 分割月牙形数据效果很差，因为它们不是球形的")

    print("\n=== 使用 DBSCAN 进行异常检测 ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  检测到 {len(anomalies)} 个异常点")
    for a in anomalies[-3:]:
        print(f"    点 {[round(v, 2) for v in a]}")
```

## 实际应用

使用 scikit-learn，上述算法只需一行代码：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

从零实现的版本让你清楚了解这些库具体计算了什么。K-Means 在分配和重新计算之间迭代。DBSCAN 从密集种子区域扩展簇。GMM 在期望和最大化之间交替。库版本增加了数值稳定性、更智能的初始化方式（K-Means++）和 GPU 加速，但核心逻辑是相同的。

## 交付成果

本节课产出了从零实现的 K-Means、DBSCAN 和 GMM。聚类代码可作为更高级无监督方法的基础，被反复复用。

## 练习

1. 实现 K-Means++ 初始化：不再是随机选取质心，而是随机选取第一个，后续每个质心以与其到最近已有质心距离的平方成正比的概率选取。比较与随机初始化的收敛速度。
2. 在代码中添加层次凝聚聚类。实现 Ward 连接并生成树状图（以嵌套合并列表的形式）。在不同层级截断并与 K-Means 结果进行比较。
3. 构建一个简单的异常检测流水线：在同一批数据上运行 DBSCAN 和 GMM，标记两种方法都认定为离群点的样本（DBSCAN 中的噪声点，GMM 中低概率点）。衡量两者的重合度，并讨论两种方法意见不一致的场景。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| 聚类 | "将相似事物分组" | 将数据划分为子集，使得组内相似度超过组间相似度，通过特定距离度量来衡量 |
| 质心 | "簇的中心" | 分配到某簇的所有点的均值；K-Means 将其用作簇的代表点 |
| 惯性 | "簇有多紧密" | 每个点到其分配质心的平方距离之和；越小越紧密 |
| 轮廓系数 | "簇分离得有多好" | 对每个点，计算 (b - a) / max(a, b)，其中 a 为簇内平均距离，b 为到最近其他簇的平均距离 |
| 核心点 | "密集区域内的点" | 在 DBSCAN 中，eps 距离内至少有 min_samples 个邻居的点 |
| EM 算法 | "软版 K-Means" | 期望最大化：迭代计算成员概率（E步）并更新分布参数（M步） |
| 树状图 | "簇的树" | 展示层次聚类中簇的合并顺序和距离的树形图 |
| 异常 | "离群点" | 不符合预期模式的数据点，由 DBSCAN 识别为噪声，或由 GMM 识别为低概率 |

## 延伸阅读

- [斯坦福 CS229 - 无监督学习](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng 关于聚类和 EM 的讲义
- [scikit-learn 聚类指南](https://scikit-learn.org/stable/modules/clustering.html) - 所有聚类算法的实用对比，配有可视化示例
- [DBSCAN 原始论文（Ester 等，1996）](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 引入基于密度聚类思想的论文
