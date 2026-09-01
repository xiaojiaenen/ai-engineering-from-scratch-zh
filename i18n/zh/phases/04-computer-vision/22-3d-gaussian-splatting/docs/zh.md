# 从零实现 3D Gaussian Splatting

> 场景是数百万个 3D 高斯球的云。每个高斯球具有位置、方向、尺度、透明度和依赖于视角的颜色。对其进行光栅化、反向传播穿过光栅化过程，完成。

**类型：** 构建
**语言：** Python
**先修知识：** 第 4 阶段 课程 13（3D 视觉与 NeRF）、第 1 阶段 课程 12（张量运算）、第 4 阶段 课程 10（扩散模型基础，可选）
**耗时：** 约 90 分钟

## 学习目标

- 解释为何 3D Gaussian Splatting 在 2026 年取代 NeRF 成为照片级真实 3D 重建的生产默认方案
- 陈述六个每高斯参数（位置、旋转四元数、尺度、透明度、球谐颜色、可选特征）及其各自贡献的浮点数数量
- 使用 `alpha` 合成从零实现一个 2D 高斯光栅化器，并展示 3D 情况如何投影到相同的循环中
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出到 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` schema

## 问题所在

NeRF 将场景存储为 MLP 的权重。每个渲染像素需要沿光线进行数百次 MLP 查询。训练需要数小时，渲染需要数秒，且权重无法编辑——如果你想移动场景中的一把椅子，就必须重新训练。

3D Gaussian Splatting（Kerbl、Kopanas、Leimkühler、Drettakis，SIGGRAPH 2023）取代了这一切。场景是一个显式的 3D 高斯集合。渲染是 GPU 光栅化，达 100+ fps。训练只需数分钟。编辑是直接的：平移一部分高斯球，你就移动了椅子。到 2026 年，Khronos 小组已批准了高斯 splat 的 glTF 扩展，OpenUSD 26.03 内置了高斯 splat schema，Zillow 和 Apartments.com 用它渲染房地产，而大多数关于 3D 重建的新研究论文都是核心 3DGS 思想的变体。

心智模型很简单，但数学有足够多的组成部分，以至于大多数介绍从光栅化开始，跳过了投影和球谐部分。本课程将构建整体——先构建 2D 版本，再扩展至 3D。

## 概念

### 高斯球携带什么

一个 3D 高斯球是空间中具有这些属性的参数化团块：

```
position         mu         (3,)    世界坐标中的中心
rotation         q          (4,)    编码方向的单位四元数
scale            s          (3,)    每轴的对数尺度（在渲染时指数化）
opacity          alpha      (1,)    后 sigmoid 透明度 [0, 1]
SH 系数          c_lm       (3 * (L+1)^2,)   视角依赖颜色
```

旋转 + 尺度构建一个 3x3 协方差矩阵：`Sigma = R S S^T R^T`。这就是高斯球在 3D 中的形状。球谐函数允许颜色随视角变化——镜面高光、微妙的光泽、视角依赖的光晕——而无需存储逐视角纹理。使用 SH 3 阶，你每个颜色通道获得 16 个系数，仅颜色就占每个高斯球 48 个浮点数。

一个典型场景包含 100 万到 500 万个高斯球。每个存储约 60 个浮点数（3 + 4 + 3 + 1 + 48 + 杂项）。五百万高斯球的场景约 240 MB——远小于带逐点纹理的等效点云，也比高分辨率重渲染的 NeRF MLP 权重重小一个数量级。

### 光栅化，而非光线步进

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯球<br/>(位置、旋转、尺度、<br/>透明度、SH 颜色)"] --> PROJ["投影到 2D<br/>(相机外参 + 内参)"]
    PROJ --> TILES["分配到瓦片<br/>(16x16 屏幕空间)"]
    TILES --> SORT["每瓦片深度排序"]
    SORT --> ALPHA["alpha 合成<br/>从前到后"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全部 GPU 友好。每个像素无需 MLP 查询。一块 RTX 3080 Ti 即可在 147 fps 渲染 600 万个 splat。

### 投影步骤

世界位置 `mu` 处、具有 3D 协方差 `Sigma` 的 3D 高斯球投影到屏幕位置 `mu'` 处、具有 2D 协方差 `Sigma'` 的 2D 高斯球：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 观察变换（相机的旋转 + 平移）
J = 透视投影在 mu' 处的雅可比矩阵
```

2D 高斯球的足迹是一个椭圆，其轴是 `Sigma'` 的特征向量。该椭圆内的每个像素都接收高斯球的贡献，加权因子为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### alpha 合成规则

对于一个像素，覆盖它的高斯球按从后到前排序（或以等效方式使用反转公式的前到后排序）。颜色使用自 1980 年代以来所有半透明光栅化器使用的相同方程合成：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       直到 i 的透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角依赖颜色
```

这与 **NeRF 的体渲染方程相同**，只是作用于显式稀疏高斯集合，而非沿光线的密集采样。这一等价性正是渲染质量能与 NeRF 匹敌的原因——两者都在积分相同的辐射场方程。

### 为何可微

每一步——投影、瓦片分配、alpha 合成、SH 求值——相对于高斯参数都是可微的。给定真值图像，计算渲染像素损失，反向传播穿过光栅化器，通过梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 次迭代，高斯球找到其正确的位置、尺度和颜色。

### 密度增加与剪枝

固定数量的高斯球无法覆盖复杂场景。训练包含两个自适应机制：

- 当梯度幅值高但尺度小时，在其当前位置 **克隆** 一个高斯球——此处重建需要更多细节。
- 当大尺度高斯球的梯度较高时，将其 **分裂** 为两个更小的高斯球——一个大高斯球太平滑，无法拟合该区域。
- **剪枝** 透明度降至阈值以下的高斯球——它们不再贡献。

密度增加每隔 N 次迭代运行一次。场景通常从约 10 万个初始高斯球（从 SfM 点初始化）增长到训练结束时的 100 万到 500 万个。

### 球谐函数一段话说清

视角依赖颜色是单位球面上的函数 `c(direction)`。球谐函数是球面的傅里叶基。截断至 L 阶，你每个通道得到 `(L+1)^2` 个基函数。对新的视角求值颜色是学习到的 SH 系数与在视角方向上求值的基函数之间的点积。0 阶 = 一个系数 = 恒定颜色。3 阶 = 16 个系数 = 足以捕捉漫反射阴影、镜面反射和轻度反射。SD 高斯 splatting 论文默认使用 3 阶。

### 2026 生产栈

```
1. 采集          智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS     COLMAP 或 GLOMAP 推导相机位姿 + 稀疏点
3. 训练 3DGS     nerfstudio / gsplat / inria 官方 / PostShot（RTX 4090 上约 10-30 分钟）
4. 编辑          SuperSplat / SplatForge（清理漂浮物、分割）
5. 导出          .ply -> glTF KHR_gaussian_splatting 或 .usd（OpenUSD 26.03）
6. 查看          Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 与生成式变体

- **4D 高斯 splatting**——高斯球是时间的函数；用于体积视频（超人 2026、A$AP Rocky 的《Helicopter》）。
- **生成式 splats**——文本到 splat 模型（World Labs 的 Marble）可幻觉出整个场景。
- **3D 高斯无迹变换**——NVIDIA NuRec 的自动驾驶仿真变体。

```figure
cv3-gaussian-splat
```

## 动手构建

### 步骤 1：一个 2D 高斯球

我们先构建一个 2D 光栅化器。3D 情况在投影后归结为此。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    返回: (G, H, W)      每个像素对每个高斯球的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 为每个（高斯球，像素）对执行二次型 `diff^T Sigma^-1 diff`。

### 步骤 2：2D splatting 光栅化器

从前到后 alpha 合成。2D 中深度无意义，因此我们使用每高斯球的标量值作为顺序依据。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     在 [0, 1] 范围内
    depths:    (G,)     每高斯球用于排序的标量
    image_size: (H, W)
    返回:    (H, W, 3)  渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快——真实实现使用基于瓦片的 CUDA 内核——但数学完全正确且完全可微。

### 步骤 3：一个可训练的 2D splat 场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D 中的单个角度
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过正确的激活函数映射。这是每个 3DGS 实现的标准模式。

### 步骤 4：将 2D 高斯球拟合到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红色圆形
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝色方形
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 步后，64 个高斯球收敛到两个形状。这就是整个思路——对显式几何基元进行梯度下降。

### 步骤 5：从 2D 到 3D

3D 扩展保持相同的循环。新增内容：

1. 每高斯球的旋转是四元数而非单个角度。
2. 协方差为 `R S S^T R^T`，其中 `R` 由四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参和透视投影在 `mu` 处的雅可比矩阵。
4. 颜色变为球谐展开；在视角方向上对其求值。
5. 深度排序来自实际相机空间 z 而非学习到的标量。

每个生产级实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都在 GPU 上通过基于瓦片的 CUDA 内核精确执行此操作。

### 步骤 6：球谐函数求值

3 阶以内的 SH 基每个通道有 16 项。求值：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   最后一维是 RGB 通道
    dirs:      (..., 3)       单位向量
    返回:     (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 3 阶项省略以保持简洁；完整 16 系数版本见代码文件
    return result
```

学习到的 `sh_coeffs` 存储该高斯球"各个方向的颜色"。渲染时，你对当前视角方向求值，得到一个 3 向量 RGB。

## 使用它

对于真实的 3DGS 工作，请使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。对于典型场景，在 RTX 4090 上运行需要 10-30 分钟。

2026 年重要的导出选项：

- `.ply`——原始高斯点云（可移植，文件最大）。
- `.splat`——PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting`——Khronos 标准，跨查看器可移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat`——USD 原生，用于 NVIDIA Omniverse 和 Vision Pro 管线。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 通过随时间变化的均值和透明度扩展了相同机制。

## 交付它

本课程产生：

- `outputs/prompt-3dgs-capture-planner.md`——一个规划捕获会话（照片数量、相机路径、光照）的提示，针对给定场景类型。
- `outputs/skill-3dgs-export-router.md`——一个技能，根据下游查看器或引擎选择合适的导出格式（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **（简单）** 在不同的合成图像上运行上述 2D splat 训练器。在 `[16, 64, 256]` 范围内变化 `num_splats`，并绘制每个的 MSE vs 步数图。找出收益递减点。
2. **（中等）** 扩展 2D 光栅化器以支持通过 2 阶谐波依赖标量"视角"的每高斯球 RGB 颜色。在一对目标图像上训练并验证模型重建两者。
3. **（困难）** 克隆 `nerfstudio` 并在你拥有的任意场景（书桌、植物、人脸、房间）的 20 张照片捕获上训练 `splatfacto`。导出到 glTF `KHR_gaussian_splatting` 并在查看器中打开（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）。报告训练时间、高斯球数量和渲染 fps。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------|---------|
| 3DGS | "高斯 splat" | 以显式场景表示，由数百万个具有每高斯球位置、旋转、尺度、透明度、SH 颜色的 3D 高斯球组成 |
| 协方差 | "高斯球的形状" | `Sigma = R S S^T R^T`；一个高斯球的方向和各向异性尺度 |
| alpha 合成 | "从后到前混合" | 与 NeRF 体渲染相同的方程，现在作用于显式稀疏集合 |
| 密度增加 | "克隆和分裂" | 在重建欠拟合处自适应添加新高斯球 |
| 剪枝 | "删除低透明度" | 移除在训练中坍缩到接近零透明度的高斯球 |
| 球谐函数 | "视角依赖颜色" | 球面上的傅里叶基；将颜色存储为视角方向的函数 |
| Splatfacto | "nerfstudio 的 3DGS" | 2026 年训练 3DGS 的最简便路径 |
| `KHR_gaussian_splatting` | "glTF 标准" | Khronos 2026 扩展，使 3DGS 跨查看器和引擎可移植 |

## 进一步阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)——原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat)——生产级 CUDA 光栅化器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)——参考训练配方
- [Khronos KHR_gaussian_splatting 扩展](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)——2026 年可移植格式
- [OpenUSD 26.03 发行说明](https://openusd.org/release/)——`UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D 高斯 splatting 2026 状态](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026)——行业概述
