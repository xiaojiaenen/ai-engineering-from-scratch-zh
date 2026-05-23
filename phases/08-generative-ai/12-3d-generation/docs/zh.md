# 3D生成

> 3D 是二维到三维杠杆效应最强的模态。2023 年的突破是 3D 高斯溅射。2024-2026 年的生成式推动力将多视角扩散 + 3D 重建叠加以从单个提示词或照片生成物体和场景。

**类型:** 学习
**语言:** Python
**前提:** 阶段 4（视觉），阶段 8 · 07（潜空间扩散）
**时间:** ~45 分钟

## 问题所在

3D 内容创作非常痛苦：

- **表示方式。** 网格、点云、体素网格、有符号距离场 (SDF)、神经辐射场 (NeRF)、3D 高斯。每种方式都有其权衡。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集 (Objaverse-XL, 2023) 大约有 1000 万个物体，大部分质量较低。
- **内存。** 一个 512³ 的体素网格包含 1.28 亿个体素；一个有用的场景 NeRF 每条光线需要 100 万个采样点。生成比重建更难。
- **监督。** 对于二维图像，你有像素。对于 3D，你通常只有一些二维视图，必须将其提升到三维。

2026 年的技术栈将这两个问题分开。首先，使用扩散模型生成 *二维多视角图像*。其次，将 *3D 表示*（通常是高斯溅射）拟合到这些图像上。

## 核心概念

![3D 生成：多视角扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示方式：3D 高斯溅射 (Kerbl et al., 2023)

将一个场景表示为约 100 万个 3D 高斯体的云。每个高斯体有 59 个参数：位置 (3)、协方差 (6，或四元数 4 + 缩放 3)、不透明度 (1)、球谐函数颜色 (3 度为 48，0 度为 3)。

渲染 = 投影 + alpha 合成。速度快（在 4090 上 1080p 约 100 fps）。可微。通过针对真实照片的梯度下降进行拟合。一个场景在消费级 GPU 上 5-30 分钟即可拟合。

2023-2024 年的两项新进展：
- **生成式高斯溅射。** 像 LGM、LRM、InstantMesh 这样的模型直接从一张或几张图像预测高斯云。
- **4D 高斯溅射。** 带有逐帧偏移的高斯体，用于动态场景。

### 多视角扩散

微调一个预训练的图像扩散模型，以从文本提示或单张图像生成同一物体的多个一致视图。Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024)。通常输出物体周围的 4-16 个视图，通过高斯溅射或 NeRF 提升到 3D。

### 文本到 3D 流程

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | 文本 | 通过 SDS 生成 NeRF | 每个资产 ~1 小时 |
| Magic3D | 文本 | 网格 + 纹理 | ~40 分钟 |
| Shap-E (OpenAI, 2023) | 文本 | 隐式 3D | ~1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | ~30 分钟 |
| LRM (Meta, 2023) | 图像 | 三平面 | ~5 秒 |
| InstantMesh (2024) | 图像 | 网格 | ~10 秒 |
| SV3D (Stability, 2024) | 图像 | 新视角 | ~2 分钟 |
| CAT3D (Google, 2024) | 1-64 张图像 | 3D NeRF | ~1 分钟 |
| TripoSR (2024) | 图像 | 网格 | ~1 秒 |
| Meshy 4 (2025) | 文本 + 图像 | PBR 网格 | ~30 秒 |
| Rodin Gen-1.5 (2025) | 文本 + 图像 | PBR 网格 | ~60 秒 |
| Tencent Hunyuan3D 2.0 (2025) | 图像 | 网格 | ~30 秒 |

2025-2026 年的方向：具有适合游戏引擎的 PBR 材质的直接文本到网格模型。对于通用物体，多视角扩散中间步骤仍然是表现最佳的方案。

### NeRF（背景知识）

神经辐射场 (Mildenhall et al., 2020)。一个微型 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿光线积分进行渲染。在质量上超越基于网格的新视角合成，但渲染速度慢 100-1000 倍。在大多数实时用途中已被高斯溅射取代，但在研究领域仍占主导地位。

## 动手实现

`code/main.py` 实现了一个玩具级的 2D “高斯溅射” 拟合：将一个合成目标图像（平滑渐变）表示为 2D 高斯溅射的总和。通过梯度下降优化位置、颜色和协方差以匹配目标。你将看到两个核心操作：正向渲染（溅射 + alpha 合成）和通过梯度下降进行拟合。

### 步骤 1：2D 高斯溅射

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：通过求和溅射来渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D 高斯溅射会按深度对高斯体排序并按顺序进行 alpha 合成。我们的 2D 玩具只是简单求和。

### 步骤 3：通过梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常见陷阱

- **视角不一致性。** 如果你独立生成 4 个视图，并且它们在物体结构上不一致，3D 拟合就会模糊。解决方法：使用具有共享注意力的多视角扩散。
- **背面幻觉。** 单张图像到 3D 必须想象出看不见的背面。质量差异巨大。
- **高斯溅射爆炸。** 无约束的训练会增长到 1000 万个溅射点并过拟合。密集化 + 修剪启发式方法（来自 3D-GS 原始论文）是必不可少的。
- **拓扑问题。** 从隐式场（SDF）生成的网格通常有孔洞或自相交。发布前运行重新网格化工具（例如 blender 的体素重新网格化）。
- **训练数据许可。** Objaverse 拥有混合许可；商业用途因模型而异。

## 使用场景

| 任务 | 2026 年选择 |
|------|-----------|
| 从照片重建场景 | 高斯溅射 (3DGS, Gsplat, Scaniverse) |
| 用于游戏的文本到 3D 物体 | Meshy 4 或 Rodin Gen-1.5 (PBR 输出) |
| 图像到 3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 从少量图像合成新视角 | CAT3D, SV3D |
| 动态场景重建 | 4D 高斯溅射 |
| 虚拟形象 / 穿衣人体 | Gaussian Avatar, HUGS |
| 研究 / 最新 SOTA | 上周刚发布的任何模型 |

用于在游戏或电商流水线中部署生产级 3D 内容：Meshy 4 或 Rodin Gen-1.5 输出可直接用于 Unity / Unreal 的 PBR 网格。

## 部署上线

保存 `outputs/skill-3d-pipeline.md`。该技能接受一个 3D 需求简报（输入：文本 / 一张图像 / 少量图像；输出：网格 / 溅射点 / NeRF；用途：渲染 / 游戏 / VR）并输出：流程（多视角扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需的材质通道。

## 练习

1. **简单。** 使用 4、16、64 个高斯体运行 `code/main.py`。报告最终与目标的 MSE。
2. **中等。** 扩展到彩色高斯体 (RGB)。确认重建结果匹配目标的颜色图案。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片拍摄中重建一个真实物体。报告拟合时间和在保留视图上的最终 SSIM。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 将场景表示为 3D 高斯体的云；可微的 alpha 合成渲染。 |
| NeRF | "神经辐射场" | 在 3D 点处输出颜色 + 密度的 MLP；通过光线积分渲染。 |
| Triplane | "三个二维平面" | 将 3D 分解为三个二维轴对齐的特征网格；比体积表示更节省。 |
| SDS | "分数蒸馏采样" | 通过使用二维扩散分数作为伪梯度来训练 3D 模型。 |
| Multi-view diffusion | "一次生成多个视图" | 输出一批一致的相机视图的扩散模型。 |
| PBR | "基于物理的渲染" | 具有反照率、粗糙度、金属度、法线通道的材质。 |
| Densification | "增长溅射点" | 3DGS 训练启发式方法：在高梯度区域分裂/克隆溅射点。 |

## 生产注意事项：3D 尚无共享基底

与图像（潜空间扩散 + DiT）和视频（时空 DiT）不同，3D 在 2026 年没有单一的主导运行时。生产决策树在表示方式上分叉：

- **NeRF / 三平面。** 推理是光线行进 + 每个样本一次 MLP 前向传播。一个 512² 的渲染需要数百万次 MLP 前向传播。积极批量处理光线采样；SDPA/xformers 适用。
- **多视角扩散 + LRM 重建。** 两阶段流程。阶段 1（多视角 DiT）是一个扩散服务器，就像课程 07 一样。阶段 2 (LRM transformer) 是在视图上进行一次前向传播。整体延迟特征是“扩散 + 一次前向传播”——相应地为每个阶段选择服务原语。
- **SDS / DreamFusion。** 每个资产的优化，而非推理。构建作业，而非请求处理器。

对于 2026 年的大多数产品，正确答案是“按请求运行多视角扩散模型，异步重建到 3DGS，为实时查看提供 3DGS 服务”。这清晰地将工作负载分配在 GPU 推理服务器（快）和离线优化器（慢）之间。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视角扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。