# 3D 生成

> 在 3D 领域，2D-to-3D 的杠杆效应最为显著。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成式推进在此基础上叠加了多视图扩散 + 3D 重建，能够从单一提示词或照片生成物体和场景。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 4（视觉），Phase 8 · 07（潜在扩散模型）
**时间：** 约 45 分钟

## 问题所在

3D 内容制作非常痛苦：

- **表示方式。** 网格、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯。每种方式都有各自的权衡。
- **数据稀缺。** ImageNet 有 1400 万张图片。最大的干净 3D 数据集（Objaverse-XL，2023 年）只有约 1000 万个物体，大多数质量较低。
- **内存开销。** 一个 512³ 体素网格包含 1.28 亿个体素；一个可用的场景 NeRF 需要每射线 100 万个样本。生成比重建更难。
- **监督信号。** 对于 2D 图像你有像素作为监督。对于 3D 你通常只有少数几张 2D 视角，需要将其提升到 3D。

2026 年的技术栈将这两个问题分离。首先，用扩散模型生成 *2D 多视图图像*。其次，用 *3D 表示*（通常是 Gaussian Splatting）去拟合这些图像。

## 概念解析

![3D 生成：多视图扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示方式：3D Gaussian Splatting（Kerbl 等，2023）

将场景表示为约 100 万个 3D 高斯分布的集合。每个高斯包含 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐函数颜色（3 阶时 48 个，0 阶时 3 个）。

渲染 = 投影 + alpha 混合。速度很快（在 4090 上 1080p 约 100 fps）。可微分。通过对真实照片进行梯度下降来拟合。一个场景在消费级 GPU 上只需 5-30 分钟即可拟合完成。

2023-2024 年在此基础上有两个创新：
- **生成式 Gaussian splats。** LGM、LRM、InstantMesh 等模型直接从一张或几张图像预测高斯云。
- **4D Gaussian Splatting。** 带有逐帧偏移的高斯分布，用于动态场景。

### 多视图扩散

微调预训练的图像扩散模型，使其能够从文本提示词或单张图像生成同一物体的多个一致视角。Zero123（Liu 等，2023）、MVDream（Shi 等，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围 4-16 个视角，再通过 Gaussian Splatting 或 NeRF 提升到 3D。

### 文本到 3D 管线

| 模型 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| DreamFusion（2022） | 文本 | 通过 SDS 生成的 NeRF | 每个资产约 1 小时 |
| Magic3D | 文本 | 网格 + 纹理 | 约 40 分钟 |
| Shap-E（OpenAI，2023） | 文本 | 隐式 3D | 约 1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | 约 30 分钟 |
| LRM（Meta，2023） | 图像 | 三平面 | 约 5 秒 |
| InstantMesh（2024） | 图像 | 网格 | 约 10 秒 |
| SV3D（Stability，2024） | 图像 | 新视角 | 约 2 分钟 |
| CAT3D（Google，2024） | 1-64 张图像 | 3D NeRF | 约 1 分钟 |
| TripoSR（2024） | 图像 | 网格 | 约 1 秒 |
| Meshy 4（2025） | 文本 + 图像 | PBR 网格 | 约 30 秒 |
| Rodin Gen-1.5（2025） | 文本 + 图像 | PBR 网格 | 约 60 秒 |
| Tencent Hunyuan3D 2.0（2025） | 图像 | 网格 | 约 30 秒 |

2025-2026 年方向：直接支持 PBR 材质的文本到网格模型，适用于游戏引擎。多视图扩散中间步骤仍是通用物体的最佳方案。

### NeRF（背景知识）

神经辐射场（Mildenhall 等，2020）。一个小型 MLP 接收 `(x, y, z, 视角方向)` 并输出 `(颜色，密度)`。通过沿射线积分进行渲染。质量优于基于网格的新视角合成，但渲染速度慢 100-1000 倍。在大多数实时应用中已被 Gaussian Splatting 取代，但在研究中仍占主导地位。

```figure
v4-3d-multiview
```

## 动手实现

`code/main.py` 实现了一个玩具级的 2D "Gaussian Splatting" 拟合：将一个合成目标图像（平滑渐变）表示为多个 2D 高斯 Splat 的和。通过梯度下降优化位置、颜色和协方差以匹配目标图像。你可以看到两个核心操作：前向渲染（Splat + alpha 混合）和通过梯度下降拟合。

### 步骤 1：2D Gaussian Splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：通过求和渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D Gaussian Splatting 会按深度排序高斯并按顺序进行 alpha 混合。我们的 2D 玩具版本直接求和。

### 步骤 3：通过梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 注意事项

- **视角不一致。** 如果你独立生成 4 个视角而它们对物体结构存在分歧，3D 拟合结果会模糊。解决方法：使用共享注意力的多视图扩散。
- **背面幻觉。** 单图转 3D 必须"发明"不可见的背面。质量差异很大。
- **高斯 Splat 爆炸。** 无约束训练会增长到 1000 万个 Splat 并过拟合。去密度化 + 剪枝启发式方法（来自 3D-GS 原始论文）必不可少。
- **拓扑问题。** 来自隐式场（SDF）的网格经常存在孔洞或自相交。发布前运行重网格化工具（如 Blender 的体素重网格）。
- **训练数据的许可证。** Objaverse 许可证混杂；商业用途因模型而异。

## 使用指南

| 任务 | 2026 年推荐 |
|------|-------------|
| 从照片重建场景 | Gaussian Splatting（3DGS、Gsplat、Scaniverse） |
| 游戏用文本到 3D 物体 | Meshy 4 或 Rodin Gen-1.5（输出 PBR） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 从少量图像合成新视角 | CAT3D、SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 虚拟形象 / 穿衣人体 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 上周刚发布的东西 |

对于在游戏中或电商管线中发布生产级 3D 内容：Meshy 4 或 Rodin Gen-1.5 输出的 PBR 网格可直接导入 Unity / Unreal。

## 完成项目

保存 `outputs/skill-3d-pipeline.md`。技能接受 3D 需求简报（输入：文本 / 一张图像 / 几张图像；输出：网格 / Splat / NeRF；用途：渲染 / 游戏 / VR），并输出：管线（多视图扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 使用 4、16、64 个高斯运行 `code/main.py`。报告最终 MSE 与目标的差异。
2. **中等。** 扩展到彩色高斯（RGB）。确认重建结果与目标颜色模式匹配。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片的采集数据重建真实物体。报告拟合时间和在保留视图上的最终 SSIM。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 3D Gaussian Splatting | "3DGS" | 将场景表示为 3D 高斯云的集合；可微 alpha 混合渲染。 |
| NeRF | "神经辐射场" | 输出 3D 点处颜色 + 密度的 MLP；通过射线积分渲染。 |
| Triplane | "三个 2D 平面" | 将 3D 分解为三个 2D 轴对齐特征网格；比体素更经济。 |
| SDS | "分数蒸馏采样" | 使用 2D 扩散分数作为伪梯度来训练 3D 模型。 |
| 多视图扩散 | "同时输出多个视角" | 输出一批一致相机视角的扩散模型。 |
| PBR | "基于物理的渲染" | 具有反照率、粗糙度、金属度、法线通道的材质。 |
| 去密度化 | "增大 Splat" | 3DGS 训练启发式方法：在高梯度区域分裂 / 克隆 Splat。 |

## 生产笔记：3D 尚无统一的运行时

与图像（潜在扩散 + DiT）和视频（时空 DiT）不同，3D 在 2026 年还没有单一的主导运行时。生产决策树因表示方式而异：

- **NeRF / Triplane。** 推理是光线步进 + 每个样本一次 MLP 前向传播。512² 渲染需要数百万次 MLP 前向传播。激进地批处理射线样本；SDPA/xformers 适用。
- **多视图扩散 + LRM 重建。** 两阶段管线。阶段 1（多视图 DiT）是像第 07 课一样的扩散服务器。阶段 2（LRM transformer）是对视角的一次性前向传播。整体延迟特征是"扩散 + 一次性"——根据各阶段选择合适的服务原语。
- **SDS / DreamFusion。** 每个资产优化，而非推理。构建任务，而非请求处理器。

对于大多数 2026 年的产品，正确的做法是"按需运行多视图扩散模型，异步重建为 3DGS，为实时查看提供服务"。这巧妙地将工作负载分割给 GPU 推理服务器（快）和离线优化器（慢）。

## 延伸阅读

- [Mildenhall 等 (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl 等 (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole 等 (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu 等 (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi 等 (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视图扩散。
- [Hong 等 (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao 等 (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
