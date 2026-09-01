# Vision Transformers 与 Patch-Token 原语

> 在多模态之前，图像必须先转变为 transformer 能够处理的 token 序列。2020 年的 ViT 论文以 16x16 像素的 patch、线性投影和位置嵌入回答了这一问题。五年后，2026 年的所有前沿模型（原生支持 2576px 的 Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）依然以此为起点——编码器从 ViT 演变为 DINOv2 再到 SigLIP 2，增加了 register token，位置编码方案升级为 2D-RoPE，但这一原语始终未变。本课将从头到尾解析 patch-token 流水线，并用 Python 标准库实现它，从而为 Phase 12 的其余内容建立关于“视觉 token”的具体心智模型。

**类型：** Learn
**语言：** Python（stdlib，patch tokenizer + geometry calculator）
**前置要求：** Phase 7（Transformers），Phase 4（Computer Vision）
**时间：** 约 120 分钟

## 学习目标

- 将 HxWx3 尺寸的图像转换为带有正确位置编码的 patch token 序列。
- 计算给定 (patch size, resolution, hidden dim, depth) 下 ViT 的序列长度、参数量和 FLOPs。
- 列举推动 ViT 从 2020 年研究走向 2026 年生产的三项升级：自监督预训练（DINO / MAE）、register token 和原生分辨率打包。
- 针对下游任务在 CLS pooling、mean pooling 和 register token 之间做出选择。

## 问题

Transformer 处理的是向量序列。文本本身已经是序列（字节或 token）。图像则是具有三个颜色通道的 2D 像素网格——并非序列。若将所有像素展平，一张 224x224 的 RGB 图像将变成 150,528 个 token，而在此长度上进行 self-attention 根本不可行（计算量随序列长度呈二次增长）。

2020 年之前的方法是在前端外挂一个 CNN 特征提取器：ResNet 生成 7x7、2048 维的特征图，将这 49 个 token 输入 transformer。这种方法可行，但继承了 CNN 的归纳偏置（平移等变性、局部感受野），且失去了 transformer 在规模扩展上的优势。

Dosovitskiy 等人（2020）直截了当地提出了一个问题：如果我们跳过 CNN 会怎样？将图像划分为固定大小的 patch（例如 16x16 像素），对每个 patch 进行线性投影得到向量，添加位置嵌入，然后将序列输入 vanilla transformer。当时这被视为异端——没有卷积的视觉处理。但在足够多的数据（JFT-300M，随后是 LAION）支持下，它在 ImageNet 上击败了 ResNet 并持续进步。

到 2026 年，ViT 原语已成为不可动摇的基础。每个开源权重的 VLM 的 vision tower 都是其后裔（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“是否应该使用 patch？”，而是“用什么 patch size、什么 resolution schedule、什么预训练目标、什么位置编码”。

## 概念

### 将 Patch 作为 Token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch size `P`，你将图像切割成 `(H/P) x (W/P)` 个不重叠的 patch 网格。每个 patch 是一个 `P x P x 3` 的像素立方体。将每个立方体展平为 `3 P^2` 维的向量。应用共享的线性投影 `W_E`（形状为 `(3 P^2, D)`）将每个 patch 映射到模型的 hidden dimension `D`。

以 ViT-B/16 的标准配置为例：
- 分辨率 224，patch size 16 → 网格 14x14 → 196 个 patch token。
- 每个 patch 包含 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加可学习的 `[CLS]` token → 序列长度为 197。

从数学上看，patch 投影等同于 kernel size 为 `P`、stride 为 `P`、输出通道数为 `D` 的 2D 卷积。生产代码正是这样实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“线性投影”是概念性描述；“卷积核”是高效实现。

### 位置嵌入

Patch 本身没有内在顺序——transformer 将它们视为一个集合。早期 ViT 添加了可学习的 1D 位置嵌入（每个位置一个 768 维向量，共 197 个）。这样可以工作，但会将模型绑定到训练分辨率：推理时若更改网格大小，必须对位置表进行插值。

现代视觉 backbone 使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或因子化 2D 位置。2D-RoPE 根据 patch 的 (row, column) 索引旋转 query 和 key 向量，使模型能够从旋转角度推断相对 2D 位置。无需位置表。模型在推理时可以处理任意网格尺寸。

### CLS Token、Pool 输出与 Register Token

什么是图像级表示？三种方案并存：

1. `[CLS]` token。在 patch 序列前添加一个可学习向量。经过所有 transformer block 后，CLS token 的 hidden state 即为图像表示。继承自 BERT。原始 ViT 和 CLIP 使用此方案。
2. Mean pool。对 patch token 的输出 hidden state 求平均。SigLIP、DINOv2 和大多数现代 VLM 使用此方案。
3. Register token。Darcet 等人（2023）观察到，未经显式 sink token 训练的 ViT 会产生高范数“伪影”patch，这些 patch 会劫持 self-attention。添加 4–16 个可学习的 register token 可以吸收这种负载，并提升 dense prediction 质量（分割、深度估计）。DINOv2 和 SigLIP 2 均内置 registers。

这一选择对下游任务至关重要。CLS 适用于分类任务。对于将 patch token 输入 LLM 的 VLM，你完全跳过 pooling——每个 patch 都成为 LLM 的输入 token。Registers 在交接给 LLM 前会被丢弃（它们是脚手架，而非内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 在 JFT-300M 上使用监督分类进行预训练。随后迅速被以下方法取代：

- CLIP（2021）：在 4 亿对图文上进行对比学习。见 Lesson 12.02。
- MAE（2021，He 等人）：掩码 75% 的 patch，重建像素。自监督，可直接在纯图像上工作。
- DINO（2021）/ DINOv2（2023）：基于 student-teacher 架构的自蒸馏，无需标签和 caption。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉 backbone，也是“dense features”用例的默认选择。
- SigLIP / SigLIP 2（2023，2025）：带有 sigmoid loss 和 NaFlex 原生宽高比支持的 CLIP。是 2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）的主流 vision tower。

你选择的预训练方式决定了 backbone 的擅长领域：CLIP/SigLIP 擅长与文本的语义匹配，DINOv2 擅长 dense 视觉特征，MAE 适合作为下游微调的起点。

### 缩放定律

ViT 缩放（Zhai 等人，2022）确立了 ViT 的质量遵循模型规模、数据规模和计算量的可预测规律。在计算量固定的前提下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch size 是序列长度与保真度之间的权衡杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型配置）比 patch 16 为每幅图像产生更多 token；更利于 OCR 和 dense 任务，但速度更慢。
- 分辨率是另一个关键杠杆。从 224 到 384 再到 512 几乎总是能带来提升，但 FLOPs 成本呈二次增长。

ViT-g/14（1B 参数，patch 14，分辨率 224 → 256 token）和 SigLIP SO400m/14（400M 参数，patch 14）是 2026 年开源 VLM 的两款主力编码器。

### ViT 的参数量计算

完整计算逻辑位于 `code/main.py`。以 224 分辨率下的 ViT-B/16 为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，先用这种方法估算每个 ViT 的参数量。backbone 的大小决定了你在任何下游 VLM 中的 VRAM 底线。

### 2026 生产环境配置

2026 年大多数开源 VLM 搭载的编码器是原生分辨率（NaFlex）下的 SigLIP 2 SO400m/14。其配置如下：
- 400M 参数。
- Patch size 14，默认分辨率 384 → 每幅图像 729 个 patch token。
- 图像级任务使用 Mean pool；VQA 任务则将所有 729 个 patch 输入 LLM。
- 4 个 register token，在交接给 LLM 前丢弃。
- 2D-RoPE 配合图像级缩放以支持原生宽高比。

该配置中的每一项决策都能追溯到你可以阅读的论文。

```figure
image-patch-tokens
```

## 实践

`code/main.py` 是一个 patch tokenizer 和几何计算器。它接受 (图像高度 H, 宽度 W, patch P, hidden D, 深度 L) 作为输入，并输出：

- 分 patch 后的网格形状和序列长度。
- 合成 8x8 像素玩具图像的 token 序列（逐步演示 flatten + project 流程）。
- 按 patch embed、position embed、transformer blocks 和 head 拆分的参数量。
- 目标分辨率下每次前向传播的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 之间的对比表格。

运行它。将参数量与公开数值对照。调整 patch size 和分辨率，感受 token 数量的成本变化。

## 交付成果

本课将生成 `outputs/skill-patch-geometry-reader.md`。给定 ViT 配置（patch size、resolution、hidden dim、depth），它输出 token 数量、参数量和 VRAM 估算值，并附带理由说明。在为 VLM 选择 vision backbone 时都应使用此技能——它能避免“token 数量爆炸导致 LLM 上下文溢出”这类意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch size 14 下的 patch-token 序列长度。这与仅使用 CLS 的表示相比如何？

2. patch 14 处理一帧 1080p 图像（1920x1080）会产生多少个 token？在 30 FPS 下播放 5 分钟视频，总共会产生多少视觉 token？哪种方式最能节省开销：pooling、帧采样还是 token merging？

3. 用纯 Python 实现对 patch token 的 mean pooling。验证对 DINOv2 输出的 196 个 token 进行 mean pooling 的结果是否与模型在请求 pooled embedding 时 `forward` 的返回值一致。

4. 阅读《Vision Transformers Need Registers》（arXiv:2309.16588）第 3 节。用两句话描述 registers 吸收的 artifact 是什么，以及它为何对下游 dense prediction 至关重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像列表，生成单个打包序列和 block-diagonal attention mask。学到 Lesson 12.06 时用于验证。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|------------------------|----------------------------------------------------------|
| Patch | “16x16 像素方块” | 输入图像的固定大小且不重叠的区域；对应一个 token |
| Patch embedding | “线性投影” | 共享的可学习矩阵（或 stride=P 的 Conv2d），将展平的 patch 像素映射为 D 维向量 |
| CLS token | “分类 token” | prepend 的可学习向量，其最终 hidden state 表示整张图像；2026 年已非必需 |
| Register token | “Sink token” | 额外的可学习 token，用于吸收 ViT 在预训练过程中产生的高范数注意力伪影 |
| Position embedding | “位置信息” | 使序列具备顺序感知能力的逐位置向量或旋转；2D-RoPE 是现代默认方案 |
| Grid | “Patch 网格” | 给定分辨率和 patch size 下，(H/P) x (W/P) 的 2D patch 数组 |
| NaFlex | “原生灵活分辨率” | SigLIP 2 特性：单一模型无需重新训练即可支持多种宽高比和分辨率 |
| Backbone | “Vision tower” | 预训练的图像编码器，其 patch-token 输出作为 VLM 中 LLM 的输入 |
| Pooling | “图像级摘要” | 将 patch token 聚合成单一向量的策略：CLS、mean、attention pool 或 register-based |
| Patch 14 vs 16 | “细粒度 vs 粗粒度网格” | Patch 14 每幅图像产生更多 token，OCR 保真度更好但更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy 等人 — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT 论文。
- [He 等人 — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab 等人 — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无标签。
- [Darcet 等人 — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与伪影分析。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认 vision tower。
- [Zhai 等人 — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验缩放定律。
