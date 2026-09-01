# Any-Resolution Vision：Patch-n'-Pack 与 NaFlex

> 真实图像并非 224x224 的正方形。一张收据是 9:16，一张图表是 16:9，一张医学扫描可能是 4096x4096，一张手机截图是 9:19.5。2024 年之前的 VLM 答案——将所有内容缩放到固定正方形——丢弃了 OCR、文档理解和高分辨率场景解析得以工作的关键信号。NaViT（Google，2023）展示了你可以将可变分辨率的 patch 打包进单个 transformer batch，配合块对角掩码。Qwen2-VL 的 M-RoPE（2024）完全丢弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像切分为 base + 子图。SigLIP 2 的 NaFlex 变体（2025）现已成为开放 VLM 的默认编码器，一个 checkpoint 即可服务所有宽高比。本课从零到一实现 patch-n'-pack。

**类型：** Build
**语言：** Python（stdlib，patch packer + 块对角掩码）
**前置知识：** Phase 12 · 01（ViT patches），Phase 12 · 05（LLaVA）
**时间：** 约 120 分钟

## 学习目标

- 将一批可变分辨率图像的 patch 打包成单一序列，并构建块对角注意力掩码。
- 针对给定任务在 AnyRes 平铺（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 在不缩放的前提下，为 OCR、图表和摄影计算 token 预算。
- 说出三种方形缩放失败模式：文字被压扁、内容被裁切、padding 浪费 token。

## 问题所在

Transformer 期望的是序列。一个 batch 是相同长度序列的堆叠。如果你的图像是 224x224，每次都得到 196 个 patch token，无需 padding，完事。在 224 上训练，在 224 上推理，再也不用关心分辨率。

但世界并不配合。文档是竖版的（8.5x11 英寸，约 2:3）。图表截图是横版的（16:9）。收据是高瘦的（1:3）。医学成像以 2048x2048 或更大尺寸交付。手机设备截图是 1170x2532（0.46:1）。

2024 年以前的三种方案及各自缺陷：

1. 缩放到固定正方形（224x224 或 336x336）。形变会扭曲文字和人脸。下采样会破坏图表标签和 OCR 内容。这是 LLaVA-1.5 之前的标准做法。
2. 裁剪到固定宽高比。你会丢弃大部分图像内容，而裁剪位置的选择本身就是一个视觉问题。
3. 填充到最长边。解决了形变问题，但对于竖版图像，超过 50% 的 token 被浪费在 padding 上。这些 padding token 带来二次方注意力开销。

2024-2025 年的答案：让 transformer 以图像原生分辨率摄入 patch，并弄清楚如何在不浪费算力的情况下将异构 batch 打包进一个序列。

## 概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani 等，2023）是证明该方法可在大规模上生效的论文。思路很机械：

1. 对 batch 中每张图像，在选定的 patch 尺寸（如 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展平为其各自的变长序列。
3. 将所有图像的 patch 拼接成 batch 的一条长序列。
4. 构建块对角注意力掩码，使图像 A 的 patch 只在图像 A 内部互相注意。
5. 携带逐 patch 位置信息（2D RoPE 或分数位置嵌入）。

一批三张图像，尺寸分别为 336x336（576 个 token）、224x224（256 个 token）和 448x336（768 个 token），被打包成一条 1600 个 token 的序列，配合 1600x1600 的块对角掩码。无 padding，无浪费算力。Transformer 可处理任意宽高比。

NaViT 还在训练中引入了分数 patch 丢弃——在整个 batch 中随机丢弃 50% 的 patch——既起到正则化效果，又加速训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是更务实的替代方案。给定一张高分辨率图像和一个固定编码器（336 下的 CLIP 或 SigLIP），对图像进行平铺：

1. 从预定义网格布局中选择一种——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——最适配图像宽高比。
2. 将整个图像平铺到该网格；每个 tile 成为一个 336x336 的裁剪块。
3. 同时生成缩略图：整张图像缩放到 336x336 作为全局上下文 token。
4. 将每个 tile 送入冻结的 336 编码器。拼接 tile token 与缩略图 token。

对于一张 672x672 的图像，采用 2x2 网格加缩略图：4 × 576 + 576 = 2880 个视觉 token。开销大但有效——LLM 同时看到局部细节和全局上下文。

AnyRes 是你编码器冻结且仅支持单一分辨率时的首选方案。对于大图像它会爆炸式增加 token 数量（1344x1344 的图像采用 4x4 网格是 9216 + 576 ≈ 9800 个 token，几乎占满 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置嵌入（Multimodal Rotary Position Embedding）。不同于 NaViT 的分数位置或 AnyRes 的 tile + 缩略图方案，每个 patch 携带 3D 位置（时间、高度、宽度）。query/key 旋转处理任意的 H、W 和时序长度。

M-RoPE 无需重新训练即可支持原生动态分辨率。在推理时你喂入任意 HxW 图像，patch 嵌入器产出 H/14 × W/14 个 token，每个 token 获得其 (t=0, r=row, c=col) 位置，RoPE 用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 沿用了这一方案。InternVL3 的 V2PE 是同一思路，按模态变编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 O(H × W / P²) 个 token——没有 tile 倍增效应。与 NaViT 不同，它仍期望每个前向传播处理单张图像。跨分辨率批处理仍需在其上层叠加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的原生灵活模式。一个模型在推理时服务多个序列长度（256、729、1024 个 token）。训练时内部使用 NaViT 风格的 patch-n'-pack，每 patch 携带绝对分数位置。卖点在于：一个 checkpoint，根据任务在推理时选择 token 预算。

语义任务（分类、检索）用 256 个 token。OCR 或图表理解用 1024 个 token。无需重新训练。

### 打包掩码

块对角掩码是大多数实现的卡点所在。对于一个长度为 `N_total` 的打包序列，覆盖图像 `i=0..B-1`，长度为 `n_i`，掩码 `M` 形状为 `(N_total, N_total)`，当且仅当两个索引落入同一图像的 block 时值为 1，否则为 0。可通过累积长度列表构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 当且仅当存在某个 b 使得 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中这只需一行，用 `torch.block_diag` 或显式 gather。FlashAttention 的变长路径（`cu_seqlens`）直接跳过掩码，利用累积长度张量在序列内部完成注意力——对于典型 batch，比密集掩码快约 10 倍。

### Token 预算

根据任务选择策略：

- OCR / 文档：1024-4096 个 token。SigLIP 2 NaFlex 取 1024，或 AnyRes 3x3 + 缩略图。
- 图表与 UI：在 384-448 原生分辨率下取 729-1024 个 token。Qwen2.5-VL 动态分辨率配合 max_pixels 上限。
- 自然照片：256-576 个 token 即可。下游 LLM 能看到足够的信息。token 花在内容密度高的地方。
- 视频：空间池化后每帧 64-128 个 token，2-8 FPS。第 12.17 课涵盖此内容。

2026 年的生产准则：按任务设置 per-token 最大像素上限，以原生宽高比编码至该上限，打包 batch，跳过 padding。Qwen2.5-VL 暴露的 `min_pixels` 和 `max_pixels` 正是为此设计的。

```figure
mm-patch-n-pack
```

## 使用方式

`code/main.py` 实现了针对异构图像 batch 的 patch-n'-pack，使用整数像素坐标。它：

- 接收一组 (H, W) 图像尺寸列表。
- 在 patch 尺寸为 14 时计算每张图像的 patch 序列长度。
- 将它们打包为总长度为 `sum(n_i)` 的单一序列。
- 构建块对角注意力掩码（为清晰起见使用密集版本）。
- 比较打包成本与方形缩放、AnyRes 平铺的差异。
- 为混合 batch（收据、图表、截图、照片）打印 token 预算表。

运行它。输出的数字正是 2026 年所有开放 VLM 采用 patch-n'-pack 的原因。

## 交付物

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比工作负载（OCR、图表、照片、视频帧）和一个总 token 预算，它能选择正确的策略（NaFlex、AnyRes、M-RoPE 或固定方形），并发出逐请求配置。在为产品规划 VLM 时使用该 skill，可避免导致延迟预算崩盘的隐性 10 倍 token 膨胀。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch 尺寸为 14 时，原生分辨率下有多少 token？缩放到 336 的方形后有多少？哪种在实际中损失更多 OCR 精度？

2. 为长度为 256、576、729、1024 的四张图像 batch 构建块对角掩码。验证注意力矩阵为 2585x2585，且非零条目数恰好为 `256² + 576² + 729² + 1024²`。

3. 对于 1792x896 的图像（patch 14），比较：（a）缩放到 336 方形后编码，（b）AnyRes 2x1 + 缩略图，（c）M-RoPE 原生编码。哪种使用最少 token？哪种保留最多细节？

4. 实现分数 patch 丢弃：给定一个打包序列，随机均匀丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文第 3.2 节（arXiv:2409.12191）。用两句话描述 `min_pixels` 和 `max_pixels` 控制了什么，以及为何两个边界都重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch-n'-pack | "NaViT 风格打包" | 将来自不同图像的变长 patch 序列拼接到一个 batch 维度 |
| 块对角掩码 | "打包掩码" | 将每张图像的 patch 约束为只相互注意，不注意打包序列中的邻居 |
| AnyRes | "LLaVA-NeXT 平铺" | 将高分辨率图像划分为固定尺寸 tile 的网格，外加全局缩略图；用固定编码器编码每个 tile |
| NaFlex | "SigLIP 2 原生灵活模式" | 单个 SigLIP 2 checkpoint，无需重新训练，推理时服务 256/729/1024 token 预算 |
| M-RoPE | "多模态 RoPE" | 3D 旋转位置编码（时间、行、列），处理任意 H、W、T，无需位置表 |
| cu_seqlens | "FlashAttention 打包" | FlashAttention 变长路径使用的累积长度张量，替代密集块对角掩码 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 按请求控制的旋钮，限制极小或极大输入的 token 数量 |
| 视觉 token 预算 | "每图像多少个 token" | 每张图像产出的 patch token 近似数量；设定 LLM 的 prompt 预算和注意力开销 |

## 延伸阅读

- [Dehghani 等 — Patch n' Pack: NaViT（arXiv:2307.06304）](https://arxiv.org/abs/2307.06304)
- [Wang 等 — Qwen2-VL（arXiv:2409.12191）](https://arxiv.org/abs/2409.12191)
- [Laurençon 等 — What matters when building vision-language models?（Idefics2，arXiv:2405.02246）](https://arxiv.org/abs/2405.02246)
- [Tschannen 等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report（arXiv:2502.13923）](https://arxiv.org/abs/2502.13923)
