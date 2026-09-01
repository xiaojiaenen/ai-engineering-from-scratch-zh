# 视频-语言模型：时间 Token 与接地

> 视频不是照片的堆叠。一个 5 秒的片段具有因果顺序、动作动词和事件时序，这些是图像模型无法表示的。Video-LLaMA（Zhang 等，2023年6月）发布了第一个带音视频接地的开源视频-LLM。VideoChat 和 Video-LLaVA 扩展了这一模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与顶尖闭源模型的差距。每个系统都以不同方式解决了时间 token 问题——Q-former 按片段、concat-pool 按帧、TMRoPE 按 token。本课阅读这些模式，构建均匀-动态帧采样器，并在时间接地任务上进行评估。

**类型：** 构建
**语言：** Python（stdlib、帧采样器 + 时间接地评估器）
**前置知识：** Phase 12 · 08（LLaVA-OneVision）
**时间：** 约 180 分钟

## 学习目标

- 解释为何时间位置编码会独立于视觉编码器影响视频 VLM 性能。
- 比较均匀采样、动态 FPS 和事件驱动帧采样在 token/秒与接地准确率上的表现。
- 描述 Q-former-per-clip（Video-LLaMA）与 pooled-per-frame（Video-LLaVA）与 M-RoPE-per-token（Qwen2.5-VL）三种设计。
- 说出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题所在

一个 30 FPS 的 1 分钟视频包含 1800 帧。按每帧 196 个视觉 token（ViT-B at 224）计算，共计 35.2 万 token——超过任何 2024 年时代 LLM 的上下文窗口。

有三种降维策略：

1. 对帧进行子采样（根据内容 1-8 FPS）。
2. 对每帧的 patch token 进行激进池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩：输入 16 帧片段，输出 64 个 token。

每种权衡各不相同。子采样损失时间细节。池化损失空间细节。Q-former 两者都损失一点但节省 token。

时间位置编码是另一个维度：模型如何知道第 5 帧在第 6 帧之前？选项包括简单的一维时间 RoPE（Video-LLaMA）、可学习的时间嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整三维）。

## 概念

### Video-LLaMA：每个片段的 Q-former + 音频分支

Video-LLaMA（2023）是第一个开源视频-LLM。架构：

- 16 帧片段，2 FPS（即 8 秒）。
- 每帧 ViT 特征 → 跨所有 16 帧交叉注意力的 Video Q-former → 32 个学习查询 → LLM。
- 并行音频分支：波形 → ImageBind 音频编码器 → Audio Q-former → 32 个查询 → LLM。

优势：音视频联合推理。劣势：固定片段长度，无法支持任意时间接地。

### VideoChat 与 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的想法但去掉了音频并简化了结构。Video-LLaVA（Lin 等，2023）在图像和视频帧上训练单一视觉编码器（"先对齐后投影"），给出统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

它们都不处理长视频。都是 8-16 帧系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——时间-模态旋转位置嵌入。每个 patch token 携带 (t, h, w) 位置，其中 t 是实际时间戳（而非帧索引）。

与简单时间嵌入的关键区别：

- 绝对时间，而非索引。模型看到的是"在 4.2 秒时"而非"在第 15 帧"。
- 按 token 旋转，而非按片段。每个视觉 token 按其时间戳独立旋转。
- 兼容动态 FPS。如果你在 2 FPS 处采样、在 4 FPS 处采样，TMRoPE 原生处理不均匀间隔。

TMRoPE 支持"猫在几秒时跳起来？"查询。模型可以输出"在 4.2 秒"。Video-LLaMA 只能说"在片段早期"。

### 帧采样策略

均匀采样：在时长内均匀抽取 N 帧。简单，但会丢失运动峰值。

动态 FPS：根据运动强度自适应采样。光流或帧差法选择高运动段进行密集采样。Qwen2.5-VL 在此上训练。

事件驱动：运行轻量级检测器，在行动发生处增加采样密度。VideoAgent 使用此方法。

关键帧 + 上下文：在镜头边界采样 + 少量相邻帧。用于电影内容。

### 每帧池化

在 1 FPS 且每帧 576 个 token 的情况下，5 分钟片段共 172,800 个 token。Qwen2.5-VL-72B 的 128k 上下文可以处理但代价昂贵。

3x3 双线性池化降至每帧 64 个 token → 5 分钟共 19,200 个 token。大多数任务的甜点。

更激进地池化（6x6 → 每帧 16 个 token）用于对空间细节要求较低的智能体工作流。

### 四个视频基准

- VideoMME：综合视频理解，短 + 中 + 长。
- TempCompass：细粒度时间推理，"之前/之后"类问题。
- EgoSchema：长视界第一人称视频。
- Video-MMMU：多模态多学科视频问题。

完整的视频-VLM 评估需覆盖全部四个基准。它们在不同维度上施加压力——TempCompass 专注排序，EgoSchema 专注 3+ 分钟推理，VideoMME 跨越多个时长。

### 接地输出格式

时间接地的输出格式：

- 自由文本："猫大约在 4 秒处跳起来。"易于解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练此格式。
- 基于 token：特殊 `<time>4.1</time>` token 与答案交错。Qwen2.5-VL 的内部格式。

基于 token 的方式对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 最佳实践

2026 年视频 VLM 的最佳实践：

- 编码器：SigLIP 2 + M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（1-4，取决于运动）+ 最大帧数上限。
- 每帧池化：3x3 双线性。
- 输出：带时间与事件字段的结构化 JSON。
- 基准：VideoMME + TempCompass 用于通用；EgoSchema 用于长视界。

```figure
video-temporal-patches
```

## 使用

`code/main.py` 包含：

- 均匀与动态 FPS 帧采样器。
- 玩具时间接地评估器：给定时间 T 处的"真实"事件与模型输出，以容差评分准确度。
- Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）之间的对比。

## 交付

本课生成 `outputs/skill-video-vlm-frame-planner.md`。给定视频任务（监控、动作识别、时间接地、摘要），它选择帧采样器、池化因子、输出格式和预期准确率层级。

## 练习

1. 对于 3 分钟烹饪演示，选择均匀采样 vs 动态 FPS。用 token 数量论证。

2. TMRoPE 相比简单时间嵌入表额外增加了什么能力？

3. 编写一个 VLM 可学习输出的时间接地 JSON Schema。包含错误情况。

4. 阅读 Video-LLaVA 第 3 节"Alignment Before Projection"。为何这比分别训练图像和视频编码器更好？

5. 根据 VideoMME 排行榜，截至 2026 年顶级开源模型与顶级闭源模型之间的差距是多少？该差距中有多少可归因于时间编码，多少归因于基础 LLM 规模？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 时间接地 | "时间局部化答案" | VLM 输出事件发生的具体时间范围 |
| TMRoPE | "Time-Multimodal RoPE" | 带绝对时间戳的三维旋转位置编码，由 Qwen2.5-VL 使用 |
| 动态 FPS | "运动感知采样" | 在高运动段采样更多帧，在静止段采样更少 |
| 帧池化 | "每帧空间压缩" | 在送入 LLM 前用双线性插值减少每帧的 patch 数 |
| 视频 Q-former | "片段压缩器" | 交叉注意力瓶颈，将 N 帧映射到 K 个学习查询 |
| VideoMME | "视频基准" | 综合短/中/长视频基准，2500+ 样本 |

## 延伸阅读

- [Zhang 等 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
