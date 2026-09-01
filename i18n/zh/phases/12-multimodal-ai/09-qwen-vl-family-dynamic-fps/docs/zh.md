# Qwen-VL 系列与动态 FPS 视频

> Qwen-VL 系列——Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025）——是 2026 年最具影响力的开源视觉-语言模型 lineage。每一代都做出一个决定性的架构押注，随后整个开源生态在十二个月内跟进：通过 M-RoPE 实现原生动态分辨率、带绝对时间对齐的动态 FPS 采样、ViT 中的窗口注意力、以及结构化代理输出格式。到 Qwen3-VL 时，配方已经稳定：一个带有原生宽高比输入能力的 2D-RoPE-ViT 编码器、通过 MLP 投射器接入大型 Qwen3 语言基座，以及将 OCR、定位和代理行为作为一等公民目标的训练阶段。本节按时间顺序阅读该系列，帮助你理解为何每个超参数都设在当前位置。

**类型：** 学习
**语言：** Python（stdlib、M-RoPE 编码器 + 动态 FPS 采样器）
**前置知识：** Phase 12 · 06（patch-n'-pack）
**预计时间：** 约 120 分钟

## 学习目标

- 计算 M-RoPE 的三个轴旋转（时间、高度、宽度）并解释为何三者缺一不可。
- 为视频选择合适的动态 FPS 采样策略，并推理 tokens/second 与事件检测精度之间的关系。
- 按顺序说出 Qwen-VL 的四代升级及其各自带来的能力。
- 接线 Qwen2.5-VL 风格的 JSON 代理输出格式，并从 VLM 响应中解析结构化工具调用。

## 问题背景

Qwen-VL 于 2023 年 8 月发布，直接回应了 LLaVA-1.5 和 BLIP-2。Qwen 团队瞄准的差距有三方面：分辨率、视频和结构化输出。

**分辨率：** LLaVA-1.5 运行在 336x336。对照片来说够用，但对中文发票或密集的电子表格截图来说毫无用处。Qwen-VL 的第一个创新是 448x448 分辨率和接地框（bounding-box）输出，让模型能够"指认"物体。

**视频：** Video-LLaMA 堆叠逐帧编码器并将它们喂给 LLM。这对短片段有效，但对多分钟视频无效，因为时间轴才是信号。Qwen 团队希望拥有一个理解时间的单一编码器。

**结构化输出：** LLaVA 输出自由文本。代理需要 JSON。Qwen-VL 在显式 JSON 输出格式上进行训练，包括将边界框坐标作为文本。

每一代 Qwen-VL 都扩展这三个轴之一。

## 概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为编码器（25 亿参数），LLama 兼容的 Q-Former（1 步，256 个查询），Qwen-7B 作为基座。贡献包括：

- 448x448 分辨率（当时开源 VLM 的 SOTA）。
- 定位能力：在带有显式坐标 token 输出的图文对上训练。"The cat is at <box>(112, 204), (280, 344)</box>"。
- 从一开始就支持中文 + 英文多语言训练。

当时的基准测试：在英文上与 GPT-4V 竞争，在中文上占主导地位。定位监督才是真正的亮点。

### Qwen2-VL（2024 年 9 月）—— M-RoPE 与原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器替换了固定分辨率 + Q-Former 的堆栈。关键变更包括：

- **原生动态分辨率。** ViT 接受任何能被 28 整除的 HxW（patch 14 配合 2x 空间合并）。一张 1120x672 的图像（40x24 合并 patch）产生 960 个视觉 token。无需调整大小，无需分块，无需缩略图。
- **M-RoPE（多模态 RoPE）。** 每个 token 携带 3D 位置 (t, h, w)，而非 1D。对于图像 t=0，对于视频 t = frame_index。RoPE 按轴频率旋转 query/key 向量。无需位置嵌入表。
- **MLP 投射器。** 丢弃 Q-Former；使用在合并 patch token 上的 2 层 MLP。
- **带动态 FPS 的视频。** 视频默认以 1-2 FPS 采样，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多项多模态基准上与 GPT-4o 持平，并在 DocVQA 上超越它（94.5 vs 88.4）。架构变更是决定性的一步。

### Qwen2.5-VL（2025 年 2 月）—— 动态 FPS + 绝对时间

Qwen2.5-VL 的重大转变在于视频。动态 FPS 不仅仅是"需要在时多采样帧"。论文形式化了以下内容：

- **绝对时间 token。** 不使用位置索引（第 0、1、2 帧...），而是使用实际时间戳。"在 0:04，猫跳了起来。"模型会看到与帧 token 交错的 `<time>0.04</time>` token。
- **动态 FPS。** 慢速画面以 1 FPS 采样，动作画面以 4+ FPS 采样。由用户或训练者选择；M-RoPE 自适应。
- **ViT 中的窗口注意力。** 空间注意力在块内局部窗口化以提升吞吐量；每隔几层添加全局注意力。
- **显式 JSON 输出格式。** 在工具调用数据上训练：`"{\"tool\": \"click\", \"coords\": [380, 220]}"`。开箱即用的代理就绪。
- **MRoPE-v2 缩放。** 位置随最大输入尺寸缩放，因此 10 分钟视频不会超出频率范围。

基准测试：Qwen2.5-VL-72B 在大多数视频基准上超越 GPT-4o，在文档上与 Gemini 2.0 持平，并开创了 GUI 定位的开源模型 SOTA（ScreenSpot：84% 准确率 vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是一次增量升级，重在整合而非重新发明：更大的 LLM 主干（Qwen3-72B）、扩充的训练数据、改进的 OCR、通过 Qwen3 "思考模式"增强的推理能力。ViT 和 M-RoPE 保持不变。论文侧重于数据和训练改进而非架构。

该 lineage 的要点：到 2025 年，Qwen-VL 架构已经稳定。后续代数扩展的是算力和数据，而非基本组件。

### M-RoPE 的数学表达

经典 RoPE 使用成对坐标通过位置 `m` 旋转维度为 `d` 的 query `q`：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 将隐藏维度拆分为三个频带。设 `d = 96`，分配 32 维给时间、32 维给高度、32 维给宽度。每个频带由其自身的轴位置旋转。位于 (t=5, h=10, w=20) 的 patch 会对其三个频带应用旋转 `R_t(5)`、`R_h(10)`、`R_w(20)`。

文本 token 使用 `t = text_index, h = 0, w = 0`（或归一化选择），保持兼容性。视频帧使用 `t = frame_time, h = row, w = col`。单张图像使用 `t = 0`。

优势：一套位置编码同时处理文本、图像和视频，无需分支代码或不同的位置表。

### 动态 FPS 采样逻辑

给定一个时长为 `T` 秒的视频和目标 token 预算 `B`：

1. 计算可承受的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择一个满足 `fps <= fps_max` 的目标 FPS。
3. 如果运动剧烈（基于光流启发式或显式用户请求），选择较高 FPS。如果运动平缓，选择较低 FPS。
4. 以选定 FPS 均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式训练此逻辑；在推理时用户通过 `fps` 参数控制。一段 60 秒的动作序列以 4 FPS 采样，每帧 81 个 token = 19440 个 token，在 32k 上下文中可管理。

### 结构化代理输出

Qwen2.5-VL 的代理训练明确针对结构化工具调用：

```json
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出执行 `JSON.parse`。与需要正则表达式和歧义处理的自由格式 "click at (1024, 512)" 相比，这就是为什么 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跃升到 84%。

```figure
mm-mrope-axes
```

## 使用方式

`code/main.py` 实现了：

- 混合文本、图像 patch 和视频帧的打包序列的 M-RoPE 位置计算。
- 动态 FPS 采样器：给定 (duration, budget, motion_level)，选择 FPS 并生成帧时间戳。
- 玩具级 Qwen2.5-VL JSON 输出解析器，处理包含坐标字段的工具调用响应。

运行它，然后在 5 分钟视频上将固定 FPS 切换为动态 FPS，感受其中的差异。

## 交付物

本课生成 `outputs/skill-qwen-vl-pipeline-designer.md`。给定一个视频任务（监控、代理、动作识别、无障碍），它会输出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口注意力标志、代理输出模式）和延迟估算。每当为视频产品部署 Qwen-VL 系列模型时都应使用此文件。

## 练习

1. 计算位于 (t=3, h=5, w=7) 的 patch 在隐藏维度 48（每频带 16，基础 theta 10000）下的 M-RoPE 旋转。展示每个频带中前三对对应的旋转角度。

2. 一段 10 分钟的安防摄像头录像以 1 FPS 采样会产生多少帧？在 384 分辨率下使用 3x 池化，总共多少 token？Qwen2.5-VL 默认的 32k 上下文能处理吗？

3. 分别为 30 秒的网球对打、30 秒的食谱演示、30 秒的 UI 代理录制选择 FPS。用动态 FPS 逻辑为每个选择提供理由。

4. Qwen2.5-VL 完全移除了 Q-Former。为什么简单的 MLP 在 2025 年可行而在 2023 年不行？（提示：数据规模和编码器质量。）

5. 将三个 Qwen2.5-VL JSON 工具调用输出解析为 Python 字典。 malformed JSON 会失败在哪里？Qwen cookbook 推荐什么恢复策略？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| M-RoPE | "多模态 RoPE" | 具有时间、高度和宽度频带的 3D 旋转位置嵌入 |
| Dynamic FPS | "智能采样" | 根据运动、时长和 token 预算为每段视频选择的帧采样率 |
| Absolute time token | "时间戳 token" | 交错在序列中的 `<time>t</time>`，使模型看到的是实际秒数而非帧索引 |
| Window attention | "局部注意力" | 空间自注意力限制在小窗口内以提升速度；周期性添加全局注意力 |
| Structured agent output | "JSON 模式" | 训练数据监督教导 VLM 输出包含坐标和工具名的可解析 JSON |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 按请求控制 bounding 总像素数，从而控制 token 数 |
| Grounding | "指认" | 将边界框坐标作为文本 token 输出；自 Qwen-VL v1 起使用 |

## 延伸阅读

- [Bai 等 — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang 等 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen 团队 — Qwen2.5-VL 技术报告 (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen 团队 — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu 等 — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
