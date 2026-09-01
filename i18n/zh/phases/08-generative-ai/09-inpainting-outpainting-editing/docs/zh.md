# 图像修补、扩展与编辑

> 文生图创造新内容。修补负责修复旧内容。在生产环境中，70% 的可计费图像工作都是编辑——替换背景、去除标志、扩展画布、重绘手指。修补正是扩散模型发挥价值的地方。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 8 · 07（潜空间扩散），Phase 8 · 08（ControlNet & LoRA）
**预计时间：** 约 75 分钟

## 问题所在

客户发来一张完美的产品照片，但背景里有个干扰视线的标牌。你想擦除这个标牌，同时保持画面其余部分像素级一致。你不能从头运行文生图——结果会有不同的颜色、光影和产品角度。你想要重新生成的是*仅掩码区域*，且生成过程需尊重周围上下文。

这就是修补（Inpainting）。其变体包括：

- **修补（Inpainting）**。在掩码区域内重新生成，保留掩码外像素。
- **扩展（Outpainting）**。在掩码区域外（或画布之外）重新生成，保留掩码内部分。
- **图像编辑（Image editing）**。重新生成整张图像，但保持与原始图像语义或结构上的一致（SDEdit、InstructPix2Pix）。

2026 年的每一个扩散管线都内置了修补模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们的工作原理相同。

## 核心概念

![修补：具备上下文感知与保留机制的掩码感知去噪](../assets/inpainting.svg)

### 朴素方案（以及为何它是错的）

对标准文生图施加掩码。在每个采样步中，用正向扩散后的干净图像替换噪声潜变量中非掩码区域的值。这种方法……效果很差。边界伪影会渗入画面，因为模型对掩码区域内部的内容毫无信息。

### 标准的修补模型

训练一个修改过的 U-Net，使其输入通道数变为 9 而非 4：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码源图像的副本与单通道掩码。训练时，随机掩码图像区域，并训练模型仅对掩码区域进行去噪，同时为非掩码区域提供干净的 conditioning 信号。推理时，模型能够“看到”掩码区域周围的上下文，从而生成连贯的补全结果。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 均采用这种 9 通道（或等效）输入。对应 Diffusers 库中的 `StableDiffusionInpaintPipeline` 与 `FluxFillPipeline`。

### SDEdit（Meng 等，2022）—— 免训练编辑

向源图像加噪至某一中间步 `t`，随后使用新提示词从 `t` 反向运行至 0。无需重新训练。起始 `t` 的选择在保真度与创作自由度之间权衡：

- `t/T = 0.3` → 与源图几乎一致，仅做小幅风格调整
- `t/T = 0.6` → 中度编辑，保留粗略结构
- `t/T = 0.9` → 接近纯噪声起始，基本不保留源图内容

### InstructPix2Pix（Brooks 等，2023）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，同时以输入图像和文本指令（如“改成日落”、“加一条龙”）作为条件。需设置两组 CFG 比例：图像比例与文本比例。

### RePaint（Lugmayr 等，2022）

保留标准无条件扩散模型。在每个反向步中重采样——偶尔跳回更嘈杂的状态并重新生成。可避免边界伪影。适用于没有训练好修补模型的场景。

```figure
inpaint-mask-reinject
```

## 动手构建

`code/main.py` 在 5 维数据上实现了一个玩具级的一维修补方案。我们在 5-D 混合数据上训练 DDPM，每个样本由来自两个聚类之一的 5 个浮点数组成。推理时，我们“掩码”5 个维度中的 2 个，在每一步注入未掩码的三个维度的正向加噪版本，仅重新生成掩码维度。

### 步骤 1：5-D DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤 2：在所有 5 个维度上训练去噪器

标准 DDPM。网络对 5-D 噪声输入输出 5-D 噪声预测。

### 步骤 3：推理时的掩码感知反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 用干净源图像最新加噪的版本替换非掩码维度
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后对 x_t 执行常规反向步
```

这就是朴素方案，它在玩具级 1-D 数据上可行。真实图像修补采用 9 通道输入，因为纹理连贯性更为重要。

### 步骤 4：扩展（Outpainting）

扩展是掩码反转的修补：掩码新创建的区域（原本不存在的画布），其余部分保留原图。训练目标完全相同。

## 常见陷阱

- **接缝（Seams）。** 朴素方案会留下可见边界，因为梯度信息无法跨掩码流动。解决方法：将掩码膨胀 8-16 像素，或使用标准修补模型。
- **掩码泄漏。** 若条件图像的非掩码区域质量低或含噪，会污染掩码内的生成结果。适当去噪或轻微模糊即可。
- **CFG 与掩码尺寸相互作用。** 小掩码配高 CFG = 补丁饱和过曝。小幅编辑请降低 CFG。
- **SDEdit 保真度断崖。** 从 `t/T = 0.5` 提高到 `t/T = 0.6` 可能导致主体身份丢失。建议网格搜索并保存检查点。
- **提示词不匹配。** 提示词应描述*整张*图像，而非仅新内容。应为“一只猫坐在椅子上”，而非“一只猫”。

## 使用指南

| 任务 | 管线方案 |
|------|----------|
| 移除物体，小掩码 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + “blue sky at sunset” |
| 扩展画布 | SDXL outpaint 模式（8px 羽化）或带 outpaint mask 的 Flux-Fill |
| 重绘手部/面部 | SD-Inpaint 配合重新描述主体的提示词 + ControlNet-Openpose |
| 更改单区域风格 | 对掩码区域使用 `t/T=0.5` 的 SDEdit |
| “变成日落” | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 掩码 → SD-Inpaint |
| 超高分辨率保真 | Flux-Fill 或 GPT-Image（托管服务）处理最复杂案例 |

SAM（Meta 的 Segment Anything，2023）+ 扩散修补构成了 2026 年主流的背景移除管线。SAM 2（2024）支持视频处理。

## 交付规范

保存至 `outputs/skill-editing-pipeline.md`。该技能接收原图 + 编辑描述 + 可选掩码（或 SAM 提示词），输出：掩码生成方案、基础模型、CFG 比例（图像 + 文本）、SDEdit-t 或修补模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将掩码维度比例从 0.2 调整至 0.8。在何种比例下，修补质量（掩码维度的残差）与无条件生成相当？
2. **中等。** 实现 RePaint：每第 10 个反向步跳回 5 步（加噪）并重新去噪。测量其是否能减少掩码边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 对比：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在 20 个面部重绘任务上的表现。分别评估姿态一致性评分与身份保留评分。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Inpainting | “填洞” | 在掩码内重新生成，保留掩码外像素。 |
| Outpainting | “扩展画布” | 在画布外重新生成，保留掩码内部分。 |
| 9-channel U-Net | “标准修补模型” | 以 `noisy \| encoded-source \| mask` 为输入的 U-Net。 |
| SDEdit | “带噪声等级的图生图” | 加噪至时间 `t`，用新提示词去噪。 |
| InstructPix2Pix | “纯文本指令编辑” | 在 (图像, 指令, 输出) 三元组上微调的扩散模型。 |
| RePaint | “无需重新训练” | 反向过程中定期重加噪以降低接缝。 |
| SAM | “Segment Anything” | 通过点击或框选生成掩码的工具；常与修补配合使用。 |
| Flux-Kontext | “带上下文的编辑” | Flux 变体，接受参考图像 + 指令以进行编辑。 |

## 生产环境注意：编辑管线对延迟敏感

编辑图像的用户期望端到端响应在 5 秒以内。在 L4 上，1024² 分辨率下运行 30 步 SDXL-Inpaint 需 3-4 秒，加上 SAM 掩码生成（约 200 ms）和 VAE 编解码（合计约 500 ms）。在生产环境中，这属于 TTFT（首 Token 延迟）瓶颈而非吞吐量瓶颈——批大小设为 1，低并发，并最小化每个阶段：

- **SAM-H 是耗时瓶颈。** 1024² 下 SAM-H 约 200 ms；SAM-ViT-B 约 40 ms，画质损失极小。SAM 2（视频版）会增加时间维度开销；单图编辑请勿使用它。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 将图像编码为潜变量。若你已有上一轮生成的潜变量（迭代编辑 UI 中的典型场景），直接通过 `latents=...` 传入以跳过一次 VAE 编码。
- **掩码膨胀同样影响吞吐量。** 小掩码意味着 U-Net 前向计算的大部分被浪费（非掩码像素无论如何都会被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 始终运行完整 U-Net；只有 9 通道标准修补变体能利用掩码计算优化。
- **Flux-Kontext 是 2025 年的答案。** 对 `(source_image, instruction)` 单次前向通过——无需独立掩码，无需 SDEdit 噪声扫描。在 H100 上仅需约 1.5 秒即可输出一张编辑图。架构启示：合并管线阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 免训练的图像修补。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit 方法。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令驱动的图像编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，掩码生成源头。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频版 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力层级编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年工具集。
