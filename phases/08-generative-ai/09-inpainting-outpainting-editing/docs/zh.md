# 图像修复、扩展与编辑

> 文本到图像生成新事物，图像修复则修复旧事物。在实际生产中，70%的可计费图像工作是编辑——更换背景、移除标识、扩展画布、重新生成手部。图像修复正是扩散模型证明其价值的领域。

**类型：** 构建
**语言：** Python
**先决条件：** Phase 8 · 07 (潜在扩散), Phase 8 · 08 (ControlNet & LoRA)
**时间：** ~75 分钟

## 问题所在

客户发送了一张完美的产品照片，但背景中有一个分散注意力的标志。你想擦除这个标志，并保持其他所有部分的像素完全一致。你不能从头开始运行文本到图像生成——结果会有不同的颜色、不同的光照、不同的产品角度。你只想重新生成*被遮罩的区域*，并且希望重新生成的内容能尊重周围环境的上下文。

这就是图像修复。其变体包括：

- **图像修复。** 在遮罩内部重新生成，保持外部像素不变。
- **图像扩展。** 在遮罩外部（或画布之外）重新生成，保持内部不变。
- **图像编辑。** 重新生成整张图像，但保持与原始图像在语义或结构上的保真度（SDEdit, InstructPix2Pix）。

2026年的每个扩散模型管道都提供图像修复模式。Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit。它们的工作原理相同。

## 核心概念

![图像修复：基于上下文保留的遮罩感知去噪](../assets/inpainting.svg)

### 朴素方法（及其错误之处）

运行标准的文本到图像生成，并带上一个遮罩。在每个采样步骤中，用正向扩散后的干净图像替换噪声潜变量中的未遮罩区域。它能工作……但效果很差。边界伪影会渗透出来，因为模型对遮罩区域内有什么内容没有信息。

### 正确的图像修复模型

训练一个修改过的U-Net，它接受9个输入通道而不是4个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是经过VAE编码的源图像的一个副本加上一个单通道遮罩。在训练时，你随机遮罩图像的区域，并训练模型只去噪被遮罩的区域，同时将未遮罩区域作为干净的条件信号提供。在推理时，模型可以“看到”遮罩区域周围的内容，并产生连贯的补全。

SD-Inpaint, SDXL-Inpaint, Flux-Fill 都使用这种9通道（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`, `FluxFillPipeline`。

### SDEdit (Meng 等人, 2022) — 免费编辑

向源图像添加噪声直到某个中间时间步 `t`，然后用新的提示词运行反向链从 `t` 降到 0。无需重新训练。起始时间步 `t` 的选择权衡了保真度与创作自由：

- `t/T = 0.3` → 几乎与源图像相同，有微小的风格变化
- `t/T = 0.6` → 中等程度编辑，保留粗略结构
- `t/T = 0.9` → 从近乎纯噪声生成，源图像保留极少

### InstructPix2Pix (Brooks 等人, 2023)

在 `(input_image, instruction, output_image)` 三元组上微调一个扩散模型。在推理时，以输入图像和一个文本指令（“变成日落”，“添加一条龙”）为条件。使用两个CFG缩放因子：图像缩放因子和文本缩放因子。

### RePaint (Lugmayr 等人, 2022)

保持一个标准的无条件扩散模型。在每个反向步骤中，进行重采样——偶尔跳回到一个更噪的状态并重新生成。避免边界伪影。在你没有训练好的图像修复模型时使用。

## 动手构建

`code/main.py` 实现了一个在5维数据上的玩具级1-D图像修复方案。我们在5维混合数据上训练一个DDPM，其中每个样本是来自两个聚类之一的5个浮点数。在推理时，我们“遮罩”5个维度中的2个，在每一步注入未遮罩三个维度的噪声正向版本，并只重新生成被遮罩的维度。

### 步骤1：5维DDPM数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤2：在所有5个维度上训练去噪器

标准DDPM。网络为5维噪声输入输出5维噪声预测。

### 步骤3：推理时的遮罩感知反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，它在玩具级1-D数据上有效。真实的图像修复使用9通道输入，因为纹理连贯性更重要。

### 步骤4：图像扩展

图像扩展是遮罩反转后的图像修复：遮罩新的（之前不存在的）画布，其余部分用原始图像填充。训练目标相同。

## 常见陷阱

- **接缝。** 朴素方法会留下可见的边界，因为梯度信息无法跨遮罩流动。解决方法：将遮罩膨胀8-16像素，或使用正确的图像修复模型。
- **遮罩泄漏。** 如果条件图像的未遮罩区域质量低或带有噪声，它会污染遮罩内的生成结果。稍微去噪或模糊处理。
- **CFG与遮罩大小相互作用。** 对小遮罩使用高CFG = 饱和色块。对小编辑降低CFG。
- **SDEdit保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能会丢失主体的身份。进行参数扫描和检查点保存。
- **提示词不匹配。** 描述的应该是*整*幅图像，而不仅仅是新内容。“一只猫坐在椅子上”而不是“一只猫”。

## 实际使用

| 任务 | 管道 |
|------|------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + “日落时的蓝天” |
| 扩展画布 | SDXL 扩展模式（8px羽化）或带扩展遮罩的 Flux-Fill |
| 重新生成手/脸 | SD-Inpaint + 重新描述主体的提示词 + ControlNet-Openpose |
| 改变一个区域的风格 | 对遮罩区域在 `t/T=0.5` 处应用 SDEdit |
| “变成日落” | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）处理最难案例 |

SAM (Meta的Segment Anything, 2023) + 扩散修复是2026年的背景移除管道。SAM 2 (2024) 可处理视频。

## 部署

保存 `outputs/skill-editing-pipeline.md`。该技能接收原始图像 + 编辑描述 + 可选遮罩（或SAM提示），并输出：遮罩生成方法、基础模型、CFG缩放因子（图像 + 文本）、SDEdit-t 或修复模式，以及质量检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将被遮罩的维度比例从0.2变到0.8。在什么比例下，修复质量（被遮罩维度的残差）等于无条件生成？
2. **中等。** 实现RePaint：每10个反向步骤，跳回5步（添加噪声）并重新去噪。测量它是否减少了遮罩边缘的边界残差。
3. **困难。** 使用Hugging Face diffusers比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在20个面部重新生成任务上的表现。分别评估姿态遵循度和身份保持度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 图像修复 | “填补空洞” | 在遮罩内部重新生成；保持外部像素不变。 |
| 图像扩展 | “扩展画布” | 在画布外部重新生成；保持内部不变。 |
| 9通道U-Net | “正确的修复模型” | 以 `noisy | encoded-source | mask` 为输入的U-Net。 |
| SDEdit | “带噪声级别的Img2img” | 噪声添加到时间步 `t`，用新提示词去噪。 |
| InstructPix2Pix | “纯文本编辑” | 在（图像，指令，输出）三元组上微调的扩散模型。 |
| RePaint | “无需重训练” | 反向过程中周期性添加噪声以减少接缝。 |
| SAM | “Segment Anything” | 通过点击或框选生成遮罩；与修复配对。 |
| Flux-Kontext | “带上下文编辑” | Flux变体，接受参考图像+指令进行编辑。 |

## 生产注意事项：编辑管道对延迟敏感

编辑图像的用户期望低于5秒的往返时间。1024²分辨率的30步SDXL-Inpaint在L4上需要3-4秒，加上SAM遮罩生成（~200毫秒）和VAE编码/解码（合计~500毫秒）。在生产环境中，这是首字节时间受限而非吞吐量受限——批量为1，低并发，需最小化每个阶段：

- **SAM-H较慢。** 1024²的SAM-H约200毫秒；SAM-ViT-B约40毫秒，质量损失较小。SAM 2（视频）增加了时间开销；不要用于单张图像编辑。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 编码为潜变量。如果你有上一代生成的潜变量（在迭代编辑UI中常见），通过 `latents=...` 直接传递以跳过一次VAE编码。
- **遮罩膨胀也影响吞吐量。** 小遮罩意味着大部分U-Net前向传播是浪费的（未遮罩像素无论如何都会被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论遮罩大小都运行完整的U-Net；只有9通道的正确修复变体才利用遮罩计算。
- **Flux-Kontext是2025年的答案。** 对 `(source_image, instruction)` 进行单次前向传播——无需单独遮罩，无需SDEdit噪声扫描。在H100上，它能在约1.5秒内完成一次编辑。架构上的启示：将各阶段合并。

## 扩展阅读

- [Lugmayr 等人 (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的图像修复。
- [Meng 等人 (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov 等人 (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩来源。
- [Ravi 等人 (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频SAM。
- [Hertz 等人 (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力层编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024工具链。