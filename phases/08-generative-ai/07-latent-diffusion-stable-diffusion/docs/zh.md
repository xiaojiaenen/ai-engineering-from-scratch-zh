# 潜在扩散与稳定扩散

> 在512×512图像的像素空间进行扩散是计算上的战争罪行。Rombach等人(2022)注意到，生成图像并不需要全部78.6万个维度——只需足够捕捉语义结构，其余部分交给一个单独的解码器。在VAE的潜在空间中运行扩散。这一个想法就是稳定扩散。

**类型:** 构建
**语言:** Python
**前置课程:** 第8阶段 · 02 (VAE)，第8阶段 · 06 (DDPM)，第7阶段 · 09 (ViT)
**时间:** ~75分钟

## 问题所在

在512²像素空间进行扩散意味着U-Net需要在形状为`[B, 3, 512, 512]`的张量上运行。对于一个5亿参数的U-Net，每个采样步骤大约需要100 GFLOPS。五十个步骤就是每张图像5 TFLOPS。在十亿图像上进行训练，计算成本是荒谬的。

这些FLOPS大部分用于推动感知上不重要的细节通过网络——那些有损VAE本可以压缩掉的高频纹理。Rombach的想法：训练一个VAE一次(第一阶段)，将其冻结，然后完全在4通道64×64的潜在空间中运行扩散(第二阶段)。相同的U-Net。像素减少到1/16。FLOPS减少约64倍，同时保持可比的质量。

这就是稳定扩散的配方。SD 1.x / 2.x在`64×64×4`潜在空间上使用860M参数的U-Net，SDXL在`128×128×4`上使用2.6B参数的U-Net，SD3将U-Net替换为具有流匹配的扩散Transformer(DiT)。Flux.1-dev(Black Forest Labs, 2024)搭载了12B参数的DiT-MMDiT。所有都运行在相同的双阶段基础架构上。

## 核心概念

![潜在扩散：VAE压缩 + 潜在空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1.  **第一阶段 — VAE。** 编码器`E(x) → z`，解码器`D(z) → x`。目标压缩：每个空间轴下采样8倍 + 调整通道，使得总潜在大小约为像素数的1/16。损失 = 重建(L1 + LPIPS感知) + KL(小权重，使`z`不会被强制太接近高斯分布，因为我们不需要从`z`进行精确采样)。通常使用对抗损失训练，使解码后的图像清晰。

2.  **第二阶段 — 在`z`上进行扩散。** 将`z = E(x_real)`视为数据。训练一个U-Net(或DiT)来对`z_t`去噪。在推理时：通过扩散采样`z_0`，然后`x = D(z_0)`。

**文本条件。** 另外两个组件。一个冻结的文本编码器(SD 1.x用CLIP-L，SD 2/XL用CLIP-L+OpenCLIP-G，SD3和Flux用T5-XXL)。一个交叉注意力注入：每个U-Net块接收`[Q = image features, K = V = text tokens]`并将其混合进来。token是文本影响图像的唯一途径。

**损失函数与第06课相同。** 相同的DDPM/流匹配的MSE噪声损失。只是更换了数据领域。

## 架构变体

| 模型 | 年份 | 骨干网络 | 潜在形状 | 文本编码器 | 参数量 |
|------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + 精炼器 | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏版 | 128×128×4 | 同上 | 1-4步采样 |
| SD3 | 2024 | MMDiT (多模态DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT蒸馏版 | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4步 |

趋势：用DiT(在潜在patch上的transformer)替代U-Net，扩大文本编码器(T5在提示遵循上优于CLIP)，增加潜在通道数(从4增加到16提供更多细节空间)。

## 构建它

`code/main.py`将一个玩具级的1-D"VAE"(恒等编码器和解码器，仅作演示；真正的VAE会是卷积网络)堆叠在第06课的DDPM之上，并添加了使用无分类器引导的类别条件。它展示了相同的扩散损失在原始1-D值或编码值上运行都有效——这是关键洞察。

### 步骤1：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真正的VAE具有训练好的权重。为了教学，这个线性映射足以说明扩散在`z`上操作，而无需关心原始数据空间。

### 步骤2：在`z`空间中进行扩散

与第06课相同的DDPM。网络看到的数据是`z = E(x)`。采样得到`z_0`后，用`D(z_0)`解码。

### 步骤3：无分类器引导

训练时，10%的时间丢弃类别标签(替换为空token)。推理时，同时计算`ε_cond`和`ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导(最大多样性)，`w = 3` = 默认值，`w = 7+` = 饱和/过度锐化。

### 步骤4：文本条件(概念，非代码)

用冻结的文本编码器输出替换类别标签。通过交叉注意力将文本嵌入输入到U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类条件扩散模型与稳定扩散之间唯一实质性的区别。

## 常见陷阱

- **VAE缩放不匹配。** SD 1.x的VAE在编码后应用一个缩放常数(`scaling_factor ≈ 0.18215`)。忘记这一点会导致U-Net在方差严重错误的潜在空间上训练。每个检查点都附带这个常数。
- **文本编码器静默错误。** SD3需要T5-XXL并支持至少128个token，回退到仅CLIP是有损的。始终检查`use_t5=True`，否则提示保真度会急剧下降。
- **混合潜在空间。** SDXL、SD3、Flux都使用不同的VAE。在SDXL潜在空间上训练的LoRA在SD3上无法工作。Hugging Face diffusers 0.30+会拒绝加载不匹配的检查点。
- **CFG值过高。** `w > 10`会产生饱和、油腻的图像，并以牺牲多样性为代价过度拟合提示。最佳范围是`w = 3-7`。
- **负面提示泄露。** 空的负面提示变成空token；有内容的负面提示变成`ε_uncond`。这两者并不相同；有些管线会静默地默认使用空token。

## 使用它

2026年的生产技术栈：

| 目标 | 推荐骨干网络 |
|--------|----------------------|
| 窄领域，有配对数据，从头训练模型 | SDXL 微调 (LoRA / 全参数) — 上线最快 |
| 开放域文本生成图像，开放权重 | Flux.1-dev (12B, Apache / 非商业) 或 SD3.5-Large |
| 最快的推理，开放权重 | Flux.1-schnell (1-4步, Apache) 或 SDXL-Lightning |
| 最佳提示遵循，托管服务 | GPT-Image / DALL-E 3 (仍是), Midjourney v7, Imagen 4 |
| 编辑工作流 | Flux.1-Kontext (2024年12月) — 原生支持图像 + 文本 |
| 研究，基线 | SD 1.5 — 古老但被充分研究 |

## 部署它

保存`outputs/skill-sd-prompter.md`。该技能接收文本提示 + 目标风格，输出：模型 + 检查点、CFG缩放值、采样器、负面提示、分辨率、可选的ControlNet/IP-Adapter组合，以及逐步的QA检查清单。

## 练习

1.  **简单。** 以引导值`w ∈ {0, 1, 3, 7, 15}`运行`code/main.py`。记录按类别的平均样本。在`w`取什么值时，类别均值偏离超过真实数据均值？
2.  **中等。** 将玩具线性编码器替换为带有重建损失的tanh-MLP编码器/解码器对。在新的潜在空间上重新训练扩散。样本质量有变化吗？
3.  **困难。** 使用diffusers搭建一个真实的稳定扩散推理环境：加载`sdxl-base`，使用CFG=7运行30步欧拉采样并计时。然后切换到`sdxl-turbo`，运行4步，CFG=0。相同的主题，不同的质量——描述发生了什么变化以及原因。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 第一阶段 | "那个VAE" | 训练好的编码器/解码器对；将512²压缩为64²。 |
| 第二阶段 | "那个U-Net" | 在潜在空间上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调节条件强度。 |
| 空token | "空提示嵌入" | 用于`ε_uncond`的无条件嵌入。 |
| 交叉注意力 | "文本如何进入" | 每个U-Net块将文本token作为K和V进行注意力计算。 |
| DiT | "扩散Transformer" | 用潜在patch上的Transformer替代U-Net；扩展性更好。 |
| MMDiT | "多模态DiT" | SD3的架构：具有联合注意力的文本和图像流。 |
| VAE缩放因子 | "魔法数字" | 将潜在向量除以约5.4，使扩散在单位方差空间中操作。 |

## 生产说明：在8GB消费级GPU上运行Flux-12B

参考的Flux集成是经典的"我有消费级GPU，能部署吗？"配方。诀窍是生产推理文献中列出的三旋钮配方应用于扩散DiT：

1.  **交错加载。** Flux有三个网络不需要同时存在于显存中：T5-XXL文本编码器(fp32约10GB)、CLIP-L(小模型)、12B MMDiT和VAE。首先编码提示，*删除*编码器，加载DiT，去噪，*删除*DiT，加载VAE，解码。消费级8GB GPU一次只能装下一个阶段。
2.  **通过bitsandbytes进行4-bit量化。** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`应用于T5编码器和DiT。内存减少8倍，根据Aritra的基准测试(链接在笔记本中)，文本到图像的质量下降难以察觉。
3.  **CPU卸载。** `pipe.enable_model_cpu_offload()`在每次前向传播推进时，自动在CPU和GPU之间交换模块。增加10-20%的延迟，但使管线得以运行。

内存分配为：`10 GB T5 / 8 = 1.25 GB`量化，`12 B params × 0.5 bytes = ~6 GB`量化DiT，加上激活值。用stas00的术语来说，这是TP=1推理的极端情况——没有模型并行，最大量化。对于生产环境，你会在H100上运行TP=2或TP=4；对于单个开发笔记本，这就是配方。

## 扩展阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — 稳定扩散。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1系列。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述所有检查点的参考实现。