# Latent Diffusion & Stable Diffusion

> 在 512×512 图像的像素空间中做扩散，是一场计算战争罪行。Rombach 等人（2022）注意到，你不需要全部 786k 个维度来生成图像——你只需要足够捕捉语义结构的维度，再加一个独立的解码器来处理其余部分。在 VAE 的潜空间内运行扩散。就是这一个想法，成就了 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8 · 02（VAE）、阶段 8 · 06（DDPM）、阶段 7 · 09（ViT）
**时间：** 约 75 分钟

## 问题所在

512² 分辨率的像素空间扩散意味着 U-Net 需要在形状为 `[B, 3, 512, 512]` 的张量上运行。每个采样步对于 5 亿参数的 U-Net 约需 100 GFLOPS。50 步即每张图片 5 TFLOPS。在一百亿张图像上训练，计算账单天文数字。

大部分 FLOPS 都耗费在通过网络推送感知上不重要的细节——那些高频率纹理本可由有损 VAE 压缩掉。Rombach 的想法：先训练一个 VAE（*第一阶段*），冻结它，然后在 4 通道 64×64 的潜空间中完全运行扩散（*第二阶段*）。同样的 U-Net，1/16 的像素，质量相当的情况下 FLOPS 减少约 64 倍。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 在 `64×64×4` 潜空间上使用 8.6 亿参数的 U-Net，SDXL 在 `128×128×4` 上使用 26 亿参数的 U-Net，SD3 将 U-Net 换成了带有流匹配的扩散 Transformer（DiT）。Flux.1-dev（Black Forest Labs，2024）搭载 120 亿参数的 DiT-MMDiT。它们都在同样的两阶段基础上运行。

## 核心概念

![潜扩散：VAE 压缩 + 潜空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **阶段 1 — VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴下采样 8 倍，并调整通道数使总潜尺寸约为像素数的 1/16。损失 = 重建损失（L1 + LPIPS 感知）+ KL 散度（权重很小，因为 `z` 不需要被强制成正态分布——我们不需要从 `z` 精确采样）。通常还会加对抗损失使解码图像更锐利。

2. **阶段 2 — 在 `z` 上的扩散。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）来去噪 `z_t`。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件注入。** 两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L + OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。以及交叉注意力注入：每个 U-Net 区块接收 `[Q = 图像特征，K = V = 文本 token]` 并对其进行混合。token 是文本影响图像的唯一途径。

**损失函数与第 06 课相同。** 同样是 DDPM / 流匹配的噪声 MSE。你只是更换了数据域。

## 架构变体

| 模型 | 年份 | 骨干网络 | 潜空间形状 | 文本编码器 | 参数量 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 个 token） | 8.6 亿 |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 8.65 亿 |
| SDXL | 2023 | U-Net + 精炼器 | 128×128×4 | CLIP-L + OpenCLIP-G | 26 亿 + 66 亿 |
| SDXL-Turbo | 2023 | 蒸馏版 | 128×128×4 | 同上 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 20 亿 / 80 亿 |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 120 亿 |
| Flux.1-schnell | 2024 | MMDiT 蒸馏版 | 128×128×16 | T5-XXL + CLIP-L | 120 亿，1-4 步 |

趋势：用 DiT 替换 U-Net（基于潜 patch 的 transformer），扩展文本编码器（T5 在提示词遵循度上优于 CLIP），增加潜通道数（4 → 16 提供更多细节余量）。

```figure
noise-schedule
```

## 动手构建

`code/main.py` 将 toy 1-D "VAE"（恒等编码器和解码器，仅作演示；真实 VAE 会是卷积网络）叠在第 06 课的 DDPM 上，并加入类别条件与分类器自由引导。它展示了同样的扩散损失无论你在原始 1-D 值上运行还是在编码值上运行都同样有效——这是关键洞察。

### 步骤 1：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真实 VAE 拥有训练过的权重。为教学目的，这个线性映射足以展示扩散在 `z` 上运作，而不关心原始数据空间。

### 步骤 2：在 `z` 空间中做扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样完 `z_0` 后，用 `D(z_0)` 解码。

### 步骤 3：分类器自由引导（CFG）

训练时，10% 的时间丢弃类别标签（替换为空 token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样性），`w = 3` = 默认值，`w = 7+` = 饱和/过锐。

### 步骤 4：文本条件（概念，非代码）

将类别标签替换为冻结文本编码器的输出。通过交叉注意力将文本嵌入馈送到 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型与 Stable Diffusion 之间唯一的实质性差异。

## 陷阱

- **VAE 尺度不匹配。** SD 1.x 的 VAE 在编码后会应用一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这一点会使 U-Net 在方差 wildly wrong 的潜上训练。每个 checkpoint 都附带这个因子。
- **文本编码器静默错误。** SD3 需要 T5-XXL 且 token 数 ≥ 128，回退到仅 CLIP 是有损的。始终检查 `use_t5=True`，否则提示词忠实度会崩盘。
- **混用潜空间。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL 潜上训练的 LoRA 无法在 SD3 上工作。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的 checkpoint。
- **CFG 过高。** `w > 10` 会产生饱和、油腻的图像，并以牺牲多样性为代价过度拟合提示词。甜点区是 `w = 3-7`。
- **负向提示词泄漏。** 空的负向提示词变为空 token；填写的负向提示词变为 `ε_uncond`。二者不同——某些管线会静默回退到空 token。

## 如何使用

2026 年的生产栈：

| 目标 | 推荐骨干 |
|--------|----------------------|
| 狭窄领域，配对数据，从头训练模型 | SDXL 微调（LoRA / 全量）—— 交付最快 |
| 开放域文生图，开源权重 | Flux.1-dev（120 亿，Apache / 非商用）或 SD3.5-Large |
| 最快推理，开源权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示词遵循度，托管服务 | GPT-Image / DALL-E 3（仍在）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）—— 原生接受图像 + 文本 |
| 研究，基线 | SD 1.5 —— 虽老但研究充分 |

## 交付成果

保存 `outputs/skill-sd-prompter.md`。技能接收文本提示 + 目标风格，输出：模型 + checkpoint、CFG 尺度、采样器、负向提示词、分辨率、可选 ControlNet/IP-Adapter 组合，以及逐步 QA 检查清单。

## 练习

1. **简单。** 用引导 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录各类别的平均样本。在什么 `w` 值下，类别均值会偏离真实数据均值？
2. **中等。** 将 toy 线性编码器替换为带重建损失的 tanh-MLP 编码器/解码器对。在新的潜上重新训练扩散。样本质量有变化吗？
3. **困难。** 用 diffusers 搭建真实 Stable Diffusion 推理：加载 `sdxl-base`，用 CFG=7 跑 30 步 Euler 采样，计时。现在切换到 `sdxl-turbo` 用 4 步和 CFG=0。同一主题，质量不同——描述发生了什么变化以及原因。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 第一阶段 | "VAE" | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 | "U-Net" | 在潜空间上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调节条件强度。 |
| 空 token | "空提示词嵌入" | 用于 `ε_uncond` 的条件嵌入。 |
| 交叉注意力 | "文本如何进入" | 每个 U-Net 区块将文本 token 作为 K 和 V 进行注意力计算。 |
| DiT | "扩散 Transformer" | 用基于潜 patch 的 transformer 替换 U-Net；扩展性更好。 |
| MMDiT | "多模态 DiT" | SD3 的架构：文本和图像流带联合注意力。 |
| VAE 缩放因子 | "魔法数字" | 将潜除以约 5.4，使扩散在单位方差空间上运行。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是标准的"我有消费级 GPU，能部署这个吗？"配方。窍门与生产推理文献中列出的适用于扩散 DiT 的三旋钮配方相同：

1. **交错加载。** Flux 有三个网络永远不需要同时存在于 VRAM 中：T5-XXL 文本编码器（fp32 下约 10 GB）、CLIP-L（小）、120 亿参数的 MMDiT，以及 VAE。先编码提示词，*删除* 编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只能装一个阶段。
2. **通过 bitsandbytes 进行 4-bit 量化。** 在 T5 编码器和 DiT 上都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存减少 8 倍，质量下降在文生图场景下几乎不可察觉（根据 Aritra 的基准测试，链接在笔记本中）。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 会自动在 CPU 和 GPU 之间交换模块，随着每次前向传播推进。增加 10-20% 延迟但使管线得以运行。

内存账目如下：`10 GB T5 / 8 = 1.25 GB` 量化后，`120 亿参数 × 0.5 字节 = 约 6 GB` 量化 DiT，加上激活值。按 stas00 的说法，这是 TP=1 推理的极端情况——无模型并行，最大量化。对于生产环境你会在 H100 上运行 TP=2 或 TP=4；对于单台开发笔记本，这就是配方。

## 延伸阅读

- [Rombach 等人（2022）。《使用潜扩散模型进行高分辨率图像合成》](https://arxiv.org/abs/2112.10752) —— Stable Diffusion。
- [Podell 等人（2023）。《SDXL：改进潜扩散模型用于高分辨率图像合成》](https://arxiv.org/abs/2307.01952) —— SDXL。
- [Peebles & Xie（2023）。《可扩展的 Transformer 扩散模型（DiT）》](https://arxiv.org/abs/2212.09748) —— DiT。
- [Esser 等人（2024）。《扩展整流流 Transformer 用于高分辨率图像合成》](https://arxiv.org/abs/2403.03206) —— SD3，MMDiT。
- [Ho & Salimans（2022）。《分类器自由扩散引导》](https://arxiv.org/abs/2207.12598) —— CFG。
- [Labs（2024）。《Flux.1 — Black Forest Labs 公告》](https://blackforestlabs.ai/announcing-black-forest-labs/) —— Flux.1 系列。
- [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers/index) —— 上述每个 checkpoint 的参考实现。
