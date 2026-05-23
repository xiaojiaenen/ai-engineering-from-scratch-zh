# ControlNet、LoRA与条件控制

> 纯文本是一种笨拙的控制信号。ControlNet让你能克隆一个预训练的扩散模型，并用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA让你通过仅训练1000万参数来微调一个20亿参数的模型。它们共同将Stable Diffusion从玩具变成了2026年每个代理商都在使用的图像生产流水线。

**类型:** 构建
**语言:** Python
**先决条件:** 第8阶段 · 第07课（潜在扩散），第10阶段（从头实现大语言模型 —— 为LoRA打基础）
**时间:** ~75分钟

## 问题所在

像“一位穿着红裙的女士在繁忙的街道上遛狗”这样的提示词，无法向模型提供关于*狗在哪里*、*女士是何种姿态*或*街道的透视*的任何信息。文本大约只能锚定你需要指定的图像的10%。其余部分是视觉性的，无法用语言高效描述。

为每一种信号（姿态、深度、边缘、分割）从头训练一个新的条件模型是不切实际的。你希望保持拥有26亿参数的SDXL主干网络冻结，附加一个小型侧网络来读取条件输入，并让它推动主干网络的中间特征。这就是ControlNet。

你还希望教模型学习新概念（你的脸、你的产品、你的风格）而无需重新训练整个模型。你想要一个100倍小的增量。这就是LoRA——插入现有注意力权重中的低秩适配器。

ControlNet + LoRA + 文本 = 2026年从业者的工具包。大多数生产图像流水线在SDXL / SD3 / Flux基础模型之上，叠加2-5个LoRA、1-3个ControlNet以及一个IP-Adapter。

## 核心概念

![ControlNet克隆编码器；LoRA添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet (Zhang 等人, 2023)

取一个预训练的Stable Diffusion模型。*克隆*U-Net的编码器部分。冻结原始编码器。训练克隆体以接受额外的条件输入（边缘、深度、姿态）。通过*零卷积*跳跃连接（初始化为零的1×1卷积——初始时为无操作，通过学习得到增量）将克隆体连接回原始模型的解码器部分。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着ControlNet起始于恒等映射——即使在训练之前也不会有害。使用标准扩散损失在100万（提示词，条件，图像）三元组上进行训练。

每种模态的ControlNet作为小型侧模型提供（SDXL约360M，SD 1.5约70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu 等人, 2021)

对于模型中的任何线性层`W ∈ R^{d×d}`，冻结`W`并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中`r << d`。对于注意力机制，秩4-16是标准配置，对于重度微调，秩64-128是标准配置。新参数的数量：`2 · d · r`，而不是`d²`。对于具有`d=640`, `r=16`的SDXL注意力机制：每个适配器2万参数，而不是41万——减少了20倍。整个模型而言：一个LoRA通常为20-200MB，而基础模型为5GB。

在推理时，你可以缩放LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5`是常规设置。多个LoRA叠加时是可加的（当然要注意它们会以非线性方式相互影响）。

### IP-Adapter (Ye 等人, 2023)

一个微小的适配器，接受*图像*作为条件（与文本并行）。使用CLIP图像编码器生成图像token，并将它们与文本token一起注入交叉注意力中。每个基础模型约20MB。让你无需LoRA即可实现“按照此参考图像的风格生成图像”。

## 可组合性矩阵

| 工具 | 控制什么 | 大小 | 何时使用 |
|------|----------|------|----------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格化 |
| IP-Adapter | 参考图像的风格或主体 | 20MB | 无法用文本描述的外观 |
| Textual Inversion | 将单一概念作为新token | 10KB | 遗留技术，大多被LoRA取代 |
| DreamBooth | 对主体进行完全微调 | 2-5GB | 强身份保持，高计算开销 |
| T2I-Adapter | 更轻量的ControlNet替代方案 | 70MB | 边缘设备、推理预算有限 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。两者兼用。

## 构建它

`code/main.py`在一维上模拟这两种机制：

1.  **LoRA。** 一个预训练的线性层`W`。冻结它。训练一个低秩的`B @ A`，使得`W + BA`匹配目标线性层。证明`r = 1`足以完美学习一个秩为1的修正。
2.  **ControlNet精简版。** 一个“冻结的基础”预测器和一个读取额外信号的“侧网络”。侧网络的输出由一个可学习的标量控制，该标量初始化为零（我们的零卷积版本）。训练并观察该门控值逐渐增大。

### 步骤1：LoRA数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤2：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在步骤0，输出与基础输出完全相同。早期训练`gate`更新缓慢——不会发生灾难性偏移。

## 常见陷阱

- **过度缩放LoRA。** `α = 2` 或 `α = 3` 是常见的“让它更强”的技巧，但会产生过度风格化/损坏的输出。保持`α ≤ 1.5`。
- **ControlNet权重冲突。** 将Pose ControlNet权重设为1.0，Depth ControlNet权重也设为1.0，通常会导致效果过强。权重之和≈1.0是一个安全的默认值。
- **在错误的基础模型上使用LoRA。** SDXL的LoRA在SD 1.5上会静默无效，因为注意力维度不匹配。Diffusers 0.30+版本会发出警告。
- **Textual Inversion漂移。** 在一个检查点上训练的token在另一个检查点上会严重漂移。LoRA更具可移植性。
- **LoRA权重合并与存储。** 你可以将LoRA烘焙到基础模型权重中以获得更快的推理速度（无需运行时添加），但你会失去在运行时缩放`α`的能力。请保留两个版本。

## 应用场景

| 目标 | 2026年流水线 |
|------|---------------|
| 复现某品牌艺术风格 | 使用约30张精选图像、秩为32训练的LoRA |
| 将我的脸放入生成图像中 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图像 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + Inpainting (第09课) |
| 快速单步风格化 | LCM-LoRA 用于 SDXL-Turbo |

## 部署

保存`outputs/skill-sd-toolkit-composer.md`。该技能接收一个任务（输入资源：提示词，可选的参考图像，可选的姿态，可选的深度，可选的涂鸦），并输出工具栈、权重和可复现的随机种子协议。

## 练习

1.  **简单。** 在`code/main.py`中，将LoRA秩`r`从1变到4。在哪个秩时，LoRA能精确匹配一个秩为2的目标增量？
2.  **中等。** 在两个目标变换上分别训练两个LoRA。同时加载它们并展示它们的可加性交互。交互在何时会破坏线性关系？
3.  **困难。** 使用diffusers进行堆叠：SDXL基础模型 + Canny-ControlNet (权重0.8) + 风格LoRA (α 0.8) + IP-Adapter (权重0.6)。测量随着堆叠权重变化时，FID与提示词遵循度之间的权衡。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ControlNet | “空间控制” | 克隆的编码器 + 零卷积跳跃连接；读取条件图像。 |
| 零卷积 | “起始时为恒等映射” | 初始化为零的1×1卷积；ControlNet起始于无操作。 |
| LoRA | “低秩适配器” | `W + B @ A`, `r << d`；比完全微调少100倍的参数。 |
| 秩 r | “那个旋钮” | LoRA的压缩度；4-16为典型值，64+用于重度个性化。 |
| α | “LoRA强度” | LoRA增量的运行时缩放因子。 |
| IP-Adapter | “参考图像” | 通过CLIP图像token实现的小型图像条件适配器。 |
| DreamBooth | “主体完全微调” | 在某个主体的约30张图像上训练整个模型。 |
| Textual Inversion | “新token” | 仅学习新的词嵌入；遗留技术，大多已被取代。 |

## 生产注意事项：LoRA交换、ControlNet通道、多租户服务

一个真正的文生图SaaS服务，会在同一个基础检查点上提供数百个LoRA和十多个ControlNet。服务问题看起来很像LLM多租户问题（生产文献在连续批处理、LoRAX / S-LoRA下涵盖了LLM的案例）：

- **热交换LoRA，不要合并。** 将`W' = W + α·B·A`合并到基础模型中可以提高约3-5%的单步推理速度，但会冻结`α`和基础模型。将LoRA作为秩r增量保留在显存中；diffusers提供了`pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`用于按请求激活。交换成本是`2 · d · r · num_layers`的权重——兆字节级，亚秒级。
- **ControlNet作为第二个注意力通道。** 克隆的编码器与基础模型并行运行。两个权重均为1.0的ControlNet = 每步两次额外的前向传播，而不是一次合并的传播。批处理大小的余量呈二次方下降。预算约为每个活跃ControlNet额外1.5倍的步骤成本。
- **量化LoRA同样适用。** 如果你量化了基础模型（参见第07课，Flux在8GB显存上运行），LoRA增量也可以干净地量化到8位或4位。QLoRA式的加载允许你在4位Flux基础模型之上堆叠5-10个LoRA，而不会耗尽内存。

Flux特例：Niels的Flux-on-8GB笔记本将基础模型量化到4位；在该量化基础模型上堆叠风格LoRA（`pipe.load_lora_weights("user/style-lora")`）并设置`weight_name="pytorch_lora_weights.safetensors"`仍然有效。这是2026年大多数SaaS代理商部署的方案。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初为LLM设计；已移植到扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet的轻量级替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线。