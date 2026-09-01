# ControlNet、LoRA 与条件控制

> 仅靠文本是一种笨拙的控制信号。ControlNet 让你可以克隆一个预训练的扩散模型，并用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA 让你只需训练 1000 万个参数就能微调一个 20 亿参数的模型。二者结合，将 Stable Diffusion 从一个玩具变成了 2026 年每家机构都使用的图像管线。

**类型：** 构建
**语言：** Python
**前置知识：** 第 8 阶段 · 第 07 课（潜在扩散），第 10 阶段（从零构建 LLM — LoRA 基础）
**时间：** 约 75 分钟

## 问题所在

一个提示词如"一个穿红裙的女人在繁忙的街道上遛狗"并没有告诉模型 *狗在哪里*、*女人的姿态是什么*，或 *街道的视角*。文本只能确定你描述一张图像所需信息的约 10%。其余的是视觉信息，无法用文字高效描述。

为每种信号（姿态、深度、Canny、分割）从头训练一个新的条件模型代价高昂。你想要冻结 26 亿参数的 SDXL 主干网络，附加一个读取条件的小侧网络，并让它微调主干网络的中间特征。这就是 ControlNet。

你还想教会模型新概念（你的脸、你的产品、你的风格），而无需重新训练整个模型。你想要一个 100 倍的更小增量。这就是 LoRA —— 低秩适配器，插到已有的注意力权重中。

ControlNet + LoRA + 文本 = 2026 实践者的工具包。大多数生产图像管线会在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等人，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器部分。冻结原始网络。训练克隆网络以接受额外的条件输入（边缘、深度、姿态）。将克隆网络连接回原始网络的解码器部分，使用 *零卷积* 跳跃连接（初始化为零的 1×1 卷积 —— 起始时作为恒等变换，学习增量）。

```
SD U-Net 解码器：  ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 起始时是恒等的 —— 即使在训练前也不会造成损害。在 100 万 (提示词, 条件, 图像) 三元组上使用标准扩散损失进行训练。

每种模态的 ControlNet 作为小型侧模型提供（SDXL 约 3.6 亿参数，SD 1.5 约 7000 万参数）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等人，2021）

对于模型中的任何线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常取秩 4-16，重度微调取秩 64-128。新参数数量：`2 · d · r` 而非 `d²`。对于 SDXL 注意力 `d=640`，`r=16`：每个适配器 2 万个参数，而非 41 万 —— 减少 20 倍。在整个模型上：一个 LoRA 通常 20-200MB，而基础模型 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常范围。多个 LoRA 可以叠加（但有非线性交互的常见注意事项）。

### IP-Adapter（Ye 等人，2023）

一个微小的适配器，接受 *图像* 作为条件（ alongside 文本）。使用 CLIP 图像编码器生成图像 token，将它们注入到交叉注意力中，与文本 token 并列。每个基础模型约 20MB。让你可以做到"生成一张参考此图像的风格的图"而无需 LoRA。

## 可组合性矩阵

| 工具 | 控制什么 | 大小 | 何时使用 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 从参考图像获取风格或主体 | 20MB | 文本无法描述的样貌 |
| 文本反转（Textual Inversion） | 单个概念作为新 token | 10KB | 过时，多数被 LoRA 取代 |
| DreamBooth | 对主题的全量微调 | 2-5GB | 强身份、高算力 |
| T2I-Adapter | 更轻量的 ControlNet 替代 | 70MB | 边缘设备、推理预算 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者都用。

```figure
v4-controlnet-zero
```

## 构建它

`code/main.py` 在 1-D 上模拟这两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练低秩的 `B @ A` 使得 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习秩-1 修正。

2. **ControlNet-lite。** 一个"冻结基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由初始化为零的可学习标量门控（我们版本的零卷积）。训练并观察门控值上升。

### 步骤 1：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 是冻结的；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤 2：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在第 0 步，输出与基础网络相同。早期训练中 `gate` 缓慢更新 —— 避免灾难性漂移。

## 常见陷阱

- **LoRA 缩放过度。** `α = 2` 或 `α = 3` 是常见的"让它更强"的 hack，会产生过度风格化 / 损坏的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 在权重 1.0 使用姿态 ControlNet 和权重 1.0 使用深度 ControlNet 通常会超标。权重之和 ≈ 1.0 是安全默认值。
- **在错误的基础上用 LoRA。** SDXL LoRA 在 SD 1.5 上会静默失效，因为注意力维度不匹配。Diffusers 0.30+ 会发出警告。
- **文本反转漂移。** 在一个检查点上训练的 token 在另一个检查点上会严重漂移。LoRA 更便携。
- **LoRA 权重合并与存储。** 你可以将 LoRA 烘焙进基础模型权重以获得更快的推理（无运行时加法），但会失去在运行时缩放 `α` 的能力。保留两个版本。

## 使用它

| 目标 | 2026 管线 |
|------|---------------|
| 复现品牌艺术风格 | 在 ~30 张精选图像上以秩 32 训练的 LoRA |
| 将我的脸放入生成图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知的构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 修补（第 9 课） |
| 快速 1 步风格化 | SDXL-Turbo 上的 LCM-LoRA |

## 部署

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接受任务（输入资产：提示词、可选参考图像、可选姿态、可选深度、可选涂鸦）并输出工具栈、权重和可复现的种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 变到 4。在什么秩下 LoRA 恰好匹配秩-2 的目标增量？
2. **中等。** 在两个目标变换上分别训练两个 LoRA。一起加载它们并展示它们的叠加交互。交互在何时破坏线性？
3. **困难。** 使用 diffusers 堆叠：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。当堆叠权重变化时，测量 FID 与提示词遵循性的权衡。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------------------|----------|
| ControlNet | "空间控制" | 克隆编码器 + 零卷积跳跃；读取条件图像。 |
| 零卷积 | "起始为恒等" | 初始化为零的 1×1 卷积；ControlNet 起始时是不起作用的。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；参数量比全量微调少 100 倍。 |
| 秩 r | "旋钮" | LoRA 压缩；通常 4-16，重度个性化 64+。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 的小图像条件适配器。 |
| DreamBooth | "完整主题微调" | 用 ~30 张主体图像训练整个模型。 |
| 文本反转 | "新词" | 仅学习新词嵌入；过时，多数已被取代。 |

## 生产笔记：LoRA 热交换、ControlNet 车道、多租户服务

一个真实的文本到图像 SaaS 在同一个基础检查点上服务数百个 LoRA 和 dozen 个 ControlNet。这个服务问题看起来很像 LLM 多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下覆盖 LLM 案例）：

- **热交换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并进基础模型可获得约 3-5% 更快的每步推理，但会冻结 `α` 和基础模型。将 LoRA 作为秩-r 增量热置于 VRAM 中；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。交换代价是 `2 · d · r · num_layers` 权重 —— MB 级，亚秒级。
- **ControlNet 作为第二个注意力车道。** 克隆编码器与基础网络并行运行。两个 ControlNet 各权重 1.0 = 每步两个额外的前向传播，而非一个合并的前向。批量大小余量呈二次下降。为每个活动的 ControlNet 预算约 1.5× 步代价。
- **量化 LoRA 也可以。** 如果你对基础模型量化了（见第 7 课，8GB 上的 Flux），LoRA 增量也干净地量化为 8-bit 或 4-bit。QLoRA 式加载让你可以在 4-bit Flux 基础上堆叠 5-10 个 LoRA 而不会撑爆内存。

Flux 特定：Niels 的 8GB Flux 笔记本将基础模型量化为 4-bit；在该量化基础上堆叠风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）在 `weight_name="pytorch_lora_weights.safetensors"` 下仍然有效。这是 2026 年大多数 SaaS 机构交付的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) —— ControlNet。
- [Hu 等人 (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) —— LoRA（最初为 LLM 设计；移植到扩散模型）。
- [Ye 等人 (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) —— IP-Adapter。
- [Mou 等人 (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) —— ControlNet 的轻量替代。
- [Ruiz 等人 (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) —— DreamBooth。
- [HuggingFace Diffusers —— ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet) —— 参考管线。
