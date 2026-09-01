# Transfusion：一个 Transformer 中的自回归文本 + 扩散图像

> Chameleon 和 Emu3 把赌注全押在离散 token 上。它们有效，但量化瓶颈显而易见——图像质量低于连续空间扩散模型的水平。Transfusion（Meta，Zhou 等人，2024 年 8 月）采取了相反的赌注：保留图像的连续性，完全抛弃 VQ-VAE，用一个 transformer 跑两个损失。文本 token 使用下一个 token 预测（NTP），图像 patch 使用 flow-matching / 扩散损失。两个目标函数优化同一组权重。Stable Diffusion 3 背后的架构（MMDiT）是它的近亲。本课阅读 Transfusion 的论文、构建一个双损失玩具训练器，并追踪让一个 transformer 同时完成两项任务的注意力掩码。

**类型：** 构建
**语言：** Python（标准库，MNIST 规模玩具上的双损失训练器）
**前置知识：** 第 12 课 · 第 11 课（Chameleon）、第 8 课（生成式 AI）
**时间：** 约 180 分钟

## 学习目标

- 在一个 transformer 主干上实现双损失（文本 token 的 NTP、图像 patch 的扩散 MSE）的接线。
- 解释为什么图像 patch 的双向注意力加上文本 token 的因果注意力是正确的掩码选择。
- 在计算量、质量和代码复杂度上，对比 Transfusion 风格（连续图像 + 扩散损失）与 Chameleon 风格（离散图像 + NTP）。
- 指出 MMDiT 的贡献：每个 block 具有模态专属权重，残差流中执行联合注意力。

## 问题

离散 vs 连续图像 token 的争论比 LLM 更悠久。连续表示（原始像素、VAE 潜变量）保留了细节。离散 token（VQ 索引）契合 transformer 的原生词表，但在量化步骤丢失了细节。

Chameleon / Emu3 选择了离散方案：一个损失、一个架构，但图像保真度受限于 tokenizer 的质量。

扩散模型选择了连续方案：出色的图像质量，但与 LLM 是独立的模型，需要复杂的噪声调度工程，且无法与文本生成干净集成。

Transfusion 提出了一个问题：我们能两者兼得吗？保留图像的连续性，仍训练一个模型，用两个损失缝合进一个梯度步骤。

## 概念

### 双损失架构

一个单独的 decoder-only transformer 处理一个包含以下内容的序列：

- 文本 token（离散的，来自 BPE 词表）。
- 图像 patch（连续的，16×16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标记标记连续 patch 的位置。

前向传播只运行一次。损失函数根据 token 类型选择两个头之一：

- 对于文本 token：词表 logits 头上的标准交叉熵。
- 对于图像 patch：连续 patch 上的扩散损失——预测加到每个 patch 上的噪声。

梯度流经共享的 transformer 主体。两个损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token  attends to 未来文本，否则 teacher forcing 会失效。然而图像 patch 代表同一帧快照；它们应在同一图像块内双向互相 attend。

掩码规则：

```
M[i, j] = 1 当：
  (i 是文本且 j 是文本且 j <= i)          # 文本使用因果掩码
  OR (i 是图像且 j 是图像且 same_image_block(i, j))  # 图像块内双向
  OR (i 是文本且 j 是图像且 j < i_image_end)       # 文本 attend 之前的图像
  OR (i 是图像且 j 是文本且 j < i_image_start)     # 图像 attend 前面的文本
```

在训练和推理中以分块三角形式实现。

### Transformer 内部的扩散损失

扩散损失是标准的：向图像 patch 添加噪声，让模型预测噪声（等价地，预测干净 patch）。Transfusion 的版本使用 flow matching——预测从噪声到干净数据的速度场。

训练时：
1. 对每个图像 patch x0，采样随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同序列的文本 NTP 损失一起反向传播。

推理时，生成方式为：
- 文本 token：标准自回归采样。
- 图像 patch：扩散采样循环（典型 10-30 步），以先验文本 token 为条件。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024 年 3 月）在 Transfusion 同期推出了 MMDiT（多模态扩散 Transformer）。两种架构是近亲。

MMDiT 的关键差异：

- 每个 block 使用模态专属权重。每个 transformer block 对文本 token 和图像 patch 有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其余部分为模态专属。
- Rectified flow 训练。一种特定的 flow-matching 变体，采样更简单、数学比 DDPM 更简洁。
- 规模。MMDiT 是 SD3 的主干（2B 和 8B 参数变体）。Transfusion 的论文扩展到 7B。

两者都收敛于同一个核心思想：一个 transformer 对文本做 NTP，对连续图像表示做扩散。

### 为何优于 Chameleon 风格

连续扩散与离散 NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B 参数下，FID 比同等大小的 Chameleon 风格模型优 3-5 个点。
- 无需训练 tokenizer——图像编码器更简单（线性投影到隐藏层，与 ViT 输入层相同）。
- 推理时可以并行化图像 patch 的去噪，区别于自回归图像 token。

缺点：Transfusion 是双损失模型，训练动态更复杂。需要调优损失权重。NTP 与扩散之间的调度不匹配可能导致一个头占主导。

### 后续发展

Janus-Pro（第 12.15 课）通过将理解和生成的视觉编码器解耦来细化 Transfusion 的思路——前者用 SigLIP，后者用 VQ——同时共享 transformer 主体。Show-o（第 12.14 课）用离散扩散（掩码预测）替换了连续扩散。Transfusion 之后，统一生成家族迅速分支。

2026 年的生产级 VLM 发射图像——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用了该家族的后代。细节属于商业秘密。

```figure
cfg-guidance-scale
```

## 动手实现

`code/main.py` 在一个小型 MNIST 风格的问题上构建了一个玩具版 Transfusion：

- 文本描述是短整数序列，描述一个数字（0-9）。
- 图像是 4×4 的字节网格。
- 一对共享权重的线性投影充当 transformer 替身；文本上用 NTP 损失，噪声 patch 上用 MSE 损失。
- 训练循环交替两个损失，注意力掩码是显式的。
- 生成在一次前向传播中输出文本描述和 4×4 图像。

这个 transformer 是玩具级别的。双损失管线、注意力掩码构建和推理循环才是真正的核心工件。

## 交付物

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本+图像、文本+音频、文本+视频），设计双损失调度（损失权重、掩码形状、共享 vs 模态专属 block）并标记实现风险。

## 练习题

1. 一个 Transfusion 风格模型训练时包含 70% 文本 token 和 30% 图像 patch。图像扩散损失约为文本 NTP 损失 magnitude 的 10 倍。什么损失权重可以平衡它们？

2. 为一个序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现分块三角掩码。标注每个位置为 0 或 1。

3. MMDiT 具有模态专属的 QKV 权重。相比 Transfusion 完全共享的 transformer，这会带来多少参数开销？在 7B 参数规模下是否值得？

4. 生成时：给定文本 prompt，模型先运行 50 个 token 的 NTP，然后遇到 `<image>`，接着对 256 个 patch 运行 20 步去噪的扩散。总共多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述 rectified flow 以及为何它比 DDPM 在更少推理步数下收敛。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 双损失训练 | "NTP + 扩散" | 单个 transformer 在同一梯度步骤中同时对文本 token 优化交叉熵、对连续图像 patch 优化 MSE |
| Flow matching | "Rectified flow" | 扩散变体，预测从噪声到干净数据的速度场；数学比 DDPM 更简洁 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 的架构：联合注意力，模态专属 MLP 和归一化 |
| 分块三角掩码 | "因果文本 + 双向图像" | 文本区域因果、图像区域内双向的注意力掩码 |
| 连续图像表示 | "无 VQ" | 图像 patch 为实值向量，而非整数码本索引 |
| 速度预测 | "v-parameterization" | 网络输出是噪声与数据之间的速度场，而非噪声本身 |

## 延伸阅读

- [Zhou 等人 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等人 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等人 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
