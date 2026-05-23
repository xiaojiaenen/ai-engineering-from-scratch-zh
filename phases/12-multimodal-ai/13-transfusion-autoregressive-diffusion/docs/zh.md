# Transfusion：一个Transformer中的自回归文本+扩散图像

> Chameleon和Emu3将一切押注在离散token上。它们有效，但量化瓶颈显而易见——图像质量低于连续空间扩散模型的水平。Transfusion（Meta, Zhou et al., 2024年8月）采取了相反的策略：保持图像连续性，完全摒弃VQ-VAE，并用两种损失训练一个transformer。文本token使用下一个token预测（NTP）。图像块使用流匹配/扩散损失。两个目标优化同一组权重。支撑Stable Diffusion 3（MMDiT）的架构是其近亲。本课程阅读Transfusion论文，构建一个玩具级的双损失训练器，并追踪允许一个transformer同时处理两项任务的注意力掩码。

**类型：** 构建
**语言：** Python（标准库，基于MNIST规模玩具数据的双损失训练器）
**先决条件：** 阶段 12 · 11 (Chameleon)，阶段 8 (生成式 AI)
**时间：** 约 180 分钟

## 学习目标

- 构建一个在同一个主干网络上运行两种损失（文本token的NTP，图像块的扩散均方误差）的transformer。
- 解释为什么在图像块上使用双向注意力，在文本token上使用因果注意力是正确的掩码选择。
- 比较Transfusion风格（连续图像，扩散损失）与Chameleon风格（离散图像，NTP）在计算量、质量和代码复杂度上的差异。
- 说明MMDiT的贡献：每个块中的模态特定权重，残差流中的联合注意力。

## 问题

离散与连续图像token之争比LLM的历史还要久。连续表示（原始像素、VAE隐空间）保留细节。离散token（VQ索引）契合transformer的原生词表，但在量化步骤中丢失细节。

Chameleon/Emu3走离散路线：单一损失，单一架构，但图像保真度受限于分词器质量。

扩散模型走连续路线：卓越的图像质量，但模型独立于LLM，需要复杂的噪声调度工程，并且与文本生成无法干净地集成。

Transfusion提问：我们能兼得吗？保持图像连续性，仍训练一个模型，将两种损失缝合进同一个梯度步骤。

## 概念

### 双损失架构

一个单一的仅解码器transformer处理一个序列，该序列包含：
- 文本token（离散，来自BPE词表）。
- 图像块（连续，16x16像素块通过线性嵌入投影到隐藏维度——与ViT编码器的输入相同）。
- `<image>` 和 `</image>` 标签标记连续块的位置。

前向传播运行一次。损失函数为每个token选择一个头：
- 对于文本token：在词表logits头上计算标准交叉熵。
- 对于图像块：在连续块上计算扩散损失——预测添加到每个块上的噪声。

梯度流经共享的transformer主体。两种损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本token必须是因果的——你不能让一个文本token注意到未来的文本，否则教师强制会失效。然而，图像块代表一个快照；它们应该在同一图像块内彼此双向关注。

掩码：
```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

在训练和推理时实现为一个块状三角掩码。

### Transformer内的扩散损失

扩散损失是标准的：给图像块添加噪声，要求模型预测噪声（或等效地，预测干净的块）。Transfusion的版本使用流匹配——预测从噪声到干净的向量场。

训练期间：
1. 对于每个图像块 x0，采样一个随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与来自同一序列的文本NTP损失一起反向传播。

推理时，生成过程是：
- 文本token：标准自回归采样。
- 图像块：扩散采样循环（通常10-30步），以之前的文本token为条件。

### MMDiT：Stable Diffusion 3的变体

Stable Diffusion 3（Esser et al., 2024年3月）大约与Transfusion同一时间发布了MMDiT（多模态扩散Transformer）。两者是兄弟架构。

MMDiT的关键区别：
- 每个块的模态特定权重。每个transformer块对于文本token和图像块有独立的Q, K, V和MLP权重。注意力是联合的（跨模态）；其他一切都是模态特定的。
- 整流流训练。一种特定的流匹配变体，具有已知的采样特性，数学比DDPM更简单。
- 规模。MMDiT是SD3（20亿和80亿参数变体）的主干。Transfusion的论文扩展到70亿参数。

两者都收敛于同一个核心思想：一个transformer对文本运行NTP，对连续图像表示运行扩散。

### 为什么这优于Chameleon风格

在图像生成上，连续扩散与离散NTP之间的质量差距是可测量的。Transfusion论文报告：
- 在70亿参数时，在FID指标上比同等大小的Chameleon风格模型优3-5分。
- 无需训练分词器——图像编码器更简单（线性投影到隐藏维度，与ViT的输入层相同）。
- 推理时可以并行化图像块去噪，这与自回归图像token不同。

缺点：Transfusion是双损失模型，使训练动态更加棘手。损失权重需要调整。NTP和扩散之间的调度不匹配可能导致一个头主导训练。

### 后续发展

Janus-Pro（课程12.15）通过将用于理解和生成的视觉编码器解耦来改进Transfusion的想法——一个用SigLIP，另一个用VQ——同时共享transformer主体。Show-o（课程12.14）将扩散替换为离散扩散（掩码预测）。统一生成家族在Transfusion之后迅速分支。

2026年能够生成图像的生产VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7的图像生成路径——几乎可以肯定使用了这个家族的某个后代。具体细节是专有的。

## 使用它

`code/main.py` 在一个小型类MNIST问题上构建了一个玩具级的Transfusion：
- 文本描述是描述一个数字（0-9）的短整数序列。
- 图像是4x4的字节网格。
- 一对共享权重的线性投影充当transformer的替代；文本上的NTP损失，噪声块上的MSE损失。
- 训练循环交替使用两种损失，注意力掩码是显式的。
- 生成过程在一个前向传播中输出一个文本描述和一个4x4的图像。

这个transformer是玩具级的。真正的产物是双损失流水线、注意力掩码构建和推理循环。

## 交付它

本课程产生`outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本+图像，文本+音频，文本+视频），它设计双损失调度（损失权重、掩码形状、共享块与模态特定块），并标记实施风险。

## 练习

1.  一个Transfusion风格的模型训练70%的文本token和30%的图像块。图像扩散损失的幅度大约是文本NTP损失的10倍。如何设置损失权重来平衡它们？

2.  为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块状三角掩码。将每个条目标记为0或1。

3.  MMDiT具有模态特定的QKV权重。与Transfusion的完全共享transformer相比，这增加了多少参数开销？在70亿参数规模下，这值得吗？

4.  生成：给定一个文本提示，模型运行NTP生成50个token，然后遇到`<image>`，然后对256个块运行20个去噪步骤的扩散。总共有多少次前向传播？

5.  阅读SD3论文第3节。描述整流流，并解释为什么它在推理步骤中比DDPM收敛更快。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 双损失训练 | "NTP + 扩散" | 一个单一transformer在同一个梯度步骤中同时优化文本token的交叉熵和连续图像块的均方误差 |
| 流匹配 | "整流流" | 一种扩散变体，预测从噪声到干净数据的向量场；数学比DDPM更简单 |
| MMDiT | "多模态DiT" | Stable Diffusion 3的架构：联合注意力，模态特定的MLP和归一化层 |
| 块状三角掩码 | "因果文本 + 双向图像" | 注意力掩码，在文本中是因果的，但在图像区域内是双向的 |
| 连续图像表示 | "无VQ" | 图像块表示为实值向量，而非整数码本索引 |
| 速度预测 | "v参数化" | 网络输出是噪声与数据之间的向量场，而非噪声本身 |

## 扩展阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)