# CLIP 与对比视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了单个核心创意足以支撑未来五年发展：仅使用嘈杂的网络图像-标题对和对比损失，将图像编码器和文本编码器的输出对齐到同一向量空间。零监督标签。4 亿对样本。由此产生的嵌入空间实现了零样本分类、图像-文本检索，并成为 2026 年所有 VLM 中的视觉塔。SigLIP 2（2025）用 Sigmoid 替换了 Softmax，并以更低成本超越了 CLIP。本课从 InfoNCE 到 Sigmoid 成对损失的推导出发，用标准 Python 实现训练步骤。

**类型：** 构建
**语言：** Python（stdlib，InfoNCE + Sigmoid 损失实现）
**前置知识：** 第 12 阶段 · 01（ViT 补丁），第 7 阶段（Transformer）
**时间：** 约 180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 Sigmoid 成对损失（SigLIP）能在 batch 32768+ 规模上扩展，而无需 Softmax 要求的 all-gather 通信开销。
- 通过构建文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP/SigLIP 预训练给你的四个杠杆：batch 大小、温度、提示模板、数据质量。

## 问题背景

CLIP 之前的视觉模型依赖监督学习。收集带标签的数据集（ImageNet：120 万张图像，1000 个类别），训练 CNN，交付模型。标注成本高昂，标签偏向标注者能达成一致的内容，且在不微调的情况下无法迁移到新任务。

网络上的图像-标题数据有十亿级 loosely-labeled 配对可免费提供。一张金毛犬的照片配文"my dog Max in the park"就携带了监督信号——文本描述了图像。关键问题是：能否将其转化为有效的训练方法？

CLIP 的答案：将图像-标题配对视为匹配任务。给定 N 张图像和 N 个标题的批次，学习将每张图像匹配到其对应的标题，而非其他 N-1 个干扰项。监督信号是"这两个东西属于一起；其他 N-1 个则不匹配。"无需类别标签，无需人工标注。只需一个对比损失。

由此产生的嵌入空间超越了 CLIP 的设计初衷。ImageNet 零样本之所以可行，是因为"一张猫的照片"在嵌入空间中靠近从未被显式标注为"猫"的猫的图片。这是催生 2026 年所有 VLM 的核心赌注。

## 核心概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，每个标题输出一个 D 维向量。

两个编码器都将输出归一化为单位长度。相似度计算为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于 N 个（图像，标题）配对的批次，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习的温度参数（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 使用关于行和列的对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正例是其对应标题
loss_t2i = CE(S^T, labels=identity)   # 每个标题的正例是其对应图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。交叉熵中的 Softmax 迫使每张图像与其对应标题的匹配度高于批次中其他所有标题。"负例"是批次中的所有其他项。更大的批次 = 更多负例 = 更强的信号。CLIP 在 batch 32k 下训练；规模至关重要。

### 温度参数

`tau` 控制 Softmax 的尖锐程度。低 tau → 分布更尖锐，产生硬负例挖掘效果。高 tau → 分布更平滑，所有样本均有贡献。CLIP 学习 log(1/tau)，并通过裁剪防止坍塌。SigLIP 2 固定初始 tau，改用可学习的偏置。

### 为什么 Sigmoid 扩展性更好（SigLIP）

Softmax 需要同步整个相似度矩阵。在分布式训练中，你必须通过 all-gather 将所有嵌入同步到每个副本，然后执行 Softmax。通信开销与 world size 呈二次方关系。

SigLIP 用逐元素 Sigmoid 替换了 Softmax：对于每对 `(i, j)`，损失是对"这些是否是匹配的配对？"的二分类。正例标签是对角线，其余均为负例。损失函数为：

```
L = -1/N * Σ(i,j) [ y_ij * log(sigmoid(S[i,j])) + (1-y_ij) * log(sigmoid(-S[i,j])) ]
```

`y_ij = 1` 当 `i == j`，否则为 0。每对的损失相互独立。无需 all-gather。每个 GPU 计算其局部块并求和。SigLIP 2 可扩展到 batch 32k-512k，成本较低，而 CLIP 需要按比例增加通信开销。

### 零样本分类

给定 N 个类别名称，为每个类别构建文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。对余弦相似度取 argmax = 预测类别。无需在目标类别上训练。

提示模板很关键。CLIP 原始论文使用了每个类别 80 个模板（普通、艺术、照片、绘画等），并对嵌入取平均。ImageNet 准确率提升 3 个百分点。现代使用通常选取一到两个模板。

### 线性探测与微调

零样本是基线。线性探测（在冻结的 CLIP 特征之上训练一个线性层用于你的目标类别）在域内任务上优于零样本。全量微调在域内优于线性探测，但可能损害零样本迁移能力。三种范式，三种权衡。

### SigLIP 2：NaFlex 与稠密特征

SigLIP 2（2025）新增了：
- NaFlex：单模型处理可变长宽比和分辨率。
- 更好的稠密特征，用于分割和深度估计，目标是作为 VLM 中的冻结骨干。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅限英语。
- 10 亿参数规模，而 CLIP 最高为 4 亿。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍是纯图像-文本检索场景的默认选择，特别是在 LAION-2B 训练分布与你的查询模式匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的思路，18 亿对规模，90% 为噪声数据。证明了噪声数据可规模化。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，多种规模，是首选的开源检查点。EVA-CLIP：从掩码图像建模初始化；对 VLM 而言是强骨干网络。BASIC：Google 的 CLIP+ALIGN 混合模型。同属一个家族，不同数据与调参。

### 零样本上限

CLIP 类模型在 ImageNet 零样本上上限约为 76%（CLIP-G、OpenCLIP-G）。超越此极限需要更多数据（SigLIP 2 可达 80%+）或架构改进（监督头、更多参数）。该基准正在趋近饱和；真正价值在于下游 VLM 所消费的嵌入空间。

```figure
multimodal-fusion
```

## 实践应用

`code/main.py` 实现了：

1. 玩具双编码器（基于哈希的图像特征、文本字符特征），无需 numpy 即可观察 InfoNCE 的形状。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定性）。
3. 用于对比的 Sigmoid 成对损失。
4. 零样本分类流程：计算与一组文本提示的余弦相似度，取 argmax 作为预测结果。

运行它并观察损失曲线。绝对数值是玩具级的；曲线形状与真实 CLIP 训练器输出的相符。

## 交付物

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它会使用 CLIP 模板构建文本提示，用指定检查点（如 `openai/clip-vit-large-patch14`）嵌入双方，返回 top-1 / top-5 预测及相似度分数。该技能不会针对提示列表之外的类别做出断言。

## 练习

1. 手工实现 4 对批次的 InfoNCE。构造 4×4 相似度矩阵，执行 Softmax，提取对角线，计算交叉熵。用 Python 实现与此手工计算进行验证。

2. SigLIP 除了温度外还使用偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批次存在大类不平衡（每行负例远多于正例）时，`b` 起到什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫 vs 狗构建零样本分类器。尝试两种提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。多模板集成是否优于单模板？

4. 计算 512 GPU、batch 32k 下 Softmax InfoNCE 与 Sigmoid 成对损失的通信开销。哪个随 O(N) 扩展，哪个随 O(N²) 扩展？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 扩展律论文（arXiv:2212.07143，Cherti 等）。从图中复现其数据扩展结论：在固定模型规模下，ImageNet 零样本准确率与训练数据规模之间是否存在对数线性关系？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| InfoNCE | "对比损失" | 对批次相似度矩阵的交叉熵；每个项的正例是其配对项，负例是其余所有项 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对二元交叉熵；无 Softmax，无 all-gather，在分布式训练中低成本扩展 |
| 温度 | "tau" | 在 Softmax/Sigmoid 之前缩放 logits 的标量；控制分布的尖锐程度 |
| 零样本 | "无需微调的分类" | 使用文本提示构建类别嵌入，通过余弦相似度进行分类；无需在目标类别上训练 |
| 提示模板 | "a photo of a ..." | 围绕类别名称的文本框架；影响零样本准确率 1-5 个百分点 |
| 双编码器 | "双塔" | 一个图像编码器 + 一个文本编码器，输出到共享 D 维空间 |
| 硬负例 | "困难干扰项" | 与正例足够相似，以至于模型必须努力才能区分的负例 |
| 线性探测 | "冻结 + 一层" | 仅在冻结特征之上训练线性分类器；测量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力：接收任意长宽比和分辨率的图像而无需重采样 |
| 温度缩放 | "log 参数化 tau" | CLIP 参数化为 `log(1/tau)` 以获得良好的梯度行为；通过裁剪防止趋近于零的坍塌 |

## 延伸阅读

- [Radford 等 — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai 等 — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等 — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用噪声网络数据扩展规模。
- [Cherti 等 — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 扩展律。
