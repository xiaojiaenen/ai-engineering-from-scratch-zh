# 视觉Transformer与Patch-Token基础范式

> 在实现多模态功能之前，图像必须先转化为Transformer能处理的token序列。2020年的ViT论文通过16x16像素块、线性投影和位置嵌入解决了这个问题。五年后的2026年，所有前沿模型（原生2576px的Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）依然沿用此范式——编码器从ViT演进到DINOv2再到SigLIP 2，引入了寄存器token，位置编码升级为2D-RoPE，但基本范式依然稳固。本课将端到端解读patch-token处理流程，并使用标准库Python实现它，为第12阶段建立具体的"视觉token"认知模型。

**类型：** 学习
**语言：** Python（标准库，patch分词器与几何计算器）
**前置课程：** 第7阶段（Transformer）、第4阶段（计算机视觉）
**时间：** 约120分钟

## 学习目标

- 将HxWx3图像转换为带有正确位置编码的patch token序列
- 计算给定（patch尺寸、分辨率、隐藏维度、深度）配置的ViT序列长度、参数量和浮点运算量
- 说明使ViT从2020年研究原型发展到2026年生产方案的三大升级：自监督预训练（DINO/MAE）、寄存器token和原生分辨率打包
- 针对下游任务选择CLS池化、均值池化或寄存器token方案

## 问题背景

Transformer处理的是向量序列。文本本身就是序列（字节或token）。图像是三维网格（二维空间+RGB通道）——并非序列。若将每个像素展平，224x224的RGB图像将产生150,528个token，此时自注意力的计算复杂度（序列长度的平方）将不可接受。

2020年前的做法是在前端附加CNN特征提取器：ResNet生成7x7×2048维的特征图，将这49个token输入Transformer。此方法有效，但继承了CNN的归纳偏置（平移等变性、局部感受野），且无法发挥Transformer对规模的适应能力。

Dosovitskiy等人（2020年）提出一个直接的问题：能否跳过CNN？将图像分割为固定大小的块（如16x16像素），线性投影每个块为向量，添加位置嵌入，然后将序列输入标准Transformer。这在当时被视为离经叛道——没有卷积的视觉处理。但在充足数据（JFT-300M，后来是LAION）支撑下，它在ImageNet上击败了ResNet并持续改进。

到2026年，ViT范式已成为无可争议的基石。所有开源VLM的视觉塔都是其衍生模型（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题已从"是否使用patch"转变为"选择什么patch尺寸、什么分辨率调度、什么预训练目标、什么位置编码"。

## 核心概念

### Patch即token

给定形状为`x`的图像`(H, W, 3)`和patch尺寸`P`，可将图像分割为`(H/P) x (W/P)`个互不重叠的patch网格。每个patch是`P x P x 3`的像素立方体。将每个立方体展平为`3 P^2`向量，应用共享线性投影`W_E`（形状为`(3 P^2, D)`），将每个patch映射到模型隐藏维度`D`。

以ViT-B/16标准配置为例：
- 分辨率224，patch尺寸16 → 网格14x14 → 196个patch token
- 每个patch包含`16 x 16 x 3 = 768`个像素值，投影到`D = 768`维
- 添加可学习的`[CLS]` token → 序列长度197

Patch投影在数学上等价于二维卷积（核尺寸`P`，步长`P`，输出通道`D`）。生产代码正是这样实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"是概念性表述，"卷积核"是高效实现方式。

### 位置嵌入

Patch本身没有顺序——Transformer将其视为无序集合。早期ViT添加可学习的一维位置嵌入（每个位置一个768维向量，共197个）。此方法有效，但将模型绑定到训练分辨率：推理时若改变网格尺寸需插值位置表。

现代视觉骨干采用2D-RoPE（Qwen2-VL的M-RoPE、SigLIP 2的默认方案）或分解式二维位置编码。2D-RoPE根据patch的（行，列）索引旋转查询和键向量，使模型从旋转角度推断相对二维位置。无需位置表，模型在推理时可处理任意网格尺寸。

### CLS token、池化输出与寄存器token

如何获得图像级表示？三种方案并存：

1. `[CLS]` token：在patch序列前添加可学习向量。所有Transformer块处理后，CLS token的隐藏状态即为图像表示。继承自BERT，被原始ViT和CLIP采用。
2. 均值池化：对patch token的输出隐藏状态取平均。被SigLIP、DINOv2和大多数现代VLM采用。
3. 寄存器token：Darcet等人（2023年）发现，未使用显式汇聚token训练的ViT会产生高范数"伪影"patch，干扰自注意力。添加4-16个可学习寄存器token可吸收此负载，提升密集预测质量（分割、深度估计）。DINOv2和SigLIP 2均内置寄存器。

方案选择影响下游任务。CLS适用于分类任务。对于将patch token输入LLM的VLM，则完全跳过池化——每个patch成为LLM输入token。寄存器token在交接前被丢弃（它们是脚手架而非内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020年ViT通过JFT-300M监督分类预训练。很快被以下方案取代：

- CLIP（2021年）：4亿图文对的对比学习（课程12.02）
- MAE（2021年，He等）：掩码75% patch并重建像素。自监督，纯图像方案
- DINO（2021年）/ DINOv2（2023年）：师生模型自蒸馏，无需标签或描述文本。2023年的DINOv2 ViT-g/14是最强的纯视觉骨干，是"密集特征"场景的默认选择
- SigLIP / SigLIP 2（2023年，2025年）：带sigmoid损失和NaFlex原生宽高比的CLIP。2026年开源VLM的主流视觉塔（Qwen、Idefics2、LLaVA-OneVision）

预训练选择决定了骨干的擅长领域：CLIP/SigLIP用于文本语义匹配，DINOv2用于密集视觉特征，MAE作为下游微调的起点。

### 缩放规律

ViT缩放研究（Zhai等，2022年）确立了ViT质量在模型规模、数据规模和计算量上的可预测规律。在固定计算量下：
- 更大模型 + 更多数据 → 更优质量
- Patch尺寸是序列长度与保真度的调节杠杆。Patch 14（DINOv2/SigLIP SO400m的典型配置）比patch 16产生更多token；对OCR和密集任务更有利，但速度更慢
- 分辨率是另一关键杠杆。从224提升到384再到512几乎总是有益的，但FLOPs呈平方增长

ViT-g/14（10亿参数，patch 14，分辨率224 → 256 token）和SigLIP SO400m/14（4亿参数，patch 14）是2026年开源VLM的两个主力编码器。

### ViT参数量计算

完整计算过程见`code/main.py`。以ViT-B/16@224为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

加载检查点前应以这种方式估算所有ViT参数量。骨干规模决定了任何下游VLM的显存基线。

### 2026年生产配置

2026年大多数开源VLM搭载的编码器是原生分辨率（NaFlex）的SigLIP 2 SO400m/14，其特点包括：
- 4亿参数
- Patch尺寸14，默认分辨率384 → 每图像729个patch token
- 图像级任务采用均值池化；VQA任务中全部729个patch流入LLM
- 4个寄存器token，LLM交接前丢弃
- 支持原生宽高比的2D-RoPE（图像级缩放）

该配置的每个决策都有可追溯的论文依据。

## 实践应用

`code/main.py`是一个patch分词器和几何计算器，输入（图像高H、宽W、patch尺寸P、隐藏维度D、深度L）并输出：
- 分patch后的网格形状和序列长度
- 合成8x8像素玩具图像的token序列（演示展平+投影流程）
- 参数量分解（patch嵌入、位置嵌入、Transformer块、输出头）
- 目标分辨率下单次前向传播的FLOPs
- ViT-B/16@224、ViT-L/14@336、DINOv2 ViT-g/14@224、SigLIP SO400m/14@384的对比表

运行程序，将参数量与公开数据对比。尝试不同patch尺寸和分辨率，感受token数量的代价。

## 输出成果

本课将生成`outputs/skill-patch-geometry-reader.md`。输入ViT配置（patch尺寸、分辨率、隐藏维度、深度），即可输出token数量、参数量和显存预估（附带推导依据）。为VLM选择视觉骨干时使用此技能，可避免"token爆炸导致LLM上下文溢出"的意外情况。

## 练习

1. 计算Qwen2.5-VL原生1280x720输入、patch尺寸14时的patch-token序列长度。与仅CLS表示相比如何？

2. 1080p帧（1920x1080）在patch 14下产生多少token？以30FPS播放5分钟视频，总共多少视觉token？池化、帧采样、token合并哪种节省最多？

3. 用纯Python实现patch token的均值池化。验证对DINOv2输出的196个token进行均值池化的结果是否与模型`forward`返回的池化嵌入一致。

4. 阅读《视觉Transformer需要寄存器》第3节（arXiv:2309.16588）。用两句话描述寄存器吸收的伪影是什么，以及为何对下游密集预测重要。

5. 修改`code/main.py`支持patch-n'-pack：输入不同分辨率的图像列表，生成单个打包序列和分块对角注意力掩码。待学至课程12.06时进行验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|----------|
| Patch | "16x16像素块" | 输入图像中固定大小的不重叠区域；转化为一个token |
| Patch嵌入 | "线性投影" | 共享学习矩阵（或步长=P的Conv2d），将展平的patch像素映射到D维向量 |
| CLS token | "类别token" | 前置的可学习向量，其最终隐藏状态代表整幅图像；2026年已非必需 |
| 寄存器token | "汇聚token" | 额外可学习token，吸收ViT预训练中产生的高范数注意力伪影 |
| 位置嵌入 | "位置信息" | 每位置向量或旋转，使序列具备顺序感知能力；2D-RoPE是现代默认方案 |
| 网格 | "patch网格" | 给定分辨率和patch尺寸下，(H/P)×(W/P)的二维patch阵列 |
| NaFlex | "原生灵活分辨率" | SigLIP 2特性：单模型支持多种宽高比和分辨率，无需重新训练 |
| 骨干 | "视觉塔" | 预训练图像编码器，其patch-token输出作为VLM中LLM的输入 |
| 池化 | "图像级摘要" | 将patch token转化为单向量的策略：CLS、均值、注意力池化或基于寄存器的方案 |
| Patch 14 vs 16 | "精细网格 vs 粗略网格" | Patch 14产生更多token/图像，OCR保真度更高但速度更慢；patch 16是经典默认值 |

## 延伸阅读

- [Dosovitskiy等 — 一张图像相当于16x16个单词（arXiv:2010.11929）](https://arxiv.org/abs/2010.11929) — 原始ViT
- [He等 — 掩码自编码器是可扩展的视觉学习器（arXiv:2111.06377）](https://arxiv.org/abs/2111.06377) — MAE自监督预训练
- [Oquab等 — DINOv2（arXiv:2304.07193）](https://arxiv.org/abs/2304.07193) — 规模化自蒸馏，无需标签
- [Darcet等 — 视觉Transformer需要寄存器（arXiv:2309.16588）](https://arxiv.org/abs/2309.16588) — 寄存器token与伪影分析
- [Tschannen等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786) — 2026年默认视觉塔
- [Zhai等 — 视觉Transformer的缩放规律（arXiv:2106.04560）](https://arxiv.org/abs/2106.04560) — 经验缩放规律