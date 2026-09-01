# 视觉 Transformer（ViT）

> 图像是像素块的网格。句子是 token 的网格。同一个 transformer 都能处理。

**类型：** 构建
**语言：** Python
**先修：** 第 7 阶段 · 05（完整 Transformer）、第 4 阶段 · 03（CNN）、第 4 阶段 · 14（视觉 Transformer 导论）
**时间：** 约 45 分钟

## 问题背景

2020 年之前，计算机视觉就是卷积天下。ImageNet、COCO 和检测榜单上的所有 SOTA 模型都使用 CNN 骨干网络。Transformer 则是为语言设计的。

Dosovitskiy 等人（2020）的论文《An Image is Worth 16x16 Words》表明，你可以完全去掉卷积。将图像切成固定大小的块（patches），把每个块线性投影到嵌入向量，然后将序列喂给标准的 transformer encoder。在足够大的规模下（ImageNet-21k 预训练或更大），ViT 能够匹配甚至超越基于 ResNet 的模型。

ViT 开启了 2026 年更广泛的一个模式：一个架构，多种模态。Whisper 对音频分词，ViT 对图像分块，机器人用动作 token，视频用像素 token。Transformer 不挑 —— 给它一个序列，它就能学习。

到了 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了大部分视觉任务。CNN 仍然在边缘设备和延迟敏感任务上占优。其他一切场景，模型栈的某个位置总有一个 ViT。

## 核心概念

![图像 → 分块 → token → transformer](../assets/vit.svg)

### 步骤 1 —— 分块（patchify）

将一个 `H × W × C` 的图像拆分成 `N × (P·P·C)` 的平坦块序列。典型配置：`224 × 224` 图像，`16 × 16` 的块 → 196 个块，每个块包含 768 个值。

```
图像 (224, 224, 3) → 14 × 14 网格的 16x16x3 块 → 196 个长度为 768 的向量
```

块大小是关键杠杆。更小的块 = 更多 token、更好分辨率、但注意力计算成本是平方级的。更大的块 = 更粗糙、更便宜。

### 步骤 2 —— 线性嵌入

一个可学习的矩阵将每个平坦的块投影到 `d_model` 维度。这等价于核大小为 `P`、步长为 `P` 的卷积。在 PyTorch 中，这就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` —— 两行代码就能实现。

### 步骤 3 —— 插入 `[CLS]` token，添加位置嵌入

- 在序列开头插入一个可学习的 `[CLS]` token。它的最终隐藏状态用作图像表示，用于分类。
- 添加可学习的位置嵌入（原始 ViT）或正弦 2D 位置编码（后续变体）。
- 2024 年后，RoPE 扩展到 2D 位置，有时不需要显式的位置嵌入。

### 步骤 4 —— 标准 transformer 编码器

堆叠 L 个由 `LayerNorm → 自注意力 → 残差连接 → LayerNorm → MLP → 残差连接` 组成的块。与 BERT 完全相同。没有任何视觉专用层。这正是这篇论文的精髓所在。

### 步骤 5 —— 分类头

分类任务：取 `[CLS]` 的隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用块的嵌入向量。

### 重要的变体

| 模型 | 年份 | 改进 |
|------|------|------|
| ViT | 2020 | 原版，固定块大小，全图全局注意力 |
| DeiT | 2021 | 知识蒸馏；仅需 ImageNet-1k 即可有效训练 |
| Swin | 2021 | 层次化结构，带偏移窗口，计算复杂度为亚二次方 |
| DINOv2 | 2023 | 自监督学习（无需标签），2026 年最佳通用视觉特征 |
| ViT-22B | 2023 | 220 亿参数；遵循缩放定律 |
| SigLIP | 2023 | ViT + 语言配对，使用 sigmoid 对比损失 |
| SAM 3 | 2025 | 通用分割；ViT-Large + 可提示掩码解码器 |

### 为什么进展较慢

ViT 需要**大量数据**才能匹敌 CNN，因为它缺少 CNN 的归纳偏置（平移不变性、局部性）。没有超过 1 亿张标注图像或强力的自监督预训练，CNN 在相同计算预算下仍然胜出。DeiT 在 2021 年通过蒸馏技巧解决了这个问题；DINOv2 在 2023 年通过自监督学习永久解决了这个问题。

```figure
n5-patch-stream
```

## 动手实现

参见 `code/main.py`。纯标准库实现的分块 + 线性嵌入 + 验证检查。不包含训练 —— ViT 在任何实际规模下都需要 PyTorch 和数小时的 GPU 时间。

### 步骤 1：构造假图像

一个 24 × 24 的 RGB 图像，以 `(R, G, B)` 元组的行列表表示。我们使用 6×6 的块 → 得到 16 个块，每个块的嵌入向量为 108 维。

### 步骤 2：分块

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅扫描顺序：按行优先遍历网格。所有 ViT 都使用这种排序方式。

### 步骤 3：线性嵌入

将每个平坦的块乘以随机的 `(patch_flat_size, d_model)` 矩阵。验证在插入 `[CLS]` 后输出形状为 `(N_patches + 1, d_model)`。

### 步骤 4：计算 realistic ViT 的参数数量

打印 ViT-Base 的参数数量：12 层、12 个注意力头、d=768、patch=16。与 ResNet-50（约 2500 万）比较。ViT-Base 约 8600 万参数。ViT-Large 约 3.07 亿。ViT-Huge 约 6.32 亿。

## 使用预训练模型

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 个块
cls_emb = out[:, 0]                       # 图像表示向量
```

**DINOv2 嵌入是 2026 年的图像特征默认选择。** 冻结骨干网络，训练一个简单的分类头即可。适用于分类、检索、检测和图像描述。Meta 的 DINOv2 检查点在除文本外的所有视觉任务上都优于 CLIP。

**块大小的选择。** 小模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 部署配置

参见 `outputs/skill-vit-configurator.md`。该技能根据数据集规模、分辨率和计算预算，为新视觉任务选择合适的 ViT 变体和块大小。

## 练习题

1. **简单。** 运行 `code/main.py`。验证块的数量等于 `(H/P) * (W/P)`，且平坦块的维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置嵌入 —— 对每个块的 `row` 和 `col` 分别计算独立的正弦编码，然后拼接。将其喂入小型 PyTorch ViT，在 CIFAR-10 上比较与可学习位置嵌入的准确率。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上使用 4×4 块进行训练。测量测试准确率。现在对同样的 1,000 张图像进行 DINOv2 风格预训练（简化版：仅训练编码器从遮挡块中重建原始嵌入）。准确率是否提升？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch（块） | "视觉 transformer 的 token" | 图像中 `P × P × C` 区域的像素值展平向量。 |
| Patchify（分块） | "切分 + 展平" | 将图像切成不重叠的块，每个块展平为向量。 |
| `[CLS]` token | "图像摘要" | 插入在序列开头的可学习 token；其最终嵌入即为图像表示。 |
| Inductive bias（归纳偏置） | "模型的先验假设" | ViT 比 CNN 少的先验假设；需要更多数据来弥合差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师网络在无标签情况下训练。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的继任者" | ViT + 文本编码器，使用 sigmoid 对比损失训练；在同等计算下优于 CLIP。 |
| Swin | "窗口化 ViT" | 层次化 ViT，带局部注意力 + 偏移窗口；亚二次方复杂度。 |
| Register tokens | "2023 年的技巧" | 少量额外的可学习 token，用于吸收注意力热点；改善 DINOv2 特征质量。 |

## 延伸阅读

- [Dosovitskiy 等（2020）。An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) —— ViT 原论文。
- [Touvron 等（2021）。Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) —— DeiT。
- [Liu 等（2021）。Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) —— Swin。
- [Oquab 等（2023）。DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) —— DINOv2。
- [Darcet 等（2023）。Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) —— 修复 DINOv2 的 register token。
