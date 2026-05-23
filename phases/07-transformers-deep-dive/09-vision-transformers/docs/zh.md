# 视觉Transformer (ViT)

> 图像是图像块的网格。句子是token的网格。同一个Transformer可以同时处理两者。

**类型：** 构建
**语言：** Python
**先修知识：** 第7阶段 · 05（完整Transformer），第4阶段 · 03（卷积神经网络），第4阶段 · 14（视觉Transformer入门）
**时长：** ~45分钟

## 问题

2020年之前，计算机视觉意味着卷积。ImageNet、COCO和检测基准测试上的每个SOTA模型都使用CNN主干。Transformer则用于处理语言。

Dosovitskiy等人(2020)——"一张图像值16x16个词"——表明你可以完全抛弃卷积。将图像切成固定大小的块，将每个块线性投影到嵌入中，将序列输入标准的Transformer编码器。在足够的规模下（ImageNet-21k预训练或更大），ViT可以匹配或超越基于ResNet的模型。

ViT是2026年更广泛模式的开端：一种架构，多种模态。Whisper将音频token化。ViT将图像token化。机器人技术的动作token。视频的像素token。Transformer并不关心这些——给它一个序列，它就能学习。

到2026年，ViT及其衍生模型（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了视觉领域的大部分。CNN在边缘设备和延迟敏感任务上仍然胜出。其他一切都至少包含一个ViT组件。

## 核心概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 步骤1 — 分块

将一张`H × W × C`图像分割成`N × (P·P·C)`个展平的图像块序列。典型设置：`224 × 224`图像，`16 × 16`个图像块 → 196个包含768个值的图像块。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

图像块大小是关键杠杆。更小的图像块 = 更多token，更高的分辨率，但注意力成本呈二次方增长。更大的图像块 = 更粗糙，成本更低。

### 步骤2 — 线性嵌入

一个学习到的矩阵将每个展平的图像块投影到`d_model`维。等效于一个核大小为`P`、步长为`P`的卷积。在PyTorch中，这实际上就是`nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——只需两行代码。

### 步骤3 — 添加`[CLS]` token，加上位置嵌入

- 添加一个可学习的`[CLS]` token。它的最终隐藏状态将用作分类任务的图像表示。
- 添加可学习的位置嵌入（原始ViT）或正弦二维位置编码（后续变体）。
- 在2024年后，RoPE被扩展到二维用于位置编码，有时不使用显式的位置嵌入。

### 步骤4 — 标准Transformer编码器

堆叠L个`LayerNorm → Self-Attention → + → LayerNorm → MLP → +`块。与BERT完全相同。没有视觉特定的层。这是该论文在教学意义上的核心结论。

### 步骤5 — 头

用于分类：取`[CLS]`的隐藏状态 → 线性层 → softmax。对于DINOv2或SAM，则丢弃`[CLS]`，直接使用图像块嵌入。

### 重要的变体

| 模型 | 年份 | 变化 |
|-------|------|------|
| ViT | 2020 | 原始模型。固定图像块大小，全局注意力。 |
| DeiT | 2021 | 知识蒸馏；仅用ImageNet-1k即可训练。 |
| Swin | 2021 | 层级结构，使用滑动窗口。固定次二次方成本。 |
| DINOv2 | 2023 | 自监督（无需标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 220亿参数；缩放定律适用。 |
| SigLIP | 2023 | ViT + 语言对，使用sigmoid对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示的掩码解码器。 |

### 为何需要时间

ViT需要*大量*数据才能匹配CNN，因为它没有CNN的归纳偏置（平移不变性、局部性）。如果没有超过1亿张标记图像或强大的自监督预训练，在相同计算条件下CNN仍然胜出。DeiT在2021年通过蒸馏技巧解决了这个问题；DINOv2在2023年通过自监督永久地解决了这个问题。

## 动手构建

参见`code/main.py`。纯标准库实现分块 + 线性嵌入 + 合理性检查。没有训练——任何实际规模的ViT都需要PyTorch和数小时的GPU时间。

### 步骤1: 伪图像

一张24 × 24的RGB图像，表示为`(R, G, B)`元组行的列表。我们使用6×6的图像块 → 16个图像块，每个有108维的嵌入向量。

### 步骤2: 分块

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

光栅顺序：按行优先顺序遍历网格。所有ViT都使用这种排序。

### 步骤3: 线性嵌入

将每个展平的图像块乘以一个随机的`(patch_flat_size, d_model)`矩阵。验证在添加`[CLS]`后，输出形状为`(N_patches + 1, d_model)`。

### 步骤4: 计算真实ViT的参数量

打印ViT-Base的参数数量：12层，12个头，d=768，图像块大小=16。与ResNet-50（约2500万）比较。ViT-Base约为8600万。ViT-Large约3.07亿。ViT-Huge约6.32亿。

## 使用它

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2嵌入是2026年默认的图像特征提取器。** 冻结主干网络，训练一个微小的头部。可用于分类、检索、检测、图像描述生成。Meta的DINOv2模型在所有非文本视觉任务上都优于CLIP。

**图像块大小选择。** 小型模型使用16×16（ViT-B/16）。密集预测（分割）使用8×8或14×14（SAM, DINOv2）。超大型模型使用14×14。

## 部署它

参见`outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算，为新的视觉任务选择ViT变体和图像块大小。

## 练习

1.  **简单。** 运行`code/main.py`。验证图像块数量等于`(H/P) * (W/P)`，展平的图像块维度等于`P*P*C`。
2.  **中等。** 实现二维正弦位置编码——为每个图像块的`row`和`col`独立生成两组正弦编码，然后拼接。将它们输入一个小型的PyTorch ViT，并在CIFAR-10上比较其与可学习位置嵌入的准确率。
3.  **困难。** 构建一个3层的ViT（PyTorch），在1000张MNIST图像上使用4×4的图像块进行训练。测量测试准确率。现在在相同的1000张图像上添加DINOv2预训练（简化版：仅训练编码器从掩码图像块预测原始图像块嵌入）。准确率是否提高？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Patch (图像块) | “视觉Transformer的token” | 图像某个`P × P × C`区域的像素值展平向量。 |
| Patchify (分块) | “切碎 + 展平” | 将图像切成不重叠的块，将每个块展平成一个向量。 |
| `[CLS]` token | “图像摘要” | 添加的可学习token；其最终嵌入是图像的表示。 |
| 归纳偏置 | “模型的假设” | ViT比CNN具有更少的先验；需要更多数据来弥补差距。 |
| DINOv2 | “自监督的ViT” | 使用图像增强和动量教师进行无标签训练。2026年最佳通用图像特征。 |
| SigLIP | “CLIP的继任者” | 使用sigmoid对比损失训练的ViT + 文本编码器；在相同计算条件下优于CLIP。 |
| Swin | “窗口化ViT” | 具有局部注意力和滑动窗口的层级ViT；次二次方复杂度。 |
| Register tokens | “2023年的技巧” | 少量额外的可学习token，可以吸收注意力汇聚点；改善DINOv2特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — 针对DINOv2的寄存器token修复方案。