# Vision Transformers (ViT)

> 将图像切成 patches，把每个 patch 当作一个词，跑标准 transformer。不必回头。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 Lesson 02（Self-Attention）、Phase 4 Lesson 04（图像分类）
**时间：** 约 45 分钟

## 学习目标

- 从零实现 patch embedding、learned positional embedding、class token 和 transformer encoder block，构建一个最小化 ViT
- 解释为什么 ViT 曾被认为需要海量预训练数据，直到 DeiT 和 MAE 推翻了这一观点
- 从架构先验角度对比 ViT、Swin 和 ConvNeXt（无先验、局部窗口注意力、卷积 backbone）
- 使用 `timm` 在小型数据集上通过标准 linear-probe / fine-tune 流程微调预训练 ViT

## 背景

近十年来，卷积与计算机视觉几乎等价。CNN 具有强归纳偏置——局部性、平移等变性——无人认为它们能被替代。随后 Dosovitskiy 等人（2020）证明，将一个普通 transformer 直接应用于展平的图像 patch，完全不使用任何卷积操作，在大规模下也能匹配甚至超越最佳 CNN。

关键在于"大规模"。在 ImageNet-1k 上，ViT 输给了 ResNet；但在 ImageNet-21k 或 JFT-300M 上预训练后再在 ImageNet-1k 上微调，ViT 则胜出了。当时的结论是：transformer 缺乏有用的先验，但可以通过足够多的数据来学会这些先验。后续工作（DeiT、MAE、DINO）表明，配合正确的训练配方——强数据增强、自监督预训练、知识蒸馏——ViT 在小型数据集上同样表现良好。

到 2026 年，纯 CNN 在边缘设备上仍具竞争力（ConvNeXt 是最强之一），但 transformer 主导了其余所有领域：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 的块结构是最值得掌握的部分。

## 核心概念

### 处理流程

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["Patch 嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>(196, 768) tokens"]
    FLAT --> CAT["前置<br/>[CLS] token"]
    CAT --> POS["添加学习到的<br/>位置嵌入"]
    POS --> ENC["N 个 transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤：Patches → tokens → 注意力 → 分类器。每一种变体（DeiT、Swin、ConvNeXt、MAE 预训练）都只改动其中一两个步骤，其余保持不变。

### Patch Embedding

第一步的卷积是核心。核大小 16、步长 16，因此一张 224×224 的图像会被切分为 14×14 的网格，每个网格是 16×16 的 patch，再被线性投影到 768 维 embedding。这一个卷积同时完成了 patch 化和线性投影。

```
输入：  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, 无 padding)：
输出：  (768, 14, 14)
展平空间维度：(196, 768)
```

196 个 patches = 196 个 tokens。每个 token 的特征维度为 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### Class Token

一个前置到序列开头、可学习的全局向量：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   形状 (197, 768)
```

经过 N 个 transformer 块后，`[CLS]` 的输出即为全局图像表示。分类头仅读取这一个向量。

### Positional Embedding

Transformers 本身不具备空间位置感知能力。给每个 token 加上一个学习到的向量：

```
tokens = tokens + learned_pos_embedding   （形状同样为 (197, 768)）
```

该 embedding 是模型的可训练参数，通过梯度训练适配二维图像结构。正弦 2D 替代方案存在，但实际中很少使用。

### Transformer Encoder Block

标准结构：多头自注意力、MLP、残差连接、Pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP 为两层结构，含 GELU：Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每块 12 个注意力头，共计 86M 参数。

### 为何使用 Pre-LN

早期 transformer 使用 Post-LN（`x = LN(x + sublayer(x))`），在无需 warmup 的情况下难以训练超过 6-8 层。Pre-LN（`x = x + sublayer(LN(x))`）可以稳定训练更深的网络，无需 warmup。所有 ViT 和所有现代 LLM 均采用 Pre-LN。

### Patch 大小的权衡

- 16×16 patches → 196 个 tokens，为标准配置。
- 32×32 patches → 49 个 tokens，更快但分辨率更低。
- 8×8 patches → 784 个 tokens，更精细，但 O(n²) 的注意力计算成本急剧上升。

更大的 patch = 更少的 tokens = 更快但空间细节更少。SwinV2 在分层窗口中使用 4×4 patches。

### DeiT 的 ImageNet-1k 训练 ViT 配方

原始 ViT 需要 JFT-300M 才能击败 CNN。DeiT（Touvron 等人，2020）仅凭 ImageNet-1k 就将 ViT-B 训练到了 81.8% top-1，依靠以下四项改进：

1. 强数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. Stochastic depth（训练中随机丢弃整个块）。
3. 重复增强（同一个图像在每个 batch 中采样 3 次）。
4. 从 CNN 教师进行知识蒸馏（可选，可进一步提升准确率）。

所有现代 ViT 训练配方均源于 DeiT。

### Swin 与 ConvNeXt 对比

- **Swin**（Liu 等人，2021）—— 基于窗口的注意力。每个块仅在局部窗口内做注意力；交替块移位窗口以跨窗口混合信息。在保留注意力算子的同时引入了类似 CNN 的局部性先验。
- **ConvNeXt**（Liu 等人，2022）—— 重新设计的 CNN，在架构选择上与 Swin 对齐（深度可分离卷积、LayerNorm、GELU、倒瓶颈结构）。证明了差距不在于"注意力 vs 卷积"，而在于"现代训练配方 + 架构"。

2026 年，ConvNeXt-V2 和 Swin-V2 均达到生产级水准；选择取决于你的推理栈（ConvNeXt 在边缘设备上编译性更好）和预训练语料。

### MAE 预训练

掩码自编码器（Masked Autoencoder，He 等人，2022）：随机掩码 75% 的 patch，训练编码器仅处理可见的 25%，训练一个小解码器从编码器的输出重建被掩码的 patch。预训练完成后，丢弃解码器，仅微调编码器。

MAE 使得 ViT 仅凭 ImageNet-1k 即可有效训练，达到 SOTA，是当前默认的自监督预训练方案。

```figure
batchnorm-inference
```

## 动手实现

### Step 1：Patch Embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一次卷积、一次展平、一次转置。这就是完整的图像到 token 的转换步骤。

### Step 2：Transformer Block

Pre-LN、多头自注意力、含 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 负责头的拆分、缩放点积计算和输出投影。`batch_first=True` 保证形状为 `(N, seq, dim)`。

### Step 3：ViT 模型

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"输出: {vit(x).shape}")
print(f"参数: {sum(p.numel() for p in vit.parameters()):,}")
```

约 2.8M 参数——一个可在 CPU 上运行的微型 ViT。真实 ViT-B 为 86M；只需将 `dim=768, depth=12, num_heads=12` 传入同一类定义即可。

### Step 4：单图像推理验证

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"概率:  {logits.softmax(-1)}")
```

应能无报错运行，概率之和为 1。

## 如何使用

`timm` 提供了所有 ViT 变体的 ImageNet 预训练权重。一行搞定：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

2026 年，`timm` 是 vision transformer 的生产级首选。在同一 API 下支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等数十种模型。

对于多模态工作（图像 + 文本），`transformers` 提供了 CLIP、SigLIP、BLIP-2、LLaVA。这些模型的图像编码器均为 ViT 的变体。

## 交付成果

本课将产出：

- `outputs/prompt-vit-vs-cnn-picker.md` —— 根据数据集大小、算力和推理栈，在 ViT、ConvNeXt 或 Swin 之间做出选择的 prompt。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` —— 一项验证技能，用于检查 ViT 的 patch embedding 和 positional embedding 形状是否与模型预期的序列长度一致，捕获最常见的移植 bug。

## 练习

1. **（简单）** 打印上述微型 ViT 前向传播过程中每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` → patches `(N, 16, 192)` → 加 CLS `(N, 17, 192)` → 分类器输入 `(N, 192)` → 输出 `(N, num_classes)`。
2. **（中等）** 在 Lesson 4 的合成 CIFAR 数据集上微调预训练的 `timm` ViT-S/16，并与同数据集上的 ResNet-18 微调进行对比。报告训练时间和最终准确率。
3. **（困难）** 为微型 ViT 实现 MAE 预训练：掩码 75% 的 patch，训练编码器 + 一个小解码器以重建被掩码的 patch。在预训练前后评估在线性探测（linear probe）下的准确率。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|---------------|---------|
| Patch embedding | "第一个卷积" | 核大小 = 步长 = patch 大小的卷积；将图像转换为 token 嵌入的网格 |
| Class token | "[CLS]" | 前置到 token 序列开头的可学习向量；其最终输出为全局图像表示 |
| Positional embedding | "学习到的位置" | 添加到每个 token 的学习向量，使 transformer 知道每个 patch 的来源位置 |
| Pre-LN | "子层前的 LayerNorm" | 稳定的 transformer 变体：`x + sublayer(LN(x))`，而非 `LN(x + sublayer(x))` |
| Multi-head attention | "并行注意力" | 标准 transformer 注意力拆分为 `num_heads` 个独立子空间，最后拼接 |
| ViT-B/16 | "Base，patch 16" | 标准尺寸：dim=768, depth=12, heads=12, patch_size=16, image=224；约 86M 参数 |
| DeiT | "数据高效的 ViT" | 仅在 ImageNet-1k 上使用强数据增强训练的 ViT；证明了不一定需要大规模预训练数据集 |
| MAE | "掩码自编码器" | 自监督预训练：掩码 75% 的 patch 并重建；当前主流的 ViT 预训练方案 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) —— ViT 原始论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) —— 如何在 ImageNet-1k 上单独训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) —— MAE 预训练
- [timm documentation](https://huggingface.co/docs/timm) —— 生产中所有 vision transformer 的参考文档
