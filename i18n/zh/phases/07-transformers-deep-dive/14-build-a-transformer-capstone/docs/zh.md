# 从零构建 Transformer —— 最终项目

> 十三节课。一个模型。不走捷径。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 · 01 至 13。不要跳过。
**时间：** 约 120 分钟

## 问题

你已经读完了所有论文。你已经实现了注意力机制、多头分解、位置编码、编码器和解码器块、BERT 和 GPT 损失、MoE、KV cache。现在把它们组合起来，在一个真实任务上协同工作。

最终项目：在字符级语言建模任务上端到端训练一个小型 decoder-only transformer。它阅读莎士比亚，它生成新的莎士比亚文本。它足够小，可以在笔记本电脑上 10 分钟内完成训练。它足够正确，换用更大的数据集和更长的训练时间，你就能得到一个真正的语言模型。

这就是本课程的"nanoGPT"。它并非原创——Karpathy 的 2023 nanoGPT 教程是每个学生至少会写一次的参考实现。我们拔高了难度并围绕已讲内容重新设计。

## 概念

![从零构建 Transformer 框图](../assets/capstone.svg)

带标注的架构：

```
输入 token (B, N)
   │
   ▼
token embedding + positional embedding  ◀── 第 04 课 (可选 RoPE)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 第 05 课
│  MultiHeadAttention (causal)      │  ◀── 第 03 课 + 第 07 课 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第 05 课
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
最终 RMSNorm
   │
   ▼
lm_head（与 token embedding 共享权重）
   │
   ▼
logits (B, N, V)
   │
   ▼
移位一位交叉熵                  ◀── 第 07 课
```

### 我们将交付

- `GPTConfig` —— 一个地方配置所有超参数。
- `MultiHeadAttention` —— causal，batched，可选 Flash-style 路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代 FFN。
- `Block` —— pre-norm，残差包裹的 attention + FFN。
- `GPT` —— embeddings，堆叠 blocks，LM head，generate()。
- 带 AdamW、cosine LR、梯度裁剪的训练循环。
- Shakespeare 文本上的字符级 tokenizer。

### 我们不交付

- RoPE —— 在第 04 课中已在概念上实现。这里为简单起见使用学习型位置编码。练习要求你换入 RoPE。
- 生成时的 KV cache —— 每个生成步骤都对完整前缀重新计算 attention。较慢但更简单。练习要求你添加 KV cache。
- Flash Attention —— PyTorch 2.0+ 在输入匹配时会自动分派；我们使用 `F.scaled_dot_product_attention`。
- MoE —— 每个 block 使用单一 FFN。你在第 11 课已见过 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，一个 4 层、4 头、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从 ~4.2（随机）收敛到约 ~1.5，耗时约 6 分钟。
- 采样输出看起来有莎士比亚风格：古语、换行、"ROMEO:" 等proper names 会浮现。
- 验证损失（留出的最后 10% 文本）与训练损失紧密跟踪；在此规模/预算下不过拟合。

```figure
n5-block-stack
```

## 构建

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。见 `code/main.py`。脚本处理：

- 缺失时下载 `tinyshakespeare.txt`（或读取本地副本）。
- Byte-level 字符 tokenizer。
- 90/10 的 train/val 划分。
- 在支持硬件上使用 bf16 autocast 的训练循环。
- 训练完成后采样。

### 步骤 1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个独特字符。极小词汇表。4 字节 vocab_size 足够。不需要 BPE，不需要 tokenizer 折腾。

### 步骤 2：模型

见 `code/main.py`。block 来自第 05 课的教科书实现 —— pre-norm，RMSNorm，SwiGLU，causal MHA。4/4/128 的参数规模约 800K。

### 步骤 3：训练循环

取一个长度为 256 的随机 token 窗口批次。前向传播。移位一位交叉熵。反向传播。AdamW step。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤 4：采样

给定 prompt，反复前向传播，从 top-p logits 中采样，追加并继续。500 个 token 后停止。

### 步骤 5：阅读输出

训练 2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但具有莎士比亚形态。对于约 800K 参数和笔记本电脑上 6 分钟的训练来说，这是一个明显的胜利。

## 使用

这个最终项目是一个参考架构。三个扩展方向让它变得真实：

1. **换掉 tokenizer。** 使用 BPE（如 `tiktoken.get_encoding("cl100k_base")`）。词汇量从 65 跳到约 50,000。模型容量需要相应放大以弥补。
2. **在更大的语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单卡 A100 上，125M 参数的 GPT 训练 10B token 约需 24 小时。
3. **加入 RoPE + KV cache + Flash Attention。** 下面的练习会逐步引导你完成每项。

最终成为一个 125M 参数的 GPT，生成流畅英语。不是前沿模型。但相同的代码路径——只是更大——就是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练研究 checkpoint 所用的代码。

## 交付

见 `outputs/skill-transformer-review.md`。该 skill review 评估从头构建的 transformer 实现是否在所有 13 节先修课上都正确。

## 练习

1. **简单。** 运行 `code/main.py`。验证你的训练模型最终步的验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000 —— val loss 是否持续下降？
2. **中等。** 用 RoPE 替换学习型位置编码。在 `MultiHeadAttention` 内部对 Q 和 K 施加旋转。训练并验证 val loss 至少相当。
3. **中等。** 在采样循环中实现 KV cache。有和无 cache 各生成 500 个 token。在笔记本电脑上 wall-clock 时间应提升 5–20 倍。
4. **困难。** 为模型添加第二个头，预测 next-plus-one token（MTP —— 来自 DeepSeek-V3 的多令牌预测）。联合训练。是否有帮助？
5. **困难。** 用 4-expert MoE 替换每个 block 的单一 FFN。Router + top-2 routing。观察在匹配 active parameters 下 val loss 的变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| nanoGPT | "Karpathy 的教程仓库" | 最小化 decoder-only transformer 训练代码，约 300 行；标准参考实现。 |
| tinyshakespeare | "标准玩具语料库" | 约 1.1 MB 文本；2015 年以来的每个 character-LM 教程都使用它。 |
| Tied embeddings | "共享输入/输出矩阵" | LM head 权重 = token embedding 矩阵的转置；节省参数，提升质量。 |
| bf16 autocast | "训练精度技巧" | 前向/反向在 bf16 下运行，优化器状态保持在 fp32；2021 年以来的标准做法。 |
| Gradient clipping | "阻止尖峰" | 将全局梯度范数上限设为 1.0；防止训练爆炸。 |
| Cosine LR schedule | "2020+ 默认选择" | LR 线性上升（warmup）后按余弦形状衰减至峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 实际 FLOPs / 理论峰值；2026 年 40% dense、30% MoE 算是不错。 |
| Val loss | "留集损失" | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典标注实现。
