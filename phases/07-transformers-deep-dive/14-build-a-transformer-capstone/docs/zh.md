# 从零开始构建 Transformer —— 结业项目

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**先修要求：** 第 7 阶段 · 01 至 13 课。请勿跳过。
**时间：** 约 120 分钟

## 问题陈述

你已经读过了每一篇论文。你实现了注意力机制、多头拆分、位置编码、编码器和解码器模块、BERT 和 GPT 的损失函数、MoE、KV 缓存。现在，让它们在一个真实的任务上协同工作。

结业项目：在字符级语言建模任务上端到端地训练一个小型仅解码器的 Transformer。它阅读莎士比亚作品。它生成新的莎士比亚风格文本。它足够小，可以在笔记本电脑上在 10 分钟内完成训练。它足够准确，换上更大的数据集和更长的训练时间就能得到一个真正的语言模型。

这是本课程的“nanoGPT”。它并非原创——Karpathy 2023 年的 nanoGPT 教程是每位学生至少编写一次的参考实现。我们借鉴其结构，并根据我们已学内容进行调整。

## 概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构，附有注释：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们交付的内容

- `GPTConfig` —— 一个配置所有超参数的地方。
- `MultiHeadAttention` —— 因果的、批处理的，可选的 Flash 式路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代的 FFN（前馈网络）。
- `Block` —— 预范数、残差包裹的注意力 + FFN。
- `GPT` —— 嵌入、堆叠的模块、语言模型头、`generate()`。
- 带有 AdamW、余弦学习率、梯度裁剪的训练循环。
- 基于莎士比亚文本的字符级分词器。

### 我们未交付的内容

- **RoPE** —— 在第 4 课中概念性地实现了。这里我们为求简洁使用可学习的位置嵌入。练习要求你替换为 RoPE。
- **生成时的 KV 缓存** —— 每个生成步骤都会重新计算整个前缀的注意力。较慢但更简单。练习要求你添加 KV 缓存。
- **Flash Attention** —— PyTorch 2.0+ 会在输入匹配时自动分发；我们使用 `F.scaled_dot_product_attention`。
- **MoE** —— 每个模块使用单个 FFN。你在第 11 课中见过 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，一个 4 层、4 头、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从约 4.2（随机）下降到约 1.5，耗时约 6 分钟。
- 采样输出看起来具有莎士比亚风格：古语词、换行符、诸如“ROMEO:”的专有名词开始出现。
- 验证损失（留出的最后 10% 文本）紧密跟踪训练损失；在此规模/预算下没有过拟合。

## 构建它

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。参见 `code/main.py`。脚本处理以下事项：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 训练/验证集按 90/10 比例划分。
- 在支持的硬件上使用 bf16 自动混合精度的训练循环。
- 训练完成后的采样。

### 步骤 1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。词汇量极小。适合 4 字节的 `vocab_size`。无需 BPE，没有分词器的麻烦。

### 步骤 2：模型

参见 `code/main.py`。该模块是来自第 5 课的教科书式内容——预范数、RMSNorm、SwiGLU、因果 MHA。参数量（4/4/128 配置）：约 80 万。

### 步骤 3：训练循环

获取长度为 256 个 token 窗口的随机批次。前向传播。移位一步的交叉熵损失。反向传播。AdamW 步进。记录日志。重复。

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

给定一个提示，重复进行前向传播，从 top-p 采样 logits，追加结果，继续。在 500 个 token 后停止。

### 步骤 5：阅读输出

2,000 步之后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

并非真正的莎士比亚。但具有莎士比亚的形态。对于笔记本电脑上约 80 万参数和 6 分钟的训练来说，这是一个明显的胜利。

## 使用它

这个结业项目是一个参考架构。有三种扩展方式可以将其用于实际用途：

1. **更换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇量大小从 65 跃升至约 50,000。模型容量需要相应扩大以补偿。
2. **在更大的语料库上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单个 A100 上处理 100 亿个 token，一个 1.25 亿参数的 GPT 需要约 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习将引导你逐一完成。

最终将得到一个 1.25 亿参数的 GPT，能够生成流利的英语。不是前沿模型。但相同的代码路径——只是更大——正是 Karpathy、EleutherAI 和 Allen 研究所在 2026 年用于训练研究检查点的方法。

## 交付它

参见 `outputs/skill-transformer-review.md`。该技能检查一份从零开始构建的 Transformer 实现，验证其在之前所有 13 课内容上的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证你训练的模型最终步的验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000——验证损失是否持续改善？
2. **中等。** 用 RoPE 替换可学习的位置嵌入。将旋转应用于 `MultiHeadAttention` 内部的 Q 和 K。进行训练并验证验证损失至少同样低。
3. **中等。** 在采样循环中实现 KV 缓存。分别使用和不使用缓存生成 500 个 token。在笔记本电脑上，实际运行时间应能提升 5–20 倍。
4. **困难。** 为模型添加第二个头，用于预测下一个的下一个 token（来自 DeepSeek-V3 的 MTP——多 token 预测）。进行联合训练。它有帮助吗？
5. **困难。** 将每个模块的单个 FFN 替换为一个 4 专家的 MoE。路由器 + top-2 路由。观察在活动参数量匹配的情况下，验证损失如何变化。

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|-----------------|-----------------------|
| nanoGPT | “Karpathy 的教程仓库” | 极简的仅解码器 Transformer 训练代码，约 300 行；权威参考。 |
| tinyshakespeare | “标准的玩具语料库” | 约 1.1 MB 的文本；2015 年以来每个字符级语言模型教程都使用它。 |
| 绑定嵌入 | “共享输入/输出矩阵” | 语言模型头权重 = token 嵌入矩阵的转置；节省参数，提升质量。 |
| bf16 自动混合精度 | “训练精度技巧” | 前向/反向传播使用 bf16，优化器状态保持 fp32；2021 年以来的标准做法。 |
| 梯度裁剪 | “防止峰值” | 将全局梯度范数限制在 1.0；防止训练崩溃。 |
| 余弦学习率调度 | “2020 年后的默认选择” | 学习率线性上升（预热），然后以余弦形状衰减至峰值的 10%。 |
| MFU | “模型浮点运算利用率” | 实现的浮点运算 / 理论峰值；2026 年，密集模型 40%，MoE 模型 30% 算是强劲。 |
| 验证损失 | “留出损失” | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典的带注释实现。