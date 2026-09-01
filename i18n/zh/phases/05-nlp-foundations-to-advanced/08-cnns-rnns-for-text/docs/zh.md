```markdown
# CNNs 和 RNNs 用于文本处理

> 卷积学习 n-grams。循环网络负责记忆。两者都被注意力机制超越。但在资源受限的硬件上依然有用。

**类型：** 构建
**语言：** Python
**前置知识：** 第 3 阶段 · 11（PyTorch 简介）、第 5 阶段 · 03（词嵌入）、第 4 阶段 · 02（从零实现卷积）
**时间：** 约 75 分钟

## 问题背景

TF-IDF 和 Word2Vec 生成的扁平向量忽略了词序。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时承载着关键信号。

在 transformer 出现之前，两类架构填补了这一空白。

**用于文本的卷积神经网络（TextCNN）。** 对词嵌入序列应用 1D 卷积。宽度为 3 的滤波器是可学习的三元组检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）以检测多尺度模式。通过最大池化得到固定大小的表示。结构简单、可并行、速度快。

**循环神经网络（RNN、LSTM、GRU）。** 一次处理一个 token，维护一个向前传递信息的隐藏状态。具有序列性、携带记忆、支持灵活输入长度。在 2014 到 2017 年间主导了序列建模，然后注意力机制出现了。

本课将构建这两种模型，然后指出催生注意力机制的缺陷。

## 核心概念

**TextCNN**（Kim，2014）。词元先经过嵌入。宽度为 `k` 的 1D 卷积在嵌入序列的连续 `k`-gram 上滑动，产生特征图。对该特征图进行全局最大池化，选取最强激活值。拼接来自多个滤波器宽度的最大池化输出。送入分类器头。

为什么有效。一个滤波器是一个可学习的 n-gram。最大池化具有位置不变性，因此"not good"在评论开头或中间都会激活同一个特征。三种宽度各 100 个滤波器，共 300 个可学习的 n-gram 检测器。训练是并行的；无序列依赖。

**RNN。** 在每个时间步 `t`，隐藏状态为 `h_t = f(W * x_t + U * h_{t-1} + b)`。在不同时间步共享 `W`、`U`、`b`。时间 `T` 处的隐藏状态是整个前缀的摘要。对于分类任务，对 `h_1 ... h_T` 进行池化（最大、均值或最后状态）。

普通 RNN 面临梯度消失问题。**LSTM** 通过门控机制决定遗忘什么、存储什么、输出什么，使梯度在长序列中稳定传播。**GRU** 将 LSTM 简化为两个门；性能相近但参数更少。

**双向 RNN** 运行一个正向 RNN 和一个反向 RNN，拼接隐藏状态。每个 token 的表示都能看到左右上下文。对标注任务至关重要。

```figure
rnn-unroll
```

## 动手实现

### 步骤 1：PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。池化后的输出大小固定，与输入长度无关。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列做最大池化，而不是取最后一个状态。对于分类任务，最大池化通常优于取最后一个隐藏状态，因为在长序列末尾的信息往往主导最后一个状态。

### 步骤 3：梯度消失演示（直觉理解）

不带门控的普通 RNN 无法学习长距离依赖。考虑一个简单的任务：判断 token `A` 是否出现在序列中的任意位置。如果 `A` 在第 1 个位置，序列长度为 100，那么损失的回传梯度必须经过 99 次循环权重的乘法。如果权重小于 1，梯度会消失。如果大于 1，梯度会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 当权重=0.9、经过 100 步时：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第 100 步到第 1 步的梯度实际上为零。
```

LSTM 通过 **细胞状态** 解决了这个问题，细胞状态通过网络时只有加法交互（遗忘门以乘法缩放它，但梯度仍能沿"高速公路"流动）。GRU 以更少的参数实现了类似效果。两者都能在 100+ 步的序列上实现稳定训练。

### 步骤 4：为什么这仍然不够

即使有 LSTM，仍有三个问题持续存在。

1. **序列瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 次串行的前向/反向传播步骤。无法在时间维度上并行化。
2. **编码器 - 解码器设置中的固定大小上下文向量。** 解码器只能看到编码器的最终隐藏状态，该状态已对整个输入进行了压缩。长输入会丢失细节。第 09 课将直接讲解此问题。
3. **远距离依赖精度上限。** LSTM 比普通 RNN 表现更好，但在跨越 200+ 步传播特定信息方面仍然困难。

注意力机制解决了以上三个问题。Transformer 完全放弃了循环结构。第 10 课是转折。

## 应用实践

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已可用于生产环境。训练代码为标准写法。

Hugging Face 提供了预训练嵌入，可直接用作输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用场景检查清单。

- **边缘设备 / 端侧推理。** 使用 GloVe 嵌入的 TextCNN 比 transformer 小 10 到 100 倍。如果部署目标是手机，这就是首选技术栈。
- **流式 / 在线分类。** RNN 一次处理一个 token；transformer 需要完整序列。对于实时传入的文本，LSTM 仍然占优。
- **轻量基线模型。** 对新任务快速迭代。在 CPU 上 5 分钟内即可训练一个 TextCNN。
- **数据有限时的序列标注。** BiLSTM-CRF（第 06 课）仍然是适用于 1k-10k 标注句子的生产级 NER 架构。

其余情况一律选择 transformer。

## 交付成果

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: 根据给定约束条件选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束条件（任务、数据量、延迟预算、部署目标、计算预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、transformer 微调，或"使用预训练 transformer 作为冻结编码器 + 小型分类头"。
2. 嵌入输入：随机初始化、冻结的 GloVe / fastText，或上下文相关的 transformer 嵌入。
3. 训练方案（5 行以内）：优化器、学习率、批次大小、训练轮数、正则化。
4. 一个监控指标。对于 RNN/CNN 模型：缺乏注意力机制意味着它们会遗漏长距离依赖；检查按序列长度分组的准确率。对于 transformer：学习率过高会导致微调崩溃；检查训练损失。

当标注样本不足约 500 条时，拒绝推荐微调 transformer，除非已证明 TextCNN / BiLSTM 基线已达到性能 plateau。标记边缘部署需优先考虑架构选择。
```

## 练习

1. **简单。** 在 3 类玩具数据集（自行构造）上训练 TextCNN。验证滤波器宽度组合 (2, 3, 4) 的平均 F1 优于单一宽度 (3)。
2. **中等。** 为 LSTM 分类器实现最大池化、均值池化和末状态池化三种方式。在小型数据集上比较，记录哪种池化方式最优，并提出假设解释原因。
3. **困难。** 构建 BiLSTM-CRF NER 标注器（结合第 06 课和本课内容）。在 CoNLL-2003 上训练。与第 06 课的纯 CRF 基线以及 BERT 微调版本进行比较。报告训练时间、内存占用和 F1 分数。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| TextCNN | 用于文本的 CNN | 在词嵌入上堆叠 1D 卷积并用全局最大池化的结构。Kim (2014)。 |
| RNN | 循环神经网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控循环网络 | 增加输入门 / 遗忘门 / 输出门 + 细胞状态。在长序列上稳定训练。 |
| GRU | 简化的 LSTM | 两个门而非三个。精度相近，参数更少。 |
| 双向 | 两个方向 | 正向 + 反向 RNN 拼接。每个 token 都能看到其上下文的两侧。 |
| 梯度消失 | 训练信号衰减 | 普通 RNN 中反复乘以小于 1 的权重，导致早期时间步的梯度趋近于零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 原论文。八页。通俗易懂。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 原论文。异常清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 对所有人变得可理解的图解文章。
```
