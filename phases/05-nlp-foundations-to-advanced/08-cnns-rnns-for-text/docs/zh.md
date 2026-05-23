# 文本CNN与循环神经网络

> 卷积神经网络学习N元语法，循环神经网络具有记忆性。两者都被注意力机制超越，但在硬件受限场景下仍有价值。

**类型：** 构建项目
**语言：** Python
**前置知识：** 第三阶段·11（PyTorch入门）、第五阶段·03（词嵌入）、第四阶段·02（从零实现卷积）
**预计时间：** 约75分钟

## 问题背景

TF-IDF与Word2Vec生成的扁平向量忽略了词序。基于此构建的分类器无法区分`dog bites man`和`man bites dog`。词序有时承载着关键信号。

在Transformer架构出现之前，两类架构填补了这个空白：

**文本卷积神经网络（TextCNN）。** 在词嵌入序列上应用一维卷积。宽度为3的卷积核相当于可学习的三元语法检测器：它跨越三个词并输出得分。堆叠不同宽度（2、3、4、5）以检测多尺度模式。通过最大池化生成固定大小的表示。结构扁平、并行化、速度快。

**循环神经网络（RNN、LSTM、GRU）。** 逐个处理token，维护一个向前传递信息的隐藏状态。具有顺序性、记忆性，支持灵活输入长度。2014至2017年间主导序列建模，直到注意力机制出现。

本课将构建这两种架构，并揭示推动注意力机制发展的根本缺陷。

## 核心概念

**TextCNN**（Kim, 2014）。首先对token进行嵌入。宽度为`k`的一维卷积核在连续的`k`元嵌入语法上滑动，生成特征图。对该特征图进行全局最大池化以选取最强激活值。连接不同卷积宽度的最大池化输出，送入分类器头。

工作原理：卷积核相当于可学习的N元语法。最大池化具有位置不变性，因此无论"not good"出现在评论开头还是中部，都会触发相同特征。三个卷积宽度各使用100个滤波器，共300个可学习的N元语法检测器。训练过程并行化，无顺序依赖。

**循环神经网络。** 在每个时间步`t`，隐藏状态`h_t = f(W * x_t + U * h_{t-1} + b)`。跨时间共享`W`、`U`、`b`。时间`T`处的隐藏状态是整个前缀的摘要。分类时，对`h_1 ... h_T`进行池化（最大值、平均值或最后状态）。

普通RNN受梯度消失问题困扰。**LSTM**通过引入门控机制（决定遗忘、存储和输出的内容）稳定了长序列的梯度流动。**GRU**将LSTM简化为两个门控，以更少参数实现相近性能。

**双向RNN**同时运行正向和反向的RNN，连接两者隐藏状态。每个token的表示都能感知左右上下文。这对标注任务至关重要。

## 构建步骤

### 步骤1：使用PyTorch实现TextCNN

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

`transpose(1, 2)`将`[batch, seq_len, embed_dim]`重塑为`[batch, embed_dim, seq_len]`，因为`nn.Conv1d`将中间轴视为通道。池化输出具有固定尺寸，与输入长度无关。

### 步骤2：LSTM分类器

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

对序列进行最大池化，而非仅使用最后状态。分类任务中，最大池化通常优于取最后隐藏状态，因为长序列末端的信息往往主导最后状态。

### 步骤3：梯度消失演示（直观理解）

无门控的普通RNN无法学习长程依赖。考虑一个玩具任务：预测token`A`是否在序列中出现。如果`A`位于位置1且序列长度为100个token，损失函数的梯度需要通过99次循环权重乘法反向传播。若权重小于1，梯度会消失；若大于1，则会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM通过**细胞状态**解决此问题——该状态在网络中仅通过加法交互传递（遗忘门进行乘法缩放，但梯度仍沿"高速公路"流动）。GRU以更少参数实现类似效果。两者都能支持100+时间步的稳定训练。

### 步骤4：为何仍不足够

即使使用LSTM，仍存在三个问题：

1. **顺序瓶颈。** 在长度为1000的序列上训练RNN需要1000次串行的前向/反向传播，无法跨时间步并行化。
2. **编码器-解码器中的固定大小上下文向量。** 解码器仅能感知编码器的最终隐藏状态（压缩了整个输入信息）。长输入会丢失细节。第9课将直接探讨此问题。
3. **远距离依赖准确率上限。** LSTM优于普通RNN，但仍难以在200+步上传播特定信息。

注意力机制解决了这三个问题。Transformer完全抛弃了循环结构。第10课将介绍这一转折点。

## 应用指南

PyTorch的`nn.LSTM`、`nn.GRU`和`nn.Conv1d`已达到生产可用标准。训练代码是标准化的。

Hugging Face提供预训练嵌入，可直接作为输入层使用：

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

适用场景核查清单：

- **边缘/设备端推理。** 使用GloVe嵌入的TextCNN比Transformer小10-100倍。若部署目标是手机，这是首选技术栈。
- **流式/在线分类。** RNN逐个处理token；Transformer需要完整序列。对于实时输入的文本，LSTM仍有优势。
- **快速原型基准测试。** 在新任务上快速迭代。可在CPU上5分钟内训练TextCNN模型。
- **数据受限的序列标注。** BiLSTM-CRF（第6课）仍是处理1k-10k标注句子的生产级命名实体识别架构。

其他情况建议使用Transformer。

## 模型部署

保存为`outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **基础。** 在三分类玩具数据集（自行设计）上训练TextCNN。验证卷积宽度（2,3,4）组合在平均F1值上优于单一宽度（3）。
2. **进阶。** 为LSTM分类器实现最大池化、平均池化和最后状态池化。在小数据集上比较性能，记录最优池化方式并分析原因。
3. **挑战。** 结合第6课和本课内容，构建BiLSTM-CRF命名实体识别标注器。在CoNLL-2003数据集上训练，与第6课的纯CRF基线及BERT微调方案对比，报告训练时间、内存占用和F1值。

## 核心术语

| 术语 | 常见表述 | 实际含义 |
|------|----------|----------|
| TextCNN | 文本CNN | 基于词嵌入的一维卷积堆叠配合全局最大池化。Kim (2014)。 |
| RNN | 循环神经网络 | 在每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控循环单元 | 增加输入/遗忘/输出门及细胞状态，实现长序列稳定训练。 |
| GRU | 简化版LSTM | 两个门控替代三个。准确率相近，参数更少。 |
| 双向结构 | 双向处理 | 正向+反向RNN连接。每个token同时感知左右上下文。 |
| 梯度消失 | 训练信号衰减 | 普通RNN中权重多次乘积（<1）导致早期时间步梯度趋近于零。 |

## 扩展阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN原论文。八页篇幅，清晰可读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM原论文。逻辑异常清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 用图解让LSTM普及化的经典文章。