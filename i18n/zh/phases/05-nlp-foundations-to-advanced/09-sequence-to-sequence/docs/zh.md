```markdown
# 序列到序列模型（Sequence-to-Sequence Models）

> 两个 RNN 假装自己是翻译器。它们遇到的瓶颈，正是注意力机制存在的理由。

**类型：** 动手构建
**语言：** Python
**先修知识：** 第5阶段 · 08（用于文本的 CNN + RNN），第3阶段 · 11（PyTorch 入门）
**时间：** 约 75 分钟

## 问题所在

分类将一个变长序列映射为一个标签；翻译则将一个变长序列映射为另一个变长序列。输入和输出可能属于不同的词表、甚至是不同的语言，且长度并不保证对齐。

seq2seq 架构（Sutskever、Vinyals、Le，2014 年）用一个刻意保持简单的配方解决了这个问题。两个 RNN。一个读取源句并生成一个固定大小的上下文向量；另一个读取该向量，逐个 token 地生成目标句。和你第08课写的代码相同，只是组合方式不同。

值得学习有两个原因：第一，上下文向量的瓶颈是 NLP 中最有教学价值的失败案例，它解释了注意力机制和 Transformer 为何有效；第二，其训练方案（teacher forcing、scheduled sampling、推理时的 beam search）至今仍适用于所有现代生成系统，包括 LLM。

## 核心概念

**Encoder（编码器）。** 一个读取源句的 RNN。它的最终隐状态就是**上下文向量**——对整个输入的固定大小摘要。据说没有丢失任何信息，除了源句本身。

**Decoder（解码器）。** 另一个从上下文向量初始化的 RNN。在每个时间步，它将上一步生成的 token 作为输入，并输出在目标词表上的分布。通过采样或 argmax 选出下一个 token，反馈回去，重复直到生成 `<EOS>` token 或达到最大长度。

**训练：** 在每一个解码器时间步上计算交叉熵损失，在整个序列上求和。通过两个网络做标准的 BPTT（随时间反向传播）。

**Teacher forcing（教师强制）。** 训练时，解码器在时间步 `t` 的输入是位置 `t-1` 的**真实** token，而非解码器自身的上一轮预测。这能稳定训练；否则，早期错误会级联放大，模型无法学会。推理时必须使用模型自身的预测，因此始终存在训练/推理分布差异。这个差异被称为 **exposure bias（暴露偏差）**。

**瓶颈。** 编码器关于源句学到的所有内容都必须挤进那一个上下文向量。长句丢失细节，罕见词变得模糊。词序变换（如"chat noir"→"black cat"）需要记忆而非计算。

Attention（第10课）通过让解码器能够看到**每一个**编码器隐状态来解决这个问题，而不只是最后一个。这就是它的核心价值。

```figure
lstm-gates
```

## 动手实现

### 第1步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        # padding_idx=0：词嵌入中，索引0视为填充标记
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        # batch_first=True：输入张量格式为 [batch, seq_len, feature]
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 形状为 `[batch, seq_len, hidden_dim]`——每个输入位置对应一个隐状态。`hidden` 形状为 `[1, batch, hidden_dim]`——最后一步的隐状态。第08课说"对 outputs 做池化用于分类"；这里我们保留最后隐状态作为上下文向量，忽略逐时间步的输出。

### 第2步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        # padding_idx=0：目标词嵌入中，索引0视为填充标记
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        # batch_first=True：输入张量格式为 [batch, seq_len, feature]
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器被逐个时间步调用。输入：一批单个 token 和当前隐状态。输出：下一 token 的词表 logits 和更新后的隐状态。

### 第3步：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    # 用 BOS（begin-of-sequence）token 初始化解码器输入
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    # ignore_index=0：跳过填充 token 的损失计算
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        # 以 teacher_forcing_ratio 概率使用真实 token，否则使用模型预测
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

值得关注的两个关键参数：`ignore_index=0` 跳过填充 token 的损失；`teacher_forcing_ratio` 是每一步使用真实 token 而非模型预测的概率。建议从 1.0（完全教师强制）开始，随训练进程退火到约 0.5，以缩小暴露偏差。

### 第4步：推理循环（贪心解码）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    # 用 BOS token 初始化
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        # 遇到 EOS token 时提前停止
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每个时间步选择概率最高的 token。但它可能走偏：一旦选定某个 token，就无法撤回。**Beam search** 保留 top-`k` 个部分序列，最终选择得分最高的完整序列。beam width 取 3-5 是标准做法。

### 第5步：瓶颈演示

在复制任务上训练模型：源序列 `[a, b, c, d, e]`，目标序列 `[a, b, c, d, e]`。逐步增加序列长度，观察准确率变化。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐状态无法无损地记忆长度为 40 的输入。信息在每一个编码器时间步都存在，但解码器只能看到最后一个状态。Attention 直接解决了这个问题。

## 实际应用

PyTorch 提供了 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库内置了完整的编码器-解码器模型（BART、T5、mBART、NLLB），均在数十亿 token 上训练。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器已将 RNN 替换为 Transformer，但整体结构（编码器、解码器、逐 token 生成）与 2014 年 seq2seq 论文完全一致。各模块内部的机制不同。

### 何时仍选择基于 RNN 的 seq2seq

对新项目来说几乎不用。例外情况：

- 流式翻译：逐 token 消费输入且内存有界。
- 端侧文本生成：Transformer 的内存开销难以承受。
- 教学用途：理解编码器-解码器瓶颈是理解 Transformer 胜出原因的最快路径。

### 暴露偏差及其缓解方法

- **Scheduled sampling（计划采样）。** 训练中逐渐退火 teacher forcing 比例，让模型学会从自身错误中恢复。
- **Minimum risk training（最小风险训练）。** 训练时使用句级 BLEU 分数而非 token 级交叉熵，更贴近实际目标。
- **强化学习微调。** 用评估指标奖励序列生成器。现代 LLM 的 RLHF 就是这一思路。

以上三种方法同样适用于基于 Transformer 的生成。

## 交付

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计序列到序列（seq2seq）流水线。
phase: 5
lesson: 09
---

给定任务（翻译、摘要、改写、问题重写），输出：

1. 架构：预训练 Transformer 编码器-解码器（BART、T5、mBART、NLLB）为默认方案；仅在特定约束下使用基于 RNN 的 seq2seq。
2. 初始检查点：指定名称（如 `facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`），根据任务目标和语言覆盖范围匹配。
3. 解码策略：贪心解码用于确定性输出；beam search（宽度 4-5）用于质量优化；带温度采样的抽样用于多样性。附一句理由说明。
4. 交付前需验证的一个失败模式：暴露偏差在长输出上表现为生成漂移——抽取 20 个处于第 90 百分位长度的输出样本，人工检查。

拒绝为少于百万对平行语料的场景推荐从零训练 seq2seq 模型。将任何对面向用户的内容使用贪心解码的流水线标记为脆弱（贪心解码容易产生重复和循环）。
```

## 练习

1. **简单。** 实现玩具复制任务。在 GRU seq2seq 上训练，目标等于输入的输入-输出对。在长度 5、10、20 上测量准确率，复现瓶颈现象。
2. **中等。** 添加 beam width 为 3 的 beam search 解码。在小规模平行语料上测量 BLEU，并与贪心解码对比。记录 beam search 的优势位置（通常是末尾 token）和无明显差异的位置。
3. **困难。** 在 1 万对改写作数据集上微调 `facebook/bart-base`。将微调后模型的 beam-4 输出与基座模型在保留集上的输出进行对比。报告 BLEU 分数，并挑选 10 个定性示例。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|-----------|---------|
| Encoder（编码器） | 输入侧 RNN | 读取源句，输出逐时间步隐状态和最终上下文向量。 |
| Decoder（解码器） | 输出侧 RNN | 从上下文向量初始化，逐个生成目标 token。 |
| Context vector（上下文向量） | 摘要 | 编码器的最终隐状态，固定大小，即注意力所解决的瓶颈。 |
| Teacher forcing（教师强制） | 使用真实 token | 训练时输入上一步的真实 token，稳定学习过程。 |
| Exposure bias（暴露偏差） | 训练/测试差异 | 模型仅在训练时见过真实 token，从未练习过从自身错误中恢复。 |
| Beam search（束搜索） | 更好的解码方式 | 每步保留 top-k 个部分序列，而非贪心选择。 |

## 延伸阅读

- [Sutskever, Vinyals, Le（2014）。Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)——原始 seq2seq 论文，仅四页。
- [Cho et al.（2014）。Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)——引入 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio（2014）。Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)——注意力论文，学完本课后立即阅读。
- [PyTorch NLP from Scratch 教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)——可运行的 seq2seq + 注意力代码。
```
