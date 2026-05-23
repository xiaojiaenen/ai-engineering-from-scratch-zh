# 序列到序列模型

> 假装是翻译器的两个RNN。它们遇到的瓶颈正是注意力机制存在的原因。

**类型：** 构建
**语言：** Python
**先修要求：** 阶段5 · 08（用于文本的CNN + RNN），阶段3 · 11（PyTorch入门）
**时间：** 约75分钟

## 问题所在

分类将可变长度的序列映射到单个标签。翻译将可变长度的序列映射到另一个可变长度的序列。输入和输出存在于不同的词表中，可能来自不同的语言，并且无法保证长度相等。

Seq2seq架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的方法解决了这个问题。两个RNN。一个读取源句子并生成一个固定大小的上下文向量。另一个读取这个向量并逐个token地生成目标句子。这与你为第08课编写的代码相同，只是以不同的方式粘合在一起。

这值得研究的原因有两点。首先，上下文向量瓶颈是自然语言处理中最具教学价值的失败案例。它推动了注意力机制和Transformer所擅长的一切。其次，训练方法（教师强制、计划采样、推理时的束搜索）仍然适用于所有现代生成系统，包括大型语言模型。

## 核心概念

**编码器。** 一个读取源句子的RNN。它的最终隐藏状态就是**上下文向量**——整个输入的固定大小摘要。据说，它不会丢失源句的任何信息。

**解码器。** 另一个由上下文向量初始化的RNN。在每一步，它将之前生成的token作为输入，并输出一个关于目标词表的概率分布。采样或取argmax来选择下一个token。将其反馈回去。重复此过程，直到生成一个`<EOS>` token或达到最大长度。

**训练：** 在每个解码器步骤计算交叉熵损失，并在整个序列上求和。通过两个网络进行标准的反向传播。

**教师强制。** 在训练期间，解码器在第`t`步的输入是位置`t-1`处的*真实*token，而不是解码器自己先前的预测。这稳定了训练；没有它，早期的错误会级联放大，模型永远学不会。在推理时，你必须使用模型自身的预测，因此始终存在训练/推理分布差异。这种差异被称为**曝光偏差**。

**瓶颈。** 编码器学习到的关于源句的一切信息都必须被压缩到那个单一的上下文向量中。长句子会丢失细节。罕见词会变得模糊。语序重排（如 chat noir vs. black cat）必须靠记忆而非计算。

注意力机制（第10课）通过让解码器查看*所有*编码器隐藏状态（而不仅仅是最后一个）来解决这个问题。这就是全部要点。

## 构建它

### 步骤1：一个编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`的形状是`[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden`的形状是`[1, batch, hidden_dim]`——最后一步的状态。第08课说过“对输出进行池化用于分类”。这里我们保留最后一个隐藏状态作为上下文向量，并忽略每一步的输出。

### 步骤2：一个解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器一次被调用一步。输入：一个批次的单个token和当前隐藏状态。输出：下一个token的词表logits和更新后的隐藏状态。

### 步骤3：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个旋钮值得命名。`ignore_index=0`跳过对填充token的损失计算。`teacher_forcing_ratio`是在每一步使用真实token还是模型预测的概率。从1.0（完全教师强制）开始，并在训练过程中逐渐降低到~0.5，以缩小曝光偏差的差距。

### 步骤4：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每一步选择概率最高的token。它可能会跑偏：一旦你选定一个token，就无法收回了。**束搜索**保留得分最高的`k`个部分序列存活，并在最后选择得分最高的完整序列。束宽3-5是标准设置。

### 步骤5：瓶颈演示

在一个玩具复制任务上训练模型：源`[a, b, c, d, e]`，目标`[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个GRU隐藏状态无法无损地记忆一个40个token的输入。信息存在于编码器的每一步中，但解码器只能看到最后的状态。注意力机制直接解决了这个问题。

## 使用它

PyTorch有`nn.Transformer`和基于`nn.LSTM`的seq2seq模板。Hugging Face的`transformers`库提供了完整的编码器-解码器模型（BART、T5、mBART、NLLB），这些模型在数十亿token上训练过。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器模型用Transformer取代了RNN。其高层结构（编码器、解码器、逐token生成）与2014年的seq2seq论文完全相同。每个块内部的机制不同。

### 何时仍使用基于RNN的Seq2Seq

对于新项目，几乎从不。特定的例外情况：

- 流式翻译，其中你一次消费一个输入token，且内存有界。
- 设备端文本生成，其中Transformer的内存成本过高。
- 教学目的。理解编码器-解码器瓶颈是理解为何Transformer胜出的最快路径。

### 曝光偏差及其缓解方法

- **计划采样。** 在训练期间逐渐降低教师强制比例，使模型学会从自己的错误中恢复。
- **最小风险训练。** 在句子级别的BLEU分数而非token级别的交叉熵上进行训练。更接近你实际想要优化的目标。
- **强化学习微调。** 用一个指标奖励序列生成器。用于现代大型语言模型的RLHF。

这三种方法仍然适用于基于Transformer的生成。

## 发布它

保存为`outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1.  **简单。** 实现玩具复制任务。在输入-输出对（目标等于源）上训练一个GRU seq2seq模型。在长度5、10、20时测量准确率。重现瓶颈。
2.  **中等。** 添加束宽为3的束搜索解码。在一个小型平行语料库上，对比贪心解码测量BLEU分数。记录束搜索在哪些情况下（通常是最后几个token）表现更好，以及在哪些情况下没有区别。
3.  **困难。** 在10k对的释义数据集上微调`facebook/bart-base`。将微调后模型的束宽4输出与基础模型在保留输入上的输出进行比较。报告BLEU分数，并挑选10个定性示例。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|--------------|
| 编码器 | 输入RNN | 读取源句。生成每一步的隐藏状态和一个最终的上下文向量。 |
| 解码器 | 输出RNN | 由上下文向量初始化。一次生成一个目标token。 |
| 上下文向量 | 摘要 | 最终的编码器隐藏状态。固定大小。这是注意力机制解决的瓶颈。 |
| 教师强制 | 使用真实token | 在训练时喂入真实前一个token。稳定学习过程。 |
| 曝光偏差 | 训练/测试差距 | 模型在真实token上训练，从未练习过从自己的错误中恢复。 |
| 束搜索 | 更好的解码 | 在每一步保留得分最高的前k个部分序列存活，而不是贪心地提交。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始seq2seq论文。四页纸。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了GRU和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力机制论文。在学习本课后立即阅读。
- [PyTorch NLP从零开始教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的seq2seq + 注意力代码。