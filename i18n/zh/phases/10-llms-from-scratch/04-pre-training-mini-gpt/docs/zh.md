# 从零预训练一个迷你 GPT（124M 参数）

> GPT-2 Small 有 1.24 亿个参数。那是 12 层 transformer、12 个注意力头和 768 维嵌入向量。你可以在几小时内用单张 GPU 从零开始训练它。大多数人从未做过这件事，他们直接使用预训练的检查点。但如果你自己没有训练过一个模型，你就不会真正理解你所构建产品背后的模型内部究竟发生了什么。

**类型：** 构建
**语言：** Python（使用 numpy）
**前置知识：** Phase 10，课程 01-03（分词器、构建分词器、数据管道）
**耗时：** 约 120 分钟

## 学习目标

- 从零实现完整的 GPT-2 架构（1.24 亿参数）：token 嵌入、位置嵌入、transformer 块和语言模型头
- 使用下一个 token 预测和交叉熵损失在文本语料上训练 GPT 模型
- 实现带温度采样和 top-k/top-p 滤波的自回归文本生成
- 监控训练损失曲线并验证模型是否学会了连贯的语言模式

## 问题所在

你知道什么是 transformer。你读过相关图示，你能背诵"attention is all you need"，还能在白板上画出标着"Multi-Head Attention"的方框。

但这并不意味着你理解模型生成文本时究竟发生了什么。

GPT-2 Small 中有 124,438,272 个参数（含权值绑定）。每一个参数都是通过运行训练循环设置的：前向传播、计算损失、反向传播、更新权重。十二个 transformer 块，每个块十二个注意力头，768 维嵌入空间，50,257 个 token 的词表。每次模型生成一个 token，所有 1.24 亿个参数都参与一次矩阵乘法链，将 token ID 序列转化为下一个 token 的概率分布。

如果你从未自己构建过这样的模型，你就是在操作一个黑盒。你可以调用 API，可以微调。但当出现问题时——当模型产生幻觉、重复输出、或拒绝遵循指令时——你没有一个心智模型来理解 *为什么*。

本课程从零构建 GPT-2 Small。不用 PyTorch，用 numpy。每一次矩阵乘法都清晰可见，每一个梯度都由你的代码计算。你将亲眼看到 1.24 亿个数字如何协同工作以预测下一个词。

## 概念讲解

### GPT 架构

GPT 是一个自回归语言模型。"自回归"意味着它一次生成一个 token，每个 token 都依赖于所有前面的 token。其架构是一叠 transformer 解码器块。

以下是从 token ID 到下一个 token 概率的完整计算图：

1. Token ID 输入。形状：(batch_size, seq_len)。
2. Token 嵌入查找。每个 ID 映射到一个 768 维向量。形状：(batch_size, seq_len, 768)。
3. 位置嵌入查找。每个位置 (0, 1, 2, ...) 映射到一个 768 维向量。相同形状。
4. 添加 token 嵌入 + 位置嵌入。
5. 通过 12 个 transformer 块。
6. 最终层归一化。
7. 线性投影到词汇表大小。形状：(batch_size, seq_len, vocab_size)。
8. Softmax 得到概率。

这就是整个模型。没有卷积，没有循环。只有嵌入、注意力、前馈网络和层归一化堆叠 12 次。

```mermaid
graph TD
    A["Token IDs\n(batch, seq_len)"] --> B["Token Embeddings\n(batch, seq_len, 768)"]
    A --> C["Position Embeddings\n(batch, seq_len, 768)"]
    B --> D["Add"]
    C --> D
    D --> E["Transformer Block 1"]
    E --> F["Transformer Block 2"]
    F --> G["..."]
    G --> H["Transformer Block 12"]
    H --> I["Layer Norm"]
    I --> J["Linear Head\n(768 -> 50257)"]
    J --> K["Softmax\nNext-token probabilities"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#0f3460,color:#fff
    style C fill:#1a1a2e,stroke:#0f3460,color:#fff
    style D fill:#1a1a2e,stroke:#16213e,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
    style I fill:#1a1a2e,stroke:#16213e,color:#fff
    style J fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Transformer 块

每个块遵循相同的模式。Pre-norm 架构（GPT-2 使用 pre-norm，而非原始 transformer 的 post-norm）：

1. LayerNorm
2. Multi-Head Self-Attention
3. 残差连接（将输入加回）
4. LayerNorm
5. Feed-Forward Network（MLP）
6. 残差连接（将输入加回）

残差连接至关重要。没有它们，梯度在反向传播到达第一个块时就已经消失殆尽。有了它们，梯度可以通过"跳过"路径直接从损失流到任意层。这就是为什么你可以堆叠 12、32 甚至 96 个块（据传 GPT-4 使用了 120 个）。

### 注意力：核心机制

Self-attention 让每个 token 都能看到所有前面的 token，并决定对每个 token 关注多少。以下是数学原理。

对于每个 token 位置，从输入中计算三个向量：
- **Query (Q)**："我在寻找什么？"
- **Key (K)**："我包含什么？"
- **Value (V)**："我携带了哪些信息？"

```
Q = input @ W_q    (768 -> 768)
K = input @ W_k    (768 -> 768)
V = input @ W_v    (768 -> 768)

attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = mask(attention_scores)   # 因果掩码：未来位置为 -inf
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

因果掩码是使 GPT 自回归的关键。位置 5 可以关注位置 0-5，但不能关注 6、7、8，依此类推。这防止了模型在训练期间通过查看未来 token 来"作弊"。

**Multi-head attention** 将 768 维空间拆分为 12 个头，每个头 64 维。每个头学习不同的注意力模式。一个头可能追踪句法关系（主谓一致），另一个可能追踪语义相似性（同义词），再一个可能追踪位置邻近性（相邻词汇）。所有 12 个头的输出被拼接后重新投影回 768 维。

```mermaid
graph LR
    subgraph MultiHead["Multi-Head Attention (12 heads)"]
        direction TB
        I["Input (768)"] --> S1["Split into 12 heads"]
        S1 --> H1["Head 1\n(64 dims)"]
        S1 --> H2["Head 2\n(64 dims)"]
        S1 --> H3["..."]
        S1 --> H12["Head 12\n(64 dims)"]
        H1 --> C["Concat (768)"]
        H2 --> C
        H3 --> C
        H12 --> C
        C --> O["Output Projection\n(768 -> 768)"]
    end

    subgraph SingleHead["Each Head Computes"]
        direction TB
        Q["Q = X @ W_q"] --> A["scores = Q @ K^T / 8"]
        K["K = X @ W_k"] --> A
        A --> M["Apply causal mask"]
        M --> SM["Softmax"]
        SM --> MUL["weights @ V"]
        V["V = X @ W_v"] --> MUL
    end

    style I fill:#1a1a2e,stroke:#e94560,color:#fff
    style O fill:#1a1a2e,stroke:#e94560,color:#fff
    style Q fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#0f3460,color:#fff
    style V fill:#1a1a2e,stroke:#0f3460,color:#fff
```

除以 sqrt(d_k)——sqrt(64) = 8——是缩放操作。没有它，高维向量的点积会增长很大，将 softmax 推向梯度几乎为零的区域。这是原始"Attention Is All You Need"论文中的关键洞察之一。

### KV Cache：为什么推理速度快

在训练期间，你一次性处理整个序列。在推理期间，你一次生成一个 token。没有优化的情况下，生成第 N 个 token 需要重新计算所有 N-1 个前面 token 的注意力。这对每个生成的 token 是 O(N^2)，对整个长度为 N 的序列是 O(N^3)。

KV Cache 解决了这个问题。计算每个 token 的 K 和 V 后，将它们存储起来。当生成第 N+1 个 token 时，你只需要为新 token 计算 Q，并查表获取所有前面 token 的缓存 K 和 V。这将每个 token 的计算成本从 O(N) 降至 O(1)（针对 K 和 V 的计算）。注意力分数的计算仍然是 O(N)，因为你关注所有前面位置，但你避免了对输入的冗余矩阵乘法。

对于具有 12 层 12 个头的 GPT-2，KV cache 每个 token 存储 2 (K + V) x 12 层 x 12 个头 x 64 维 = 18,432 个值。对于一个 1024 token 的序列，在 FP32 下大约是 75MB。对于具有 128 层的 Llama 3 405B，单个序列的 KV cache 可以超过 10GB。这就是为什么长上下文推理是内存受限的。

### Prefill vs Decode：推理的两个阶段

当你向 LLM 发送 prompt 时，推理发生在两个截然不同的阶段。

**Prefill** 并行处理你的整个 prompt。所有 token 都是已知的，因此模型可以同时为所有位置计算注意力。这个阶段是计算受限的——GPU 以满吞吐量执行矩阵乘法。对于 A100 上的 1000 token prompt，prefill 大约需要 20-50ms。

**Decode** 一次生成一个 token。每个新 token 依赖于所有前面的 token。这个阶段是内存受限的——瓶颈是从 GPU 内存读取模型权重和 KV cache，而非矩阵运算本身。GPU 的计算核心大部分时间空闲，等待内存读取。对于 GPT-2，每个 decode 步骤大约花费相同的时间，无论 matmul 需要多少 FLOPs，因为内存带宽才是约束条件。

这个区别对生产系统很重要。Prefill 吞吐量随 GPU 计算能力扩展（更多 FLOPS = 更快的 prefill）。Decode 吞吐量随内存带宽扩展（更快的内存 = 更快的 decode）。这就是为什么 NVIDIA 的 H100 相比 A100 更注重内存带宽改进——它直接加速了 token 生成。

```mermaid
graph LR
    subgraph Prefill["Phase 1: Prefill"]
        direction TB
        P1["Full prompt\n(all tokens known)"]
        P2["Parallel computation\n(compute-bound)"]
        P3["Builds KV Cache"]
        P1 --> P2 --> P3
    end

    subgraph Decode["Phase 2: Decode"]
        direction TB
        D1["Generate token N"]
        D2["Read KV Cache\n(memory-bound)"]
        D3["Append to KV Cache"]
        D4["Generate token N+1"]
        D1 --> D2 --> D3 --> D4
        D4 -.->|repeat| D1
    end

    Prefill --> Decode

    style P1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 训练循环

训练 LLM 就是下一个 token 预测。给定 tokens [0, 1, 2, ..., N-1]，预测 tokens [1, 2, 3, ..., N]。损失函数是模型预测概率分布与实际下一个 token 之间的交叉熵。

一个训练步骤：

1. **前向传播**：将整个批次通过所有 12 个块。得到每个位置的 logits（softmax 前的分数）。
2. **计算损失**：logits 与目标 token（输入向右移一位）之间的交叉熵。
3. **反向传播**：使用反向传播计算所有 1.24 亿参数的梯度。
4. **优化器步骤**：更新权重。GPT-2 使用 Adam 优化器，带有学习率预热和余弦衰减。

学习率调度比你想象的更重要。GPT-2 在前 2,000 步从 0 预热到峰值学习率，然后按照余弦曲线衰减。以高学习率开始会导致模型发散。保持恒定的高学习率会在训练后期导致震荡。预热后衰减的模式被所有主要 LLM 所使用。

### GPT-2 Small：参数数量

| 组件 | 形状 | 参数数量 |
|------|------|---------|
| Token embeddings | (50257, 768) | 38,597,376 |
| Position embeddings | (1024, 768) | 786,432 |
| 每块注意力 (W_q, W_k, W_v, W_out) | 4 x (768, 768) | 2,359,296 |
| 每块 FFN (up + down) | (768, 3072) + (3072, 768) | 4,718,592 |
| 每块 LayerNorms (2x) | 2 x 768 x 2 | 3,072 |
| 最终 LayerNorm | 768 x 2 | 1,536 |
| **每块总计** | | **7,080,960** |
| **总计（12 个块）** | | **85,054,464 + 39,383,808 = 124,438,272** |

输出投影（logits 头）与 token 嵌入矩阵共享权重。这称为 weight tying——它将参数数量减少了 3800 万，并提升了性能，因为它迫使模型对输入和输出使用相同的表示空间。

## 构建它

### 第 1 步：嵌入层

Token 嵌入将 50,257 个可能的 token 中的每一个映射到 768 维向量。位置嵌入添加了每个 token 在序列中位置的信息。两者相加。

```python
import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb
```

初始化时使用的 0.02 标准差来自 GPT-2 论文。太大则初始前向传播会产生极端的值，使训练不稳定。太小则所有输入的初始输出几乎相同，使早期的梯度信号失效。

### 第 2 步：带因果掩码的 Self-Attention

先从单头注意力开始。因果掩码在 softmax 之前将未来位置设为负无穷，确保每个位置只能关注自身和更早的位置。

```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, -1, -2 if Q.ndim == 4 else 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

Softmax 实现在指数运算前减去最大值。如果没有这一步，exp(large_number) 会溢出为无穷大。这是一个数值稳定性技巧，不改变输出结果，因为 softmax(x - c) = softmax(x) 对任意常数 c 成立。

### 第 3 步：Multi-Head Attention

将 768 维输入拆分为 12 个头，每个头 64 维。每个头独立计算注意力。拼接结果并投影回 768 维。

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out
```

reshape-transpose-reshape 的变换是多头注意力中最令人困惑的部分。具体过程如下：(batch, seq_len, 768) 张量变为 (batch, seq_len, 12, 64)，然后变为 (batch, 12, seq_len, 64)。现在 12 个头中的每一个都有自己的 (seq_len, 64) 矩阵来执行注意力。注意力完成后，我们逆转这个过程：(batch, 12, seq_len, 64) 变为 (batch, seq_len, 12, 64) 再变为 (batch, seq_len, 768)。

### 第 4 步：Transformer 块

一个完整的 transformer 块：LayerNorm、带残差的 multi-head attention、LayerNorm、带残差的前馈网络。

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # GELU 近似：为简化使用 ReLU
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x
```

前馈网络将 768 维输入扩展到 3,072 维（4 倍），应用非线性激活，然后投影回 768 维。这种扩张-收缩模式让模型在每个位置拥有"更宽"的内部表示。GPT-2 使用 GELU 激活函数，但这里我们为了简化使用 ReLU——对于理解架构而言，差异很小。

### 第 5 步：完整 GPT 模型

堆叠 12 个 transformer 块。前端加嵌入层，后端加输出投影。

```python
class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total
```

注意 weight tying：`logits = x @ self.embedding.token_embed.T`。输出投影复用了 token 嵌入矩阵（转置后）。这不仅仅是节省参数的技巧。它意味着模型使用相同的向量空间来理解 token（嵌入）和预测它们（输出）。

### 第 6 步：训练循环

对于 1.24 亿参数的真实训练，你需要 GPU 和 PyTorch。此训练循环在一个纯 numpy 运行的小型模型上演示了这些机制。我们使用一个小型模型（4 层、4 头、128 维）使其可行。

```python
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=3e-4):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model
```

损失初始值接近 ln(vocab_size)——对于 256 token 的字节级词表，即 ln(256) = 5.55。随机模型给每个 token 分配相等的概率。随着训练进行，损失下降，因为模型学会了预测常见模式："th" 跟在 "t" 后面，句号后跟空格，等等。

在生产环境中，你会使用 Adam 优化器配合梯度累积、学习率预热和梯度裁剪。前向传播-损失-反向传播-更新的循环是完全相同的，只是优化器更加复杂。

### 第 7 步：文本生成

生成使用训练好的模型一次预测一个 token。每个预测从输出分布中采样（或取 argmax 贪婪解码）。

```python
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens
```

Temperature 控制随机性。Temperature 1.0 使用原始分布。Temperature 0.5 使其更尖锐（更确定——模型更频繁地选择其最优选）。Temperature 1.5 使其更平坦（更随机——低概率 token 获得更大的机会）。Temperature 0.0 是贪婪解码（总是选择概率最高的 token）。

`tokens[-seq_len:]` 窗口是必要的，因为模型有最大上下文长度（GPT-2 为 1024）。一旦超过，你必须丢弃最老的 token。这就是大家常说的"上下文窗口"。

```figure
sampling-decoder
```

## 使用它

### 完整训练与生成演示

```python
corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

model = train_mini_gpt(corpus, num_steps=200)

prompt = list("The transformer".encode("utf-8"))
output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
print(f"\nGenerated: {generated_text}")
```

在小语料和小模型上，生成的文本最多只能算半连贯。它会从训练文本中学到一些字节级模式，但无法像 GPT-2 使用 40GB 训练数据和完整 1.24 亿参数架构那样泛化。重点不在于输出质量。重点在于你可以追踪每一步：嵌入查找、注意力计算、前馈变换、logit 投影、softmax 和采样。每个操作都是可见的。

## 交付物

本课程产出 `outputs/prompt-gpt-architecture-analyzer.md`——一个用于分析任意 GPT 风格模型架构选择点的 prompt。给它一张模型卡或技术报告，它会分解参数分配、注意力设计和扩展决策。

## 练习

1. 将模型修改为 24 层 16 个头（而非 12/12）。计算参数数量。加倍深度与加倍宽度（嵌入维度）相比如何？

2. 实现 GELU 激活函数（GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))），并替换前馈网络中的 ReLU。分别用两种激活函数训练 500 步并比较最终损失。

3. 为生成函数添加 KV cache。在首次前向传播后存储每一层的 K 和 V 张量，并在后续 token 生成中复用它们。测量加速效果：分别用和不用缓存生成 200 个 token，比较实际运行时间。

4. 实现 top-k 采样（仅考虑概率最高的 k 个 token）和 top-p 采样（核采样：考虑累积概率超过 p 的最小 token 集合）。比较 temperature=0.8 下 top-k=50 与 top-p=0.95 的输出质量。

5. 构建训练损失曲线绘图器。训练模型 1000 步并绘制损失 vs 步骤曲线。识别三个阶段：快速初始下降（学习常见字节）、较慢中间阶段（学习字节模式）和平台期（在小语料上过拟合）。这条曲线的形状无论训练的是 128 维模型还是 GPT-4 都是一样的。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Autoregressive | "一次生成一个词" | 每个输出 token 都依赖于所有前面的 token——模型预测 P(token_n \| token_0, ..., token_{n-1}) |
| Causal mask | "它看不到未来" | 一个上三角的 -inf 值矩阵，防止训练期间关注未来位置 |
| Multi-head attention | "多种注意力模式" | 将 Q、K、V 拆分为并行头（如 GPT-2 的 12 个头各 64 维），使每个头能学习不同类型的关系 |
| KV Cache | "为速度而缓存" | 存储来自前面 token 的计算好的 Key 和 Value 张量，避免自回归生成时的冗余计算 |
| Prefill | "处理 prompt" | 第一个推理阶段，所有 prompt token 并行处理——GPU FLOPS 计算受限 |
| Decode | "生成 token" | 第二个推理阶段，token 逐个生成——GPU 带宽内存受限 |
| Weight tying | "共享嵌入" | 对输入 token 嵌入和输出投影头使用同一矩阵——GPT-2 节省 3800 万参数 |
| Residual connection | "跳跃连接" | 将输入直接加到子层输出（x + sublayer(x)）——在深层网络中启用梯度流动 |
| Layer normalization | "归一化激活值" | 沿特征维度归一化至均值 0 和方差 1，带有可学习的缩放和偏置参数 |
| Cross-entropy loss | "预测有多错误" | -log(分配给正确下一个 token 的概率)，对所有位置取平均——标准 LLM 训练目标 |

## 延伸阅读

- [Radford et al., 2019 -- "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) —— 引入 1.24 亿至 15 亿参数家族的 GPT-2 论文
- [Vaswani et al., 2017 -- "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) —— 原始 transformer 论文，包含缩放点积注意力和多头注意力
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) —— Meta 如何将 GPT 架构扩展到 405B 参数并使用 16K GPU
- [Pope et al., 2022 -- "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) —— 形式化 prefill vs decode 和 KV cache 分析的论文
