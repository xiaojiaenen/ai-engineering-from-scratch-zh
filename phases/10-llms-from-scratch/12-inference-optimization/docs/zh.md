# 推理优化

> LLM推理分为两个阶段。预填充阶段并行处理你的提示词——计算密集型。解码阶段每次生成一个token——内存密集型。所有优化都针对其中一个或两个阶段。

**类型：** 构建
**语言：** Python
**前置条件：** 第10阶段，第01-08课（Transformer架构，注意力机制）
**时间：** ~120分钟

## 学习目标

- 实现KV缓存以消除自回归token生成过程中的冗余计算
- 解释LLM推理的预填充与解码阶段，并说明每个阶段为何存在不同瓶颈（计算密集型 vs 内存密集型）
- 实现连续批处理和PagedAttention概念，以在并发请求下最大化GPU利用率
- 比较推理优化技术（KV缓存、推测解码、Flash Attention）及其吞吐量/延迟权衡

## 问题所在

你在4块A100 GPU上部署了Llama 3 70B模型。单个用户每秒能获得约50个token。感觉很快。然后100个用户同时访问接口。吞吐量下降到每用户3个token/秒。你每月25,000美元的GPU账单，服务的响应速度却比人打字还慢。

无论服务1个用户还是100个用户，模型本身并未改变。相同的权重、相同的架构、相同的数学运算。改变的是你如何调度工作。朴素的推理会浪费90%以上的可用GPU算力。一个等待第47个token的用户会占用整个批处理槽位，而GPU内存总线在矩阵乘法之间处于空闲状态。同时，一个新用户的2,000个token提示词本可以利用这段空闲时间进行有效计算。

这不是一个扩展问题。这是一个调度问题。本课中的技术——KV缓存、连续批处理、PagedAttention、推测解码、前缀缓存——正是区分每月25,000美元和5,000美元推理账单（服务相同流量）的关键。

vLLM在4块A100-80GB GPU上服务Llama 3 70B，在低并发下可实现每用户约50 token/秒，通过连续批处理和PagedAttention，在100个并发请求下能维持每用户15-25 TPS。没有这些优化，相同硬件在该并发下只能服务每用户5 TPS。相同的GPU，相同的模型，吞吐量相差4倍。

## 核心概念

### 预填充 vs 解码

每个LLM推理请求都有两个不同的阶段。

**预填充**处理整个输入提示词。所有token都已知，因此注意力可以在整个序列上并行计算。这是一个大规模的矩阵乘法——GPU核心保持忙碌。瓶颈在于计算：即你的硬件每秒能提供多少次浮点运算（FLOPS）。一块A100提供312 TFLOPS（BF16精度）。在单块A100上处理一个70B模型的4,096个token提示词，预填充大约需要400毫秒。

**解码**一次生成一个输出token。每个新token都会关注所有之前的token，但每次前向传播只产生一个token。权重矩阵的大小与预填充期间相同，但你是用一个向量而不是矩阵去乘它们。GPU核心在微秒内完成计算，然后等待下一批权重从内存传输过来。瓶颈在于内存带宽：即你从HBM（高带宽内存）向计算单元传输模型权重的速度。一块A100拥有2 TB/s的带宽。一个FP16精度的70B模型大小为140 GB。完整读取一次模型需要70毫秒——这是单个解码步骤的理论下限。

```mermaid
graph LR
    subgraph "Prefill (compute-bound)"
        P1["All prompt tokens"] --> P2["Parallel attention"]
        P2 --> P3["Full matmul utilization"]
    end

    subgraph "Decode (memory-bound)"
        D1["One token at a time"] --> D2["Sequential generation"]
        D2 --> D3["Waiting on memory reads"]
    end

    P3 --> D1
```

**操作字节比**（也称为算术强度）捕捉了这种权衡。它衡量你每加载一个字节内存执行了多少次操作。

```
ops:byte ratio = FLOPs per token / bytes read from memory
```

在预填充4,096个token的批次时，你每加载一个权重大约执行4,096次乘加运算。这个比率很高——你是计算密集型。在解码阶段，批次大小为1时，你每加载一个权重大约只执行1次操作。这个比率很低——你是内存密集型。

根本性的见解：*解码是内存密集型的，因为你需要读取整个模型才能生成一个token*。下面的每项优化要么减少你需要读取的内容，要么增加每次读取处理的token批次，要么完全避免读取。

### KV 缓存

在注意力计算期间，每个token的查询（Query）都会关注之前每个token的键（Key）和值（Value）向量。如果没有缓存，生成第N个token需要重新计算所有前N-1个token的K和V投影。生成第2个token时计算了第1个token的投影，然后生成第3个token时又要计算一遍，生成第4个token时再计算一遍。到生成第1,000个token时，第1个token已经被投影了999次。

KV缓存存储了所有先前token的K和V投影。当生成第N个token时，你只需要计算第N个token的K和V，然后将其与从第1到第N-1个token缓存的K/V拼接起来。

```mermaid
graph TD
    subgraph "Without KV Cache"
        A1["Token 5: recompute K,V for tokens 1-4"]
        A2["Token 6: recompute K,V for tokens 1-5"]
        A3["Token 7: recompute K,V for tokens 1-6"]
    end

    subgraph "With KV Cache"
        B1["Token 5: compute K5,V5, read K1-4,V1-4 from cache"]
        B2["Token 6: compute K6,V6, read K1-5,V1-5 from cache"]
        B3["Token 7: compute K7,V7, read K1-6,V1-6 from cache"]
    end
```

**KV缓存的内存公式：**

```
KV cache size = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_param
```

对于Llama 3 70B（80层，8个KV头使用GQA，head_dim=128，BF16精度）：

```
per token: 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes = 320 KB
at 4,096 tokens: 320 KB * 4,096 = 1.28 GB
at 128K tokens: 320 KB * 131,072 = 40 GB
```

一个128K上下文长度的Llama 3 70B对话会消耗40 GB的KV缓存——相当于一块A100内存的一半。如果有100个并发用户，每个用户4K token，仅KV缓存就需要128 GB。这就是为什么KV缓存管理是推理优化的核心挑战。

### 连续批处理

静态批处理等待N个请求到达，将它们一起处理，并等待*所有*请求完成才接受新请求。如果一个请求需要500个token，另一个需要10个，那么短请求在完成后还需要闲置490个解码步骤。

连续批处理（也称为迭代级批处理）一旦有请求完成，就立即将新请求插入到当前运行的批次中。批次在每个解码步骤都会被重新评估。一个在10个token后完成的请求会立即被一个等待中的请求替换。

```mermaid
sequenceDiagram
    participant GPU
    participant R1 as Request 1 (50 tokens)
    participant R2 as Request 2 (10 tokens)
    participant R3 as Request 3 (30 tokens)
    participant R4 as Request 4 (waiting)

    Note over GPU: Static batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    Note over R2: Wasting 40 steps...
    Note over R3: R3 done at step 30
    Note over R3: Wasting 20 steps...
    GPU->>R4: Finally start R4 at step 50

    Note over GPU: Continuous batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    GPU->>R4: Insert R4 at step 11
    Note over R3: R3 done at step 30
```

吞吐量的提升取决于输出长度的差异程度。如果长度均匀，连续批处理与静态批处理效果相同。如果长度不一（常见情况），连续批处理可以提供2-5倍的吞吐量提升，因为GPU槽位永远不会空闲。

### PagedAttention

每个请求的KV缓存是一个连续的内存块。随着请求的进出，内存会产生碎片——就像操作系统中的内存碎片一样。一个4K token的请求需要1.28 GB连续内存。即使你总共有2 GB空闲内存，也可能没有1.28 GB的*连续*空间。你要么浪费内存，要么拒绝该请求。

PagedAttention（来自vLLM）将操作系统风格的虚拟内存应用于KV缓存。它不再为每个请求分配一个连续块，而是分配固定大小的“页”（通常每页16个token）。这些页可以位于物理GPU内存的任意位置。一个页表将每个请求的逻辑序列位置映射到物理页位置。

```mermaid
graph TD
    subgraph "Contiguous allocation"
        C1["Request A: 2GB block"]
        C2["[free: 0.5GB]"]
        C3["Request B: 1GB block"]
        C4["[free: 1.5GB -- but fragmented]"]
    end

    subgraph "PagedAttention"
        P1["Page pool: 256 pages of 16 tokens each"]
        P2["Request A: pages 3,7,12,45,88..."]
        P3["Request B: pages 1,4,9,22,67..."]
        P4["No fragmentation, no waste"]
    end
```

PagedAttention还为共享前缀实现了**写时复制**。如果50个请求共享相同的系统提示词，该系统提示词的KV缓存页只存储一次，并被所有50个请求引用。只有当某个请求开始不同（例如用户消息不同）时，它才会获得自己的页。这为具有共享系统提示词的应用程序大幅降低了内存使用。

vLLM报告称，通过PagedAttention，内存浪费几乎为零（约4%，而朴素分配约为60-80%）。

### 推测解码

解码很慢，因为它是顺序的——你生成一个token，将其反馈回去，再生成下一个。但如果你能廉价地猜测接下来的5个token，然后一次性验证它们呢？

推测解码使用一个小型、快速的**草稿模型**来生成K个候选token。然后，**目标模型**在一个前向传播中处理所有K个候选token（这看起来像一次预填充——并行、计算密集型、高效）。如果目标模型同意草稿模型的预测，你就在一次目标模型前向传播的时间内接受了所有K个token。如果它在第j个位置不同意，你就接受前j-1个token，并丢弃其余的。

```mermaid
graph LR
    D["Draft model (1B)"] -->|"Generate 5 tokens<br/>~5ms"| C["Candidates: the cat sat on the"]
    C --> T["Target model (70B)"]
    T -->|"Verify all 5 in one pass<br/>~70ms"| V{"Match?"}
    V -->|"4 of 5 match"| A["Accept 4 tokens in 75ms<br/>vs 280ms sequential"]
    V -->|"Mismatch at pos 5"| R["Reject token 5<br/>Resample from target"]
```

加速比取决于**接受率**——即草稿模型的预测与目标模型匹配的频率。用Llama 3 8B为Llama 3 70B做草稿模型，在自然语言上，接受率通常在70-85%之间。这相当于解码速度提升2-3倍。

推测解码的三种方法：

| 方法 | 草稿来源 | 接受率 | 开销 |
|------|---------|--------|------|
| 草稿-目标 (Leviathan et al.) | 单独的小模型 | 70-85% | 草稿模型内存 |
| EAGLE (Li et al.) | 目标模型上的轻量头 | 75-90% | ~1% 额外参数 |
| N-gram查找 | Token n-gram 表 | 40-60% | 几乎可忽略 |

**EAGLE**在目标模型的隐藏状态之上训练一个小的自回归头。它使用目标模型倒数第二层的特征来预测下一个token的嵌入。因为它操作的是目标模型自身的表征（而不是一个单独的模型），所以它以最小的额外内存实现了更高的接受率。EAGLE-2添加了一个动态草稿树，根据上下文调整候选数量。

**N-gram推测解码**维护一个n-gram续写表，内容来自当前上下文或预构建的语料库。如果草稿与对话中之前出现的内容匹配（重复模式、代码、结构化输出），它就能以几乎为零的神经网络开销触发。平均接受率较低，但每次推测的成本几乎为零。

推测解码是*数学精确的*——输出分布与目标模型的分布完全相同。它不是近似。验证步骤确保每个被接受的token都具有目标模型本应分配的精确概率。

### 前缀缓存

许多请求共享相同的前缀。一个聊天机器人的系统提示词。一个RAG上下文块。一组少样本示例。如果没有前缀缓存，每个请求都需要从头重新计算这些共享token的KV缓存。

前缀缓存存储常见前缀的KV缓存，并在请求间重用。当一个带有已知前缀的新请求到达时，系统复制（或引用）缓存的KV条目，并只为唯一的后缀计算KV。

对于一个所有请求共享的2,000个token系统提示词，前缀缓存可以为每个请求节省约400毫秒的预填充时间。在每秒100个请求时，每秒可节省40秒的GPU计算量——相当于超过一个GPU的工作量。

SGLang的RadixAttention通过一个基数树（trie）实现前缀缓存，该树按token内容索引前缀。任何匹配存储前缀的请求都可以免费获得其KV缓存。该树支持部分前缀匹配——如果你与缓存条目共享2,000个前缀token中的1,500个，你可以重用那1,500个，只重新计算500个。

### 推理引擎

三个引擎主导着生产环境的LLM服务：

| 引擎 | 关键创新 | 最适用于 |
|------|---------|---------|
| vLLM | PagedAttention, 连续批处理 | 通用服务，兼容性最广 |
| SGLang | RadixAttention (前缀缓存), 结构化生成 | 多轮对话机器人，受限解码 |
| TensorRT-LLM | NVIDIA内核融合，FP8量化 | NVIDIA硬件上最大单卡吞吐量 |

**vLLM**是默认的起点。它支持最广泛的模型，可在任何GPU厂商（NVIDIA, AMD, Intel）的硬件上运行，并通过PagedAttention + 连续批处理实现了高吞吐量。兼容OpenAI的API意味着你可以直接用它替换任何OpenAI API调用。

**SGLang**建立在与vLLM相同的基础上，但增加了用于前缀缓存的RadixAttention和一种用于结构化LLM程序的领域特定语言。如果你的工作负载涉及多轮对话、工具使用或受限解码（JSON输出、正则表达式引导生成），SGLang通常通过前缀重用比vLLM高出2-5倍的性能。

**TensorRT-LLM**将模型编译成优化的NVIDIA GPU内核。它融合操作（注意力+线性层+激活函数在一个内核中完成），在H100 GPU上使用FP8精度，并与NVIDIA Triton推理服务器集成用于生产部署。它在NVIDIA硬件上实现了最高的单卡吞吐量，但需要更多配置，且仅适用于NVIDIA GPU。

Llama 3 70B的真实世界性能数据（4xA100-80GB，BF16）：

| 指标 | vLLM | SGLang | TensorRT-LLM |
|------|------|--------|---------------|
| 吞吐量 (1 用户) | ~50 TPS | ~55 TPS | ~65 TPS |
| 吞吐量 (100 用户) | ~2,500 总TPS | ~3,200 总TPS | ~3,000 总TPS |
| 首token延迟 | ~400ms | ~300ms (前缀命中) | ~350ms |
| 最大上下文长度 | 128K | 128K | 128K |

### 操作字节比框架

你无法优化你无法度量的东西。操作字节比告诉你当前是计算密集型还是内存密集型，这决定了哪些优化是关键。

```
Compute roof: peak FLOPS of the GPU
Memory roof:  peak bandwidth * ops:byte ratio
```

当操作字节比低时（解码，小批次），你触及的是内存带宽上限。增加更多算力（更高时钟、更多核心）没有帮助。你需要减少内存读取（量化、KV缓存压缩）或增加批次大小，将读取开销分摊到更多有效工作上。

当操作字节比高时（预填充，大批次），你触及的是算力上限。内存带宽优化没有帮助。你需要更快的GPU、内核融合或降低精度来榨取更多FLOPS。

| 场景 | 操作字节比 | 瓶颈 | 优化方法 |
|------|-----------|------|----------|
| 预填充，批次=1 | ~4,096 | 计算密集 | 内核融合，FP8 |
| 解码，批次=1 | ~1 | 内存密集 | 量化，KV压缩 |
| 解码，批次=32 | ~32 | 内存密集 | 更大批次，连续批处理 |
| 解码，批次=256 | ~256 | 过渡中 | 两者都重要 |
| 解码，批次=1024 | ~1,024 | 计算密集 | 内核融合，张量并行 |

在A100上，交叉点大约在操作字节比为156时（312 TFLOPS / 2 TB/s）。低于156，你是内存密集型。高于156，你是计算密集型。连续批处理通过在每次迭代中打包更多token，将解码阶段推向这个交叉点。

## 动手构建

### 第1步：从零实现KV缓存

我们构建一个多头KV缓存，存储每一层、每个头的键和值投影，并演示内存增长模式。

```python
import numpy as np

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=np.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.k_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.v_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        num_new = new_keys.shape[1]
        end = self.seq_len + num_new
        self.k_cache[layer_idx, :, self.seq_len:end, :] = new_keys
        self.v_cache[layer_idx, :, self.seq_len:end, :] = new_values
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :]
        )

    def advance(self, num_tokens):
        self.seq_len += num_tokens

    def memory_bytes(self):
        return self.k_cache.nbytes + self.v_cache.nbytes

    def used_bytes(self):
        per_token = 2 * self.num_layers * self.num_heads * self.head_dim * np.dtype(self.dtype).itemsize
        return per_token * self.seq_len
```

### 第2步：带KV缓存的注意力机制

一个简化的多头注意力机制，在解码步骤中使用KV缓存。

```python
def scaled_dot_product_attention(query, keys, values):
    head_dim = query.shape[-1]
    scores = np.matmul(query, keys.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    seq_len_q = scores.shape[-2]
    seq_len_k = scores.shape[-1]
    if seq_len_q > 1:
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=np.float32), k=seq_len_k - seq_len_q + 1)
        scores = scores + mask * (-1e9)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, values)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def forward(self, x, kv_cache=None, layer_idx=0):
        batch, seq_len, d_model = x.shape
        Q = np.matmul(x, self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_full, V_full = kv_cache.update(layer_idx, K[0], V[0])
            K = K_full[np.newaxis, :, :, :]
            V = V_full[np.newaxis, :, :, :]
            if seq_len == 1:
                kv_cache.advance(1)

        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, -1, d_model)
        return np.matmul(attn_out, self.W_o)
```

### 第3步：连续批处理模拟器

这模拟了静态批处理和连续批处理之间的调度差异。

```python
import heapq

class Request:
    def __init__(self, request_id, prompt_tokens, output_tokens, arrival_step):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.arrival_step = arrival_step
        self.tokens_generated = 0
        self.start_step = None
        self.end_step = None

    def is_done(self):
        return self.tokens_generated >= self.output_tokens


def simulate_static_batching(requests, batch_size):
    step = 0
    completed = []
    queue = list(requests)
    queue.sort(key=lambda r: r.arrival_step)

    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            r = queue.pop(0)
            r.start_step = max(step, r.arrival_step)
            batch.append(r)

        if batch:
            step = max(step, max(r.start_step for r in batch))
            max_output = max(r.output_tokens for r in batch)
            for r in batch:
                r.tokens_generated = r.output_tokens
                r.end_step = step + max_output
            step += max_output
            completed.extend(batch)

    return completed


def simulate_continuous_batching(requests, batch_size):
    step = 0
    completed = []
    queue = sorted(requests, key=lambda r: r.arrival_step)
    queue_idx = 0
    active = []
    waiting = []

    while queue_idx < len(queue) or active or waiting:
        while queue_idx < len(queue) and queue[queue_idx].arrival_step <= step:
            waiting.append(queue[queue_idx])
            queue_idx += 1

        while waiting and len(active) < batch_size:
            r = waiting.pop(0)
            r.start_step = step
            active.append(r)

        if not active:
            if waiting:
                step += 1
                continue
            elif queue_idx < len(queue):
                step = queue[queue_idx].arrival_step
                continue
            else:
                break

        for r in active:
            r.tokens_generated += 1

        done = [r for r in active if r.is_done()]
        for r in done:
            r.end_step = step + 1
            completed.append(r)
        active = [r for r in active if not r.is_done()]

        step += 1

    return completed


def batching_stats(completed):
    latencies = [r.end_step - r.arrival_step for r in completed]
    total_time = max(r.end_step for r in completed) - min(r.arrival_step for r in completed)
    total_tokens = sum(r.output_tokens for r in completed)
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "throughput": total_tokens / total_time if total_time > 0 else 0,
    }
```

### 第4步：前缀缓存

一个基于trie的前缀缓存，存储共享前缀的KV条目。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_data = None
        self.hit_count = 0


class PrefixCache:
    def __init__(self, max_entries=1000):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def _walk(self, token_ids):
        node = self.root
        depth = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            depth += 1
        return node, depth

    def lookup(self, token_ids):
        node, depth = self._walk(token_ids)
        if depth > 0:
            self.hits += 1
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                current.hit_count += 1
            kv_entries = []
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                if current.kv_data is not None:
                    kv_entries.append(current.kv_data)
            return depth, kv_entries
        self.misses += 1
        return 0, []

    def insert(self, token_ids, kv_per_token):
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                if self.total_entries >= self.max_entries:
                    return i
                node.children[tid] = TrieNode()
                self.total_entries += 1
            node = node.children[tid]
            if i < len(kv_per_token):
                node.kv_data = kv_per_token[i]
        return len(token_ids)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### 第5步：推测解码模拟器

我们模拟草稿-目标推测解码，接受率可配置。

```python
class DraftModel:
    def __init__(self, vocab_size, acceptance_rate=0.8):
        self.vocab_size = vocab_size
        self.acceptance_rate = acceptance_rate

    def generate(self, context, num_tokens):
        tokens = np.random.randint(0, self.vocab_size, size=num_tokens)
        return tokens

    def get_probs(self, context, token):
        probs = np.random.dirichlet(np.ones(self.vocab_size))
        return probs


class TargetModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_probs(self, context, tokens=None):
        if tokens is not None:
            return [np.random.dirichlet(np.ones(self.vocab_size)) for _ in tokens]
        return np.random.dirichlet(np.ones(self.vocab_size))


def speculative_decode(draft_model, target_model, context, num_speculative=5,
                       draft_cost=1.0, target_cost=10.0, verify_cost=12.0):
    total_tokens = 0
    total_cost = 0.0
    accepted_counts = []
    context = list(context)

    max_tokens = 100

    while total_tokens < max_tokens:
        draft_tokens = draft_model.generate(context, num_speculative)
        total_cost += draft_cost * num_speculative

        target_probs = target_model.get_probs(context, draft_tokens)
        total_cost += verify_cost

        accepted = 0
        for i, token in enumerate(draft_tokens):
            draft_p = draft_model.get_probs(context + list(draft_tokens[:i]), token)
            target_p = target_probs[i]

            r = np.random.random()
            acceptance_prob = min(1.0, target_p[token] / (draft_p[token] + 1e-10))

            if r < draft_model.acceptance_rate:
                accepted += 1
                context.append(token)
                total_tokens += 1
            else:
                new_token = np.random.choice(draft_model.vocab_size, p=target_p)
                context.append(new_token)
                total_tokens += 1
                break

        accepted_counts.append(accepted)

        if accepted == num_speculative:
            bonus_probs = target_model.get_probs(context)
            bonus_token = np.random.choice(draft_model.vocab_size, p=bonus_probs)
            context.append(bonus_token)
            total_tokens += 1

    sequential_cost = total_tokens * target_cost
    return {
        "total_tokens": total_tokens,
        "speculative_cost": total_cost,
        "sequential_cost": sequential_cost,
        "speedup": sequential_cost / total_cost if total_cost > 0 else 1.0,
        "avg_accepted": np.mean(accepted_counts),
        "acceptance_rate": np.mean(accepted_counts) / num_speculative,
    }


def compare_speculation_strategies(vocab_size=1000, num_trials=20):
    results = {}

    for name, acceptance_rate, spec_tokens in [
        ("Draft-target (8B->70B)", 0.78, 5),
        ("EAGLE", 0.85, 6),
        ("N-gram", 0.50, 4),
        ("No speculation", 0.0, 0),
    ]:
        if spec_tokens == 0:
            results[name] = {
                "speedup": 1.0,
                "acceptance_rate": 0.0,
                "avg_accepted": 0.0,
            }
            continue

        trial_results = []
        for _ in range(num_trials):
            draft = DraftModel(vocab_size, acceptance_rate=acceptance_rate)
            target = TargetModel(vocab_size)
            context = list(np.random.randint(0, vocab_size, size=10))
            result = speculative_decode(draft, target, context, num_speculative=spec_tokens)
            trial_results.append(result)

        results[name] = {
            "speedup": np.mean([r["speedup"] for r in trial_results]),
            "acceptance_rate": np.mean([r["acceptance_rate"] for r in trial_results]),
            "avg_accepted": np.mean([r["avg_accepted"] for r in trial_results]),
        }

    return results
```

### 第6步：KV缓存内存分析器

为真实模型配置计算KV缓存内存需求。

```python
MODEL_CONFIGS = {
    "Llama-3-8B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 8, "gqa": True,
    },
    "Llama-3-70B": {
        "num_layers": 80, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 70, "gqa": True,
    },
    "Llama-3-405B": {
        "num_layers": 126, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 405, "gqa": True,
    },
    "Mistral-7B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 7, "gqa": True,
    },
    "GPT-4-est": {
        "num_layers": 120, "num_kv_heads": 96, "head_dim": 128,
        "model_params_b": 1800, "gqa": False,
    },
}


def kv_cache_memory(config, seq_len, dtype_bytes=2):
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * dtype_bytes
    total = per_token * seq_len
    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def memory_budget(config, gpu_memory_gb, model_dtype_bytes=2, kv_dtype_bytes=2):
    model_memory_gb = config["model_params_b"] * 1e9 * model_dtype_bytes / (1024 ** 3)
    overhead_gb = gpu_memory_gb * 0.1
    available_for_kv = gpu_memory_gb - model_memory_gb - overhead_gb

    if available_for_kv <= 0:
        return {"error": "Model does not fit in GPU memory", "model_memory_gb": model_memory_gb}

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_dtype_bytes
    max_tokens = int(available_for_kv * (1024 ** 3) / per_token)

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "model_memory_gb": round(model_memory_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "available_for_kv_gb": round(available_for_kv, 1),
        "max_total_tokens": max_tokens,
        "max_users_at_2k": max_tokens // 2048,
        "max_users_at_4k": max_tokens // 4096,
        "max_users_at_32k": max_tokens // 32768,
    }
```

## 实际应用

使用vLLM：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain inference optimization in one paragraph."], params)
```

使用SGLang进行前缀缓存 + 结构化输出：

```python
import sglang as sgl

@sgl.function
def classify(s, text):
    s += sgl.system("You are a classifier. Output JSON only.")
    s += sgl.user(f"Classify this text: {text}")
    s += sgl.assistant(sgl.gen("result", regex=r'\{"label": "(positive|negative|neutral)"\}'))

runtime = sgl.Runtime(model_path="meta-llama/Llama-3-70B-Instruct", tp_size=4)
sgl.set_default_backend(runtime)

results = classify.run_batch([
    {"text": "This product is amazing!"},
    {"text": "Terrible experience."},
    {"text": "It was okay I guess."},
])
```

使用TensorRT-LLM：

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-70b-trt-engine/", rank=0)

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("Explain KV caching.")],
    max_new_tokens=256,
    temperature=0.7,
)
```

## 交付成果

本课程产出：
- `outputs/skill-inference-optimization.md` -- 一项诊断和优化LLM推理服务的技能

## 练习

1. 修改KV缓存分析器，比较FP16、FP8和INT4量化下的KV缓存。对于在4xA100-80GB上运行的Llama 3 70B模型、4K上下文，计算每种量化下的最大并发用户数。KV缓存量化到INT4应该能将用户容量提升约4倍。

2. 扩展连续批处理模拟器，跟踪GPU利用率（每步填充的批处理槽位比例）。绘制50个请求（输出长度服从帕累托分布，形状参数=1.5，尺度参数=20）下，静态批处理和连续批处理的利用率随时间变化图。连续批处理应能维持>80%的利用率。

3. 实现一个分组查询注意力（GQA）版本的KV缓存，其中`num_kv_heads < num_query_heads`。Llama 3 70B使用64个查询头但只有8个KV头。计算其相比全多头注意力的内存节省（KV缓存大小减少8倍）。

4. 构建一个使用LRU淘汰策略的前缀缓存。设置max_entries为500，并生成1,000个请求，其中60%共享5个常见前缀中的一个。测量命中率，并与无限缓存比较。使用良好的淘汰策略，命中率应保持在55%以上。

5. 扩展推测解码模拟器，实现树状推测（EAGLE-2风格）。不再生成单一的K个草稿token链，而是生成一棵候选树（例如，在3层的每一层分出2个分支 = 8个叶节点候选）。比较每次验证轮次中接受的总token数与线性推测的差异。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|---------|-----------|
| 预填充 | “处理提示词” | 并行计算所有输入token的注意力——计算密集型，因为完整的矩阵乘法使GPU核心保持忙碌 |
| 解码 | “生成token” | 每次前向传播生成一个token，每次都读取完整的模型权重——内存密集型，因为计算在下一个权重到达之前就完成了 |
| KV缓存 | “缓存注意力状态” | 存储所有先前token的键和值投影，避免在每个解码步骤重新计算——用内存换算力 |
| 连续批处理 | “动态批处理” | 一旦有请求完成，立即将新请求插入到运行中的批次中，在每次解码迭代时评估，而不是等待整个批次完成 |
| PagedAttention | “KV缓存的虚拟内存” | 以固定大小的页而非连续块分配KV缓存，消除内存碎片并为共享前缀启用写时复制 |
| 推测解码 | “草稿与验证” | 使用一个快速的草稿模型提议多个token，然后在一个目标模型前向传播中验证它们——数学精确，2-3倍加速 |
| EAGLE | “自推测解码” | 推测解码的一种变体，在目标模型自身的隐藏状态上训练一个轻量头，比使用单独草稿模型实现更高的接受率 |
| 前缀缓存 | “重用系统提示词KV” | 为常见前缀（系统提示词、少样本示例）存储已计算的KV缓存条目，并在请求间重用，跳过冗余的预填充 |
| 操作字节比 | “算术强度” | 计算操作与读取内存字节的比率——决定一个工作负载是计算密集型（高比率）还是内存密集型（低比率） |
| 首token延迟 | “TTFT” | 从接收到请求到产生第一个输出token的延迟——对于长提示词，主要由预填充时间决定 |

## 延伸阅读

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) -- 引入了分页KV缓存管理的vLLM论文，现已成为推理服务的行业标准
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023) -- 证明草稿-验证推测能产生精确的目标模型分布同时实现2-3倍加速的基础论文
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024) -- 通过在目标模型自身的特征上训练一个头（而不是使用单独的草稿模型）实现了更高的接受率
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2024) -- 引入了用于前缀缓存的RadixAttention和一种用于多调用LLM程序的编程模型
- Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009) -- 形式化操作字节比框架以推理计算与内存瓶颈的原始Roofline论文