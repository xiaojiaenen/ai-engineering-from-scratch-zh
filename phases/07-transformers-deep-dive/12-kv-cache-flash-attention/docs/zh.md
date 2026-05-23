# KV 缓存、Flash Attention 与推理优化

> 训练是并行且受计算量限制的。推理是串行且受内存限制的。瓶颈不同，技巧各异。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 02 (自注意力机制)，第 7 阶段 · 05 (完整 Transformer)，第 7 阶段 · 07 (GPT)
**时间：** ~75 分钟

## 问题所在

一个朴素的自回归解码器需要执行 `O(N²)` 次计算来生成 `N` 个 token：在每一步，它都对整个前缀重新计算注意力。对于一个 4K token 的响应，这意味着 1600 万次注意力操作，其中大部分是冗余的。一旦前缀 token 的每个隐藏状态被计算出来，它就是确定的——你只需要将新 token 的查询与之前所有内容的缓存键和值进行计算即可。

除此之外，注意力机制本身需要移动大量数据。标准的注意力机制会物化一个 N×N 的分数矩阵、N×d 的 softmax 输出和 N×d 的最终输出——这导致了对高带宽内存 (HBM) 的过多读写。当 N≥2K 时，注意力机制在变得受计算量限制之前，先变得受内存限制了。经典的注意力内核对现代 GPU 的利用率仅为 4–10 倍。

来自 Dao 等人的两项优化，将前沿推理从“缓慢”推向了“快速”：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力计算就是一次查询与缓存的键的计算。推理过程在每个生成步骤从 `O(N²)` 减少到 `O(N)`。
2. **Flash Attention。** 对注意力计算进行分块处理，使得完整的 N×N 矩阵永远不会写入 HBM。所有的 softmax 和矩阵乘法都发生在 SRAM 中。在 A100 上实现 2–4 倍的墙上时间加速；在 H100 上配合 FP8 可达 5–10 倍。

到 2026 年，这两项技术都将成为标配。每一个生产级推理栈 (vLLM, TensorRT-LLM, SGLang, llama.cpp) 都默认支持它们。每一个前沿模型都内置启用了 Flash Attention。

## 核心概念

![KV 缓存增长与 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存的数学计算

每个解码器层、每个 token、每个注意力头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个 70 亿参数的模型，拥有 32 层、32 个头、d_head=128，使用 fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，使用 GQA，8 个 KV 头）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文下，仅 KV 缓存就需要占用 40 GB A100 显卡的大部分显存（批大小为 1 时）。

**GQA 是 KV 缓存的胜利。** 使用 64 个头的 MHA 需要 32 GB。MLA 则压缩得更极致。

### Flash Attention——分块技巧

标准注意力机制：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM 往返。在 H100 上，HBM 带宽为 3 TB/s；SRAM 为 30 TB/s。每次 HBM 访问相比所有数据保持在片上，都会带来 10 倍的速度减慢。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个分块一次 HBM 访问。总内存占用从 `O(N²)` 降低到 `O(N)`。反向传播会重新计算正向传播中的一些值，而不是存储它们——这是另一个内存节省点。

**数值技巧。** 运行 softmax 时会在分块间维护 `(max, sum)`，因此最终的归一化是精确的。这不是近似——Flash Attention 计算出的输出与标准注意力完全相同（考虑 fp16 非结合性的误差）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 在参考硬件上的加速比 |
|------|------|----------|------------------------|
| Flash 1 | 2022 | 分块 SRAM 内核 | 在 A100 上 2 倍 |
| Flash 2 | 2023 | 更好的并行性，因果优先排序 | 在 A100 上 3 倍 |
| Flash 3 | 2024 | Hopper 异步性，FP8 | 在 H100 上 1.5–2 倍（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线，软件 exp2 技巧 | 推理优先（最初仅支持前向） |

Flash 4 在发布时仅支持前向传播。训练仍然使用 Flash 3。Flash 4 对 GQA 和可变长度的支持待定（2026 年中期）。

### 投机解码——另一个延迟优化技巧

小型模型提出 N 个 token。大型模型并行验证所有 N 个。如果验证接受了 k 个 token，那么你用一次大型模型的前向传播换来了 k 个生成结果。在代码和文本生成中，典型的 k 为 3–5。

2026 年的默认配置：
- **EAGLE 2 / Medusa。** 集成的草案头，共享验证器的隐藏状态。可在无质量损失的情况下实现 2–3 倍加速。
- **使用草稿模型的投机解码。** 在消费级硬件上可实现 2–4 倍加速。
- **前瞻解码。** 雅可比迭代；无需草稿模型。小众但免费。

### 连续批处理

经典批处理推理：等待最慢的序列完成，然后开始新的批次。当短响应提前完成时会浪费 GPU 算力。

连续批处理（首先在 Orca 中实现，现已进入 vLLM, TensorRT-LLM, SGLang）：一旦旧请求完成，立即将新请求替换进批次。对于典型的聊天工作负载，可实现 5–10 倍的吞吐量增益。

### 分页注意力——将 KV 缓存作为虚拟内存

vLLM 的标志性特性。KV 缓存以 16 个 token 的块进行分配；一个页表将逻辑位置映射到物理块。这使您可以在并行样本（波束搜索、并行采样）之间共享 KV，为提示缓存热交换前缀，并减少内存碎片。相比朴素的连续分配，吞吐量提升 4 倍。

## 动手构建

参见 `code/main.py`。我们将实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 使用 KV 缓存的解码器。
3. 一个分块 softmax，用于模拟 Flash Attention 的运行最大值算法。

### 步骤 1：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很简单：在每个层、每个头的列表中，持续增长每个 token 的 K, V 向量。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与 `softmax(qK) V` 一次性计算的结果完全相同，但任意时刻的工作集只是一个 `tile × d_head` 块，而非完整的 `N × d_head`。

### 步骤 3：对比朴素解码与缓存解码在 100 个 token 生成上的差异

统计注意力操作次数。朴素解码：`O(N²)` = 5050。缓存解码：`O(N)` = 100。代码会打印两者。

## 使用它

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产环境：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一大收益——相同的系统提示、少量示例或长上下文文档可以在多次调用中复用 KV。对于需要重复工具提示的智能体工作负载，前缀缓存通常能带来 5 倍的吞吐量增益。

## 部署它

参见 `outputs/skill-inference-optimizer.md`。该技能为新的推理部署选择注意力实现、KV 缓存策略、量化和投机解码方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器产生相同的输出；注意操作次数的差异。
2. **中等。** 实现前缀缓存：给定一个提示 P 和多个补全，对 P 运行一次前向传播以填充 KV 缓存，然后为每个补全创建分支。测量与为每个补全重新编码 P 相比的加速比。
3. **困难。** 实现一个玩具版分页注意力：将 KV 缓存分配在固定的 16 个 token 块中，并使用空闲链表。当一个序列完成时，将其块归还给池。模拟 1,000 个长度不一的聊天补全。比较与连续分配相比的内存碎片情况。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|---------------|
| KV 缓存 | “让解码变快的技巧” | 存储自每个前缀 token 的 K 和 V；新的查询将注意力计算到它们身上，而非重新计算。 |
| HBM | “GPU 主内存” | 高带宽内存；H100 上为 80 GB，B200 上为 192 GB。带宽约 3 TB/s。 |
| SRAM | “片上内存” | 每个 SM 上的高速内存，H100 上每个 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | “分块注意力内核” | 在不物化 N×N 矩阵到 HBM 的情况下计算注意力。 |
| 连续批处理 | “无等待批处理” | 将完成的序列换出，将新序列换入，而无需排空批次。 |
| 分页注意力 | “vLLM 的王牌” | KV 缓存在固定块中分配，使用页表；消除碎片。 |
| 前缀缓存 | “复用长提示” | 为跨请求的共享前缀缓存 KV；是智能体工作负载的主要成本削减手段。 |
| 投机解码 | “草案 + 验证” | 廉价的草稿模型提出 token；大模型在一次传递中验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 级流水线和软件-exp2 技巧；请阅读仓库 README 以了解本课程提到的仅前向传播启动注意事项。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文，介绍本课程引用的集成草案方法。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 并列的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token 块和页表设计的权威深入解析。