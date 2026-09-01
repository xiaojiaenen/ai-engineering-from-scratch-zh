# KV Cache、Flash Attention 与推理优化

> 训练是并行的，以 FLOP 为瓶颈。推理是串行的，以内存为瓶颈。瓶颈不同，技巧也不同。

**类型：** Build
**语言：** Python
**前置知识：** 第 7 阶段 · 02（Self-Attention）、第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 07（GPT）
**预计时间：** 约 75 分钟

## 问题所在

一个朴素的外推式解码器在生成 `N` 个 token 时需要执行 `O(N²)` 次计算：每一步都要对整个前缀重新计算注意力。对于一个 4K token 的回复，那就是 1600 万次注意力操作，其中大部分是冗余的。一旦前缀 token 的隐藏状态被计算出来，它就是确定的——你只需要用新 token 的 query 去查询之前所有 token 的缓存 key 和 value 即可。

除此之外，注意力本身的开销也很大。标准注意力会显式计算出 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 来说读写字太多。当 N≥2K 时，注意力在成为 FLOP 瓶颈之前就已经是内存瓶颈了。经典注意力核函数在现代 GPU 上的利用率比理论值低 4–10 倍。

两项来自 Dao 等人的优化，将前沿推理从"慢"推到了"快"：

1. **KV cache。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力就是对缓存 key 的一次查询。推理复杂度从每步 `O(N²)` 降为 `O(N)`。
2. **Flash Attention。** 分块计算注意力，使得完整的 N×N 矩阵从不进入 HBM。softmax + matmul 的全部计算都在 SRAM 中完成。在 A100 上实现 2–4 倍的墙钟时间加速；在搭载 FP8 的 H100 上实现 5–10 倍加速。

到 2026 年，两者已是标配。每个生产级推理框架（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认使用它们。每个前沿模型都启用了 Flash Attention。

## 核心概念

![KV cache 增长与 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV cache 计算

每个解码层、每个 token、每个 head：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K 和 V
```

对于一个 7B 模型（32 层、32 head、d_head=128、fp16）：

```
每 token 每层 = 2 * 128 * 2 = 512 bytes
每 token（32 层）= 16 KB
每 32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层、d_head=128、GQA 配 8 个 KV head）：

```
每 token 每层 = 2 * 8 * 128 * 2 = 4096 bytes（4 KB）
每 32K 上下文 = 10.4 GB
```

这就是为什么 Llama 3 70B 在 128K 上下文、batch size=1 时，KV cache 就需要占用一块 40 GB A100 的大部分显存。

**GQA 是 KV cache 的关键收益。** MHA 配 64 个 head 就是 32 GB。MLA 压缩得更厉害。

拖动维度参数，观察 cache 大小的变化。提高序列长度或 batch size，看它有多快就能撑爆一张单卡 GPU：

```figure
kv-cache-sizer
```

### Flash Attention — 分块技巧

标准注意力：

```
S = Q @ K^T          (HBM 读，N×N，HBM 写)
P = softmax(S)       (HBM 读，HBM 写)
O = P @ V            (HBM 读，HBM 写)
```

三次 HBM 往返。在 H100 上，HBM 带宽为 3 TB/s；SRAM 为 30 TB/s。每一次 HBM 往返都相当于把速度降低一个数量级，把所有数据都留在片上才是正道。

Flash Attention：

```
对 Q 的每个分块（tile 大小约 128 × 128）：
    将 Q_tile 载入 SRAM
    对 K、V 的每个分块：
        将 K_tile、V_tile 载入 SRAM
        在 SRAM 中计算 S_tile = Q_tile @ K_tile^T
        运行式 softmax 累积（在 SRAM 中）
        累加到 O_tile（在 SRAM 中）
    将 O_tile 写回 HBM
```

每个 tile 只有一次 HBM 往返。总内存占用从 `O(N²)` 降为 `O(N)`。反向传播从正向传递中重算部分值，而不是全部存储——又是一次内存收益。

**数值技巧。** 运行式 softmax 跨 tile 维护 `(max, sum)`，使最终归一化保持精确。不是近似——Flash Attention 计算出的输出与标准注意力位相同（除 fp16 非结合性外）。

**版本演进：**

| 版本 | 年份 | 关键改进 | 参考硬件加速比 |
|------|------|---------|---------------|
| Flash 1 | 2022 | 分块 SRAM 核函数 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性、因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5–2×（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 五阶段流水线、软件 exp2 | 推理优先（初始仅支持前向） |

Flash 4 在发布时仅支持前向传播。训练仍使用 Flash 3。GQA 和 varlen 支持在 Flash 4 中处于待开发状态（预计 2026 年中）。

### 投机解码 — 另一种延迟优化

小模型提出 N 个 token。大模型并行验证全部 N 个。如果验证接受了 k 个 token，则你用一次大模型前向传播换来了 k 次生成。在代码和文本场景下，典型 k=3–5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 集成的草稿头，共享验证器的隐藏状态。2–3 倍加速，无任何质量损失。
- **基于草稿模型的投机解码。** 消费级硬件上 2–4 倍加速。
- **前瞻解码。** Jacobi 迭代；无需草稿模型。小众但免费。

### 连续批处理

经典批处理推理：等最慢的序列完成后，再开始新批次。短回复提前结束时会浪费 GPU。

连续批处理（最早由 Orca 实现，现已集成于 vLLM、TensorRT-LLM、SGLang）：旧请求一完成，立即将新请求换入批次。对典型对话工作负载实现 5–10 倍的吞吐提升。

### PagedAttention — KV cache 作为虚拟内存

vLLM 的核心功能。KV cache 以 16-token 为块进行分配；页表将逻辑位置映射到物理块。支持在并行样本间共享 KV（束搜索、并行采样）、热替换前缀以实现提示缓存，以及消除内存碎片。相比朴素连续分配，吞吐量提升 4 倍。

```figure
flash-attention-memory
```

## 动手实现

参见 `code/main.py`。我们实现：

1. 一个朴素 `O(N²)` 的增量解码器。
2. 一个 `O(N)` 的 KV 缓存解码器。
3. 一个模拟 Flash Attention 运行式最大值的分块 softmax。

### 步骤 1：KV cache

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

很简单：在每个层、每个 head 的列表中持续追加 token 级的 K、V 向量。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 风格的 softmax(qK^T)V，带运行式 max/sum。"""
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

一次性计算出与 `softmax(qK) V` 位相同的输出，但在任意时刻的工作集只是一个 `tile × d_head` 的块，而不是完整的 `N × d_head`。

### 步骤 3：对比朴素解码与缓存解码在 100-token 生成上的表现

统计注意力操作次数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会同时打印两者的结果。

## 使用方式

```python
# HuggingFace transformers 在 decoder-only generate() 上自动启用 KV cache。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 可用 FA3
    torch_dtype="bfloat16",
)
# generate() 会自动使用 KV cache
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

跨请求的前缀缓存是 2026 年的重要收益——相同的系统提示、few-shot 示例或长上下文文档可以在多次调用间复用 KV。对于有大量重复 tool prompt 的智能体工作负载，前缀缓存 routinely 带来 5 倍的吞吐提升。

## 交付成果

参见 `outputs/skill-inference-optimizer.md`。该 skill 会为新的推理部署选择注意力实现方案、KV cache 策略、量化方式和投机解码。

## 练习题

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器产生相同的输出；注意操作次数的差异。
2. **中等。** 实现前缀缓存：给定提示 P 和若干补全，对 P 只运行一次前向传播来填充 KV cache，然后按各补全分支。测量与对每个补全重新编码 P 的速度提升。
3. **困难。** 实现玩具版 PagedAttention：KV cache 以固定 16-token 块存储，带空闲列表。序列完成后将其块归还到池子。模拟 1000 条长度各异的聊天补全。对比内存碎片率与连续分配的碎片率。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| KV cache | "让解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新 query 直接对其注意力，而非重新计算。 |
| HBM | "GPU 主存" | High Bandwidth Memory；H100 上 80 GB，B200 上 192 GB。带宽约 3 TB/s。 |
| SRAM | "片上内存" | 每个 SM 的高速内存，H100 上每 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | "分块注意力核函数" | 在不将 N×N 矩阵物化到 HBM 的情况下计算注意力。 |
| Continuous batching | "无等待批处理" | 完成的序列换出，新序列换入，无需排空整个批次。 |
| PagedAttention | "vLLM 的核心" | KV cache 以固定块分配，带页表；消除内存碎片。 |
| Prefix caching | "复用长提示" | 跨请求缓存共享前缀的 KV；大幅降低智能体的成本。 |
| Speculative decoding | "草稿 + 验证" | 轻量草稿模型提出 token；大模型一次前向传播验证 k 个。 |

## 延伸阅读

- [Dao 等 (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah 等 (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 发布说明（Dao-AILab，2026）](https://github.com/Dao-AILab/flash-attention) — Blackwell 五阶段流水线与软件 exp2 技巧；阅读仓库 README 了解本课提到的"仅前向传播发布"的限制说明。
- [Kwon 等 (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan 等 (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码。
- [Li 等 (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文，本课引用的集成草稿方案。
- [Cai 等 (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 并列引用的 Medusa 方案。
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token 块和页表设计的权威深入讲解。
