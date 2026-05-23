# 混合专家模型 (MoE)

> 一个密集的 70B Transformer 模型在处理每个 token 时都会激活所有参数。而一个 671B 的 MoE 模型每个 token 仅激活 37B 参数，并在所有基准测试中都超越了前者。稀疏性是本世纪最重要的扩展思路。

**类型:** 构建
**语言:** Python
**前提:** 阶段 7 · 05 (完整 Transformer), 阶段 7 · 07 (GPT)
**时间:** ~45 分钟

## 问题所在

一个密集 Transformer 模型的推理浮点运算量（FLOPs）等于其参数量（前向传播时乘以 2）。扩大密集模型的规模意味着每个 token 都要承担全部计算成本。到 2024 年，前沿模型遇到了算力瓶颈：要让模型变得更智能，每个 token 所需的浮点运算量呈指数级增长。

混合专家模型打破了这种关联。将每个前馈网络（FFN）替换为 `E` 个独立的专家和一个路由器，路由器为每个 token 选择 `k` 个专家。总参数量 = `E × FFN_size`。每个 token 的激活参数量 = `k × FFN_size`。典型的 2026 年配置为：`E=256`, `k=8`。存储量随 `E` 缩放，计算量随 `k` 缩放。

2026 年的前沿模型几乎全是 MoE：DeepSeek-V3 (总计 671B / 激活 37B)、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，排名前十的开源模型都是 MoE。

## 核心概念

![MoE 层：路由器为每个 token 从 E 个专家中选择 k 个](../assets/moe.svg)

### FFN 的替换

密集 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是一个独立的 FFN（通常是 SwiGLU）。路由器是一个简单的线性层。每个 token 选择自己的 `k` 个专家，并获得它们输出的门控混合结果。

### 负载均衡问题

如果路由器将 90% 的 token 路由到专家 3，其他专家就会闲置。目前已尝试三种解决方案：

1.  **辅助负载均衡损失**（Switch Transformer, Mixtral）。添加一个与专家使用量方差成比例的惩罚项。有效，但会增加一个超参数和第二个梯度信号。
2.  **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 跳过该层。会损害模型质量。
3.  **无辅助损失的负载均衡**（DeepSeek-V3）。为每个专家添加一个可学习的偏置，用于调整路由器的 top-k 选择。偏置的更新独立于训练损失。对主目标没有惩罚。这是 2024 年的重大突破。

DeepSeek-V3 的方法：在每个训练步骤后，对于每个专家，检查其使用量是高于还是低于目标值。将偏置调整 `±γ`。选择时使用 `scores + bias`。用于门控的专家概率是原始的 `scores`，保持不变。实现了路由与表达的解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为 *共享专家* 和 *路由专家*。每个 token 都经过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家则进行专门化。V3 运行 1 个共享专家加上 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE (GShard, Switch)：每个专家宽度与完整 FFN 相同。`E` 较小 (8–64)，`k` 较小 (1–2)。

现代细粒度 MoE (DeepSeek-V3, Qwen-MoE)：每个专家更窄 (1/8 FFN 宽度)。`E` 较大 (256+)，`k` 更大 (8+)。总参数相同，但组合方式增长快得多。每个 token 可能的"专家"组合为 `C(256, 8) = 400 trillion`。质量提升，延迟保持不变。

### 成本概况

每个 token，每层：

| 配置 | 激活参数/token | 总参数 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (密集) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎所有基准测试中都击败了 Llama 3 70B (密集)，同时**每个 token 的激活浮点运算更少**。更多参数 = 更多知识。更多激活浮点运算 = 每个 token 更多计算。MoE 将两者解耦。

### 隐患：内存

无论激活了哪些专家，所有专家都常驻在 GPU 上。一个 671B 的模型在 fp16 权重下需要约 1.3 TB 显存。前沿的 MoE 部署需要专家并行——将专家分片到多个 GPU 上，通过网络路由 token。延迟主要由全对全通信主导，而非矩阵乘法。

## 动手构建

参见 `code/main.py`。一个用纯标准库实现的紧凑 MoE 层，包含：
- `n_experts=8` 个类 SwiGLU 的专家（为便于说明，每个仅用一个线性层）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过每个专家的偏置实现的无辅助损失负载均衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置在不改变模型预测方向的情况下，纠正负载不平衡。

### 步骤 2：通过路由器运行 100 个 token

跟踪每个专家的激活频率。没有偏置时，使用分布会倾斜。通过一个偏置更新循环（对于使用过多的专家调整 `-γ`，对于使用不足的专家调整 `+γ`），几次迭代后使用分布就会趋于均匀。

### 步骤 3：参数量对比

打印 MoE 配置的"密集等效"参数量。DeepSeek-V3 结构：256 个路由专家 + 1 个共享专家，激活 8 个，d_model=7168。总参数量惊人。激活参数量仅为密集 Llama 3 70B 的七分之一。

## 实际应用

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产环境推理：vLLM 原生支持 MoE 路由。SGLang 具有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你希望以更低的每 token 推理成本获得前沿质量。
- 你拥有足够的显存/专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码），而非上下文密集型（长文档）。

**何时不选择 MoE：**
- 边缘部署——你为任何激活的浮点运算支付了全部存储成本。
- 延迟敏感的单用户服务——专家路由会增加开销。
- 小模型 (<7B)——MoE 的质量优势仅在计算阈值（约 6B 激活参数）以上才显现。

## 部署

参见 `outputs/skill-moe-configurator.md`。该技能根据给定的参数预算、训练 token 数和部署目标，为新 MoE 选择 E、k 和共享专家布局。

## 练习

1.  **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在 50 次迭代内平衡专家使用率。
2.  **中等。** 将可学习路由器替换为基于哈希的路由器（确定性，无需学习）。比较质量和平衡性。为什么可学习路由器更好？
3.  **困难。** 实现类似 GRPO 的"推演匹配路由"（DeepSeek-V3.2 的技巧）：记录推理期间激活的专家，在梯度计算期间强制执行相同的路由。在一个玩具策略梯度设置上衡量其效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 专家 | "众多 FFN 中的一个" | 一个独立的前馈网络；其参数专门用于 FFN 计算的一个稀疏切片。 |
| 路由器 | "门控" | 一个微小的线性层，为每个 token 对每个专家评分；进行 top-k 选择。 |
| Top-k 路由 | "每个 token 有 k 个激活的专家" | 每个 token 的 FFN 计算恰好经过 k 个专家，由门控加权。 |
| 辅助损失 | "负载均衡惩罚" | 额外的损失项，惩罚不均匀的专家使用。 |
| 无辅助损失 | "DeepSeek-V3 的技巧" | 通过路由器选择阶段的每个专家偏置实现平衡；无额外梯度。 |
| 共享专家 | "始终开启" | 每个 token 都经过的额外专家；捕获通用知识。 |
| 专家并行 | "按专家分片" | 将不同的专家分配到不同的 GPU；通过网络路由 token。 |
| 稀疏性 | "激活参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 中为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer 等人 (2017)。Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 原始构想。
- [Fedus, Zoph, Shazeer (2022)。Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang 等人 (2024)。Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024)。DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP。
- [Wang 等人 (2024)。Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的平衡论文。
- [Dai 等人 (2024)。DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器所用的细粒度 + 共享专家分离方法。
- [Kim 等人 (2022)。DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始共享专家论文。