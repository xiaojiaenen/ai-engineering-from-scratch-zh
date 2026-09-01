# 专家混合（MoE）

> 一个稠密的70B Transformer对每个token都会激活所有参数。而一个671B的MoE每个token只激活37B参数，却在各项基准测试中全面超越前者。稀疏性是这十年最重要的扩展理念。

**类型：** Build
**语言：** Python
**前置知识：** 第7阶段 · 05（完整Transformer）、第7阶段 · 07（GPT）
**预计时间：** 约45分钟

## 问题所在

稠密Transformer在推理时的FLOPs等于其参数量（正向传播乘以2）。放大稠密模型时，每个token都要支付全额账单。到2024年，前沿模型已触及计算瓶颈：想要显著更智能，就需要每个token呈指数级增长的FLOPs。

专家混合（Mixture of Experts）打破了这种绑定。将每个FFN替换为 `E` 个独立专家 + 一个为每个token选择 `k` 个专家的路由器。总参数 = `E × FFN_size`。每个token的激活参数 = `k × FFN_size`。典型2026配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026年的前沿几乎全是MoE：DeepSeek-V3（总计671B / 激活37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前10名开源模型全部是MoE。

## 核心概念

![MoE层：路由器为每个token从E个专家中选择k个](../assets/moe.svg)

### FFN的替换

稠密Transformer块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 为每个token从E个中选k个
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是一个独立的FFN（通常为SwiGLU）。路由器是一个单层线性网络。每个token自主挑选 `k` 个专家，并获得其输出的门控混合。

### 负载均衡问题

如果路由器将90%的token都送往专家3，其他专家就会饥饿。三种解决方案曾被尝试：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加与专家使用方差成正比的惩罚。有效，但引入了一个超参数和一个额外的梯度信号。
2. **专家容量限制 + token丢弃**（早期Switch）。每个专家最多处理 `C × N/E` 个token；超出部分的token跳过该层。会损害质量。
3. **无需辅助损失的平衡**（DeepSeek-V3）。为每个专家添加一个可学习的偏置，用于偏移路由器的top-k选择。偏置通过训练损失之外的方式更新。不影响主目标函数的损失。这是2024年的关键突破。

DeepSeek-V3的做法：在每个训练步骤之后，对于每个专家，检查其使用量高于还是低于目标值。将偏置按 `±γ` 调整。选择使用 `scores + bias`。用于门控的专家概率保持原始 `scores` 不变。将路由与输出表达解耦。

### 共享专家

DeepSeek-V2/V3还将专家拆分为*共享*和*路由*两类。每个token都会经过所有共享专家。路由专家通过top-k选择。共享专家捕获通用知识；路由专家负责 specialization。V3运行1个共享专家 + 256个路由专家中的top-8。

### 细粒度专家

经典MoE（GShard、Switch）：每个专家与完整FFN等宽。`E` 较小（8–64），`k` 较小（1–2）。

现代细粒度MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN大小）。`E` 较大（256+），`k` 也较大（8+）。总参数相同，但组合数增长快得多。`C(256, 8) = 400万亿` 个可能的token级"专家"组合。质量提升，延迟保持不变。

### 成本构成

每个token、每层：

| 配置 | 每token激活参数 | 总参数 |
|------|-----------------|--------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3在几乎所有基准测试上都超越了Llama 3 70B（稠密），同时**每个token的激活FLOPs更少**。更多参数 = 更多知识。更多激活FLOPs = 每个token更多计算。MoE将二者解耦。

### 隐患：显存

所有专家无论是否被激活都驻留在GPU上。一个671B模型需要约1.3 TB的fp16权重显存。前沿MoE部署需要专家并行——在GPU间分片专家，通过网络路由token。延迟主要由all-to-all通信主导，而非矩阵乘法。

```figure
expert-routing
```

## 动手实现

参见 `code/main.py`。一个纯标准库实现的紧凑MoE层，包含：

- `n_experts=8` 个类SwiGLU专家（每个专家仅一个线性层，用于演示）
- top-k=2 路由
- softmax归一化门控权重
- 通过每专家偏置实现无辅助损失的平衡

### 步骤1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对选中的专家使用原始score计算softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置只影响选择，不影响门控权重。这就是DeepSeek-V3的技巧——偏置纠正负载不均衡，但不引导模型的预测。

### 步骤2：让100个token通过路由器

追踪各专家的触发频率。没有偏置时，使用分布不均匀。加入偏置更新循环（对过度使用的专家 `-γ`，对使用不足的专家 `+γ`），几次迭代后使用率就会收敛到均匀分布。

### 步骤3：参数量对比

打印MoE配置的"稠密等价"参数。DeepSeek-V3架构：256个路由专家 + 1个共享专家，8个激活，d_model=7168。总参数量令人咋舌。而激活参数仅为稠密Llama 3 70B的七分之一。

## 使用它

HuggingFace加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026年生产推理：vLLM原生支持MoE路由。SGLang拥有最快的专家并行路径。两者都自动处理top-k选择和专家并行。

**何时选择MoE：**
- 你希望在更低的每token推理成本下获得前沿质量。
- 你拥有足够的VRAM / 专家并行基础设施。
- 你的工作负载是token密集型（聊天、代码）而非上下文密集型（长文档）。

**何时不选择MoE：**
- 边缘部署——任何激活FLOPs都要支付全额存储开销。
- 低延迟关键的单用户服务——专家路由带来额外开销。
- 小模型（<7B）——MoE的质量优势只在超过一定计算阈值后显现（约6B激活参数）。

## 交付

参见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练token数和部署目标，为新MoE选择E、k和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失的偏置更新如何在50次迭代内使专家使用趋于均衡。
2. **中等。** 将可学习路由器替换为基于哈希的路由器（确定性、无需学习）。比较质量与平衡性。为什么可学习路由器更好？
3. **困难。** 实现GRPO风格的"rollout匹配路由"（DeepSeek-V3.2技巧）：记录推理时哪些专家被激活，在梯度计算时强制使用相同的路由。在玩具策略梯度设置下测量其影响。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| 专家（Expert） | "众多FFN中的一个" | 一个独立的前馈网络；参数专用于FFN计算的稀疏子集。 |
| 路由器（Router） | "门控" | 一个小型线性层，对每个token计算与各专家的相关分数；进行top-k选择。 |
| Top-k路由 | "每token k个激活专家" | 每个token的FFN计算恰好经过k个专家，由门控加权。 |
| 辅助损失（Auxiliary loss） | "负载均衡惩罚" | 额外的损失项，惩罚不均衡的专家使用。 |
| 无辅助损失（Auxiliary-loss-free） | "DeepSeek-V3的技巧" | 仅通过对路由器选择的每专家偏置实现平衡；不引入额外梯度。 |
| 共享专家（Shared expert） | "始终激活" | 一个每个token都会经过的额外专家；捕获通用知识。 |
| 专家并行（Expert parallelism） | "按专家分片" | 将不同专家分配到不同GPU；通过网络路由token。 |
| 稀疏性（Sparsity） | "激活参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3约为37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 最初的理念。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的平衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本教程路由器使用的细粒度 + 共享专家拆分。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 共享专家原始论文。
