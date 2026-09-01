# 投机解码 — 草稿、验证、循环

> 自回归解码是串行的。每个 token 都要等待前一个完成。投机解码打破了这个链条：一个廉价模型起草 N 个 token，昂贵模型一次前向传播验证所有 N 个。当草稿正确时，你只需支付一次大前向传播来获得 N 个生成结果。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 07（GPT 因果语言模型）、阶段 7 · 12（KV 缓存与 Flash Attention）
**时间：** 约 60 分钟

## 问题所在

在 H100 上，一个 70B LLM 采样一个 token 需要约 30 ms。而一个 3B 的草稿模型仅需约 3 ms。如果我们让 3B 草稿模型超前 5 个 token，然后运行一次 70B 模型来验证所有 5 个，总耗时为 `5×3 + 30 = 45 ms`，最多可接受 5 个 token——相比之下，线性生成需要 `5×30 = 150 ms`。这就是投机解码的核心卖点：以少量额外的 GPU 显存（草稿模型）换取 2–4 倍的解码延迟降低。

关键是要保证分布不变。Leviathan 等人（2023）和 Chen 等人同期提出的投机采样（speculative sampling）保证输出序列与被验证模型独立采样产生的序列**具有相同的分布**。没有质量损失，只有更快的速度。

2026 年主导推理的草稿-验证器对主要有四个流派：

1. **基础投机（Leviathan 2023）**。分离的草稿模型（如 Llama 3 1B）+ 验证器（如 Llama 3 70B）。
2. **Medusa（Cai 2024）**。在验证器上部署多个解码头，并行预测位置 `t+1..t+k`。无需单独的草稿模型。
3. **EAGLE 系列（Li 2024, 2025）**。轻量级草稿模型复用验证器的隐藏状态；接受率比基础方法更高，典型加速 3–4 倍。
4. **前瞻解码（Lookahead decoding，Fu 2024）**。Jacobi 迭代；完全不需要草稿模型。自投机。属于小众方案但无依赖。

2026 年的所有生产级推理框架默认都支持投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持基础方法加 EAGLE-2。

## 核心概念

### 核心算法

给定验证器 `M_q` 和更廉价的草稿模型 `M_p`：

1. 令 `x_1..x_k` 为已解码的前缀。
2. **草稿**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，得到草稿概率 `p_1..p_N`。
3. **并行验证**：对 `M_q` 一次性运行 `x_1..x_k, d_{k+1}, ..., d_{k+N}`，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右逐 token 接受/拒绝**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次被拒绝时：从归一化的"残差"分布 `(q_j - p_j)_+` 中采样 `t_j`。位置 `j` 之后的所有草稿 token 被丢弃。
6. 若所有 `N` 个草稿都被接受：从 `q_{N+1}` 中采样一个额外的 token `t_{N+1}`（免费奖励 token）。

残差分布的技巧是保持输出分布与 `M_q` 从头采样完全一致的关键数学洞察。

### 决定加速效果的因素

令 `α` = 每个草稿 token 的期望接受率，`c` = 草稿模型与验证器的成本比。每步：

- 朴素生成对每个 token 需要 1 次大模型调用。
- 投机解码在高 `α` 下对每个 token 需要 `1/(1-α)` 次大模型调用（精确公式为 `1/( (1-α)/(1-α^{N+1})) ≈ 1/(1-α)`）。

在 `α = 0.75` 且 `N = 5` 时的典型经验法则是：大模型调用次数减少 3 倍。草稿成本是 5 倍廉价。总 wall-clock 时间降低约 2.5 倍。

**α 取决于：**

- 草稿模型对验证器的逼近程度。同家族/相同训练数据能显著提升 α。
- 解码策略。贪婪草稿对抗贪婪验证器：α 高。温度采样：更难匹配，接受率下降。
- 任务类型。代码和结构化输出更容易接受（可预测）；自由创意写作接受率较低。

### Medusa —— 无需草稿模型的草稿

Medusa 用验证器上的额外输出头替代草稿模型。在位置 `t`：

```
共享 trunk → 隐藏状态 h_t
    ├── head_0: 预测 t+1 位置的 token（标准 LM 头）
    ├── head_1: 预测 t+2 位置的 token
    ├── head_2: 预测 t+3 位置的 token
    ├── head_3: 预测 t+4 位置的 token
```

每个头输出自己的 logits。在推理时从每个头采样得到候选序列，然后用一次前向传播通过树注意力方案验证，一次性考虑所有候选延续。

优点：无需第二模型。缺点：增加可训练参数；需要监督微调阶段（约 1B tokens）；接受率略低于带优质草稿的基础投机方法。

### EAGLE —— 通过复用隐藏状态实现更好的草稿

EAGLE-1/2/3（Li 等，2024–2025）将草稿模型做成一个轻量 transformer（通常 1 层），直接摄入验证器的最后一层隐藏状态。由于草稿能看到验证器的特征表示，其预测与验证器的输出分布高度相关。接受率从约 0.6（基础方法）提升到 0.85+。

EAGLE-3（2025）加入了在候选延续上的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机路径发布。

### KV 缓存操作

验证器一次性接收 `N` 个草稿 token 进行前向传播。这会将验证器的 KV 缓存扩展 `N` 个条目。如果部分草稿被拒绝，必须将缓存回滚到已接受前缀的长度。

生产级实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）通过临时 KV 缓冲区处理此问题。先写入，接受时提交。概念上不复杂，但比较繁琐。

```figure
draft-verify-tokens
```

## 动手实现

参见 `code/main.py`。我们实现核心投机采样算法（拒绝步骤 + 残差分布），使用：

- 一个基于手工编码分布的确定性 softmax"大模型"（便于数学分析接受率）。
- 一个作为大模型扰动版本的"草稿模型"。
- 一个接受/拒绝循环，产生与直接采样相同的边缘分布。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草稿 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理表明，这个伯努利决策加上拒绝时从残差分布采样，能精确保持验证器的分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素将 `p` 从 `q` 中减去，将负值钳制为零，然后归一化。在任何拒绝时从此分布采样。

### 步骤 3：单次投机步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

5 个 token 被接受 → 1 个奖励 token → 一次验证器前向传播产生 6 个 token。

### 步骤 4：测量接受率

运行 10,000 次投机步骤，在不同草稿质量级别下。绘制接受率与草稿/验证器分布之间 KL 散度的关系图。你应该看到一个清晰的单调关系。

### 步骤 5：验证分布等价性

经验上：投机循环产生的 token 直方图应与直接从验证器采样产生的直方图匹配。这是 Leviathan 定理的实践验证。卡方检验可在采样误差范围内确认这一点。

## 使用它

生产环境：

```bash
# 使用 EAGLE 的 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# 使用基础草稿模型的 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 包装了投机解码，使用小型草稿模型。

**选择草稿模型：**

| 策略 | 适用场景 | 加速比 |
|------|----------|--------|
| 基础草稿（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa 头 | 可以微调验证器 | 2–3× |
| EAGLE-2 / 3 | 生产环境，追求最大速度 | 3–4× |
| Lookahead | 无草稿、无训练、无额外参数 | 1.3–1.6× |

**不应使用投机解码的场景：**

- 1–5 个 token 的单序列生成。开销占主导。
- 极端创造性/高温采样（α 下降）。
- 显存受限的部署（草稿模型增加 VRAM 占用）。

## 交付成果

参见 `outputs/skill-spec-decode-picker.md`。该技能为新推理工作负载选择投机解码策略（基础/Medusa/EAGLE/lookahead）和调优参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 个 token 上确认投机 token 分布与验证器的直接采样分布在卡方 p > 0.05 范围内匹配。
2. **中等。** 绘制加速比（每次大模型前向传播的 token 数）作为 `N` 的函数，针对 `α = 0.5, 0.7, 0.85`。找出每种 α 下的最优 `N`。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个轻量 Medusa：取第 14 课的毕业设计 GPT，添加 3 个额外的 LM 头，分别预测 t+2、t+3、t+4 位置。在 tinyshakespeare 上使用联合多头损失训练。将接受率与通过截断同一模型得到的基础草稿进行比较。
4. **困难。** 实现回滚：从 10 个 token 前缀的 KV 缓存开始，喂入 5 个草稿 token，模拟在位置 3 发生拒绝。验证下次迭代时你的缓存读取正确匹配"前缀 + 前 2 个已接受草稿"。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Draft model | "便宜的模型" | 提出候选 token 的小型模型；通常比验证器便宜 10–50 倍。 |
| Verifier | "大模型" | 我们要保持其分布的目标模型；每次投机步骤运行一次。 |
| Acceptance rate (α) | "草稿正确的频率" | 验证器接受草稿的逐 token 概率。典型范围 0.7–0.9。 |
| Residual distribution | "拒绝时的后备" | `(q - p)_+` 归一化；在拒绝时从此采样可保持验证器分布。 |
| Bonus token | "免费的 token" | 当所有 N 个草稿被接受时，从验证器的下一步分布中再采样一个。 |
| Medusa | "无草稿的投机" | 验证器上的多个 LM 头并行预测 t+1..t+k 位置。 |
| EAGLE | "隐藏状态草稿" | 轻量 transformer 草稿，条件基于验证器的最后一层隐藏状态。 |
| Lookahead decoding | "Jacobi 迭代" | 使用不动点迭代的自投机；无需草稿模型。 |
| Tree attention | "一次性验证多个候选" | 分支验证，同时考虑多个草稿延续。 |
| KV rollback | "撤销拒绝的草稿" | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法与等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期提出；清晰的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隐藏状态条件草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 前瞻解码，无草稿方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 包含四种策略的标准生产参考。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。
