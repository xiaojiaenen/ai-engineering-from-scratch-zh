# 推测解码 —— 草稿、验证、重复

> 自回归解码是串行的。每个 token 都需等待前一个生成。推测解码打破了这一链条：一个廉价模型草拟 N 个 token，昂贵的模型在一次前向传播中验证全部 N 个。当草稿正确时，你仅为 N 次生成支付一次大型前向传播的成本。

**类型：** 构建
**语言：** Python
**前提条件：** 第 7 阶段 · 07（GPT 因果语言模型），第 7 阶段 · 12（KV 缓存与 Flash Attention）
**时间：** 约 60 分钟

## 问题所在

一个 700 亿参数的大语言模型在 H100 上采样一个 token 需要约 30 毫秒。一个 30 亿参数的草稿模型需要约 3 毫秒。如果我们让 30 亿参数模型提前草拟 5 个 token，然后运行一次 700 亿参数模型来验证全部 5 个，总时间约为 `5×3 + 30 = 45 ms`，可获得最多 5 个被接受的 token —— 而直线生成则需要 `5×30 = 150 ms`。这就是推测解码的核心主张：用少量额外 GPU 内存（草稿模型）换取 2–4 倍的解码延迟降低。

这个技巧必须保持分布不变。由 Leviathan 等人（2023 年）和 Chen 等人同时引入的推测采样，保证了输出序列与大模型独立生成的序列**分布完全相同**。没有质量折衷。只是更快。

在 2026 年的推理中，四类草稿-验证器组合占据主导：

1. **原始推测解码（Leviathan 2023）。** 独立的草稿模型（例如 Llama 3 1B） + 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上的多个解码头并行预测 `t+1..t+k` 位置。无需独立的草稿模型。
3. **EAGLE 系列（Li 2024, 2025）。** 轻量级草稿模型，重用验证器的隐藏状态；接受率比原始方法更高；典型加速 3–4 倍。
4. **前瞻解码（Fu 2024）。** 雅可比迭代；完全不需要草稿模型。自我推测。小众但无依赖。

2026 年所有生产环境推理栈默认支持推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少支持原始方法 + EAGLE-2。

## 核心概念

### 核心算法

给定验证器 `M_q` 和更廉价的草稿模型 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **草拟**：使用 `M_p` 自回归地提议 `d_{k+1}, d_{k+2}, ..., d_{k+N}` 个 token，草稿概率为 `p_1..p_N`。
3. **并行验证**：对 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 运行一次 `M_q`，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿 token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次被拒绝时：从归一化的“残差”分布 `(q_j - p_j)_+` 中采样 `t_j`。`j` 之后的所有草稿都被丢弃。
6. 当全部 `N` 被接受时：从 `q_{N+1}`（即下一步分布）额外采样一个 token `t_{N+1}`（免费的额外 token）。

残差分布技巧是一个数学洞见，它确保了输出分布完全等同于 `M_q` 从头开始采样。

### 决定加速的因素

设 `α` = 每个草稿 token 的期望接受率。设 `c` = 草稿模型与验证器的成本比率。每一步：

- 原始生成每 token 调用一次大模型。
- 当 `α` 较高时，推测解码每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 调用一次大模型。

在 `α = 0.75` 和 `N = 5` 时的典型经验法则：大模型调用次数减少 3 倍。草稿成本低 5 倍。总挂钟时间下降约 2.5 倍。

**α 取决于：**

- 草稿模型逼近验证器的程度。同系列 / 同训练数据会显著提升 α。
- 解码策略。草稿模型和验证器都使用贪心解码：α 高。使用温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受率更高（可预测）；自由形式的创意写作接受率更低。

### Medusa —— 无需草稿模型的草稿

Medusa 用验证器上的额外输出头取代了草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的逻辑值。在推理时，你从每个头采样得到候选序列，然后使用考虑了所有候选续写的树状注意力方案，通过一次前向传播进行验证。

优点：无需第二个模型。缺点：增加了可训练参数；需要监督微调阶段（约 10 亿 token）；接受率比使用良好草稿模型的原始推测解码略低。

### EAGLE —— 通过重用隐藏状态改进草稿

EAGLE-1/2/3（Li 等人，2024–2025）将草稿模型做成一个微小的变换器（通常 1 层），它吸收验证器最后一层的隐藏状态。因为草稿模型看到了验证器的特征表示，其预测与验证器的输出分布强相关。接受率从约 0.6（原始方法）提升到 0.85 以上。

EAGLE-3（2025）增加了对候选续写的树搜索。vLLM 和 SGLang 为 Llama 3/4 和 Qwen 3 默认集成了 EAGLE-2/3 作为推测解码路径。

### KV 缓存的协调

验证过程在一次前向传播中将 `N` 个草稿 token 输入验证器。这会将验证器的 KV 缓存扩展 `N` 个条目。如果某些草稿被拒绝，你必须将缓存回滚到已接受前缀的长度。

生产实现（vLLM 的 `--speculative-model`，TensorRT-LLM 的 LookaheadDecoder）使用临时 KV 缓冲区处理此问题。先写入，接受时才提交。概念上不难，但实现细节繁琐。

## 构建它

参见 `code/main.py`。我们实现核心的推测采样算法（拒绝步骤 + 残差分布），使用：

- 一个“大模型”，它是对一个手工编写的分布进行确定性 softmax（这样我们可以解析验证接受率的数学正确性）。
- 一个“草稿模型”，它是大模型的一个扰动版本。
- 一个接受/拒绝循环，该循环产生的边际分布与直接采样完全相同。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草拟 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理证明了这个伯努利决策，加上在拒绝时从残差分布中采样，精确地保持了验证器的分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

从 `q` 中逐元素减去 `p`，将负值钳制为零，重新归一化。在任何拒绝情况下都从此分布中采样。

### 步骤 3：一次推测步骤

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

五个被接受 → 一个额外 token → 一次验证器调用产生六个 token。

### 步骤 4：测量接受率

在不同草稿质量水平下运行 10,000 个推测步骤。绘制接受率与草稿和验证器分布之间 KL 散度的关系图。你应该看到一个清晰的单调关系。

### 步骤 5：验证分布等效性

经验验证：推测循环产生的 token 直方图应该与直接从验证器采样产生的直方图相匹配。这是 Leviathan 定理的实践体现。卡方检验在抽样误差范围内确认这一点。

## 使用它

生产环境：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM 在 2026 年中拥有最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 封装了推测解码，并使用了一个小型草稿模型。

**选择草稿模型：**

| 策略 | 适用场景 | 加速比 |
|----------|--------------|---------|
| 原始草稿（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8–2.3 倍 |
| Medusa 头 | 你可以微调验证器 | 2–3 倍 |
| EAGLE-2 / 3 | 生产环境，最大加速 | 3–4 倍 |
| 前瞻解码 | 无需草稿，无需训练，无额外参数 | 1.3–1.6 倍 |

**何时不适用推测解码：**

- 生成 1–5 个 token 的单序列。开销占主导。
- 高度创意 / 高温度采样（α 下降）。
- 内存受限的部署（草稿模型增加显存占用）。

## 部署它

参见 `outputs/skill-spec-decode-picker.md`。该技能为新的推理工作负载选择推测解码策略（原始 / Medusa / EAGLE / 前瞻）和调优参数（N，草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认在 50,000 个 token 上，推测循环产生的 token 分布与验证器直接采样的分布匹配，卡方检验 p > 0.05。
2. **中等。** 对于 `α = 0.5, 0.7, 0.85`，绘制加速比（每个大模型前向传播产生的 token 数）随 `N` 变化的函数图。找出每个 α 对应的最优 `N`。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：取第 14 课的毕业设计 GPT，添加 3 个额外的 LM 头来预测 t+2, t+3, t+4 位置。使用联合多头损失在 tinyshakespeare 上训练。将接受率与一个通过截断同一模型得到的原始草稿进行比较。
4. **困难。** 实现回滚：从一个 10 token 前缀的 KV 缓存开始，输入 5 个草稿 token，模拟在位置 3 被拒绝。验证你的缓存读取在下一次迭代时正确匹配“前缀 + 前 2 个被接受的草稿”。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 草稿模型 | “便宜的那个” | 一个提出候选 token 的较小模型；通常比验证器便宜 10–50 倍。 |
| 验证器 | “大的那个” | 其分布被保留的目标模型；每个推测步骤运行一次。 |
| 接受率 (α) | “草稿正确的频率” | 验证器接受每个草稿 token 的概率。典型值为 0.7–0.9。 |
| 残差分布 | “拒绝时的后备” | `(q - p)_+` 归一化后的分布；在拒绝时从此分布采样可保持验证器的分布不变。 |
| 额外 token | “免费的那个” | 当所有 N 个草稿被接受时，从验证器的下一步分布中额外采样一个 token。 |
| Medusa | “无草稿的推测” | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | “隐藏状态草稿” | 以验证器最后一层隐藏状态为条件的微型变换器草稿。 |
| 前瞻解码 | “雅可比迭代” | 使用定点迭代进行自我推测；无需草稿模型。 |
| 树状注意力 | “一次验证多个候选” | 考虑多个草稿续写同时进行的分支验证。 |
| KV 回滚 | “撤销被拒绝的草稿” | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 核心算法与等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 同时提出的；简洁的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— Medusa 论文；树状注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1；基于隐藏状态的草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) —— 前瞻解码，无草稿方法。
- [vLLM 文档 — 推测解码](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 标准的生产参考，集成了全部四种策略。
- [SafeAILab / EAGLE 参考实现](https://github.com/SafeAILab/EAGLE) —— EAGLE-1/2/3 的参考代码。