# 投机解码与 EAGLE-3

> Phase 7 · Lesson 16 证明了数学原理：Leviathan 拒绝采样规则能够精确保留验证器的分布。本课从训练栈的视角讲解 2026 年生产环境中的投机解码。EAGLE-3 将草稿模型从一种廉价的近似方法转变为在验证器自身隐藏状态上训练的专用微型网络，并加入了训练时的测试循环（training-time test loop），使其训练分布与推理分布对齐。结果：端到端加速 3 倍至 6.5 倍，聊天场景下每 token 接受率超过 0.9，且不存在分布偏差。2026 年所有生产推理栈均默认启用此技术。

**类型：** Build
**语言：** Python (stdlib)
**前置知识：** Phase 7 · 16（投机解码数学原理）、Phase 10 · 12（推理优化）
**预计时间：** 约 75 分钟

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明投机解码循环产生的样本与验证器直接采样的分布完全一致。
- 梳理从 vanilla 投机解码（Leviathan 2023）到 EAGLE、EAGLE-2 及 EAGLE-3 的两年的演进历程，并指出每一步具体解决了什么局限性。
- 根据接受率 `α` 和草稿-验证器成本比 `c` 计算预期加速比，并为不同场景选择最优草稿长度 `N`。
- 从零实现完整的投机解码循环：草稿生成、并行验证、从残差分布采样拒绝项、拒绝时回滚 KV 缓存、全接受时输出奖励 token。

## 问题背景

在 H100 上，对 70B 模型进行自回归解码的速度大约只有 35 token/秒。GPU 远未饱和。瓶颈在于显存带宽：每个 token 都需要从 HBM 加载 70B 权重，执行一步算术运算，并输出一个浮点数。计算单元大部分时间都在空闲等待。

投机解码将这个瓶颈转化为一个真正可解决的吞吐量问题。廉价的草稿模型通过在 `N` 次轻量前向传播中提出 `N` 个 token。验证器则对前缀加上全部 `N` 个草稿 token 运行一次前向传播。如果验证器在位置 `i` 的分布与草稿达成共识（我们将给出精确的统计学定义），则接受；否则拒绝，并从残差分布中采样一个修正项。单次大模型前向传播即可产出最多 `N+1` 个被接受的 token，而非原来的一个。

核心的定理是 Leviathan、Kalman 和 Matias（ICML 2023）提出的结论：输出分布与直接从验证器采样所得的分布完全相同。不是近似相同，是精确相同。这正是投机解码能被应用于生产环境的根本原因——它是一项纯粹的延迟优化，且不牺牲任何质量。

Phase 7 · Lesson 16 为你提供了数学基础，而本课提供的是训练栈方案。一个优秀的草稿模型带来的加速效果是廉价草稿的两倍。EAGLE、EAGLE-2 和 EAGLE-3（Li 等人，2024–2025）将“草稿 = 同系列的小模型”这一做法转变为了一套精确的工程方法论。2026 年的生产推理服务器已默认采用 EAGLE-3。

## 核心概念

### 不变量：Leviathan 拒绝采样

设 `p(t)` 为给定某前缀时草稿模型对下一个 token 的分布，`q(t)` 为验证器的分布。从 `d ~ p` 中采样一个草稿 token。以概率 `min(1, q(d) / p(d))` 接受该 token。若被拒绝，则从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 中采样。最终得到的样本严格服从 `q` 分布。无论 `p` 的质量有多差，这一结论都成立——`p` 越差，拒绝次数越多，但输出分布始终保持精确。

使用一次验证器前向传播处理 `prefix + d_1 + ... + d_N`，将 `N` 次调用串联起来。验证器同时返回 `q_1, q_2, ..., q_{N+1}`。从左向右遍历，在第一个被拒绝的位置 `j` 处，从 `residual(q_j, p_j)` 中采样并停止。若全部接受，则从 `q_{N+1}` 中额外采样一个奖励 token。

### 加速比的决定因素

设 `α` 为每个草稿 token 的期望接受率，`c = cost(draft) / cost(verifier)` 为成本比。每次验证器前向传播的期望接受 token 数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的期望总墙钟时间为 `(N * c + 1) / E[accepted]`。对该式关于 `N` 求极小值即可得到最优解。当 `α = 0.8, c = 0.05` 时，最优 `N` 约为 5–7，加速比达 3.2×。当 `α = 0.95, c = 0.02` 时，最优 `N` 约为 8–10，加速比可突破 5×。

最关键的杠杆是 `α`。在固定 `N = 5` 的情况下，将 `α` 从 0.6（vanilla 草稿）提升到 0.9（EAGLE-3），每次验证器前向传播的期望接受 token 数就从 2.2 提升至 4.1。仅凭相同的验证器开销，吞吐量几乎翻倍。

### 两年的技术演进

**Vanilla 投机解码（Leviathan, 2023）。** 草稿模型是同系列中独立训练的小型 LLM。易于接入，`α ≈ 0.6`，加速比最多约 2×。

**EAGLE-1（Li 等人, 2024）。** 草稿模型是一个极小的 transformer（通常仅 1 或 2 层），以验证器最后一层的隐藏状态为输入，直接预测下一个 token。由于草稿能看到验证器的特征表示，其分布与验证器更为接近。`α` 提升至 0.7–0.8。

**EAGLE-2（Li 等人, 2024）。** 引入动态草稿树：不再提出单一的 `N` token 序列，而是提出一棵候选树，在一次前向传播中通过验证器统一打分（tree attention），并沿最高概率路径行走。草稿长度变为逐步自适应。每条被接受路径上的 token `α` 升至 0.85 以上。

**EAGLE-3（Li 等人, 2025, NeurIPS）。** 新增两项改进。第一，彻底放弃特征预测损失——EAGLE-1/2 训练草稿去拟合验证器的隐藏状态，这限制了数据的增益上限。EAGLE-3 改为直接在 token 预测上训练。第二，训练时测试（Training-Time Test, TTT）：在草稿训练期间，将草稿自身的历史预测结果多步循环反馈为输入，与推理时的运作方式保持一致。这使训练分布与测试分布对齐，并阻断了误差累积。实测加速比：聊天场景最高达 6.5×，H100 上 SGLang batch 64 时吞吐量提升 38%。

### KV 缓存回滚

验证过程在一次前向传播中将验证器的 KV 缓存扩展 `N` 个条目。若在第 `j` 个位置发生拒绝，则位置 `j-1` 之后的缓存内容现已失效。两种常见实现方式：写入临时缓冲区并在接受时提交（vLLM、TensorRT-LLM），或维护物理 KV 缓存加上逻辑长度并在拒绝时截断。无论哪种方式，回滚开销仅为每层每头的字节数，相对于前向传播成本可忽略不计。

对于 EAGLE-2 的树搜索，验证器会在尊重树拓扑结构的非因果掩码下执行注意力计算。工程实现较为繁琐，但计算本身仍是带自定义掩码的标准 flash-attention 调用。

### 2026 年的草稿架构

| 策略 | 草稿类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| Vanilla | 独立的小型 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用现有小模型） |
| Medusa | 验证器附加的额外 LM 头 | 0.65-0.75 | 2-3× | 约 1B SFT tokens |
| EAGLE-1 | 基于隐藏状态的 1 层 transformer | 0.70-0.80 | 2.5-3× | 约 60B tokens |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | 约 60B tokens |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | 约 60-200B tokens |
| Lookahead | 无草稿（雅可比迭代） | N/A | 1.3-1.6× | 无 |

2026 年生产环境现状：vLLM 和 SGLang 在可用时默认采用 EAGLE-3，否则降级至 EAGLE-2。TensorRT-LLM 对 Meta 和 NVIDIA 公开模型提供了最快的 Medusa 路径。llama.cpp 则面向 CPU 部署提供 vanilla 草稿。

```figure
l5-spec-decode-eagle
```

## 动手实现

见 `code/main.py`。这是完整的 Leviathan 投机解码循环实现，包含所有组件：N 个 token 草稿、验证器并行前向、逐位置拒绝、残差采样、奖励 token、KV 缓存回滚，以及输出分布与从 `q` 直接采样分布一致的经验验证。

### 步骤 1：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 步骤 2：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 步骤 3：完整的投机解码步

`spec_step` 函数从 `p` 中草稿生成 `N` 个 token，随后通过一次并行的 `q` 评估对所有草稿进行验证。针对每个草稿 token 应用拒绝规则，在首次拒绝时从残差分布中采样修正项。若全部接受，则从 `q_{N+1}` 中输出一个奖励 token。

### 步骤 4：KV 缓存回滚记账

模拟器为每个 worker 追踪逻辑上的 `kv_length`。成功接受 `k` 个草稿时，`kv_length += k`。若在位置 `j` 发生拒绝，缓存虽已写入到 `j` 之后，但逻辑长度被重置为 `prefix_length + j + 1`——即修正 token 之后的位置。后续读取时按逻辑长度截断。

### 步骤 5：Leviathan 定理验证

运行 50,000 次投机解码步。统计被接受 token 的经验分布，并与从 `q` 直接采样的 50,000 个样本进行对比。卡方统计量应远低于临界值。定理在实践中验证通过。

### 步骤 6：加速比与 α 的关系

通过在不同幅度下扰动 `p` 使其偏离 `q` 来扫描草稿质量。测量 `α`，然后绘制期望的每次验证器调用 token 数随 `α` 和 `N` 变化的曲线。代码会输出一张表格，展示 EAGLE-3 级别的草稿质量（`α ≈ 0.9`）如何将每次验证器调用的产出提升至 4–5 个 token。

## 生产应用

使用 EAGLE-3 的生产级 `vllm serve` 命令：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

根据 EAGLE-3 论文，在 H100 上使用 SGLang + EAGLE-3（batch 64）相比 vanilla 解码的 batch-64 吞吐量提升约 1.38 倍。

适用投机解码的场景：

- 任何对 p50 延迟敏感度高过峰值吞吐量的交互式聊天任务。
- 代码生成与结构化输出（JSON、SQL）。由于目标分布高度可预测，`α` 通常高于 0.9。
- 长文本生成（数千 token）。分摊后的加速收益持久显著。

不适用投机解码的场景：

- 极小模型（< 3B）。草稿模型的代价与验证器相差不大。
- 极小规模的 batch-1 CPU 部署。草稿模型的内存开销可能得不偿失。
- 极高温度下的创造性采样，此时 `α` 会急剧下降。

## 交付成果

本课将产出 `outputs/skill-eagle3-tuner.md`。给定推理工作负载（模型、批次大小、目标延迟、任务特征），该文件将推荐一套投机解码策略及调优参数（草稿模型家族、`N`、树深度、温度感知切换策略）。

## 练习

1. 运行 `code/main.py`。确认在 50,000 次采样下，Leviathan 分布检验的卡方统计量始终低于 95% 显著性水平的临界值。

2. 保持 `α` 为 0.9、`c` 为 0.04，将 `N` 从 1 扫描至 10。绘制每次验证器调用的期望 token 数与实际单 token 墙钟时间曲线。找出最小化墙钟时间的 `N`，并解释曲线的形状特征。

3. 修改代码以模拟 EAGLE-2 树搜索：在每个解码步，草稿提出一个形状为 `[2, 2, 2]` 的树（共八条候选路径）。验证器运行一次前向传播，概率最高的被接受路径胜出。计算每个叶节点的 `α` 及每次验证器调用的总 token 数，并与同等计算量下的线性链投机解码进行对比。

4. 实现一个支持两个并发序列的批量 KV 缓存回滚模拟器。序列 A 的所有草稿均被接受；序列 B 在第 2 个位置发生拒绝。证明每个序列的 `kv_length` 均被正确更新，且没有计算资源被浪费。

5. 阅读 EAGLE-3 论文的第四节（Training-Time Test）。用两句话说明为何缺乏 TTT 的朴素草稿训练会遭受暴露偏差（exposure bias），以及为何在训练期间将草稿自身的预测结果反馈为输入能够解决该问题。将此与 seq2seq 领域的 scheduled sampling 文献联系起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Leviathan rule | “min(1, q over p)” | 以概率 `min(1, q(d)/p(d))` 进行伯努利接受/拒绝；拒绝时从残差分布采样，可精确保持验证器分布 |
| Residual distribution | “(q minus p) plus, normalized” | 将 `(q - p)_+` 截断至零后重新归一化——拒绝时应从中采样的正确分布 |
| Acceptance rate α | “草稿命中频率” | 在拒绝规则下的期望逐 token 伯努利成功概率；控制所有加速比计算 |
| EAGLE-1 | “隐藏状态草稿” | 以验证器最后一层隐藏状态为条件的微型 transformer 草稿（Li 等人, 2024） |
| EAGLE-2 | “动态草稿树” | 在 EAGLE-1 基础上增加候选延续树，通过一次验证器前向传播的 tree attention 统一打分 |
| EAGLE-3 | “训练时测试” | 放弃特征预测损失，直接在 token 预测上训练，并在训练期间将草稿自身输出循环反馈为输入 |
| Training-time test (TTT) | “暴露偏差修正” | 在训练期间让草稿以自回归方式运行，使训练与测试的输入分布对齐——即 scheduled sampling 的直接类比 |
| KV rollback | “撤销被拒草稿” | 拒绝发生后，将验证器的 KV 缓存重置回已接受前缀长度的记账机制 |
| Bonus token | “免费赠送的 token” | 当全部 `N` 个草稿均被接受时，无需额外验证器开销，从 `q_{N+1}` 中额外采样一个 token |
| Tree attention | “一次性验证多个候选” | 使用尊重草稿树拓扑结构的非因果掩码执行注意力计算；一次前向传播即可为树中所有节点计算 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) —— 奠基性论文与分布等价性定理
- [Chen 等人 — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) —— 并发独立提出，证明简洁清晰
- [Li 等人 — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— EAGLE-1，基于隐藏状态条件的草稿模型
- [Li 等人 — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 动态树搜索
- [Li 等人 — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) —— 2026 年生产环境默认方案
- [Cai 等人 — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) —— 无需草稿的替代方案
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 涵盖所有策略的标准生产参考文档
