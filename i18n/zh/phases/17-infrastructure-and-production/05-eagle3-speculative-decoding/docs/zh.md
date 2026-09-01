# EAGLE-3 推测解码生产环境部署

> 推测解码将一个快速的草稿模型与目标模型配对。草稿模型提出 K 个 token；目标模型在一次前向传播中完成验证；被接受的 token 等效于免费。2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态上训练草稿头，而非在原始 token 上训练，将接受率 alpha 推至 0.6-0.8 区间（通用聊天场景）。正确的问题不是"草稿多快"，而是"我的流量上 alpha 是多少？"如果 alpha 低于 ~0.55，推测解码在高并发下净收益为负，因为每个被拒绝的草稿都会引发一次额外的目标前向。本课时教你先测量 alpha，再决定是否开启该功能。

**类型：** 学习
**语言：** Python（stdlib，玩具接受率模拟器）
**前置知识：** Phase 17 · 04（推理引擎内部实现），Phase 10 · 18（多 token 预测）
**时间：** ~60 分钟

## 学习目标

- 说出推测解码的三个代际，并解释 EAGLE-3 相对于 EAGLE-2 和经典草稿模型做了什么改变。
- 定义接受率 alpha，从 alpha 和 K（草稿长度）计算预期加速比，并识别目标并发下的盈亏平衡 alpha。
- 解释为何推测解码在 vLLM 2026 中是 opt-in（非默认），以及为何不测量 alpha 就开启它是生产环境反模式。
- 写出测量方案：用哪个 benchmark、哪种 prompt 分布、哪个并发点、用什么指标做 gate。

## 问题所在

解码阶段是内存带宽瓶颈。在一块 H100 上运行 Llama 3.3 70B FP8，每解码一个 token 需要读取约 140 GB/s 的权重，并输出一个 token。GPU 计算单元在解码期间几乎空闲——瓶颈是 HBM 带宽，而非 matmul 吞吐量。

推测解码利用了这个差距。用一个廉价的草稿模型生成 K 个候选 token，然后让目标模型在一次前向中一次性验证全部 K 个。每个验证通过的 token 实际上是"免费的"（摊入目标模型原本就需要做的 batch-of-K 前向中）。

经典草稿模型方案使用同家族的更小模型（Llama 3.2 1B 为 Llama 3.3 70B 起草）。它可行，但接受率一般——小模型的分布与目标模型存在偏差。EAGLE，然后是 EAGLE-2，再到 EAGLE-3 直接在目标模型内部状态上训练轻量草稿头，因此草稿分布能更紧密地跟踪目标分布。这就是 alpha 从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8 的原因。

但要注意：EAGLE-3 在 vLLM 2026 中是 opt-in。`speculative_config` 必须显式设置。不设置标志就没有加速。许多团队在未对真实流量测量 alpha 的情况下直接开启，结果尾部延迟反而变差了。

## 核心概念

### 推测解码实际带来了什么

不做推测解码时，每个 token 的代价是一次目标前向。做推测解码时，草稿长度 K、接受率 alpha，目标模型每次前向的期望输出 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿+验证的额外开销。对于 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数字通常在 2-3x 区间，因为生产流量上 alpha 很少那么高，且 epsilon 随 batch size 增大而增长。

### 为何 alpha 是唯一重要的指标

被拒绝的 token 并没有消失——它们迫使目标模型对第一个被拒绝的 token 再执行一次前向。当 alpha 降至 0.4 时，你付出的代价是：草稿开销 + 验证 + 重新生成。在高并发下（如 256 并发），解码 batch 已经足够大，使得"仅目标模型"与"目标模型+验证"之间的内存带宽差距缩小。在大多数 2026 年硬件上，alpha 低于 0.55 时，推测解码净收益为负。

Alpha 因工作负载而异。在 ShareGPT 类通用聊天上，用 ShareGPT 训练的 EAGLE-3 可达 0.6-0.8。在领域特定流量（代码、医疗、法律）上，用通用数据训练的草稿头降至 0.4-0.6。训练领域特定的草稿头可以恢复 alpha——相较于目标模型微调，这是一项轻量、快速的任务。

### EAGLE 各代一览

- **经典草稿模型**：同家族的小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，草稿每轮做 K 次前向。
- **EAGLE-1（2024）**：在目标隐藏状态（最后一层）上训练单个草稿头。Alpha ~0.5-0.6。目标模型参数略有增加。
- **EAGLE-2（2025）**：自适应草稿长度和基于树的草稿（一次目标前向验证多个分支）。Alpha ~0.6-0.7。草稿调度更复杂。
- **EAGLE-3（2025-2026）**：草稿头在多目标层上训练（不只是最后一层），对齐效果更好。通用聊天上 alpha ~0.6-0.8。

### 2026 年生产部署流程

1. 直接部署目标模型。在目标并发下测量基线 TTFT、ITL、吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行 benchmark。
3. 记录接受率 alpha。vLLM V1 通过 `spec_decode_metrics.accepted_tokens_per_request` 上报此指标。除以请求的草稿长度即得 alpha。
4. 若生产流量分布上 alpha < 0.55，关闭推测解码或训练领域特定的 EAGLE-3 草稿。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

均值 ITL 会因推测解码而下降。但若不加调优，P99 可能变差。被拒绝的草稿触发两阶段序列（草稿 + 验证失败 + 重做）。在满 batch 下，这两次前向是串行的。关注 P99 ITL，而非 P50。

### EAGLE-3 已有部署案例

Google 在 2025 年于 AI Overviews 中部署了推测解码（相同质量，更快响应）。vLLM V1 以 `speculative_config` 作为文档化接口；N-gram GPU 推测解码是 V1 中与 chunked prefill 兼容的变体。SGLang 支持 EAGLE-3 作为前缀密集型工作负载的推荐草稿路径。

### 一行总结盈亏平衡公式

预期加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 求解 alpha：`alpha_breakeven = verify_overhead / K`。典型 verify_overhead ~0.15，K=5 时：`alpha_breakeven = 0.03`。但这是纯解码数学。在高并发下，验证开销上升，且解码 batch 已在序列间摊薄了内存读取，因此实际有效 alpha_breakeven 升至 ~0.45-0.55。

### 不适用推测解码的场景

- Batch=1 离线生成，延迟无关紧要。直接用目标模型。
- 极短输出（少于 50 token）。草稿开销和验证成本占主导。
- 领域特定任务，且没有领域训练的草稿头。Alpha 过低。
- vLLM v0.18.0 + draft-model 推测解码 + `--enable-chunked-prefill`。该组合无法编译。文档化的例外是 V1 中的 N-gram GPU 推测解码。

```figure
mx-speculative-tree
```

## 动手实践

`code/main.py` 模拟了在一系列 alpha 值和草稿长度 K 下，有无推测解码的解码循环。它输出盈亏平衡 alpha、实测加速比和尾部行为。在多个 (alpha, K) 组合上运行，观察推测解码在何处不再划算。

## 产出物

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和目标并发，生成一份分阶段 EAGLE-3 部署计划——基线 benchmark、启用配置、测量 alpha、以 alpha >= 0.55 为 gate、监控 P99 ITL。

## 练习题

1. 运行 `code/main.py`。K=5 时，达到 2x 加速需要多少 alpha？3x 加速呢？这个结果对 verify_overhead 有多敏感？
2. 假设生产流量 70% 通用聊天、30% 代码。EAGLE-3 在 ShareGPT 上训练后，通用聊天 alpha=0.7；代码 alpha=0.4。混合 alpha 是多少？推测解码是否净正收益？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram），以及哪种与 chunked prefill 兼容。
4. 你发现开启 EAGLE-3 后均值 ITL 下降了 25%，但 P99 ITL 上升了 15%。诊断原因并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的显存开销。与运行 Llama 3.2 1B 作为经典草稿相比如何？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| 推测解码 | "草稿+验证" | 用廉价模型提议 K 个 token，在一次目标前向中验证全部 K 个 |
| 接受率 alpha | "spec accept rate" | 目标模型接受的草稿 token 比例；唯一重要指标 |
| 草稿长度 K | "spec k" | 草稿每轮向目标模型提议的 token 数；典型值 4-8 |
| 验证开销 epsilon | "spec overhead" | 验证+重做相比单纯目标前向的额外开销；随 batch 增大而增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 变体；在多目标层上训练草稿头；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec config" | vLLM V1 中的显式 opt-in；无默认意味着无加速 |
| N-gram 推测解码 | "N-gram 草稿" | GPU 端利用 prompt 中 N-gram 查表生成草稿；与 chunked prefill 兼容 |
| 盈亏平衡 alpha | "no-op alpha" | 推测解码加速比为 1 时的 alpha 值；在生产并发下关注此值 |
| 被拒绝草稿的双次前向 | "reroll cost" | 草稿被拒绝时触发两次目标前向；驱动 P99 尾部恶化 |

## 延伸阅读

- [vLLM — 推测解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — `speculative_config` 及 V1 中 chunked prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 完整字段说明。
- [EAGLE 论文 (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE 草稿头方法。
- [EAGLE-2 论文 (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应草稿与树结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 含推测解码的高效 LLM 系统。
- [BentoML — 推测解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产部署检查清单。
