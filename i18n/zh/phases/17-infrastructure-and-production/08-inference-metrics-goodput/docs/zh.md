# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否有效。TTFT = prefill + 队列 + 网络。TPOT（即 ITL）是显存受限的 decode 每个 token 开销。端到端延迟 = TTFT + TPOT × 输出长度。吞吐量是整个集群聚合后的 tokens/秒。但产品真正关心的是 goodput —— 同时满足所有 SLO 的请求比例。吞吐高而 goodput 低，意味着你在处理永远不会及时到达用户的 token。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数据：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。始终报告 P50、P90、P99，不要只报平均值。注意测量陷阱：GenAI-Perf 在计算 ITL 时排除了 TTFT，LLMPerf 则包含；同一份运行数据中两个工具对 TPOT 给出不同结果。

**类型：** 学习
**语言：** Python（stdlib、玩具百分位计算器和 goodput 报告器）
**前置知识：** Phase 17 · 04（推理引擎内部）
**时间：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量、goodput，并指出每个指标测量的是哪个组件。
- 解释为什么均值不适合 LLM 推理，以及如何读解 P50/P90/P99。
- 构造 SLO 多约束（例如 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s）并据此计算 goodput。
- 指出两个在同一运行中对 TPOT 给出不同结果的基准测试工具，并解释原因。

## 问题所在

"吞吐量 15,000 tokens/秒。"那又怎样？如果有 40% 的请求端到端延迟超过 2 秒，用户会直接放弃会话。仅凭吞吐量无法告诉你产品是否可用。

推理存在多个延迟维度，每一维度的失败模式各不相同。Prefill 受计算瓶颈制约，随提示词长度缩放。Decode 受显存带宽瓶颈制约，随 batch size 缩放。排队延迟是调度层面的运营问题。网络延迟是物理距离问题。你需要为每个维度分别度量，还需要百分位数，以及一个能回答"用户是否得到了预期体验"的单一综合指标 —— 这就是 goodput。

## 概念讲解

### TTFT —— time to first token（首 token 延迟）

`TTFT = queue_time + network_request + prefill_time`

提示词越长，prefill 占比越大。在 H100 上运行 FP8 格式的 Llama-3.3-70B，32k token 的提示词纯 prefill 就耗时约 800 ms。排队延迟是负载下的调度器行为。网络延迟包括 TLS 握手在内的线上传输时间。TTFT 是用户在收到任何流式输出之前经历的延迟。

### TPOT / ITL —— inter-token latency（token 间延迟）

名称各异，实质相同。`TPOT`（time per output token，每输出 token 时间）、`ITL`（inter-token latency）、"decode latency per token（decode 每 token 延迟）"—— 指的是同一个量。它是第一个 token 之后，相邻两个流式 token 之间的间隔时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同等 Llama-3.3-70B H100 堆栈 + chunked prefill 配置下，TPOT 均值约 7 ms。若关闭 chunked prefill，在邻近序列发生长 prefill 期间，TPOT 可飙升至 50 ms。关注 P99，而不是均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），E2E 由 TPOT 主导。对于短输出但提示词很长的请求，E2E 由 TTFT 主导。应按输出长度条件来报告 E2E。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标。反映集群效率。无法反映单个请求的健康状况。

### Goodput —— 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多约束的组合。一个请求只有满足所有约束条件才算"通过"。Goodput 就是满足条件的请求占比。在高吞吐但只有 60% goodput 的情况下，这是失败。降低吞吐、达到 99% goodput，才是目标。

2026 年，goodput 已被纳入 MLPerf Inference v6.0 提交规范，也是 AI 平台提供商内部 SLA 跟踪的基准指标。

### 为什么均值是错误的统计量

LLM 延迟分布呈右偏（长尾）。一次 decode batch 中，若有一个长 prefill 邻居，可能以 TPOT ≈ 7 ms 输送 500 个 token，又以 TPOT ≈ 60 ms 输送 20 个 token。均值 TPOT 为 9 ms，但 P99 TPOT 为 65 ms。用户实际遭遇的是 P99 —— 这就是他们离开的理由。

始终报告三值组（P50、P90、P99）。从用户体验角度，P99 是你应该优化的指标。

### 参考数据 —— 2026 年 Llama-3.1-8B-Instruct on TRT-LLM

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：因 chunked-prefill 配置不同，在 10–25 ms 范围内变化。

这些是 NVIDIA 的官方参考点。它们随模型规模（70B 会是 3–5 倍）、硬件（H100 vs B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年最常用的两个基准测试工具对同一运行给出的 TPOT 结果不一致：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT，ITL 从第 2 个 token 开始计时。
- **LLMPerf**：包含 TTFT，ITL 从第 1 个 token 开始计时。

对一个 TTFT 为 500 ms、总 decode 时长 700 ms、输出 100 个 token 的请求：GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择直接影响数值。

务必注明所用工具，务必公布定义。

### 构造 SLO

2026 年一个面向消费者的 70B 对话模型的合理 SLO：

- TTFT P99 <= 800 ms
- TPOT P99 <= 25 ms
- 输出 <300 token 时，E2E P99 <= 3 s
- Goodput 目标 >= 99%

企业级 SLO 收紧 TTFT（200–400 ms），放宽 E2E。关键是将其书面化，对三个指标都进行度量，并以 goodput 作为单一综合指标持续跟踪。

### 如何测量

- 使用真实流量或高仿真合成流量（LLMPerf 加 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试并发数设定为目标峰值并发量的 2 倍。
- 运行 30–50 轮迭代，对所有样本合并后取百分位数。
- 报告中务必注明：工具名称、工具版本、模型、硬件、并发量、提示词分布。

```figure
throughput-latency
```

## 动手实践

`code/main.py` 是一个玩具 goodput 计算器。生成合成延迟分布，应用 SLO，然后计算 goodput。同时也演示了同一追踪数据上 GenAI-Perf 与 LLMPerf 的 TPOT 差异。

## 交付物

本课产出 `outputs/skill-slo-goodput-gate.md`。给定一个工作负载和 SLO，生成一份 CI/CD 可用的基准测试配方，将部署闸门设在 goodput 而非吞吐量上。

## 练习

1. 运行 `code/main.py`。生成含 1% 长尾尖峰的分布。将 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 某厂商声称"在 Llama 3.3 70B H100 上达到 15,000 tok/s"。在采信之前，列出三个应追问的问题。
3. 为什么 chunked prefill 保护的是 P99 TPOT 而非平均 TPOT？
4. 为一个语音助手（首 token 被听到，而非被阅读）设计消费者 SLO。哪个指标对用户最可见？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出另外三个两工具存在分歧的指标。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|------------------------|
| TTFT | "首 token 时间" | 队列 + 网络 + prefill；长提示词下由 prefill 主导 |
| TPOT | "每输出 token 时间" | 首 token 之后的显存受限 decode 每 token 开销 |
| ITL | "token 间延迟" | 多数工具中与 TPOT 相同（并非全部 —— 见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT × 输出长度；再加上响应侧网络延迟 |
| Throughput | "tok/s" | 集群效率；脱离延迟百分位数毫无意义 |
| Goodput | "SLO 满足率" | 同时满足所有 SLO 约束的请求占比 |
| P99 | "尾部" | 最坏 1/100 延迟；用户体感指标 |
| SLO 多约束 | "联合条件" | 三个延迟上限的 AND 组合；任一条违反则请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 两工具对 ITL 是否包含 TTFT 存在分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) —— TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) —— 备选定义与测量方案。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) —— 真实部署中的实践性测量。
- [LLMPerf](https://github.com/ray-project/llmperf) —— 基于 Ray 的开源基准测试工具。
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) —— NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) —— 业界公认的以 goodput 为核心的基准测试。
