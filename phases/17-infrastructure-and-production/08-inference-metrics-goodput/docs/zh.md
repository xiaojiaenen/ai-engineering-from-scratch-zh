# 推理指标 — TTFT、TPOT、ITL、有效吞吐量、P99

> 四项指标决定了推理部署是否正常运行。TTFT 是预填充、队列和网络延迟之和。TPOT（等同于 ITL）是每个 token 的内存受限解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐量是整个集群聚合的每秒 token 数。但对产品而言真正重要的是**有效吞吐量** — 即同时满足所有 SLO 的请求比例。高吞吐量但低有效吞吐量意味着你正在处理永远无法及时到达用户的 token。2026年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数据：平均 TTFT 162 毫秒，平均 TPOT 7.33 毫秒，平均端到端延迟 1,093 毫秒。务必报告 P50、P90、P99 — 绝不要只报告平均值。注意测量陷阱：GenAI-Perf 在 ITL 计算中排除了 TTFT，而 LLMPerf 包含了它；对于同一运行，两个工具对 TPOT 的结果不一致。

**类型：** 学习
**语言：** Python（标准库、玩具版百分位数计算器和有效吞吐量报告器）
**前置条件：** 第 17 阶段 · 04（vLLM 服务内部机制）
**时间：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、端到端延迟、吞吐量和有效吞吐量，并指出每个指标衡量的组成部分。
- 解释为什么平均值不适合衡量 LLM 服务，以及如何解读 P50/P90/P99。
- 构建一个 SLO 多重约束（例如 TTFT<500 毫秒且 TPOT<15 毫秒且端到端延迟<2 秒），并基于此计算有效吞吐量。
- 指出两个对同一运行 TPOT 结果不一致的基准测试工具，并解释原因。

## 问题所在

“我们的吞吐量是每秒 15,000 个 token。” 那又怎样？如果 40% 的请求端到端延迟超过了 2 秒，用户就会放弃会话。仅凭吞吐量无法告诉你产品是否运行正常。

推理具有多个延迟轴，且每个轴的失效模式不同。预填充是计算受限的，且随提示词长度缩放。解码是内存受限的，且随批次大小缩放。队列延迟是调度问题。网络是物理距离问题。你需要为每个部分设置不同的指标，并且需要百分位数，还需要一个单一的复合指标来表明“用户是否得到了他们期望的东西” — 这就是有效吞吐量。

## 核心概念

### TTFT — 首 token 时间

`TTFT = queue_time + network_request + prefill_time`

当提示词较长时，预填充占主导。在 H100 上运行 FP8 精度的 Llama-3.3-70B 时，一个 32k 提示词需要约 800 毫秒的纯预填充时间。队列时间是负载下的调度器行为。网络请求是包括 TLS 在内的线路传输时间。TTFT 是用户在看到任何流式内容返回之前所感知到的延迟。

### TPOT / ITL — token 间延迟

同一个量有多种称呼。`TPOT`（每个输出 token 的时间）、`ITL`（token 间延迟）、`decode latency per token` — 都是同一回事。它指的是第一个 token 之后，连续流式输出 token 之间的时间间隔。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在相同的 Llama-3.3-70B H100 堆栈上，使用分块预填充时，TPOT 平均值约为 7 毫秒。在没有分块预填充的情况下，当相邻序列进行长预填充时，TPOT 可能会飙升到 50 毫秒。关注 P99，而不是平均值。

### 端到端延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 个 token），端到端延迟受 TPOT 主导。对于短输出但长提示词的场景，端到端延迟受 TTFT 主导。应报告按输出长度条件划分的端到端延迟。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群的效率。不能反映单个请求的健康状况。

### 有效吞吐量 — 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是一个多重约束。一个请求只有在每个约束条件都满足时才是“好”的。有效吞吐量就是这类请求的比例。60% 有效吞吐量下的高吞吐量是失败的。99% 有效吞吐量下的较低吞吐量才是目标。

到 2026 年，有效吞吐量是 MLPerf Inference v6.0 提交和 AI 平台提供商内部 SLA 跟踪中使用的指标。

### 为什么平均值是错误的统计量

LLM 的延迟分布是右偏的。一个解码批次中，如果有一个长预填充的邻居序列，可能会在约 7 毫秒的 TPOT 下输出 500 个 token，然后在约 60 毫秒的 TPOT 下输出 20 个 token。平均 TPOT 是 9 毫秒。P99 TPOT 是 65 毫秒。用户会经常遇到 P99 的情况 — 这就是他们离开的原因。

务必报告三元组 (P50, P90, P99)。对于用户体验，你需要优化的是 P99。

### 参考数据 — Llama-3.1-8B-Instruct 在 TRT-LLM 上，2026 年

- 平均 TTFT：162 毫秒
- 平均 TPOT：7.33 毫秒
- 平均端到端延迟：1,093 毫秒
- P99 TPOT：根据分块预填充配置的不同，在 10-25 毫秒之间变化。

这些是已发布的 NVIDIA 参考点。它们会随模型大小（70B 模型可能会显示 3-5 倍）、硬件（H100 对比 B200 约 3 倍差异）和负载而变化。

### 测量陷阱

两个最常用的 2026 年基准测试工具对同一运行的 TPOT 结果不一致：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除了 TTFT。ITL 从第 2 个 token 开始计算。
- **LLMPerf**：包含了 TTFT。ITL 从第 1 个 token 开始计算。

对于一个 TTFT 为 500 毫秒、100 个输出 token 总解码时间为 700 毫秒的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变结果数字。

务必说明使用的工具。务必公布定义。

### 构建 SLO

2026 年面向消费者的 70B 聊天模型的一个合理 SLO：

- TTFT P99 <= 800 毫秒。
- TPOT P99 <= 25 毫秒。
- 端到端延迟 P99 <= 3 秒（适用于输出少于 300 个 token 的情况）。
- 有效吞吐量目标 >= 99%。

企业 SLO 会收紧 TTFT（200-400 毫秒）并放宽端到端延迟。关键是将其书面化，测量所有三个指标，并将有效吞吐量作为单一复合指标进行跟踪。

### 如何测量

- 运行真实流量或逼真的合成流量（使用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150` 的 LLMPerf）。
- 基准测试运行的目标并发数应为峰值并发数的 2 倍。
- 运行 30-50 次迭代，从合并样本中取百分位数。
- 发布时需注明工具名称、工具版本、模型、硬件、并发数、提示词分布。

## 使用它

`code/main.py` 是一个玩具版有效吞吐量计算器。它生成一个合成延迟分布，应用 SLO，然后计算有效吞吐量。同时展示了同一跟踪中 GenAI-Perf 与 LLMPerf 在 TPOT 上的差异。

## 交付它

本课程产出 `outputs/skill-slo-goodput-gate.md`。给定一个工作负载和 SLO，它会生成一个可用于 CI/CD 的基准测试方案，该方案根据有效吞吐量（而非吞吐量）来决定部署是否通过。

## 练习

1. 运行 `code/main.py`。生成一个包含 1% 尾部尖峰的分布。当你将 P99 TPOT 从 30 毫秒收紧到 15 毫秒时，有效吞吐量如何变化？
2. 某供应商声称“在 Llama 3.3 70B H100 上达到 15,000 tok/s”。在信任它之前，需要问哪三个问题？
3. 为什么分块预填充能保护 P99 TPOT，但不能保护平均 TPOT？
4. 为一个语音助手（首个 token 是被听到而非读到）构建一个消费者 SLO。哪个指标对用户最可见？
5. 阅读 LLMPerf 的 README 和 GenAI-Perf 的文档。找出另外三个工具之间存在分歧的指标。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| TTFT | “首 token 时间” | 队列 + 网络 + 预填充；长提示词时预填充占主导 |
| TPOT | “每个输出 token 的时间” | 首个 token 之后的每个 token 的内存受限解码成本 |
| ITL | “token 间延迟” | 大多数工具中等同于 TPOT（并非全部 — 见 GenAI-Perf） |
| 端到端延迟 | “端到端” | TTFT + TPOT * 输出长度；外加响应端的网络延迟 |
| 吞吐量 | “tok/s” | 集群效率；没有延迟百分位数则毫无意义 |
| 有效吞吐量 | “SLO 满足率” | 同时满足所有 SLO 约束的请求比例 |
| P99 | “尾部” | 最差的 1% 延迟；用户体验指标 |
| SLO 多重约束 | “联合体” | 三个延迟界限的 AND 关系；任一违反则请求失败 |
| GenAI-Perf 与 LLMPerf | “工具陷阱” | 工具在 ITL 是否包含 TTFT 上存在分歧 |

## 扩展阅读

- [NVIDIA NIM — LLM 基准测试指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM 服务基准测试指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方案。
- [BentoML — LLM 推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 在实际部署中的应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试工具。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 业界公认的基于有效吞吐量的基准测试。