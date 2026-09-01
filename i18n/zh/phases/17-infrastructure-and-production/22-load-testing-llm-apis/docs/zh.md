# 负载测试 LLM API —— 为什么 k6 和 Locust 会说谎

> 传统负载测试工具并非为流式响应、可变输出长度、token 级指标或 GPU 饱和度而设计。两个陷阱会坑害大多数团队。GIL 陷阱：Locust 的 token 级测量会在 Python GIL 下运行分词，在高并发下与请求生成争抢资源；分词积压会导致报告的 token 间延迟膨胀——你的客户端才是瓶颈，而非服务器。提示词均匀性陷阱：循环中发送相同提示词只会测试 token 分布的一个点；真实流量具有可变长度和多样化的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 修复此问题。2026 年工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式传输，通过 TestRun/PrivateLoadZone CRD 在 Kubernetes 上原生分布式部署，最适合 CI/CD 门禁；Vegeta 用于 Go 语言的恒定速率饱和；仅带 LLM-Locust 扩展的 Locust 2.43.3 才适合流式传输。负载模式：稳态、爬坡、尖峰（自动扩缩容测试）、浸泡（内存泄漏测试）。

**类型：** 构建
**语言：** Python（标准库，玩具级真实提示词生成器 + 延迟收集器）
**前置条件：** 阶段 17 · 08（推理指标）、阶段 17 · 03（GPU 自动伸缩）
**时间：** 约 75 分钟

## 学习目标

- 解释两种反模式（GIL 陷阱、提示词均匀性陷阱），它们使通用负载测试工具在 LLM API 上产生虚假数据。
- 根据给定目的选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、爬坡、尖峰、浸泡），并指出每种模式能捕获的故障模式。
- 使用输入 token 的均值 + 标准差而非固定长度，构建真实提示词分布。

## 问题所在

你用 k6 在 500 个并发用户下测试了 LLM 端点，它扛住了。你发布了。但在生产中 200 个实际用户时服务崩溃了——P99 TTFT 爆炸，GPU 占满。

发生了两件事。首先，k6 发送了 500 条完全相同的提示词——你的请求合并和前缀缓存使其看起来像在处理 500 个并发解码，而实际上只处理了一个。其次，k6 无法以人眼感知的方式跟踪流式响应的 token 间延迟；它看到的是一条 HTTP 连接，而非 500 个以不同间隔到达的 token。

LLM 负载测试是一门独立的学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python 并在 GIL 下在客户端运行分词。在高并发下，分词器队列排在请求生成之后。报告的 token 间延迟包含了客户端分词积压。你以为服务器慢；其实是测试框架慢。

修复方案：LLM-Locust 扩展将分词移至独立进程，或使用编译型语言框架（k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示词均匀性陷阱

所有已知负载测试工具都允许你配置一条提示词。在 10,000 次迭代的循环测试中，每次发送完全相同的提示词。服务器每次都看到相同的前缀——前缀缓存命中率接近 100%，吞吐量看起来很棒。

修复方案：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样化的长度，多样化的内容。

### 四种负载模式

1. **稳态**——持续 30-60 分钟的恒定 RPS。捕获：基线性能回归。
2. **爬坡**——在 15 分钟内从 0 到目标值线性增加 RPS。捕获：容量断点、预热异常。
3. **尖峰**——突然增加到 3-10 倍 RPS 持续 2 分钟然后恢复。捕获：自动伸缩延迟、队列饱和、冷启动影响。
4. **浸泡**——持续 4-8 小时的稳态。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）—— Python 但后端为 Rust。均值/标准差提示词。支持流式传输。性能测试的最佳默认选择。

**NVIDIA GenAI-Perf**—— NVIDIA 的参考工具。使用 Triton 客户端；指标覆盖全面。注意其 ITL 不包含 TTFT；LLMPerf 的包含。两种工具对同一服务器会产生不同的 TPOT。

**LLM-Locust**（TrueFoundry）—— 修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm**—— 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译型，无 GIL）添加了支持流式传输的指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门禁和 SLA 测试。

**Vegeta**—— Go 语言，比 k6 更简单。恒定速率 HTTP 饱和。非 LLM 专用，但对网关/限流测试很有用。

**Locust 2.43.3 原版**—— 对 LLM 存在 GIL 陷阱。仅配合 LLM-Locust 扩展使用。

### CI 中的 SLA 门禁

在 PR 上运行 k6：

- 每个基线 RPS 运行 30-50 次迭代。
- 门禁条件：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 违反时中断构建。

### 真实提示词分布

基于真实流量样本（如果有）或已发布的分布（例如 ShareGPT 提示词用于聊天，HumanEval 用于代码）构建。将均值 + 标准差提供给 LLMPerf。避免任何形式的“单提示词循环”。

### 应记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：支持流式传输的指标。
- 典型 LLMPerf 运行：100-1000 次请求，并发数 X。
- 典型 CI 门禁：每个 PR 30-50 次迭代。
- 四种模式：稳态、爬坡、尖峰、浸泡。

```figure
load-pattern-waves
```

## 动手实践

`code/main.py` 模拟了带有真实提示词分布的负载测试，测量有效 TPOT，并演示了均匀提示词陷阱。

## 交付物

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较均匀分布与真实分布——差距在哪里？
2. 编写 CI 门禁的 k6 脚本：100 并发下 TTFT P95 < 800 ms，运行时长 5 分钟。
3. 浸泡测试显示内存每小时增长 50 MB。列出三种原因以及用于区分它们的监控手段。
4. 尖峰测试从 10 RPS 到 100 RPS。如果已部署 Karpenter + vLLM 生产栈（阶段 17 · 03 + 18），预期的恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 报告同一服务器 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| LLMPerf | “LLM 测试框架” | Anyscale 基准测试工具，支持流式传输 |
| GenAI-Perf | “NVIDIA 工具” | NVIDIA 参考测试框架 |
| LLM-Locust | “用于 LLM 的 Locust” | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | “合成基准测试” | 大规模合成工具 |
| k6 Operator | “K8s k6” | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | “Python 客户端开销” | 分词积压会膨胀报告的延迟 |
| 提示词均匀性陷阱 | “单提示词谎言” | 循环使用相同提示词会命中缓存，虚增吞吐量 |
| 稳态 | “恒定负载” | N 分钟内的平坦 RPS |
| 爬坡 | “线性上升” | 在指定时长内从 0 到目标值 |
| 尖峰 | “突发测试” | 突然倍增然后恢复 |
| 浸泡 | “长时间测试” | 持续数小时用于检测泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
