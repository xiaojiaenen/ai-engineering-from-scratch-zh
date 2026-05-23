# 负载测试大语言模型API——为什么k6和Locust会“说谎”

> 传统负载测试工具并非为流式响应、可变输出长度、token级别指标或GPU饱和度而设计。多数团队会陷入两个陷阱。GIL陷阱：Locust的token级测量在Python GIL下运行分词，在高并发下与请求生成竞争；分词积压随后会膨胀上报的token间延迟——瓶颈在客户端而非服务器。提示均匀性陷阱：循环测试中的相同提示仅测试token分布中的一个点；真实流量具有可变长度和多样化的前缀匹配。LLMPerf通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026年工具图谱：LLM专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于token级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025年9月)** ——流式感知、Kubernetes原生、通过TestRun/PrivateLoadZone CRD进行分布式测试，最适合CI/CD门禁；Vegeta用于Go恒定速率饱和测试；Locust 2.43.3仅通过LLM-Locust扩展支持流式测试。负载模式：稳态、斜坡、尖峰（自动扩缩容测试）、浸泡（内存泄漏）。

**类型：** 构建
**语言：** Python（标准库、玩具级真实提示生成器 + 延迟收集器）
**前置条件：** 第17阶段 · 08（推理指标），第17阶段 · 03（GPU自动扩缩容）
**时间：** 约75分钟

## 学习目标

- 解释导致通用负载测试工具在LLM API上“说谎”的两个反模式（GIL陷阱、提示均匀性陷阱）。
- 根据特定目的选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI门禁）、guidellm（大规模合成）、GenAI-Perf（NVIDIA参考实现）。
- 设计四种负载模式（稳态、斜坡、尖峰、浸泡）并说明各自捕获的失败模式。
- 使用输入token的均值 + 标准差构建真实的提示分布，而非固定长度。

## 问题所在

你用k6在500并发用户下测试了LLM端点。它通过了。你发布了。在生产环境中实际200个用户时，服务崩溃了——P99 TTFT暴增，GPU满载。

发生了两件事。第一，k6发送了500个相同的提示——你的请求合并和前缀缓存让它看起来像在处理500个并发解码，而实际上只处理了一个。第二，k6不按人眼体验的方式跟踪流式响应的token间延迟；它只看到一个HTTP连接，而不是500个以不同间隔到达的token。

为LLM进行负载测试是它自己的专业领域。

## 核心概念

### GIL陷阱（Locust）

Locust使用Python并在客户端GIL下运行分词。在高并发下，分词器在请求生成之后排队。上报的token间延迟包含了客户端的分词积压。你以为服务器慢了；其实是测试工具本身的问题。

修复方案：LLM-Locust扩展将分词移到独立进程，或使用编译语言测试工具（k6，使用tokenizers.rs的LLMPerf）。

### 提示均匀性陷阱

所有已知负载测试工具都允许你配置一个提示。在10,000次迭代的循环测试中，每次发送完全相同的提示。服务器每次都看到相同的前缀——前缀缓存命中率接近100%，吞吐量看起来很棒。

修复方案：从提示分布中采样。LLMPerf使用 `--mean-input-tokens 500 --stddev-input-tokens 150` ——多样化的长度和内容。

### 四种负载模式

1.  **稳态** —— 30-60分钟内保持恒定的RPS。捕获：基线性能回归。
2.  **斜坡** —— 在15分钟内将RPS从0线性增加到目标值。捕获：容量断点、预热异常。
3.  **尖峰** —— 突然将RPS提高3-10倍，持续2分钟然后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4.  **浸泡** —— 稳态运行4-8小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026年工具图谱

**LLMPerf** (Anyscale) —— Python编写但使用Rust支持的分词。支持均值/标准差提示。流式感知。性能运行的最佳默认选择。

**NVIDIA GenAI-Perf** —— NVIDIA的参考实现。使用Triton客户端；指标覆盖全面。注意其ITL排除TTFT；LLMPerf的包含它。两个工具对同一服务器会产出不同的TPOT。

**LLM-Locust** (TrueFoundry) —— 修复了GIL陷阱的Locust扩展。熟悉的Locust DSL + 流式指标。

**guidellm** —— 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025年9月)**：
- k6本身（Go编写、编译型、无GIL）添加了流式感知指标。
- k6 Operator使用TestRun / PrivateLoadZone CRD实现Kubernetes原生分布式测试。
- 最适合CI/CD门禁和SLA测试。

**Vegeta** —— Go编写，比k6简单。恒定速率HTTP饱和测试。不具备LLM感知能力，但适用于网关/速率限制测试。

**Locust 2.43.3原生版** —— 在LLM测试中存在GIL陷阱。仅通过LLM-Locust扩展使用。

### CI中的SLA门禁

在PR上运行k6：

- 每个基准RPS下运行30-50次迭代。
- 门禁条件：P50/P95 TTFT、5xx错误率 < 5%、TPOT低于阈值。
- 违反条件则中断构建。

### 真实的提示分布

从真实流量样本（如果有的话）或公开分布（如用于聊天的ShareGPT提示、用于代码的HumanEval）构建。将均值 + 标准差输入LLMPerf。务必避免使用单一提示的循环测试。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025年9月。
- k6 v2026.1.0：流式感知指标。
- 典型LLMPerf运行：并发数X下进行100-1000个请求。
- 典型CI门禁：每个PR 30-50次迭代。
- 四种模式：稳态、斜坡、尖峰、浸泡。

## 动手使用

`code/main.py` 使用真实提示分布模拟负载测试，测量有效TPOT，并演示均匀提示陷阱。

## 交付产出

本课程产出 `outputs/skill-load-test-plan.md`。给定工作负载和SLA，选择工具并设计四种负载模式。

## 练习

1.  运行 `code/main.py`。比较均匀分布与真实分布——差距在哪里？
2.  编写用于CI门禁的k6脚本：100并发下TTFT P95 < 800毫秒，运行时间5分钟。
3.  你的浸泡测试显示内存每小时增长50MB。说出三种可能原因以及用于区分它们的检测方法。
4.  从10 RPS尖峰到100 RPS。如果已部署Karpenter + vLLM生产栈（第17阶段 · 03 + 18），预期恢复时间是多久？
5.  GenAI-Perf报告TPOT=6ms；LLMPerf对同一服务器报告TPOT=11ms。请解释。

## 关键术语

| 术语 | 人们怎么叫 | 实际含义 |
|------|------------|----------|
| LLMPerf | "LLM测试工具" | Anyscale基准测试工具，流式感知 |
| GenAI-Perf | "NVIDIA工具" | NVIDIA参考测试工具 |
| LLM-Locust | "用于LLM的Locust" | 修复GIL陷阱的Locust扩展 |
| guidellm | "合成基准测试" | 大规模合成工具 |
| k6 Operator | "K8s k6" | 基于CRD的分布式k6 |
| GIL陷阱 | "Python客户端开销" | 分词积压膨胀上报延迟 |
| 提示均匀性陷阱 | "单一提示谎言" | 使用相同提示的循环测试命中缓存，膨胀吞吐量 |
| 稳态 | "恒定负载" | N分钟内平坦的RPS |
| 斜坡 | "线性上升" | 在一段时间内从0增加到目标值 |
| 尖峰 | "突发测试" | 突然倍增然后恢复 |
| 浸泡 | "长时间测试" | 运行数小时用于检测泄漏 |

## 延伸阅读

- [TianPan — 负载测试LLM应用](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — 2026年负载测试LLM](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — LLM推理基准测试入门](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)