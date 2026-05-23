# 毕业项目 14 —— 投机解码推理服务器

> EAGLE-3 在 vLLM 0.7 中实现实时流量下 2.5-3 倍吞吐量提升。P-EAGLE (AWS 2026) 进一步推进了并行推测技术。SGLang 的 SpecForge 实现了大规模草稿头训练。Red Hat 的 Speculators 中心发布了针对常见开源模型的对齐草稿。TensorRT-LLM 使投机解码在 NVIDIA 平台上成为一等公民。2026 年的生产服务堆栈采用 vLLM 或 SGLang 配合 EAGLE 系列草稿、FP8 或 INT4 量化，以及基于队列等待时间的 HPA。本毕业项目旨在以完整尾延迟报告的形式，在 2.5 倍以上基线吞吐量下服务两个开源模型。

**类型：** 毕业项目
**语言：** Python (服务端)，C++ / CUDA (内核检查)，YAML (配置)
**先决条件：** 阶段 3 (深度学习)，阶段 7 (transformers)，阶段 10 (从零构建 LLMs)，阶段 17 (基础设施)
**涉及阶段：** P3 · P7 · P10 · P17
**时间：** 30 小时

## 问题

投机解码在 2026 年已成为标准化技术。EAGLE-3 草稿头基于目标模型的隐藏状态进行训练，提前预测 N 个 token；目标模型在单次前向传播中验证。60-80% 的接受率转化为 2-3 倍的端到端吞吐量提升。vLLM 0.7 原生集成了此功能。SGLang + SpecForge 提供训练流水线。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布了对齐草稿。

关键在于服务运维，而非模型本身。接受率随流量分布（ShareGPT 与代码及领域数据对比）而波动。拒绝情况下的尾延迟比无推测时更差——你必须在不同批次大小下报告 p99，而不仅仅是稳态 token/秒。每百万 token 成本与 Anthropic / OpenAI API 的对比是建立可信度的杠杆。

## 概念

投机解码包含两个层次。一个**草稿**模型（EAGLE-3 头、ngram 或更小的目标对齐模型）每步提议 k 个候选 token。**目标**模型在单次前向传播中验证所有 k 个；任何被接受的前缀都会替换贪婪路径。接受率取决于草稿-目标对齐度和输入分布。

EAGLE-3 在大多数流量下优于 ngram 草稿。P-EAGLE 运行并行推测以构建更深的草稿树。权衡在于：拒绝时的 P99 延迟更高，因为验证前向传播更大。服务配置必须报告按批次大小分桶的延迟以揭示这一点。

部署采用 Kubernetes。vLLM 0.7 每 GPU 或张量并行分片运行一个副本。HPA 基于队列等待时间而非 CPU 进行自动扩缩。FP8 (Marlin) 和 INT4 (AWQ) 量化将 GPU 内存占用控制在 H100 / H200 容量范围内。端到端报告包括吞吐量、接受率、批次 1/8/32 下的 p50/p99，以及每百万 token 成本。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务端：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 草稿头、P-EAGLE 并行推测、ngram 回退
- 草稿训练：SpecForge (SGLang) 或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8 (Marlin)、INT4 AWQ
- 部署：Kubernetes + NVIDIA 设备插件；基于队列等待指标的 HPA
- 评估：使用 ShareGPT、MT-Bench-v2、GSM8K、HumanEval 测量跨领域接受率
- 参考：TensorRT-LLM 投机解码作为供应商基线

## 构建

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 1xH100（或 2x 张量并行）上使用 vLLM 0.7 部署。
2. **草稿来源。** 从 Red Hat Speculators 获取对齐的 EAGLE-3 草稿头（或通过 SpecForge 训练一个）。加载到 vLLM 的投机解码配置中。
3. **基线数据。** 未启用推测时：记录批次 1/8/32 下的 token/s、p50/p99 延迟、GPU 利用率。发布结果。
4. **启用 EAGLE-3。** 切换配置；运行相同基准测试。报告加速比、接受率、p99 尾延迟变化。
5. **P-EAGLE。** 启用并行推测；测量更深草稿树与串行 EAGLE-3 的对比。报告 P-EAGLE 何时有益、何时有害的拐点。
6. **领域流量。** 通过同一服务器运行 ShareGPT、HumanEval 和领域特定流量。测量各分布下的接受率。识别草稿漂移时机。
7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同流水线。草稿更复杂（MoE 路由噪声）。进行报告。
8. **K8s HPA。** 使用跟踪 `queue_wait_ms` 的 HPA 在 K8s 下部署。演示负载增至三倍时的扩缩容。
9. **成本对比。** 计算每百万 token 成本，与相同评估集上的 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 对比。发布结果。

## 使用

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付

`outputs/skill-inference-server.md` 描述了交付成果：一个配备投机解码的实测服务堆栈、一份完整基准报告和一个 K8s 部署方案。

| 权重 | 准则 | 测量方法 |
|:-:|---|---|
| 25 | 与基线的实测加速比 | 两个模型在匹配质量下达到 2.5 倍以上吞吐量 |
| 20 | 真实流量下的接受率 | 各分布接受率报告 |
| 20 | P99 尾延迟规范性 | 有无推测时批次 1/8/32 下的 p99 值 |
| 20 | 运维 | K8s 部署、基于队列等待的 HPA、平滑发布 |
| 15 | 报告与方法论 | 清晰说明变更内容及原因 |
| **100** | | |

## 练习

1. 测量草稿版本落后目标模型一个版本时（如 Llama 3.3 → 3.4 漂移）接受率的下降。构建监控告警。
2. 实现 ngram 回退：若 EAGLE-3 接受率低于阈值，切换至 ngram 草稿。报告可靠性提升。
3. 运行受控 MoE 实验：相同 Qwen3-Coder-30B 模型，对比注入与不注入路由噪声的情况。测量草稿接受率敏感性。
4. 扩展至 H200 (141 GB)。报告每副本模型容量提升以及是否可服务未量化的 Llama 3.3 70B。
5. 在相同 H100 硬件上测试 TensorRT-LLM 投机解码。报告其相对于 vLLM 的优胜场景。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 草稿模型 | "推测器" | 提议 N 个 token 供目标模型验证的小模型 |
| EAGLE-3 | "2026 草稿架构" | 基于目标隐藏状态训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行推测" | 草稿分支树，单次目标前向传播验证 |
| 接受率 | "命中率" | 草稿 token 无需重采样即被接受的比例 |
| 量化 | "FP8 / INT4" | 降低权重精度以在 GPU 内存中容纳更大模型 |
| 队列等待 | "HPA 指标" | 请求在推理开始前在待处理队列中的等待时间 |
| Speculators 中心 | "对齐草稿" | Red Hat Neural Magic 为常见开源模型提供的 EAGLE 草稿中心 |

## 延伸阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务堆栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机解码论文及集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿中心
- [TensorRT-LLM 投机解码](https://nvidia.github.io/TensorRT-LLM/) — 供应商替代方案
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 论文 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 代码库](https://github.com/vllm-project/vllm) — 代码与基准测试