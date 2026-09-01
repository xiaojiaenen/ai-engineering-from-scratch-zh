# Capstone 14 — 推测解码推理服务器

> 推测解码 —— 廉价的草稿模型提出 token，目标模型一次性验证 —— 现已成为生产就绪的优化手段，而非研究技巧。vLLM 0.7 中的 EAGLE-3 在实际流量下实现了 2.5-3x 的吞吐量提升。P-EAGLE（AWS 2026）进一步推动了并行推测。SGLang 的 SpecForge 实现了大规模训练草稿头。Red Hat 的 Speculators hub 为常见开源模型发布了已对齐的草稿。TensorRT-LLM 使 NVIDIA 上的推测解码成为一等公民。2026 年的生产服务栈是 vLLM 或 SGLang 搭配 EAGLE 家族草稿、FP8 或 INT4 量化，以及基于队列等待的 HPA。本次 Capstone 的目标是以 2.5x+ 的基线吞吐量服务两个开源模型，并输出完整的尾部延迟报告。

**类型：** Capstone
**语言：** Python（服务）、C++ / CUDA（内核检查）、YAML（配置）
**先决条件：** Phase 3（深度学习）、Phase 7（Transformer）、Phase 10（从零实现 LLM）、Phase 17（基础设施）
**涉及的阶段：** P3 · P7 · P10 · P17
**时间：** 30 小时

## 问题

推测解码在 2026 年已成为大宗商品。EAGLE-3 草稿头在目标模型的隐藏状态上训练，可提前预测 N 个 token；目标模型单次前向传播即可验证。60-80% 的接受率可转化为 2-3x 的端到端吞吐量。vLLM 0.7 已原生集成此功能。SGLang + SpecForge 提供训练管线。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布了对齐草稿。

关键在于服务运维，而非模型本身。接受率会随流量分布（ShareGPT vs 代码 vs 领域数据）而漂移。拒绝情况下的尾部延迟比不使用推测时更差 —— 你必须在多个 batch size 下报告 p99，而非仅仅稳态 tokens/sec。与 Anthropic / OpenAI API 相比的每 1M token 成本是关键的可信度指标。

## 概念

推测解码有两层。一个 **草稿** 模型（EAGLE-3 头、ngram 或较小的目标对齐模型）在每个步骤提出 k 个候选 token。**目标** 模型在一次前向中验证所有 k 个；任何前缀被接受后即可替换贪心路径。接受率取决于草稿 - 目标对齐程度和输入分布。

EAGLE-3 在大多数流量上优于 ngram 草稿。P-EAGLE 运行并行推测以实现更深的草稿树。权衡在于：拒绝时的 P99 延迟更高，因为验证前向更大。服务配置必须按 batch size 桶化延迟报告以揭示这一问题。

部署使用 Kubernetes。vLLM 0.7 每个 GPU 运行一个副本或 tensor-parallel 分片。HPA 基于队列等待而非 CPU 进行自动扩缩容。FP8（Marlin）和 INT4（AWQ）量化将 GPU 显存控制在 H100 / H200 范围内。端到端报告包括吞吐量、接受率、batch 1/8/32 下的 p50/p99，以及 $/1M token 成本。

## 架构

```
请求入口
    |
    v
vLLM 服务器 (0.7) 或 SGLang (0.4)
    |
    +-- 草稿：EAGLE-3 头 | P-EAGLE 并行 | ngram 回退
    +-- 目标：Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     量化为 FP8-Marlin 或 INT4-AWQ
    |
    v
验证前向：将 k 个草稿 token 批量通过目标模型
    |
    v（接受前缀；对被拒绝的后缀重新采样）
    v
token 流返回客户端
    |
    v
Prometheus 指标：吞吐量、接受率、队列等待、延迟 p50/p99
    |
    v
基于队列等待指标的 HPA
```

## 技术栈

- 服务框架：vLLM 0.7 或 SGLang 0.4
- 推测方法：EAGLE-3 草稿头、P-EAGLE 并行推测、ngram 回退
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；基于队列等待指标的 HPA
- 评测：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于跨领域接受率测量
- 参考：TensorRT-LLM 推测解码用于厂商基线对比

```figure
cf-spec-decode
```

## 构建指南

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 1xH100（或 2x tensor-parallel）上的 vLLM 0.7 下部署。

2. **草稿来源。** 从 Red Hat Speculators 拉取已对齐的 EAGLE-3 草稿头（或通过 SpecForge 自行训练）。加载到 vLLM 的推测解码配置中。

3. **基线数据。** 启用推测前：记录 batch 1/8/32 下的 tokens/s、p50/p99 延迟、GPU 利用率。发布基线。

4. **启用 EAGLE-3。** 切换配置；重新运行相同基准测试。报告加速比、接受率、p99 尾部延迟变化。

5. **P-EAGLE。** 启用并行推测；测量更深草稿树与串行 EAGLE-3 的对比。报告 P-EAGLE 由益转损的拐点。

6. **领域流量。** 将 ShareGPT vs HumanEval vs 特定领域流量通过同一服务器。测量每种分布下的接受率。识别草稿漂移何时发生。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同管线。草稿更棘手（MoE 路由噪声）。报告结果。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。演示负载三倍时的自动扩缩容。

9. **成本对比。** 在同一评测集上计算 $/1M token 与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 的对比。发布报告。

## 使用方式

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 已启用
[decode]    bs=8, 每步接受 token 数=3.2, 接受率=0.76
[latency]   首 token 延迟 42ms, 完整响应 980ms（620 tokens）
[cost]      持续吞吐量下每 1M 输出 token $0.34
```

## 交付物

`outputs/skill-inference-server.md` 描述交付内容。包含带推测解码的实测服务栈、完整基准测试报告及 K8s 部署。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 相对于基线的实测加速比 | 两个模型在质量匹配下实现 2.5x+ 吞吐量 |
| 20 | 真实流量下的接受率 | 按分布的报告接受率 |
| 20 | P99 尾部延迟规范 | 有/无推测时 batch 1/8/32 的 p99 |
| 20 | 运维能力 | K8s 部署、队列等待 HPA、平滑发布 |
| 15 | 文档与方法论 | 清晰说明变化内容与原因 |
| **100** | | |

## 练习

1. 测量草稿比目标模型晚一个版本时的接受率退化（如 Llama 3.3 -> 3.4 漂移）。构建监控告警。

2. 实现 ngram 回退：当 EAGLE-3 接受率低于阈值时切换到 ngram 草稿。报告可靠性提升。

3. 运行受控的 MoE 实验：相同 Qwen3-Coder-30B 分别注入与不注入路由噪声。测量草稿接受率敏感性。

4. 扩展至 H200（141 GB）。报告每个副本可获得的模型容量余量，以及是否能服务未量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上对 TensorRT-LLM 推测解码进行基准测试。报告其相对于 vLLM 的优势场景。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 草稿模型 | "Speculator" | 提出 N 个 token 供目标模型验证的小型模型 |
| EAGLE-3 | "2026 草稿架构" | 在目标隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行推测" | 一批草稿分支在一次目标前向中验证 |
| 接受率 | "命中率" | 无需重采样即被接受的草稿 token 比例 |
| 量化 | "FP8 / INT4" | 降低精度以在 GPU 显存中容纳更大模型 |
| 队列等待 | "HPA 指标" | 请求在推理开始前于待处理队列中等待的时间 |
| Speculators hub | "对齐草稿" | Red Hat Neural Magic 的 EAGLE 草稿 hub，覆盖常见开源模型 |

## 延伸阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行推测解码论文与集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练管线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿 hub
- [TensorRT-LLM 推测解码](https://nvidia.github.io/TensorRT-LLM/) — 厂商替代方案
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm) — 代码与基准测试
