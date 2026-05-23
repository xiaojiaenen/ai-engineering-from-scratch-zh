# 分离式预填充/解码 — NVIDIA Dynamo 与 llm-d

> 预填充受计算限制；解码受内存限制。在同一块 GPU 上运行两者会浪费资源。分离架构将它们分配到不同的资源池，并通过 NIXL（RDMA/InfiniBand 或 TCP 备选）在它们之间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，1.0 GA 版）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA 规划器可自动调整预填充:解码比率以满足 SLO。NVIDIA 公布的吞吐量提升幅度如下：developer.nvidia.com（2025 年 6 月）显示在中等延迟场景下，DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上可获得约 6 倍提升；Dynamo 产品页面（developer.nvidia.com，未注明日期）声称在 GB300 NVL72 + Dynamo 上 MoE 吞吐量相比 Hopper 最高可提升 50 倍。"30 倍"这一数字是社区根据完整 Blackwell + Dynamo + DeepSeek-R1 报告得出的综合数据；我们尚未找到确切提到 30 倍的单一原始来源，因此应将其视为方向性参考。llm-d（Red Hat + AWS）是 Kubernetes 原生方案：预填充 / 解码 / 路由器作为独立的 Service，按角色配置 HPA。llm-d 0.5 增加了分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容至零功能。经济效益：内部汇总多个客户披露的信息显示，在保持相同 SLA 的情况下，从共置服务切换到使用 Dynamo 的分离架构，200 万美元级别的推理支出可节省 30–40%（即每年 60-80 万美元）；该 200 万→60-80 万美元的数字是内部综合数据，非单一公开案例研究——将其视为数量级参考，而非引用来源。短提示词（<512 token，短输出）无法证明传输成本的合理性。

**类型：** 学习
**语言：** Python（标准库，演示性分离式与共置式模拟器）
**前置课程：** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 08（推理指标）
**时间：** 约 75 分钟

## 学习目标

- 解释为何预填充和解码需要不同的最优 GPU 分配，并量化共置模式下的资源浪费。
- 绘制分离式架构图：预填充池、解码池、通过 NIXL 传输 KV、路由器。
- 说明分离式架构不适用的情况（短提示词、短输出）。
- 区分 NVIDIA Dynamo（栈上层方案）与 llm-d（Kubernetes 原生方案），并匹配各自适用场景。

## 问题所在

您在 8 块 H100 上运行 Llama 3.3 70B。在混合负载（长提示词 + 短输出）下，由于大部分计算已用于预填充，解码期间 GPU 处于空闲状态。在不同负载（短提示词 + 长输出）下，情况则相反。共置式预填充 + 解码意味着您需要同时过度配置两者。

预算影响：20-40% 的 GPU 时间浪费在错误资源上。您为运行受内存限制的解码购买 H100 计算资源，或为运行受计算限制的预填充购买 H100 HBM 带宽。两者都是昂贵的浪费。

分离式架构将预填充和解码分配到各自瓶颈优化的独立资源池。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 核心概念

### 为何瓶颈不同

**预填充** — 在一次前向传播中处理整个输入提示词的 Transformer。矩阵乘法占主导；受计算限制。H100 FP8 可提供约 2000 TFLOPS 的有效吞吐量。批处理效率高——一次前向传播处理多个 token。

**解码** — 每次迭代生成一个 token，需读取完整权重。受内存带宽限制。HBM3 提供约 3 TB/s。批处理仅在并发量高时效率才好——权重读取成本可在批内分摊。

共置模式：您购买为两者优化的 GPU。H100 两者都擅长，但成本相同。在规模上，您需要预填充池使用 H100 / 计算密集型；解缩池使用 H200 / 内存密集型，或采用激进量化。

### 架构图

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输层。在可用时使用 RDMA/InfiniBand，否则回退到 TCP。传输延迟真实存在——在 70B FP8 模型上，4K token 提示词的 KV 缓存传输通常需要 20-80 毫秒。这是短提示词不适用分离式架构的原因：传输成本超过节省收益。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA 版）：
- 位于 vLLM、SGLang、TRT-LLM 之上，作为编排器。
- Planner Profiler 测量工作负载，SLA 规划器自动配置预填充:解码比率。
- Rust 核心，Python 可扩展性。
- 吞吐量提升：NVIDIA 报告在中等延迟场景下，DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上可获得 6 倍提升（developer.nvidia.com，2025 年 6 月）；社区报告的"最高 30 倍"提升基于完整 Blackwell + Dynamo + DeepSeek-R1 技术栈，缺乏单一原始来源，应视为方向性参考。
- GB300 NVL72 + Dynamo：据 Dynamo 产品页面（developer.nvidia.com，未注明日期），MoE 吞吐量相比 Hopper 最高可提升 50 倍。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充 / 解码 / 路由器作为独立的 Kubernetes Service。
- 按角色配置 HPA，使用队列深度（预填充）/ KV 利用率（解码）作为信号。
- `topologyConstraint packDomain: rack` 在同一机架内打包预填充+解码组合，以实现高速 KV 传输。
- llm-d 0.5（2026 年）：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容至零。

如果您需要托管的栈上层编排器，请使用 Dynamo。如果您需要 Kubernetes 原生原语并致力于 CNCF 生态系统，请使用 llm-d。

### 经济效益

内部综合数据（非单一公开案例研究——数量级参考）：
- 年推理支出 200 万美元，采用共置服务。
- 切换到使用 Dynamo 的分离式架构。
- 相同请求量，相同 P99 延迟 SLA。
- 报告年节省：60 万–80 万美元（降低 30–40%）。
- 无需新硬件。

我们综合多个客户披露而非单一可引用案例研究得出此数据；最接近的公开数据点是 Baseten 的报告：使用 Dynamo KV 路由后 TTFT 快 2 倍、吞吐量提高 61%（baseten.co，2025 年 10 月），以及 VAST + CoreWeave 的预测：在 40–60% KV 命中率下，每美元 token 数提高 60–130%（vastdata.com，2025 年 12 月）。节省源于为每个资源池进行合理配置；预填充密集型负载（如带有 8K+ 前缀的 RAG）比平衡型负载受益更多。

### 何时不应使用分离式架构

- 提示词 <512 token 且输出 <200 token：传输成本占主导，收益有限。
- 小型集群（<4 块 GPU）：资源池多样性不足。
- 团队无法按角色扩展运维两个 GPU 资源池：Dynamo 有所帮助，但并不轻松。
- 无 RDMA 网络：TCP 传输成本更高。

### 路由器与 Phase 17 · 11 集成

分离式路由器是 KV 缓存感知的（Phase 17 · 11）。请求首先落在持有其前缀的解码池上——若无匹配，则流向预填充 → 解码。命中率与分离式架构相互增强——缓存感知路由器决定了是否需要新的预填充操作。

### MoE 在 Blackwell 上的性能才是真正的数据来源

GB300 NVL72 + Dynamo 显示 MoE 吞吐量相比 Hopper 基线提升 50 倍。MoE 专家路由在预填充阶段计算密集，在解码阶段内存密集（专家缓存），因此分离式架构是双重收益。2026 年前沿模型服务将是 MoE 主导（DeepSeek-V3、未来 GPT-5 变体）。

### 需要记住的数字

基准测试数字会变化——NVIDIA 和推理技术栈每季度都会发布更新结果。引用前请重新核实。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：在中等延迟场景下，吞吐量相比基线约提升 6 倍（developer.nvidia.com，2025 年 6 月）；社区在完整 Blackwell + Dynamo 技术栈上报告的"最高 30 倍"提升是方向性综合数据，无单一原始来源。
- GB300 NVL72 + Dynamo：MoE 吞吐量相比 Hopper 最高提升 50 倍（developer.nvidia.com，未注明日期）。
- 节省基准（内部综合数据，非单一案例研究）：在相同 SLA 下，年支出从 200 万降至 60-80 万美元。
- 分离式架构阈值：提示词 >512 token + 输出 >200 token。
- 通过 NIXL 传输 KV：在 70B FP8 模型上，4K 提示词的 KV 传输耗时 20-80 毫秒。

## 动手实践

`code/main.py` 模拟共置式与分离式服务。报告吞吐量、每请求成本以及提示词长度临界点。

## 交付成果

本课程产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否采用分离式架构。

## 练习

1. 运行 `code/main.py`。在什么提示词长度下，分离式架构优于共置式架构？
2. 为 RAG 服务设计预填充池和解码池，P99 前缀长度 8K，输出长度 300。
3. Dynamo vs llm-d：为一家无 Python 运行时偏好的纯 Kubernetes 环境选择其一。
4. 计算 KV 传输成本：70B FP8 模型上的 4K 预填充产生约 500 MB KV。RDMA 100 GB/s 时，传输耗时 5 毫秒。TCP 10 GB/s 时，耗时 50 毫秒。哪个影响您的 SLA？
5. MoE 专家路由改变了 KV 访问模式。对于每个 token 激活不同专家的 MoE，分离式架构表现如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 分离式服务 | “拆分预填充/解码” | 为每个阶段使用独立的 GPU 资源池 |
| NIXL | “NVIDIA 传输层” | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | “编排器” | 位于 vLLM/SGLang/TRT-LLM 之上的协调器 |
| llm-d | “Kubernetes 原生” | Red Hat + AWS 的 K8s 分离式技术栈 |
| Planner Profiler | “Dynamo 自动配置” | 测量工作负载，配置资源池比率 |
| SLA 规划器 | “Dynamo 策略” | 自动调整预填充:解码比率以满足 SLO |
| `packDomain: rack` | “llm-d 拓扑” | 在同一机架内打包预填充+解码以实现快速 KV 传输 |
| UCCL | “统一集合通信” | llm-d 0.5 的网络层，支持缩容至零 |
| MoE 专家路由 | “每 token 一个专家” | DeepSeek-V3 模式；分离式架构有益 |

## 扩展阅读

- [NVIDIA — 介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — 在 Kubernetes 上部署分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 分离式服务技术博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 发行说明](https://github.com/llm-d/llm-d/releases)