# Disaggregated Prefill/Decode — NVIDIA Dynamo and llm-d

> Prefill 是计算密集型；decode 是内存密集型。在同一块 GPU 上运行两者会浪费一种资源。Disaggregation（分离式部署）将它们分配到独立的资源池，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在它们之间传输 KV cache。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）运行在 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 会自动匹配 prefill:decode 比例以满足 SLO。NVIDIA 公布吞吐量提升数据在这个量级——developer.nvidia.com（2025-06）显示 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 的中延迟场景下约提升 6 倍，Dynamo 产品页（developer.nvidia.com，无日期）宣称在 GB300 NVL72 + Dynamo 上相比 Hopper 可将 MoE 吞吐量提升最高 50 倍。"30 倍"这个数字是社区对全栈 Blackwell + Dynamo + DeepSeek-R1 报告的汇总——我们未找到单一原始来源明确指出恰好 30x，因此应将其视为方向性声明。llm-d（Red Hat + AWS）是 Kubernetes 原生的：prefill / decode / router 作为独立服务，每个角色配有 HPA。llm-d 0.5 增加了分层 KV offloading、cache-aware LoRA routing、UCCL 网络、scale-to-zero。经济影响：多个客户披露的内部汇总表明，在保持相同 SLA 的情况下从共置部署切换到带 Dynamo 的分离式部署，可为 200 万美元级别推理支出节省 30–40%（即每年 60-80 万美元）——具体 $2M→$600-800K 数字是内部合成数据，并非单一已发布案例研究——应将其作为数量级参考，而非引用来源。短提示（<512 tokens，短输出）不值得承担传输成本。

**类型：** 学习
**语言：** Python（stdlib，分离式 vs 共置模拟器玩具程序）
**前置知识：** Phase 17 · 04（推理引擎内部原理），Phase 17 · 08（推理指标）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 有不同的最优 GPU 配置，并量化共置下的浪费。
- 绘制分离式架构：prefill 池、decode 池、通过 NIXL 的 KV 传输、router。
- 说出分离式部署不值得的场景（短提示、短输出）。
- 区分 NVIDIA Dynamo（栈上方编排器）和 llm-d（Kubernetes 原生），并匹配各自适用场景。

## 问题

你在 8 张 H100 上运行 Llama 3.3 70B。在混合工作负载（长提示 + 短输出）下，GPU 在 decode 阶段空闲，因为大部分计算已消耗在 prefill 上。在另一种工作负载（短提示 + 长输出）下，情况相反。共置 prefill + decode 意味着你两种资源都过度配置了。

预算影响：20-40% 的 GPU 时间浪费在不匹配的资源上。你购买 H100 算力来运行内存密集的 decode，或购买 H100 HBM 带宽来运行计算密集的 prefill。两者都是昂贵的浪费。

分离式部署将 prefill 和 decode 拆分到分别按各自瓶颈调优的池上。KV cache 通过高带宽互联从 prefill 池传输到 decode 池。

## 概念

### 为什么瓶颈不同

**Prefill** —— 对完整输入提示执行一次前向传播，运行 transformer。矩阵乘法占主导；计算密集。H100 FP8 可提供约 2000 TFLOPS 的有效吞吐量。批处理效率高——一次前向传播处理多个 token。

**Decode** —— 每次生成一个 token，每个迭代读取完整权重。内存带宽瓶颈。HBM3 提供约 3 TB/s。只有在高并发下批处理效率才好——权重读取分摊到整个 batch。

共置两者：你购买的 GPU 需同时优化两种能力。H100 两种都擅长，但价格相同。在大规模场景下，你希望 prefill 池使用 H100 / 计算密集型；decode 池使用 H200 / 内存密集型，或使用激进量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (仅提示)                        │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (计算)        │                │  (内存)       │
            └──────────────┘                └──────┬───────┘
                                                   │ token
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输协议。在有 RDMA/InfiniBand 时使用 RDMA，否则回退到 TCP。传输延迟是真实的——对于 70B FP8 模型上 4K token 提示的 KV cache，通常为 20-80 ms。这就是短提示不值得分离的原因：传输税超过了收益。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 作为编排器运行在 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比例。
- Rust 核心，Python 扩展。
- 吞吐量提升：NVIDIA 报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 中延迟场景下约 6 倍（developer.nvidia.com，2025-06）；社区报告"最高 30 倍"的全 Blackwell + Dynamo + DeepSeek-R1 堆栈缺乏单一原始来源，应视为方向性数据。
- GB300 NVL72 + Dynamo：相比 Hopper 高达 50 倍 MoE 吞吐量（根据 Dynamo 产品页，developer.nvidia.com，无日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / router 作为独立的 Kubernetes 服务。
- 每个角色配 HPA，信号源为队列深度（prefill）/ KV 利用率（decode）。
- `topologyConstraint packDomain: rack` 将 prefill+decode 集群打包到同一机架以实现高带宽 KV 传输。
- llm-d 0.5（2026）：分层 KV offloading、cache-aware LoRA routing、UCCL 网络、scale-to-zero。

如果你想要一个托管的栈上方编排器，使用 Dynamo。如果你想要 Kubernetes 原生原语并 committed to CNCF 生态，使用 llm-d。

### 经济影响

内部合成数据（非单一已发布案例研究——数量级参考）：

- 共置部署的推理支出：200 万美元/年。
- 切换到带 Dynamo 的分离式部署。
- 相同请求量，相同 P99 延迟 SLA。
- 报告节省：每年 60 万–80 万美元（降低 30–40%）。
- 无需新硬件。

我们通过多个客户披露综合得出此数字，而非单一可引用案例研究；最接近的已发布数据点是 Baseten 的 Dynamo KV 路由使 TTFT 快 2 倍 / 吞吐量提高 61%（baseten.co，2025-10），以及 VAST + CoreWeave 预计在 40–60% KV hit rate 下每美元可多获取 60–130% tokens（vastdata.com，2025-12）。节省来自各池的精确调优；prefill 密集的工作负载（带 8K+ 前缀的 RAG）比均衡型获益更多。

### 何时不应分离

- 提示 < 512 tokens 且输出 < 200 tokens：传输税超过收益。
- 小规模集群（< 4 GPU）：池多样性不足。
- 团队无法运营两个 GPU 池加按角色伸缩：Dynamo 有帮助，但并非一蹴而就。
- 无 RDMA  fabric：TCP 传输税更重。

### Router 与 Phase 17 · 11 集成

分离式 router 是 KV cache 感知的（Phase 17 · 11）。请求落在持有其前缀的 decode 池上——若无匹配，则流向 prefill → decode。hit rate 和分离式部署相互叠加——cache-aware router 决定是否需要新的 prefill。

### MoE on Blackwell 是真正数据的来源

GB300 NVL72 + Dynamo 相比 Hopper 基线显示 50 倍 MoE 吞吐量。MoE 专家路由在 prefill 时计算密集，在 decode 时内存密集（专家缓存），因此分离式部署是双重收益。2026 前沿模型推理是 MoE 主导的（DeepSeek-V3、未来 GPT-5 变体）。

### 需要记住的数字

基准数字会漂移——NVIDIA 和推理栈每季度发布更新结果。引用前请重新核实。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：中延迟场景下约 6 倍吞吐量对比基线（developer.nvidia.com，2025-06）；社区"最高 30 倍"的全 Blackwell + Dynamo 堆栈声明是缺乏单一原始来源的方向性汇总。
- GB300 NVL72 + Dynamo：相比 Hopper 高达 50 倍 MoE 吞吐量（developer.nvidia.com，无日期）。
- 节省参考（内部合成，非单一案例研究）：200 万美元/年支出节省 60-80 万美元/年，SLA 不变。
- 分离阈值：提示 > 512 tokens + 输出 > 200 tokens。
- NIXL KV 传输：70B FP8 上 4K 提示的 KV cache 传输为 20-80 ms。

```figure
prefill-decode-split
```

## 应用

`code/main.py` 模拟共置与分离式推理。报告吞吐量、单次请求成本，以及提示长度交叉点。

## 交付

本课产出 `outputs/skill-disaggregation-decider.md`。根据工作负载和集群，决定是否分离。

## 练习

1. 运行 `code/main.py`。在什么提示长度下分离式超过共置？
2. 为 P99 前缀长度 8K、输出 300 的 RAG 服务设计 prefill 池和 decode 池。
3. Dynamo vs llm-d：为一个纯 Kubernetes 环境、无 Python 运行时偏好选择其一。
4. 计算 KV 传输成本：70B FP8 上 4K prefill = ~500 MB KV。在 RDMA 100 GB/s 下，传输 = 5 ms。在 TCP 10 GB/s 下 = 50 ms。哪个对你的 SLA 更关键？
5. MoE 专家路由改变 KV 访问模式。当 MoE 对不同 token 激活不同专家时，分离式如何表现？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Disaggregated serving | "split prefill/decode" | 将每个阶段分配到独立的 GPU 池 |
| NIXL | "NVIDIA transport" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "the orchestrator" | vLLM/SGLang/TRT-LLM 之上的栈编排协调器 |
| llm-d | "Kubernetes native" | Red Hat + AWS K8s 分离式堆栈 |
| Planner Profiler | "Dynamo auto-config" | 测量工作负载，配置池比例 |
| SLA Planner | "Dynamo policy" | 自动匹配 prefill:decode 速率以满足 SLO |
| `packDomain: rack` | "llm-d topology" | 将 prefill+decode 打包到同一机架以实现快速 KV 传输 |
| UCCL | "unified collective" | llm-d 0.5 的 scale-to-zero 网络层 |
| MoE expert routing | "expert per token" | DeepSeek-V3 模式；分离式部署有益 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
