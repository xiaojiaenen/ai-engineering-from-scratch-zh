# Kubernetes 上的 GPU 自动伸缩——Karpenter、KAI Scheduler、Gang 调度

> 三层架构，而非单一方案。Karpenter 动态提供节点（<1 分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 Gang 调度、拓扑感知和分层队列——它避免了"7/8 部分分配"陷阱，即七个节点因缺少一个 GPU 而空转烧钱。应用级自动伸缩器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理特定信号伸缩——请求队列深度、KV cache 利用率——而非 CPU/DCGM 占空比。经典 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是占空比测量：100% 可能是 10 个请求或 100 个请求。vLLM 预分配 KV cache 内存，因此内存永远不会触发缩容。本课教你组合这三层，并避免使用默认的 Karpenter `WhenEmptyOrUnderutilized` 策略导致正在运行的推理任务被终止。

**类型：** 学习
**语言：** Python（标准库，玩具队列深度自动伸缩模拟器）
**先修知识：** Phase 17 · 02（推理平台经济学）、Phase 17 · 04（推理引擎内部原理）
**时间：** 约 75 分钟

## 学习目标

- 绘制三层自动伸缩架构图（节点供应、Gang 调度、应用层），并指出每层使用的工具。
- 解释为何 `DCGM_FI_DEV_GPU_UTIL` 不适合作为 vLLM 的 HPA 指标，并说出两个替代方案（队列深度、KV cache 利用率）。
- 描述 Gang 调度及 KAI Scheduler 所防范的部分分配失败模式（8 个 GPU 中 7 个空闲）。
- 指出 Karpenter 会终止正在运行 GPU 任务的整合策略（`WhenEmptyOrUnderutilized`），并说明 2026 年的安全替代方案。

## 问题所在

你的团队在 Kubernetes 上部署了一个 LLM 推理服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号配置了 HPA。服务在工作时间内利用率始终卡在 100%。HPA 从不扩缩——因为它以为你已经满载。你手动增加副本后，TTFT 确实下降了，但 HPA 仍然没有响应。信号在欺骗你。

另外，你使用 Cluster Autoscaler 管理节点。凌晨 2 点收到一个 1M token 的请求；集群花费 3 分钟才提供新节点，请求超时。

再另外，你部署了一个 70B 模型，需要跨 2 个节点使用 8 个 GPU。集群有 7 个空闲 GPU，另有 1 个分散在 3 个节点上。Cluster Autoscaler 为缺失的那个 GPU 提供了一个节点。七个节点空转等待 4 分钟，白白烧钱，直到 Kubernetes  provision 出最后一个 GPU。

三层架构，三种不同的故障模式。2026 年的 GPU 感知自动伸缩不是"打开 HPA"，而是组合节点供应、Gang 调度和应用信号自动伸缩。

## 核心概念

### 第一层——节点供应（Karpenter）

Karpenter 监视待调度的 pod，并在约 45–60 秒内提供节点（Cluster Autoscaler 为 GPU 节点通常需 90–120 秒）。它根据 `NodePool` 约束动态选择实例类型——如果你的 pod 需要 8 个 H100，而集群没有匹配的节点，Karpenter 会直接 provision 一个新节点，而不是扩容现有组。

**整合陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池来说很危险。它会终止运行中的 GPU 节点，将 pod 迁移到更便宜的合适规格实例上。对于推理工作负载，这意味着驱逐正在运行的请求，并在新节点上重新加载 70B 模型。损失包括数分钟容量和请求失败。

GPU 池的安全配置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

这样 Karpenter 可在 1 小时后合并真正空转的节点，但永不驱逐正在运行的任务。

### 第二层——Gang 调度（KAI Scheduler）

KAI Scheduler（最初名为"Karp"）处理默认 kube-scheduler 无法做到的事情：

**Gang 调度**——全有或全无。一个需要 8 个 GPU 的分布式推理 pod，要么 8 个同时启动，要么全部不启动。没有它，就会陷入部分分配陷阱：7/8 个 pod 启动，无限期等待，白白烧钱。

**拓扑感知**——了解哪些 GPU 共享 NVLink，哪些位于同一机架，哪些之间有 InfiniBand 连接。据此放置 pod。一个 DeepSeek-V3 67B tensor-parallel 工作负载必须保持在同一个 NVLink 域内；KAI Scheduler 会尊重这一点。

**分层队列**——多个团队竞争同一个 GPU 池，带有优先级和配额。Team A 的生产突发只有在优先级规则允许的情况下才会被 Team B 的训练任务抢占。

KAI 作为辅助调度器与 kube-scheduler 并行部署；你只需给 workload 打注解即可使用它。Ray 和 vLLM 生产栈均集成。

### 第三层——应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标——它测量在每个采样间隔 GPU 是否在忙。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 无论如何都在忙。基于占空比伸缩等于盲目伸缩。

更糟的是，vLLM 等引擎会预分配 KV cache 内存（最多到 `--gpu-memory-utilization`）。即使只有一个请求，内存使用率也维持在 90% 左右。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待 prefill 的请求数量）。
- KV cache 利用率（已分配给活跃序列的 block 占比）。
- 每个副本的 P99 TTFT（你的 SLO 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并伸缩副本。它们完全替代了 LLM 推理场景下的 HPA。

### 何时使用何工具

| 伸缩决策 | 工具 |
|----------------|------|
| 增/减节点 | Karpenter |
| 调度多 GPU 任务 | KAI Scheduler |
| 增/减副本 | Dynamo Planner / llm-d WVA（或在队列深度上自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 拆分离预填充/解码使一切复杂化

如果你运行拆分离预填充/解码（Phase 17 · 17），你有两类 pod，伸缩触发器不同：prefill pod 基于队列深度伸缩，decode pod 基于 KV cache 压力伸缩。llm-d 将这些暴露为单独的 `Services`，每个角色有独立的 HPA。不要试图用一个 HPA 同时覆盖两者。

### 冷启动问题同样重要

冷启动缓解（Phase 17 · 10）正是节点供应时间变成用户可见的时候。Karpenter 45–60 秒预热 + 20GB 模型加载 + 引擎初始化，意味着从零开始的请求需要 2–5 分钟。为 SLO 关键路径保留一个热池（`min_workers=1`），或在应用层使用类似 Modal 的 checkpointing。

### 你应该记住的数字

- Karpenter 节点供应：约 45–60 秒，Cluster Autoscaler 约 90–120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费——7/8 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：已失效；改用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：会终止正在运行的 GPU 任务。推理场景请使用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 动手实践

`code/main.py` 模拟了三层自动伸缩器在突发 GPU 工作负载下的表现。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI-Gang 调度伸缩。报告未满足请求数、空闲 GPU 分钟数和综合评分。

## 交付成果

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，它设计一份三层自动伸缩方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 丢弃了多少请求是队列深度 HPA 能够接住的？差异来自哪里？
2. 为一个使用 H100 SXM5 推理 Llama 3.3 70B FP8 的集群设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个阻止非 GPU 工作负载落到这些节点上的 taint。
3. 你的团队报告部署卡在 Pending，提示"GPU 可用但 pod 无法调度"。诊断——这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以确认？
4. 为拆分离 prefill pod 和 decode pod 各选一个伸缩信号。分别给出理由。
5. 计算 `WhenEmptyOrUnderutilized` 整合陷阱在 24×7 生产服务上的成本：该服务平均每天发生 60 次请求丢弃事件，P99 TTFT > 10 秒。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | "节点供应器" | Kubernetes 节点自动伸缩器；亚分钟级供应 |
| Cluster Autoscaler | "旧式伸缩器" | Kubernetes 节点自动伸缩器的前身；较慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 Gang + 拓扑 + 队列的辅助调度器 |
| Gang 调度 | "全有或全无" | 原子调度 N 个 pod 或全部推迟 |
| 拓扑感知 | "机架感知" | 基于 NVLink/IB/机架放置放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比指标；不作为 LLM 的伸缩信号 |
| 队列深度 | "等待请求" | prefill 绑定伸缩的正确 HPA 信号 |
| KV cache 利用率 | "内存压力" | decode 绑定伸缩的正确 HPA 信号 |
| 整合（Consolidation） | "Karpenter 整合" | 终止节点以迁移到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全整合" | 不会驱逐正在运行的 GPU 任务的政策 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——设计文档和配置示例。
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/)——整合策略语义和 GPU 安全默认值。
- [NVIDIA —— 在 Kubernetes 上部署拆分离 LLM 推理工作负载](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)——Dynamo Planner 伸缩信号。
- [Ray 文档 —— 用于 RayClusters 的 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html)——Ray 集成模式。
- [AWS EKS 计算与自动伸缩最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)——托管 Kubernetes 专属指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d)——Workload Variant Autoscaler 设计。
