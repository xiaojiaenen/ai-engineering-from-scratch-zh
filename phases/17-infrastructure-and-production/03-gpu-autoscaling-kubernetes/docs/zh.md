# Kubernetes 上的 GPU 自动扩展 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层架构，而非一层。Karpenter 动态配置节点（低于一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 Gang Scheduling、拓扑感知和分层队列 — 它防止了“7/8 部分分配”陷阱，即七个节点因缺少一个 GPU 而空闲等待和消耗资源。应用级自动扩展器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）根据推理特定信号进行扩展 — 队列深度、KV 缓存利用率 — 而非 CPU/DCGM 工作周期。经典的 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是工作周期度量：100% 可能对应 10 个请求或 100 个。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课程教你组合这三层架构，并避免 Karpenter 默认的 `WhenEmptyOrUnderutilized` 策略，该策略会在推理中途终止正在运行的 GPU 作业。

**类型：** 学习
**语言：** Python（标准库，简易队列深度自动扩展器模拟器）
**先决条件：** 阶段 17 · 02（推理平台经济学），阶段 17 · 04（vLLM 服务内部原理）
**时间：** 约 75 分钟

## 学习目标

- 绘制三层自动扩展架构图（节点配置、Gang Scheduling、应用级），并指出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 的错误 HPA 信号，并给出两个替代方案（队列深度、KV 缓存利用率）。
- 描述 Gang Scheduling 以及 KAI Scheduler 所防止的部分分配故障模式（8 个 GPU 中有 7 个空闲）。
- 说出 Karpenter 的合并策略（`WhenEmptyOrUnderutilized`），该策略会终止正在运行的 GPU 作业，并说明 2026 年的安全替代方案。

## 问题所在

你的团队在 Kubernetes 上部署了一个 LLM 服务。你设置了 HPA，使用 `DCGM_FI_DEV_GPU_UTIL` 作为信号。该服务在业务高峰时段占用率固定在 100%。HPA 从不扩容 — 它已经认为你的集群已满。你手动添加了一个副本；TTFT 下降了。HPA 仍然不扩容。信号在欺骗你。

另一方面，你使用 Cluster Autoscaler 进行节点管理。凌晨 2 点，一个 1M-token 的提示到达；集群花了 3 分钟配置一个节点，请求超时了。

再次分开来说，你部署了一个需要跨 2 个节点共 8 个 GPU 的 70B 模型。集群有 7 个空闲 GPU，另有一个 GPU 分散在 3 个节点上。Cluster Autoscaler 为缺失的 1 个 GPU 配置一个节点。七个节点等待了 4 分钟，在 Kubernetes 启动最后一个 GPU 期间白白消耗金钱。

三层架构，三种不同的故障模式。2026 年的 GPU 感知自动扩展不是“开启 HPA”。而是组合节点配置、Gang Scheduling 和应用信号自动扩展。

## 概念

### 第 1 层 — 节点配置（Karpenter）

Karpenter 监视待处理的 Pod，并在约 45-60 秒内配置节点（Cluster Autoscaler 通常需要 90-120 秒来配置 GPU 节点）。它根据 `NodePool` 约束动态选择实例类型 — 如果你的 Pod 需要 8 个 H100，且集群中没有匹配的节点，Karpenter 会直接配置一个，而不是扩展现有的节点组。

**合并陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 策略对 GPU 资源池是危险的。它会终止一个正在运行的 GPU 节点，以便将 Pod 迁移到更便宜、规模合适的实例。对于推理工作负载，这意味着驱逐正在运行的请求并在新节点上重新加载 70B 模型。损失是数分钟的容量加上请求失败。

GPU 资源池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在一小时后合并真正空闲的节点，但从不驱逐正在运行的作业。

### 第 2 层 — Gang Scheduling（KAI Scheduler）

KAI Scheduler（项目原名“Karp”，后更名）处理默认 kube-scheduler 无法处理的事项：

**Gang Scheduling** — 全部调度或全部不调度。一个需要 8 个 GPU 的分布式推理 Pod，要么 8 个全部一起启动，要么一个都不启动。没有这个，你会陷入部分分配陷阱：8 个 Pod 中有 7 个启动了，无限期等待，浪费金钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些位于同一个机架，哪些之间有 InfiniBand 连接。据此放置 Pod。一个 DeepSeek-V3 67B 张量并行工作负载必须保持在一个 NVLink 域内；KAI Scheduler 会遵守这一点。

**分层队列** — 多个团队以不同的优先级和配额竞争同一 GPU 资源池。只有在优先级规则允许的情况下，团队 A 的生产紧急任务才会被团队 B 的训练作业抢占。

KAI 作为辅助调度器与 kube-scheduler 一起部署；你可以通过注解工作负载来使用它。Ray 和 vLLM production-stack 都进行了集成。

### 第 3 层 — 应用级信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个工作周期指标 — 它测量 GPU 在每个采样间隔是否在执行工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 无论如何都很忙。基于工作周期的扩展是盲目的扩展。

更糟的是，vLLM 和类似引擎会预分配 KV 缓存内存（最多 `--gpu-memory-utilization`）。即使只有一个请求，内存使用率也会保持在 90% 左右。基于内存的 HPA 永远不会缩容。

**2026 年的替代信号**：

- 队列深度（等待预填充的请求数量）。
- KV 缓存利用率（已分配给活动序列的缓存块比例）。
- 每个副本的 P99 TTFT（你的 SLA 信号）。
- 有效吞吐量（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩展副本。对于 LLM 服务，它们完全取代了 HPA。

### 何时使用什么

| 扩展决策         | 工具 |
|----------------|------|
| 添加/移除节点   | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 添加/移除副本   | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型   | Karpenter NodePool |
| 抢占低优先级作业 | KAI Scheduler 队列 |

### 分离式预填充/解码使一切复杂化

如果你运行分离式预填充/解码（阶段 17 · 17），你将有两类具有不同扩展触发器的 Pod：预填充 Pod 根据队列深度扩展，解码 Pod 根据 KV 缓存压力扩展。llm-d 将它们暴露为独立的 `Services`，并具有每个角色的 HPA。不要尝试在它们前面放置一个单一的 HPA。

### 冷启动在这里也很重要

冷启动缓解（阶段 17 · 10）是节点配置时间对用户可见的地方。Karpenter 的 45-60 秒预热，加上一个 20GB 模型加载和引擎初始化，意味着从零开始的请求需要 2-5 分钟。为 SLO 关键路径保持一个热池（`min_workers=1`），或者在应用层使用类似 Modal 的检查点技术。

### 需要记住的数字

- Karpenter 节点配置：约 45-60 秒，而 Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 — 8 中有 7 的陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：有问题；使用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的 GPU 作业。对于推理，使用 `WhenEmpty + consolidateAfter: 1h`。

## 动手实践

`code/main.py` 模拟一个三层自动扩展器在突发性 GPU 工作负载上的表现。比较朴素的 HPA（工作周期）、队列深度 HPA 和 KAI Gang Scheduling 扩展。报告未满足的请求数、空闲 GPU 分钟数和综合分数。

## 部署上线

本课程产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，它将设计一个三层自动扩展计划。

## 练习

1.  运行 `code/main.py`。在突发性工作负载下，朴素的工作周期 HPA 会丢失多少请求，而队列深度 HPA 能捕捉到？差异从何而来？
2.  为一个在 H100 SXM5 上运行 Llama 3.3 70B FP8 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个阻止非 GPU 工作负载部署到这些节点的 Taint。
3.  你的团队报告部署卡在 Pending 状态，原因是“有可用 GPU，但 Pod 不调度”。诊断 — 这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以确认？
4.  选择一个信号来自动扩展分离式预填充 Pod，并选择一个不同的信号用于解码 Pod。说明理由。
5.  计算 `WhenEmptyOrUnderutilized` 合并陷阱在 24x7 生产服务上的成本，该服务平均每天有 60 次请求丢失事件，且 P99 TTFT > 10 秒。

## 关键术语

| 术语             | 人们怎么说       | 实际含义 |
|----------------|----------------|--------|
| Karpenter      | "节点配置器"     | Kubernetes 节点自动扩展器；亚分钟级配置 |
| Cluster Autoscaler | "旧的扩展器"   | Kubernetes 节点自动扩展器的前身；更慢，基于节点组 |
| KAI Scheduler  | "GPU 调度器"     | 用于 Gang Scheduling + 拓扑感知 + 队列的辅助调度器 |
| Gang scheduling | "全有或全无"     | 原子性地调度 N 个 Pod，或全部延迟 |
| Topology awareness | "机架感知"   | 基于 NVLink/InfiniBand/机架位置放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL`    | "GPU 利用率"     | 工作周期指标；**不是** LLM 的扩展信号 |
| Queue depth    | "等待的请求数"   | 预填充绑定扩展的正确 HPA 信号 |
| KV cache utilization | "内存压力" | 解码绑定扩展的正确 HPA 信号 |
| Consolidation  | "Karpenter 合并" | 将节点终止以换成更便宜的实例类型 |
| `WhenEmpty + 1h`    | "安全合并"       | 不会驱逐正在运行的 GPU 作业的策略 |

## 扩展阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩展信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 的特定指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。