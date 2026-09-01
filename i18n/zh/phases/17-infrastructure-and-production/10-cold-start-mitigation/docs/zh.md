# Serverless LLM 冷启动缓解

> 一个 20 GB 的模型镜像，从冷启动到提供服务需要 5-10 分钟（7B 模型）到 20+ 分钟（70B 模型）。在一个真正的 serverless 世界中，这不是预热——这是一次宕机。缓解措施在五个层次上运作：预置节点镜像（AWS 上的 Bottlerocket、双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal checkpoints，重启速度提升 10 倍）、热池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 流水线，延迟降低 10-200 倍），以及实时迁移——只传输输入 token（KB）而非 KV cache（GB）。Modal 公布的冷启动时间为 2-4 秒，作为理论下限；Baseten 默认 5-10 秒，预加热后可达亚秒级。本课教你测量、预算并叠加这五个层次的缓解措施。

**类型：** Learn
**语言：** Python（标准库，玩具级冷启动路径模拟器）
**先修条件：** Phase 17 · 02（推理平台经济学）、Phase 17 · 03（GPU 自动伸缩）
**时间：** 约 60 分钟

## 学习目标

- 列举冷启动缓解的五层结构，并说出每层的一个工具或模式。
- 计算 70B 模型的总冷启动时间：(节点供应) + (权重下载) + (权重加载到 HBM) + (引擎初始化)。
- 解释为什么实时迁移传输的是输入 token（KB）而非 KV cache（GB），以及其代价是什么（重新计算）。
- 说出热池的权衡（为空闲 GPU 付费还是接受冷启动尾部延迟），以及 `min_workers > 0` 成为强制选项的 SLA 阈值。

## 问题所在

你的 serverless LLM 端点过夜缩容到零。早上 8 点流量激增。首个请求需要等待：

1. Karpenter 供应 GPU 节点：45-60 秒。
2. 容器拉取 30 GB 含权重的镜像：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图、KV cache 池、分词器：10-30 秒。

总计：220-510 秒（约 3-8 分钟）后才返回第一个 token。而你的 SLA 是 2 秒。你配置了热池（`min_workers=1`），问题似乎消失——但现在你每天 24 小时都要为一块空闲 GPU 付费。如果你的服务有 5 个产品，每个产品有一个热副本，那就是每月 5 × 24 × 30 = 3,600 GPU 小时，不管有没有用户调用。

冷启动缓解就是如何在保持 serverless 经济性的同时，逼近持续运行的延迟表现。

## 概念解析

### 第一层 — 预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。将数据卷快照以预拉取你的容器镜像；在 `EC2NodeClass` 中引用该快照 ID。新节点启动时权重已存在于本地 NVMe——第 2 步和部分第 3 步消失。原生兼容 Karpenter。典型节省：大型模型每台冷启动节省 2-4 分钟。

GCP 上的等价方案：预烘焙容器层的自定义 VM 镜像。Azure 上：使用相同模式的管理磁盘快照。

### 第二层 — 模型流式加载（Run:ai Model Streamer）

无需等待完整文件加载完毕再响应首个请求——按层将权重流式传输到 GPU 内存，一旦第一个 transformer 块就位即可开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生支持。兼容 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，大型模型的权重加载时间大致减半。

### 第三层 — GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA 图、KV cache 区域）进行快照。后续重启时直接反序列化到 HBM——比重新初始化快 10 倍。这最接近"2 秒启动一个热 GPU"。权衡：快照依赖 GPU 拓扑，如果 Karpenter 将你迁移到不同 SKU，则需要重新快照。

### 第四层 — 热池（min_workers=1）

最简单的缓解措施：保持一个副本始终就绪。代价是每天 24 小时支付一块 GPU 的小时费用。对小模型来说算术很残酷（支付 $0.85-$1.50/小时来避免 30 秒冷启动），对大模型则很友好（支付 $4/小时避免 5 分钟冷启动）。热池变为强制选项的 SLA 阈值：通常是 70B+ 模型上 TTFT P99 < 60 秒。

### 第五层 — 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为层级结构：NVMe（快但容量大）、DRAM（中等且分层）、HBM（极小但即取即用）。权重预加载到 DRAM；按需加载到 HBM。论文报告显示冷启动延迟相比朴素的磁盘到 HBM 方式降低 10-200 倍。生产采用尚处早期，但已有与 vLLM 的集成。

### 第六层 — 实时迁移（补充模式）

当节点不可用（抢占式实例驱逐、节点排空）时，传统做法是冷启动另一个副本并排空请求队列。实时迁移将输入 token（千字节）传输到已加载模型的目的地，并在目的地重新计算 KV cache。重新计算比在网络上传输 GB 级的 KV cache 更便宜。适用于解耦部署。

### 热池的数学

对于 P99 TTFT SLA 为 2 秒的服务，问题不是"热池用还是不用"，而是"需要多少个热副本，哪些路径需要它们"。

- 高价值交互式路径（实时聊天、语音助手）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：接受缩容到零，可容忍 5-10 分钟冷启动。
- 高级 tier：按租户配置 `min_workers`，使用专用容量。

### 先测量再优化

70B 模型在空闲节点上的冷启动解剖（示意数据）：

| 阶段 | 时间 | 缓解措施 |
|------|------|---------|
| 节点供应 | 50 秒 | Bottlerocket + 预置镜像，热池 |
| 镜像拉取 | 180 秒 | 预置数据卷（消除） |
| 权重加载到 HBM | 75 秒 | 模型流式加载（减半）；GPU 快照（消除） |
| 引擎初始化 | 20 秒 | 持久化 CUDA 图缓存 |
| 首次前向传播 | 3 秒 | 固有最低延迟 |
| **总冷启动时间** | **328 秒** | |
| **叠加缓解措施后** | **约 15 秒** | 22 倍降低 |

### 应记住的数字

- Modal 冷启动：2-4 秒（带 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预加热后可达亚秒级。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载速度约 2 倍加速。
- ServerlessLLM 分层加载：10-200 倍延迟降低（论文数据）。

```figure
cold-start-pipeline
```

## 动手实践

`code/main.py` 建模了有无各层缓解措施的冷启动路径。报告总冷启动时间、热池成本，以及热池回本所需的最低请求速率。

## 交付物

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形状，选择要叠加的缓解措施组合。

## 练习

1. 运行 `code/main.py`。计算热副本比通过额外请求丢弃来支付冷启动税更经济的临界请求速率。
2. 你部署了一个 13B 模型，P99 TTFT SLA 为 3 秒。选择达到该目标所需的最少缓解层数（最少层数）。
3. Bottlerocket 预置消除了镜像拉取，但权重仍需从快照加载到 HBM。如果快照支持的 NVMe 读取速度为 7 GB/s，计算 70B 模型的 wall-clock 时间。
4. 你的 serverless 提供商提供 GPU 快照（Modal），但你的团队拒绝，理由是"快照会泄露 PII"。请论证双方——现实风险是什么，缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计分层热池策略：付费用户、试用用户和批处理工作负载各需要多少个热副本？展示计算过程。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Cold start | "长时间停顿" | 从请求到达至新副本返回第一个 token 的时间 |
| Warm pool | "始终运行的最小值" | `min_workers >= 1`，保持至少一个副本就绪 |
| Pre-seeded image | "预烘焙 AMI" | 权重已存在于节点镜像中的节点镜像 |
| Bottlerocket | "AWS 节点 OS" | 支持双卷快照的 AWS 容器优化 OS |
| Model streamer | "流式加载" | 将权重 I/O 与计算设置重叠 |
| GPU snapshot | "快照到 HBM" | 序列化加载后的 GPU 状态；重启时反序列化 |
| Tiered loading | "NVMe + DRAM + HBM" | 存储层级结构；按需加载 |
| Live migration | "迁移 token" | 传输输入（KB），在目的地重新计算 KV |
| `min_workers` | "热副本" | serverless 最小保活数量 |
| Scale-to-zero | "纯 serverless" | 空闲时无成本；接受完整冷启动代价 |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 公布的基准测试和快照架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — 预加热操作手册。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 解耦部署的实时迁移。
