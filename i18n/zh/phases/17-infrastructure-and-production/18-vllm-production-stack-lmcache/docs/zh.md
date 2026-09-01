# 生产服务栈 — KV 卸载与缓存感知路由

> 生产服务栈将路由器、推理引擎和可观测性组件集成到一个 Kubernetes 部署中，并将 KV 缓存视为可以脱离 GPU 的资源。KV 卸载从 GPU 内存中提取 KV 缓存，并在多个查询和引擎之间复用（CPU DRAM，然后是磁盘/Ceph）。vLLM 的生产栈是参考部署方案；LMCache 是卸载层。vLLM 0.11.0 KV 卸载连接器（2026 年 1 月）通过连接器 API（v0.9.0+）使其支持异步和可插拔方式。卸载路径通常在请求路径上隐藏，但缓存未命中和晋升操作可能会增加端到端延迟。即使在没有共享前缀的情况下，LMCache 也依然有价值——当 GPU 的 KV 槽位用尽时，被抢占的请求可以从 CPU 恢复而非重新进行 prefill。在 4 个 a3-highgpu-4g 实例上的 16x H100（80GB HBM）发布的基准测试表明：当 KV 缓存超过 HBM 容量时，原生 CPU 卸载和 LMCache 都能显著提升吞吐量；而在 KV 占用较小的情况下，所有配置都与基线持平且仅有少量额外开销。

**类型：** 学习
**语言：** Python（stdlib、玩具级 KV 溢出模拟器）
**前置知识：** 阶段 17 · 04（推理引擎内部原理）、阶段 17 · 06（SGLang/RadixAttention）
**预计时间：** 约 60 分钟

## 学习目标

- 绘制 vLLM 生产栈的分层结构：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载连接器 API（v0.9.0+），以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 在何时有益（KV > HBM）vs 何时增加额外开销（KV 足够小到可放入 HBM）。
- 根据部署约束，在原生 vLLM CPU 卸载和 LMCache 连接器之间做出选择。

## 问题所在

你的 vLLM 服务在并发量上升时，GPU 的 HBM 使用率达到 100% 并频繁出现抢占事件。请求被驱逐、重新排队，而你在同一分钟内对相同的 2K token 提示词重复 prefill 四次。GPU 计算能力被浪费在冗余的 prefill 上；有效吞吐量远低于原始吞吐量。

增加 GPU 数量需要线性成本。无法增加 HBM。但 CPU DRAM 很便宜——单插槽即可提供 512 GB 以上容量，其延迟比 HBM 差几个数量级，但对于"临时温暖"的 KV 缓存来说完全可用。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求能够快速恢复，并使跨引擎的重复前缀能够共享缓存，避免每个引擎单独 re-prefill。

## 概念解析

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考的 Kubernetes 部署方案：

- **路由器** —— 缓存感知（阶段 17 · 11）。消费 KV 事件。
- **引擎** —— vLLM 工作节点。每张 GPU 或每组 TP/PP 一个。
- **KV 缓存卸载** —— LMCache 部署或原生连接器。
- **可观测性** —— Prometheus 抓取、Grafana 仪表板、OTel 追踪。
- **控制面** —— 服务发现、配置管理、滚动更新。

以 Helm chart + operator 形式发布。

### KV 卸载连接器 API（v0.9.0+）

vLLM 0.9.0 引入了用于可插拔 KV 缓存后端的连接器 API。你的引擎将 block 卸载到连接器；连接器负责存储它们（RAM、磁盘、对象存储、LMCache）。当请求需要某个 block 时，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）新增了异步卸载路径——卸载可以在后台进行，因此在常见情况下引擎不会在此阻塞。端到端延迟和吞吐量仍然取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 官方说明指出，自定义内核卸载在低命中率下可能降低吞吐量，且异步调度与投机解码存在已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地化。将 KV block 存储在主机 RAM 中。实现快速，无网络跳数。不跨引擎共享。

**LMCache 连接器**：集群级。将 block 存储在共享的 LMCache 服务器上（CPU DRAM + Ceph/S3 分层）。任何引擎都可以访问这些 block。已有 16x H100 基准数据发布。

当单个引擎面临 HBM 压力时，选择原生方案。当多个引擎共享前缀时（RAG 共享系统提示词、多租户共享模板），选择 LMCache。

### 基准测试结果

在 4 个 a3-highgpu-4g 实例上分布的 16x H100（80 GB HBM）测试：

- **低 KV 占用**（短提示词、低并发）：所有配置与基线持平，LMCache 增加约 3-5% 额外开销。
- **中等占用**：LMCache 开始在跨引擎前缀复用方面发挥作用。
- **KV 超出 HBM 容量**：原生 CPU 卸载和 LMCache 都能显著提升吞吐量；LMCache 增益更大，因为支持跨引擎共享。

### LMCache 发挥决定性作用的场景

- 多租户服务，系统提示词在租户间共享。
- RAG 场景，文档块在查询间重复。
- 基于相同基础模型微调的变体（LoRA），可复用基础模型的 KV 缓存以减少冗余计算。
- 高抢占工作负载：从 CPU 恢复比重新 prefill 代价更低。

### 不建议启用的场景

- HBM 压力较小——你付出额外开销却无收益。
- 短上下文（<1K token）——传输时间超过 re-prefill 时间。
- 单租户单提示词工作负载——无可复用的内容。

### 与解耦推理的集成

阶段 17 · 17 的解耦推理与 LMCache 叠加：如果 prefill 池到 decode 池的 KV 传输未被使用，它们会落入 LMCache；后续查询直接从 LMCache 拉取。阶段 17 · 11 的缓存感知路由器可以路由到本地缓存或 LMCache 共享缓存匹配的引擎。

### 你需要记住的关键数字

- vLLM 0.9.0：连接器 API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（不是绝对保证）。
- 16x H100 基准测试：当 KV 占用超过 HBM 时，LMCache 发挥作用。
- HBM 压力较小时：3-5% 额外开销且无收益。

```figure
zero-sharding
```

## 动手实践

`code/main.py` 模拟了一个高抢占工作负载，分别在有和没有 LMCache 的情况下运行。报告避免的 re-prefill 次数、吞吐量增益以及盈亏平衡点处的 HBM 利用率。

## 交付物

本课产出 `outputs/skill-vllm-stack-decider.md`。根据工作负载形状和 vLLM 部署配置，决定使用原生卸载、LMCache 还是不启用任何卸载。

## 练习题

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始产生收益？
2. 一个租户在每个小时内对 200 次查询共享一个 6K token 的系统提示词。计算每个租户预期的 LMCache 节省量。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生方案）。
4. LMCache 将数据存储在 Ceph 机械磁盘上。对于一个 4K token 的 KV（70B FP8，500 MB），读取时间与重新 prefill 相比如何？
5. 论证 vLLM 0.11.0 的异步路径是否真正"免费"——额外开销藏在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Production-stack | "参考部署方案" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 的可插拔 KV 存储接口 |
| Native CPU offload | "引擎本地溢出" | 将 KV 存储在同一个引擎的主机 RAM 中 |
| LMCache | "集群 KV 缓存" | 基于 CPU DRAM + 磁盘的跨引擎 KV 缓存服务器 |
| 0.11.0 async | "非阻塞卸载" | 卸载隐藏在引擎流后面 |
| Preemption | "驱逐以腾出空间" | HBM 满时的 KV 缓存洗牌 |
| Prefix reuse | "相同的系统提示词" | 多个查询共享开头部分；命中缓存 |
| Ceph tier | "磁盘分层" | 缓存层次结构中位于 DRAM 之下的持久存储 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径详细说明。
