# 使用 LMCache KV 卸载的 vLLM 生产栈

> vLLM 的 production-stack 是参考级 Kubernetes 部署方案 —— 将路由器、引擎和可观测性组件整合在一起。LMCache 是一个 KV 卸载层，它将 KV 缓存从 GPU 显存中提取出来，并在查询和引擎之间复用（先使用 CPU 内存，再扩展到磁盘/Ceph）。vLLM 0.11.0 的 KV 卸载连接器（2026年1月）通过 Connector API（v0.9.0+）实现了异步且可插拔的功能。卸载延迟对用户不可见。即使没有共享前缀，LMCache 也很有价值 —— 当 GPU 的 KV 槽位耗尽时，被抢占的请求可以从 CPU 恢复，而不是重新计算预填充。在 16x H100（80GB HBM）跨 4 个 a3-highgpu-4g 发布的基准测试显示：当 KV 缓存超过 HBM 容量时，原生 CPU 卸载和 LMCache 都能显著提升吞吐量；当 KV 占用较低时，所有配置的性能与基线相当，仅增加少量开销。

**类型：** 学习  
**语言：** Python（标准库，简易 KV 溢出模拟器）  
**前置课程：** Phase 17 · 04（vLLM 服务内部原理），Phase 17 · 06（SGLang/RadixAttention）  
**时长：** ~60 分钟

## 学习目标

- 图示 vLLM 生产栈的各层结构：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载连接器 API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-内存何时有益（KV > HBM）以及何时会增加开销（KV 小到足以放入 HBM）。
- 根据部署约束，在原生 vLLM CPU 卸载和 LMCache 连接器之间做出选择。

## 问题所在

您的 vLLM 服务显示 GPU 的 HBM 利用率达到 100%，每当并发量上升时就会发生抢占事件。请求被驱逐、重新排队，并且您在一分钟内对同一个 2K token 的提示进行了四次预填充。GPU 算力被浪费在重复的预填充上；有效吞吐量远低于原始吞吐量。

增加更多 GPU 会带来线性成本。增加 HBM 容量不可行。但 CPU 内存很便宜 —— 一个插槽拥有 512 GB+ 容量，其延迟比 HBM 差了几个数量级，但对于“临时缓存”的 KV 来说足够了。

LMCache 将 KV 缓存提取到 CPU 内存，以便被抢占的请求能够快速恢复，并且跨引擎的重复前缀可以共享缓存，无需每个引擎重新预填充。

## 核心概念

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考级 Kubernetes 部署方案：

- **路由器** —— 缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎** —— vLLM 工作进程。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载** —— LMCache 部署或原生连接器。
- **可观测性** —— Prometheus 抓取，Grafana 仪表盘，OTel 追踪。
- **控制平面** —— 服务发现、配置、滚动更新。

以 Helm chart + operator 形式提供。

### KV 卸载连接器 API (v0.9.0+)

vLLM 0.9.0 引入了用于可插拔 KV 缓存后端的 Connector API。您的引擎将数据块卸载到连接器；连接器存储它们（内存、磁盘、对象存储、LMCache）。当请求需要某个数据块时，连接器将其加载回来。

vLLM 0.11.0（2026年1月）增加了异步卸载路径 —— 卸载可以在后台进行，因此在常见情况下引擎不会被阻塞。端到端延迟和吞吐量仍取决于工作负载特征、KV 缓存命中率和系统压力；vLLM 自己的说明指出，在命中率低时自定义内核卸载可能降低吞吐量，并且异步调度与推测性解码存在已知的交互问题。

### 原生 CPU 卸载与 LMCache 对比

**原生 vLLM CPU 卸载**：引擎本地化。将 KV 块存储在主机内存中。实现快速，无网络跳数。不跨引擎共享。

**LMCache 连接器**：集群规模。将数据块存储在共享的 LMCache 服务器中（CPU 内存 + Ceph/S3 层）。数据块可被任何引擎访问。已有 16x H100 基准测试发布。

当单个引擎面临 HBM 压力时选择原生方案。当多个引擎共享前缀（例如使用共同系统提示的 RAG、共享模板的多租户场景）时选择 LMCache。

### 基准测试表现

16x H100（80 GB HBM）分布在 4 个 a3-highgpu-4g 上的测试：

- 低 KV 占用（短提示，低并发）：所有配置与基线匹配，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用方面发挥作用。
- KV 超过 HBM 容量：原生 CPU 卸载和 LMCache 都显著提升吞吐量；LMCache 收益更大，因为存在跨引擎共享。

### LMCache 决定性优势的场景

- 多租户服务，其中系统提示在租户间共享。
- RAG 场景，文档块在查询间重复出现。
- 基于同一基础模型的微调变体（LoRA），基础模型 KV 复用可减少冗余计算。
- 抢占频繁的工作负载：从 CPU 恢复比重计算预填充成本更低。

### 何时不应启用

- HBM 压力较小 —— 您支付开销却没有收益。
- 短上下文（<1K tokens）—— 传输时间 > 重计算预填充时间。
- 单租户单提示工作负载 —— 没有可捕获的复用机会。

### 与解耦服务的集成

Phase 17 · 17 的解耦服务 + LMCache 组合效果：从预填充池到解码池的 KV 传输如果没有立即使用，会落入 LMCache；后续查询可以从 LMCache 拉取。Phase 17 · 11 的缓存感知路由器可以将请求路由到本地缓存或 LMCache 共享缓存匹配的引擎。

### 需要记住的关键数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026年1月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（非绝对保证）。
- 16x H100 基准测试：当 KV 占用超过 HBM 时，LMCache 有益。
- HBM 压力较小时：增加 3-5% 开销且无收益。

## 使用方法

`code/main.py` 模拟了一个抢占频繁的工作负载，分别在有和没有 LMCache 的情况下运行。报告避免的重预填充次数、吞吐量增益以及 HBM 利用率的盈亏平衡点。

## 部署实施

本课程产出 `outputs/skill-vllm-stack-decider.md`。根据工作负载特征和 vLLM 部署情况，决定采用原生方案、LMCache 还是两者都不需要。

## 练习题

1. 运行 `code/main.py`。HBM 利用率达到多少时，LMCache 开始产生收益？
2. 一个租户在 200 次查询/小时中共享一个 6K token 的系统提示。计算每个租户预期的 LMCache 节省量。
3. LMCache 服务器是单点故障。设计高可用策略（副本、降级到原生方案）。
4. LMCache 将数据存储到使用机械硬盘的 Ceph。对于 70B FP8 模型（500 MB）的 4K token KV 缓存，读取时间与重新预填充时间对比如何？
5. 论述 vLLM 0.11.0 异步路径是否“免费” —— 开销隐藏在哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Production-stack | “参考部署方案” | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | “KV 后端接口” | vLLM 0.9.0+ 的可插拔 KV 存储接口 |
| 原生 CPU 卸载 | “引擎本地溢出” | 将 KV 存储在同一引擎的主机内存中 |
| LMCache | “集群 KV 缓存” | 跨引擎的 KV 缓存服务器，使用 CPU 内存 + 磁盘 |
| 0.11.0 异步 | “非阻塞卸载” | 卸载隐藏在引擎流之后 |
| 抢占 | “驱逐以腾出空间” | HBM 满时进行的 KV 缓存重新分配 |
| 前缀复用 | “相同系统提示” | 多个查询共享开头部分；缓存命中 |
| Ceph 层 | “磁盘层” | 缓存层次结构中位于 DRAM 之下的持久存储 |

## 延伸阅读

- [vLLM 博客 — KV 卸载连接器（2026年1月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM 生产栈 GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [用于企业级 LLM 推理的 LMCache (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现。
- [vLLM 0.11.0 发布说明](https://github.com/vllm-project/vllm/releases) — 异步路径细节。