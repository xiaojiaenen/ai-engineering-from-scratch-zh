# 多区域 LLM 推理服务与 KV Cache 局部性

> 轮询负载均衡对带缓存的 LLM 推理是主动有害的。未命中缓存节点前缀的请求需支付完整预填充代价——长提示下 P50 约 800 ms，而有缓存命中时仅约 80 ms。2026 年生产模式是感知缓存的路由器（Rust 实现的 vLLM Router、llm-d router），它们消费 KV-cache 事件并按前缀哈希匹配进行路由。最新研究（GORGO）将跨区域网络延迟纳入路由目标函数。商业"跨区域推理"产品（Bedrock 跨区域推理、GKE 多集群网关）将推理视为不透明——它们处理可用性而非 TTFT。摩根大通和梅奥诊所于 2024 年 11 月执行了 us-east-1 故障转移演练，耗时约 22 分钟。容灾现实：32% 的 LLM 容灾失败是因为团队备份了权重但忘记备份 tokenizer 文件或量化配置。

**类型：** 学习
**语言：** Python（标准库，玩具版前缀缓存感知路由器模拟器）
**前置知识：** 第 17 阶段 · 04（vLLM 推理服务），第 17 阶段 · 06（SGLang RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 解释为何轮询负载均衡会破坏缓存推理，并量化 TTFT 惩罚。
- 绘制缓存感知路由器图：输入（KV-cache 事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）。
- 说出 LLM 32% 容灾失败的根本原因（缺少 tokenizer 文件 / 量化配置），并给出三份文件的容灾清单。
- 区分商业跨区域产品（Bedrock CRI、GKE 多集群网关）与 KV 感知路由。

## 问题所在

你的服务部署在 us-east-1、us-west-2 和 eu-west-1。你在前面挂了 ALB 并使用轮询策略。生产环境的前缀缓存命中率降至 8%。TTFT P50 增至三倍。你的 vLLM 日志显示每条请求都在支付完整预填充代价。

轮询对无状态服务是最优的。LLM 推理天生是有状态的——KV 缓存编码了模型看到的一切。盲目路由等于将请求送入错误的缓存。

另外，你的团队有一份容灾计划。你将模型权重备份到跨区域 S3。某个区域发生故障；你尝试故障转移；副本拒绝启动。你忘了 tokenizer.json、量化配置和 RoPE scaling 配置在另一个你未同步的桶中。

多区域 LLM 推理是一个缓存问题、一个路由问题，也是一个容灾规范问题——而不是负载均衡器问题。

## 概念

### 缓存感知路由

请求带着提示词到达。路由器计算前缀哈希（例如前 512 个 token）；它向每个副本询问"你缓存了这个前缀吗？"。副本在分配和驱逐 block 时通过 pub/sub 通道发布 KV-cache 事件。路由器选择有匹配项的副本；若无匹配，则回退到基于 GPU 利用率的平局决胜。

**vLLM Router**（Rust 实现，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护 prefix-hash → 副本索引映射，以 O(1) 查找进行路由。无匹配时回退到队列深度最小策略。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（第 17 阶段 · 06）是副本内部的等价物。跨副本路由严格属于上游层职责。

### 数字

2K token 提示下，Llama 3.3 70B FP8，H100 的 TTFT P50：
- 缓存命中（同副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷预填充）：约 800 ms。

10 倍差距。如果路由器在各副本间实现 60-80% 的前缀缓存命中率，你可用 N 副本容量逼近单副本性能。如果命中率仅 10%，你只得到原始线性扩展。

### 跨区域带来新约束——网络延迟

跨区域 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

若路由器将 us-east-1 的请求路由到 ap-southeast-1 的热点前缀，节省的预填充时间（800 → 80 ms）被 440 ms 往返延迟完全淹没。GORGO（2026 研究）将此明确化——联合最小化 `prefill_time + network_latency`，而非仅优化预填充。通常答案是在 massive multi-MB 前缀（预填充占主导）之外保持路由区域性。

### 商业"跨区域推理"对此无帮助

AWS Bedrock 跨区域推理在容量压力时自动将请求路由到其他区域。它优化的是可用性而非 TTFT，并将推理视为不透明。GKE Multi-Cluster Gateway 亦然——服务级故障转移，无 KV cache 感知。

即使使用这些产品，你仍需要在应用层部署缓存感知路由器。它们处理"us-east-1 起火"的情况。缓存感知路由处理 TTFT 情况。

### 容灾规范——32% 缺文件问题

广泛引用的 2026 统计数据：32% 的 LLM 容灾失败是因为团队备份了权重但忘记：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE scaling、attention masks、chat templates）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA adapter 清单）

修复方案是最小三份文件的容灾清单：

1. HF model repo 下所有文件（权重 + 配置 + tokenizer）。
2. 引擎特定推理配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定）。

另外：每季度执行一次容灾演练。摩根大通 us-east-1 演练在 2024 年 11 月仅因 playbook 反复排练而达到 22 分钟恢复。

### 数据驻留正交独立

欧盟客户 PHI 不得离开欧盟。若你的缓存感知路由器将巴黎发起的请求路由到 us-east-1 以匹配前缀，无论 TTFT 增益多大，你都违反了 GDPR。在优化缓存前先按驻留边界划分路由器。

### 需要记住的数字

- 缓存命中与未命中的 TTFT 差距：约 10 倍（2K 提示下 80 ms vs 800 ms）。
- 跨区域 RTT 美欧：约 75 ms。
- 容灾失败：32% 因缺少 tokenizer/量化配置。
- 摩根大通 us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

```figure
cache-aware-router
```

## 实践

`code/main.py` 在多区域工作负载上模拟三种路由策略（轮询、缓存感知区域、缓存感知全局）。报告缓存命中率、TTFT P50/P99 及跨区域流量费用。

## 交付产出

本课程产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计一份路由方案。

## 练习

1. 运行 `code/main.py`。给定 75 ms RTT，提示词长度达到多少时跨区域路由优于纯本地路由？
2. 你的缓存命中率从 70% 降至 12%。诊断三种可能原因，以及各自对应的观测指标。
3. 为 vLLM 托管的 70B AWQ 量化模型（含 5 个 LoRA adapter）设计一份容灾清单。列出所有文件和配置。
4. 论证 Bedrock 跨区域推理对于具有严格 TTFT SLO 的金融科技企业是否"够用"。引用具体行为。
5. 一条巴黎发起的请求匹配到 us-east-1 的前缀。你是否路由？写出你的策略。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 缓存感知路由 | "智能 LB" | 按前缀哈希匹配路由到有 KV cache 的副本 |
| KV-cache 事件 | "缓存 pub-sub" | 副本发布 block 新增/驱逐；路由器建索引 |
| 前缀哈希 | "缓存键" | 前 N 个 token 的哈希，用作路由器查找键 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| 跨区域推理 | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| 容灾清单 | "备份清单" | 恢复所需的全部文件——不只是权重 |
| 数据驻留 | "GDPR 边界" | 用户数据可见区域的法律约束 |
| RTT | "往返时间" | 网络延迟；美欧 75 ms，美亚 220 ms |
| LLM 感知负载均衡 | "缓存命中 LB" | 缓存感知路由器作为产品类别 |

## 延伸阅读

- [BentoML — 多云与跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) —— 带网络延迟项的跨区域 KV-cache 复用。
- [TianPan — 多区域 LLM 推理缓存局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock 跨区域推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) —— 可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) —— 缓存感知路由器源码。
