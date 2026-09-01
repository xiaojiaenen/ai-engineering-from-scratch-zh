# Prefix-Cache Serving — RadixAttention 与 KV 复用

> 将 KV 缓存视为一等公民的可复用资源，存储在前缀树（radix tree）中，并据此调整调度策略：不像 vLLM 那样采用先来先服务（FCFS）调度，而是采用感知缓存的调度器，优先服务具有更长共享前缀的请求——这等效于深度优先的前缀树遍历，让热门分支始终驻留在 HBM 中。SGLang 就是围绕这个理念构建的服务引擎。在 Llama 3.1 8B 上使用类 ShareGPT 的 1K prompts，SGLang 达到约 16,200 tok/s，而 vLLM 仅约 12,500，领先约 29%。在前缀密集的 RAG 工作负载上，优势可达 6.4 倍。在语音克隆类工作负载上，缓存命中率超过 86%。2026 年已部署在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 的 40 万+ GPU 上。需要注意的是，6.4 倍这个数字在前后缀排序不一致时会消失——排序是工程师的杠杆。

**类型：** 学习
**语言：** Python（stdlib，玩具前缀树缓存 + 缓存感知调度器）
**前置知识：** 阶段 17 · 04（服务引擎内部原理），阶段 14（智能体 RAG）
**时间：** 约 75 分钟

## 学习目标

- 图解 RadixAttention：前缀如何存储在前缀树中，KV 块如何在同一分支上的序列间共享。
- 解释缓存感知调度以及为什么 FCFS 对前缀密集流量是错误的。
- 根据前缀缓存命中率和 prompt 长度分布，计算给定工作负载的预期加速比。
- 说出使 6.4 倍成为现实 vs 错失优势所需的 prompt 排序规范。

## 问题所在

经典服务将每个请求的 prompt 视为不透明数据。即使 5,000 个 RAG 请求都以相同的 2,000-token 系统 prompt 加上相同的检索前缀开头，vLLM 也会对该 2,000-token 前缀预填充 5,000 次。GPU 反复做相同的计算。

关键观察：在智能体和 RAG 工作负载中，prompt 几乎总是共享很长的前缀。系统 prompt、工具模式、少样本示例、检索头部、对话历史——所有这些都在请求之间重复。如果你只需计算一次该前缀的 KV 缓存并复用，就不需要再次预填充。

RadixAttention 正是这样做的。Token 以前缀树索引；每个节点拥有其路径上 token 序列对应的 KV 块。新请求遍历这棵树：任何 token 匹配的节点都会复用该节点的 KV 块。预填充成本正比于"新增"后缀，而非完整 prompt。

挑战在于调度。如果两个请求共享 2,000-token 前缀，而第三个请求仅共享该前缀中的 200 个 token，你希望一起服务那两个长共享请求，以便长前缀留在 HBM 中。FCFS 则相反——它服务先到达的请求，可能在下一个长前缀请求到来之前就淘汰了热门分支。

## 核心概念

### 前缀树作为 KV 索引

前缀树（紧凑 Trie）存储 token 序列。每个节点拥有一个 token 范围以及为该范围计算的 KV 块。子节点将一个或多个 token 扩展到序列上。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个包含系统 prompt + "Context: <doc A>" + "Question: Carol" 的新请求到来。调度器遍历：系统前缀匹配（复用 124 个块），doc-A 分支匹配（复用 31 个块），然后只为 "Question: Carol" 分配新块（4 个块）。预填充成本：仅 4 个块的新 token。没有这棵树则需要 160 个块。预填充节省约 40 倍。

### 缓存感知调度

如果缓存不断换出，前缀树支持的复用就毫无意义。两个关键策略：

1. **深度优先派发**。从队列中选择下一个请求时，优先选择与当前运行集合在同一分支上的请求。这样可保持热门分支常驻。
2. **分支级 LRU，而非块级**。淘汰整个分支（从使用最短的叶节点开始），而非单个块，使缓存形状与前缀树形状匹配。

FCFS 违反了两者。一个共享 2,000 个 token 的请求排在共享 50 个 token 的请求后面，导致 2,000-token 分支被换出以腾出空间给那个 50-token 请求。

### 你需要记住的基准数字

- Llama 3.1 8B，H100，ShareGPT 1K prompts：SGLang ~16,200 tok/s vs vLLM ~12,500（领先约 29%）。
- 前缀密集的 RAG（相同系统 + 相同文档，不同问题）：SGLang 最高 6.4 倍。
- 语音克隆工作负载：86.4% 前缀缓存命中率。
- SGLang 客户的生产命中率：50-99%，取决于 prompt 规范。
- 2026 年已部署在 40 万+ GPU 上。

### 排序陷阱

6.4 倍这个数字依赖于一致的 prompt 模板排序。如果你的客户端在某些请求中按 `[system, tools, context, history, question]` 构造 prompt，而在其他请求中按 `[system, context, tools, history, question]` 构造，树就无法找到共享前缀。对人类而言看似共享前缀的内容，对前缀树而言却是两个不同的序列。

工程师的杠杆：你的 prompt 模板就是缓存键。固定排序。将所有不可变部分（系统 prompt、工具、schema）放在最前面。将检索上下文放在中间。将用户问题放在最后。不要把动态内容穿插到前缀中。

研究中的一个真实案例：将动态内容移出可缓存前缀，使一个部署的缓存命中率从 7% 跃升至 74%。

### RadixAttention 的胜负场景

优势场景：
- RAG（相同检索前缀，不同问题）。
- Agent（相同工具 schema，不同查询）。
- 带有长系统 prompt 的对话。
- 带有重复 preamble 的语音/视觉工作负载。

劣势场景（退化为 vLLM 级别吞吐）：
- 单次生成且 prompt 唯一（代码补全、无系统 prompt 的开放式聊天）。
- 动态 prompt，每个请求都将唯一内容穿插到前缀中。

### 为什么这是调度问题而不只是内核问题

你可以将 KV 复用实现为内核技巧。SGLang 的洞察在于，只有当调度器保持热门分支常驻时，复用才值得。一个简单的"可用即复用"策略在混合负载下会导致缓存频繁换出。前缀树索引的调度器才是将内核技巧转化为 29% 生产优势的关键。

### 与 vLLM 的关系

两者并非严格的竞争关系。2026 年 vLLM 添加了前缀缓存（`--enable-prefix-caching`）和缓存感知路由（Rust 实现的 vLLM Router）。差距缩小但并未完全消除——SGLang 的整个栈以前缀树为核心；vLLM 则是将其嫁接上去。对于前缀复用占主导的工作负载，SGLang 仍是首选。对于没有强前缀模式的通用服务，vLLM 仍然相当或更好。

```figure
roofline
```

## 实践

`code/main.py` 实现了一个玩具前缀树 KV 缓存，以及带有两种策略的调度器：FCFS 和缓存感知。将同一工作负载分别通过两种策略运行，报告前缀缓存命中率和吞吐差异。然后运行"乱序排序"工作负载以展示 6.4 倍优势的消失。

## 成果

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定工作负载描述（prompt 模板形状、检索模式、并发租户数量），生成 prompt 排序建议和 SGLang 采用的可行性判断。

## 练习

1. 运行 `code/main.py`。在同一工作负载上比较 FCFS 和缓存感知策略。差异来自哪里——预填充节省、解码节省还是队列延迟？
2. 修改工作负载，使 prompt 随机打乱 `[system, tools, context]` 的顺序。重新运行。命中率会怎样？为什么？
3. 计算在 Llama 3.1 8B 上将 2,000-token 系统 prompt 作为一个前缀树分支驻留所需的 HBM 成本。与 16 序列批次不使用前缀复用的成本进行比较。
4. 阅读 SGLang RadixAttention 论文。用三句话解释为什么在前缀密集负载下，树形 LRU 淘汰优于块形 LRU。
5. 某客户报告仅有 8% 的缓存命中率。列出三个可能的原因，以及针对每个原因的诊断方法。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| RadixAttention | "SGLang 的东西" | KV 缓存以前缀树索引，使共享前缀能复用块 |
| 前缀树 (Radix tree) | "紧凑 Trie" | 每个节点拥有一个 token 范围及其 KV 块的树结构 |
| 缓存感知调度器 | "热门分支优先" | 优先服务与常驻分支共享的请求的调度器 |
| 前缀缓存命中率 | "你的 prompt 有多少是免费的" | 从复用 KV 块服务的 prompt token 比例 |
| FCFS | "先来先服务" | 破坏前缀局部性的默认调度策略 |
| 分支级 LRU | "淘汰叶节点" | 与前缀树形状匹配的淘汰策略 |
| Prompt 模板排序 | "缓存键" | prompt 的组件顺序决定了树能共享什么 |
| 系统 prompt 驻留 | "常驻前缀" | 保持不可变系统部分常驻以避免换出抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源码和文档。
- [SGLang 文档](https://sgl-project.github.io/) — RadixAttention 和调度细节。
- [SGLang 论文 — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — 设计参考。
- [LMSYS 博客 — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — 基准数字和调度器原理。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的类前缀树实现，供对比参考。
