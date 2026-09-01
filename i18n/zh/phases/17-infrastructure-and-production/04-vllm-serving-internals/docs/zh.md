# 推理引擎内部原理 — PagedAttention、连续批处理、分块 Prefill

> 现代推理引擎的吞吐量建立在三个叠加的默认机制之上，而非单一技巧。PagedAttention 始终开启；连续批处理在解码迭代之间将新请求注入活跃批次；分块 Prefill 将长提示切片，使解码 token 永不被饿死。三者同时开启，Llama 3.3 70B FP8 在单张 H100 SXM5 上可达成 2,200-2,400 tok/s（128 并发）——较 vLLM 自身默认值高出约 25%，比朴素 PyTorch 循环快 3-4 倍。本课以可画成示意图的粒度，解读 vLLM（三项技术的参考实现）中的调度器与注意力内核，并以 `code/main.py` 中的玩具连续批处理器收尾，该批处理器以 vLLM 的方式调度 prefill 与 decode。

**类型：** Learn
**语言：** Python（stdlib，玩具连续批处理调度器）
**前置知识：** Phase 17 · 01 (Model Serving)、Phase 11 (LLM Engineering)
**用时：** 约 75 分钟

## 学习目标

- 将 PagedAttention 解释为 KV cache 分配器：说明 block、block table 的含义，并解释为何在生产负载下碎片率可控制在 4% 以下。
- 从迭代层面绘制连续批处理的时序图：已结束序列如何退出批次、新序列如何在不清空批次的前提下加入。
- 用一句话描述分块 Prefill，并指出它保护的是哪项延迟指标（提示：是 TTFT 尾端，而非平均吞吐）。
- 说出 2026 年 vLLM v0.18.0 中把三项优化全开时会踩的一个坑。

## 问题背景

朴素 PyTorch 推理循环一次只处理一个请求：分词 → prefill → 直到 EOS 的 decode → 返回。面对单个用户时还能工作，面对上百个用户就变成一群耐心排队的请求。显而易见的修复是静态批处理 —— 把窗口内每个请求都补齐到最长提示，把每次 decode 都补齐到预期最长输出，让整个批次卡在慢序列上。你为从未用到的 padding 付费，快速请求为慢请求等待。

vLLM 一次性解决了三个问题。PagedAttention 让 KV cache 碎片不再像经典连续分配那样吃掉 60-80% 的 GPU 显存；连续批处理允许请求在每个 decode 迭代之间加入或离开批次，使批次始终满载“真正的工作”；分块 prefill 将 32k token 的提示拆成 ~512 token 的切片，与 decode 交替进行，避免长提示冻住 GPU 上所有解码 token。

2026 年的生产默认值是三项全开。你需要理解每项的作用，因为故障模式都出现在调度器里，而非模型本身。

## 概念解析

### PagedAttention 作为虚拟内存系统

一个序列的 KV cache 大小为 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。以 Llama 3.3 70B 在 8192 token、BF16 精度为例，每条序列约 1.25 GB。若为每个请求预分配 8192 个槽位，但平均请求只用 1500 token，就会浪费大约 82% 已保留的 HBM。经典批处理就要承受这笔浪费。

PagedAttention 借鉴了操作系统虚拟内存的思路。KV cache 不再按序列连续存放，而是以固定大小的 block（默认 16 token）分配。每条序列拥有一张 block table，将其逻辑 token 位置映射到物理 block ID。当序列增长超过已分配的 block 时，再追加一个 block；序列结束时其 block 归还到池子。

碎片率从 60-80%（经典方案）降至不足 4%（PagedAttention）。你不需要用 flag 来开启 PagedAttention —— 它是 vLLM 唯一的分配器。可调参数是 `--gpu-memory-utilization`（默认 0.9），用于在加载权重与激活后预留多少 HBM 给 KV block。

### 迭代级的连续批处理

旧式“动态批处理”会等待一段时间（例如 10 ms）凑满一个批次，然后运行 prefill + decode + decode + decode，直到所有序列结束。快序列提前结束却闲置着，GPU 仍在为慢序列跑完。

连续批处理在每个 decode 步骤之间运作。记当前正在运行的序列集合为 `RUNNING`。每次迭代：

1. 任何刚刚到达 EOS 或 max_tokens 的 `RUNNING` 序列会被移除。
2. 调度器查看等待队列；若有空闲 KV block，则接纳新序列（prefill 或恢复）。
3. 前向传播在当前 `RUNNING` 集合上执行，为每条序列输出一个新 token。

批次大小从不补齐到固定值。处于输出不同位置的序列共享一次融合前向。2026 年的 vLLM 称此机制为 `V1 scheduler`。关键不变式：调度器在每个 decode 迭代运行一次，而非每个请求一次。

### 分块 Prefill 保护 TTFT 尾端

Prefill 是计算密集型任务。在单张 H100 上，Llama 3.3 70B 对 32k token 提示进行纯 prefill 约需 800 ms。prefill 运行时，批次内其他序列的解码 token 全部等待。在一次推理循环中，一条长提示的首 token 延迟（TTFT）会成为数十个其他用户的 token 间延迟（ITL）尖峰。

分块 prefill 将 prefill 拆成固定大小的 chunk（默认 512 token）并按单元调度。chunk 之间调度器可以让 decode 序列推进一个 token。你以少量绝对 prefill 延迟损失（每个 chunk 几毫秒）换取更低的解码抖动。在混合负载的基准测试中，P99 ITL 从约 50 ms 降至约 15 ms。

### 三项机制相互依赖

三项特性互相假设对方的存在。PagedAttention 为调度器提供细粒度的 KV 资源以进行交换；连续批处理依赖这种细粒度，使接纳新序列不会引发全局重排；分块 prefill 是调度器在同一份 `RUNNING` 列表上做出的决策 —— 它只是另一条调度策略，并非独立系统。

你不必记住所有 flag。你需要理解调度器在优化什么：在 KV block 预算与分块 prefill 切片的约束下，最大化 goodput。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，`--enable-chunked-prefill` 与 draft-model 投机解码（`--speculative-model`）不可组合。文档化的例外是 V1 调度器中的 N-gram GPU 投机解码。不看发布说明就全部打开 flag 的团队会在启动时收到运行时错误，而非性能软性退化。如果你的投机收益值得为此开启分块 prefill，请重新审视该选择 —— 2026 年的正确答案往往是 EAGLE-3 不加分块 prefill，而不是能配合分块 prefill 工作的 draft model 方案。

### 应记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三项全开：2,200-2,400 tok/s。
- 同模型，默认 vLLM（无分块 prefill）：约 1,800 tok/s。
- 同模型，朴素 PyTorch 前向循环：约 600 tok/s。
- PagedAttention 在生产负载下的 KV 碎片浪费：<4%。
- 混合负载下 P99 ITL：启用分块 prefill 约 15 ms，未启用约 50 ms。

### 调度器的大致结构

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 一次批次里同时调度 prefill chunk 与 decode
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # 如 512 token
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # 一次融合 GPU 调用
```

`code/main.py` 就是用 stdlib Python 写成的等价循环，含虚假 token 计数与虚假前向延迟。运行它能直观看到分块 prefill 如何在长 prefill 期间让 decode 序列保持存活。

```figure
tensor-parallel
```

## 动手实践

`code/main.py` 模拟了一个可切换功能的类 vLLM 调度器。运行它以观察：

- `NAIVE` 模式：一次一请求，无批处理。
- `STATIC` 模式：补齐并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级的准入与释放。
- `CONTINUOUS + CHUNKED` 模式：prefill 切片与 decode 交错。

输出包含总吞吐量（虚拟秒/ token）、TTFT 均值与 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 行应明显领先。

## 产出交付

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定一份推理配置（批大小、KV 内存利用率、分块 prefill 大小、投机配置），它输出一份调度器诊断，明确指出三项默认机制中哪一项构成瓶颈以及应调整哪些参数。

## 练习

1. 运行 `code/main.py`。在混合短长请求负载下对比 `STATIC` 与 `CONTINUOUS`。吞吐差距来自 prefill 效率、decode 效率还是尾延迟？
2. 修改玩具调度器加入 `--max-num-batched-tokens`。针对跑 Llama 3.3 70B FP8 的 H100，合适的取值是多少？（提示：它是 KV block 大小与空闲 block 数量的函数，而非裸 HBM。）
3. 重读 vLLM v0.18.0 发布说明。哪些 flag 组合互斥？列出它们。
4. 针对均值为 1,500、标准差为 600 的 1,000 条请求轨迹，分别计算：(a) 在最大 8192 的按请求连续分配下的 KV cache 碎片浪费；(b) 使用 16 token block 的 PagedAttention 下的浪费。
5. 用一段话说明为何分块 prefill 能帮助 P99 ITL 却不在孤立情况下提升吞吐。实际中的吞吐增益来自何处？

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | “KV 小 tricks” | KV cache 的固定大小 block 分配器；碎片率 <4% |
| Block table | “页表” | 每条序列从逻辑 token 位置到物理 KV block 的映射 |
| Continuous batching | “正确版的动态批处理” | 每个 decode 迭代都做准入/释放决策 |
| Chunked prefill | “prefill 拆分” | 把长 prefill 拆成 512 token 的切片并与 decode 交错 |
| TTFT | “首 token 时间” | prefill + 排队 + 网络；长提示下被 prefill 主导 |
| ITL | “token 间延迟” | 连续 decode token 之间的时间；被批次大小主导 |
| Goodput | “符合 SLO 的吞吐” | 满足每条请求 TTFT 与 ITL 目标的 tokens/sec |
| V1 scheduler | “新调度器” | vLLM 2026 调度器；N-gram 投机解码是与分块 prefill 兼容的路径 |
| `--gpu-memory-utilization` | “内存旋钮” | 加载权重与激活后，预留用于 KV block 的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) —— 关于分块 prefill 与投机解码兼容性的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) —— 2026 年发布节奏与版本特有行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) —— 仍定义我们如何思考该分配器的原始写作。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) —— 碎片分析与调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) —— 带火焰图的详细 V1 调度器走读。
