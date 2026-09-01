# 生产环境量化 — AWQ、GPTQ、GGUF K量化、FP8、MXFP4/NVFP4

> 量化格式并非万能选择 — 它是硬件、推理引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 统治 CPU 和边缘侧，通过 llama.cpp 和 Ollama 交付。在 vLLM 中需要同一基座上的多 LoRA 时，GPTQ 胜出。AWQ 搭配 Marlin-AWQ 内核在 7B 级别模型上可实现 ~741 tok/s，并在 INT4 中获得最佳 Pass@1 — 这是 2026 年数据中心生产环境的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上充当中间地带 — 近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）更为激进，需要逐块验证。两个陷阱会咬伤团队：校准数据集必须匹配部署领域，且 KV cache 与权重量化是分开的 — AWQ 的教训"我的模型现在是 4 GB"忽略了生产批大小下 10-30 GB 的 KV cache。

**类型：** 学习
**语言：** Python（标准库，玩具级内存和吞吐量跨格式对比）
**前置知识：** 阶段 10 · 13（量化基础）、阶段 17 · 04（推理引擎内部原理）
**时间：** 约 75 分钟

## 学习目标

- 说出 2026 年的六种生产量化格式及其适用场景。
- 根据硬件（CPU vs GPU、Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规对话、推理、多 LoRA）选择格式。
- 计算选定格式的权重内存节省量和未受影响的 KV cache。
- 指出因校准数据集不匹配而导致量化模型在领域流量上性能下降的陷阱。

## 问题所在

量化减少内存占用和 HBM 带宽，这正是 decode 阶段所需。一个 FP16 的 70B 模型有 140 GB 权重。将权重量化为 INT4（AWQ 或 GPTQ）后模型变为 35 GB — 可装入单张 H100 并为 KV cache 留出空间，这很重要，因为在 128 并发序列、2k 上下文的场景下，仅 KV cache 就需要 20-30 GB。

但量化并非没有代价。激进的量化会降低质量，尤其在依赖推理的任务上。不同格式适配不同引擎。不同硬件原生支持不同精度。2026 年的格式生态真实存在，你不能照搬别人的选择 — 你必须根据你自己的技术栈来做选择。

## 概念解析

### 六种格式

| 格式 | 位数 | 适用场景 | 引擎 |
|------|------|---------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产环境 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘默认

GGUF 是一种文件格式，而非量化方案本身 — 它将多种 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认 — 在 4-5 位下接近 BF16 质量。对 CPU 或边缘推理来说是最优选择，因为 llama.cpp 是无可争议的最快 CPU 推理引擎。

在 vLLM 中的吞吐量惩罚：7B 模型约 93 tok/s — 该格式并未针对 GPU 内核优化。仅在部署目标是 CPU/边缘时使用 GGUF。其他情况不用。

### GPTQ — vLLM 中的多 LoRA

GPTQ 是一种带校准步骤的训后量化算法。Marlin 内核使其在 GPU 上高速运行（比非 Marlin GPTQ 快 2.6 倍）。7B 模型约 712 tok/s。

独特优势：GPTQ-Int4 支持 vLLM 中的 LoRA 适配器。如果你要同时服务一个基座模型加 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认

激活感知权重量化（Activation-aware Weight Quantization）。在量化过程中保护约 1% 最重要的权重。Marlin-AWQ 内核：比朴素实现快 10.9 倍。7B 模型约 741 tok/s，是 INT4 格式中 Pass@1 最高的。

除非你需要多 LoRA（选 GPTQ）或激进的 Blackwell FP4（选 NVFP4），否则为新 GPU 推理选择 AWQ。

### FP8 — 可靠的中间方案

8 位浮点数。近无损。广泛支持。Hopper Tensor Cores 原生加速 FP8。Blackwell 继承支持。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年的安全默认。内存节省是 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4 — Blackwell 激进方案

微缩放 FP4。每组权重的块都有自己的缩放因子。激进但由 Blackwell Tensor Cores 硬件加速。相比 FP8，每 token 字节数减半 — 这是阶段 17 · 07 中的经济效益所在。

注意事项：
- 暂不支持 LoRA（2026 年初）。
- 在依赖推理的工作负载上质量下降明显。
- 需针对你的评测集逐模型验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集 — 通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网页文本上校准会让算法对哪些权重需要保护做出错误决策。HumanEval 上的 Pass@1 可能下降数个百分点。

修复方法：在领域数据上进行校准。数百条领域样本通常足够。在发布前在评测集上测试。

### KV cache 陷阱

AWQ 将权重量化到 4 位。KV cache 是独立的，保持 FP16/FP8。对于一个用 AWQ 的 70B 模型：

- 权重：~35 GB（INT4，从 140 GB 缩减）。
- 128 并发 × 2k 上下文的 KV cache：~20 GB。
- 激活值：~5 GB。
- 总计：~60 GB — 可装入 H100 80GB。

-naively "我把模型量化到了 4 GB" 忘记了另外 30-50 GB。整体规划 HBM 预算。

另外，KV cache 量化（FP8 KV 或 INT8 KV）是不同的选择，有其自身的权衡 — 它直接影响注意力精度，不是免费收益。

### AWQ INT4 对推理存在风险

思维链、数学、长上下文代码生成 — 这些对激进量化有明显负面反应。AWQ INT4 在 MATH 上损失约 3-5 分。对于依赖推理的工作负载，使用 FP8 或 BF16 ；接受内存代价。

### 2026 选型指南

- CPU/边缘推理：GGUF Q4_K_M。完成。
- GPU 推理，常规对话，无需 LoRA：AWQ。
- GPU 推理，多 LoRA：带 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不确定：在每种候选格式上跑 1,000 样本评测。

```figure
gpu-memory-breakdown
```

## 实践

`code/main.py` 计算六种格式在多种模型尺寸下的内存占用（权重 + KV + 激活值）和相对吞吐量。展示 KV cache 何时占主导、权重压缩何时带来收益、FP8 何时是安全选择。

## 交付物

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型尺寸、工作负载类型和质量容忍度，选择一个格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 128 并发、2k 上下文的 70B 模型，计算每种格式的总 HBM。哪种格式能让你装入单张 H100 80GB？
2. 你有一个 7B 的代码模型。选择一个格式并论证。如果对质量容忍度判断错误，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据未必更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为何 AWQ 在 7B 上达到 741 tok/s 而原始 GPTQ 仅约 712。
5. 何时应该组合使用 AWQ 权重和 FP8 KV cache，而非保持 KV 为 BF16？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant 中等；生产环境 GGUF 默认 |
| GPTQ | "gee pee tee q" | 带校准的训后 INT4；在 vLLM 中支持 LoRA |
| AWQ | "a w q" | 激活感知 INT4；Marlin 内核；INT4 中 Pass@1 最佳 |
| Marlin 内核 | "快 INT4 内核" | Hopper 上 INT4 的自定义 CUDA 内核；10 倍加速 |
| FP8 | "八位浮点" | Hopper/Ada/Blackwell 上的安全精度默认 |
| MXFP4 / NVFP4 | "微缩放四" | Blackwell 4 位浮点，带逐块缩放因子 |
| 校准数据集 | "cal data" | 用于选择量化参数的输入文本；必须匹配领域 |
| KV cache 量化 | "KV INT8" | 与权重分开选择；影响注意力精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 各格式吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选型指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 原始论文。
