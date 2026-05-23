# 生产量化技术 — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> 量化格式并非通用选择 — 它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘设备，通过 llama.cpp 和 Ollama 交付。当您需要在同一基础上使用多个 LoRA 时，GPTQ 在 vLLM 中胜出。配备 Marlin-AWQ 内核的 AWQ 在 7B 级模型上可实现约 741 tok/s，并在 INT4 精度下达到最佳 Pass@1 —— 这是 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 架构上仍是中间选择 —— 近乎无损且得到广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）更为激进，需要按块验证。团队常会陷入两个陷阱：校准数据集必须与部署领域匹配，且 KV 缓存与权重量化是分开的 —— AWQ 的经验教训“我的模型现在只有 4 GB”忽略了在生产批处理大小下 10-30 GB 的 KV 缓存。

**类型：** 学习
**语言：** Python (标准库，用于不同格式的简单内存和吞吐量比较)
**先决条件：** 阶段 10 · 13（量化基础），阶段 17 · 04（vLLM 服务内部机制）
**时间：** 约 75 分钟

## 学习目标

- 列举 2026 年的六种生产量化格式及其最佳应用场景。
- 根据硬件（CPU vs GPU，Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多 LoRA）选择格式。
- 计算所选格式节省的权重内存以及未受影响的 KV 缓存大小。
- 指出导致量化模型在领域流量上性能下降的校准数据集陷阱。

## 问题所在

量化减少了内存和 HBM 带宽，这正是解码所需要的。一个 FP16 的 70B 模型占用 140 GB 权重。将权重量化为 INT4（AWQ 或 GPTQ）后，模型占用 35 GB —— 可以放入一块 H100 并有空间容纳 KV 缓存，这一点很重要，因为在 128 个并发序列、2k 上下文的场景下，仅 KV 缓存就需要 20-30 GB。

但量化并非没有代价。激进的量化会降低质量，尤其是在推理密集型任务上。不同的格式与不同的引擎配合。不同的硬件原生支持不同的精度。2026 年的格式生态是真实存在的，你不能直接照搬别人的选择 —— 你必须根据自己的技术栈进行选择。

## 核心概念

### 六种格式

| 格式 | 位数 | 最佳应用场景 | 支持引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本电脑 | llama.cpp, Ollama |
| GPTQ | 4-8 | 在 vLLM 上支持多 LoRA | vLLM, TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |

### GGUF — CPU/边缘默认选择

GGUF 是一种文件格式，本身并非量化方案 —— 它将 K-quant 变体（Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认值 —— 在 4-5 位精度下接近 BF16 质量。由于 llama.cpp 是目前最快的 CPU 推理引擎，它是 CPU 或边缘服务的最佳选择。

在 vLLM 中的吞吐量惩罚：7B 模型上约 93 tok/s —— 该格式未针对 GPU 内核优化。当部署目标是 CPU/边缘时使用 GGUF。否则不要使用。

### GPTQ — vLLM 中的多 LoRA 方案

GPTQ 是一种带校准过程的训练后量化算法。Marlin 内核使其在 GPU 上运行很快（相比非 Marlin GPTQ 提速 2.6 倍）。7B 模型上约 712 tok/s。

独特优势：GPTQ-Int4 支持 vLLM 中的 LoRA 适配器。如果你正在服务一个基础模型加上 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的选择。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认选择

激活感知权重量化。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ 内核：相比朴素实现提速 10.9 倍。7B 模型上约 741 tok/s，在 INT4 格式中 Pass@1 最佳。

除非需要多 LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4），否则在 GPU 服务方面首选 AWQ。

### FP8 — 可靠的中间选择

8 位浮点。近乎无损。支持广泛。Hopper Tensor Core 原生加速 FP8。Blackwell 继承此特性。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年的安全默认选择。内存节省量是 INT4 的一半，但质量风险要低得多。

### MXFP4 / NVFP4 — Blackwell 的激进方案

微缩放 FP4。每一组权重都有自己的缩放因子。很激进，但在 Blackwell Tensor Core 上有硬件加速。相比 FP8，每 token 的字节数减半 —— 这是阶段 17 · 07 中提到的经济性优势。

注意事项：
- 截至 2026 年初，尚不支持 LoRA。
- 在推理密集型工作负载上可见质量下降。
- 使用你自己的评估集对每个模型进行验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集 —— 通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网络文本上校准会让算法在保护哪些权重方面做出错误决策。HumanEval 上的 Pass@1 可能下降好几个百分点。

解决方法：在领域内数据上校准。通常几百个领域样本就足够了。在发布前在评估集上进行测试。

### KV 缓存陷阱

AWQ 将权重缩减到 4 位。KV 缓存是分开的，保持在 FP16/FP8。对于使用 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB 压缩到 INT4）。
- 128 并发 × 2k 上下文下的 KV 缓存：约 20 GB。
- 激活值：约 5 GB。
- 总计：约 60 GB —— 可放入 H100 80GB。

天真地说“我把模型量化到了 4 GB”忽略了另外 30-50 GB。要从整体上规划 HBM 预算。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是另一个选择，有其自身的权衡 —— 它直接影响注意力准确性，并非免费的收益。

### AWQ INT4 对推理任务有害

长上下文下的思维链、数学推理、代码生成 —— 这些任务在激进量化下会明显受损。AWQ INT4 在 MATH 基准上损失约 3-5 个百分点。对于推理密集型工作负载，使用 FP8 或 BF16；接受内存成本。

### 2026 年选择指南

- CPU/边缘服务：GGUF Q4_K_M。搞定。
- GPU 服务，常规聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：带 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不确定：在每个候选格式上运行一个 1000 样本的评估。

## 动手实践

`code/main.py` 计算一系列模型大小在六种格式下的内存占用（权重 + KV + 激活）和相对吞吐量。展示 KV 缓存占主导地位、权重量化带来收益以及 FP8 是安全选择的场景。

## 实践部署

本课程输出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，它会选择一种格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于一个在 128 并发、2k 上下文下的 70B 模型，计算每种格式所需的总 HBM。哪种格式能让你放入一块 H100 80GB？
2. 你有一个 7B 代码生成模型。选择一种格式并说明理由。如果你对质量容忍度的判断错误，恢复路径是什么？
3. 计算用于校准一个医疗领域模型的 AWQ 校准数据集所需大小。为什么数据并非总是越多越好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为什么 AWQ 在 7B 模型上能达到 741 tok/s，而原始 GPTQ 只能达到约 712。
5. 在什么情况下，将 AWQ 权重与 FP8 KV 缓存结合使用，比保持 KV 在 BF16 更合理？

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| GGUF | “llama.cpp 格式” | 打包 K-quant 变体的文件格式；CPU/边缘默认选择 |
| Q4_K_M | “Q4 K M” | 4 位中等质量 K-quant；GGUF 生产默认值 |
| GPTQ | “gee pee tee q” | 带校准的训练后 INT4 量化；在 vLLM 中支持 LoRA |
| AWQ | “a w q” | 激活感知 INT4；使用 Marlin 内核；INT4 精度下 Pass@1 最佳 |
| Marlin 内核 | “快速 INT4 内核” | 针对 Hopper 的 INT4 定制 CUDA 内核；提速 10 倍 |
| FP8 | “八位浮点” | Hopper/Ada/Blackwell 上的安全精度默认选择 |
| MXFP4 / NVFP4 | “微缩放四位” | Blackwell 的带每块缩放因子的 4 位 FP |
| 校准数据集 | “校准数据” | 用于选择量化参数的输入文本；必须与领域匹配 |
| KV 缓存量化 | “KV INT8” | 与权重分开的选择；影响注意力准确性 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式的吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选择指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和参数。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原始 AWQ 公式。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原始 GPTQ 公式。