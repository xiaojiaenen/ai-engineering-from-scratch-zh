# 面向硬件的推理编译——Blackwell 上的 FP8 与 NVFP4

> 面向硬件的推理编译以可移植性换取吞吐量，而 TensorRT-LLM——专用于 NVIDIA、针对 Blackwell 调优——是该取舍发挥价值的典型范例。在 GB200 NVL72 上配合 Dynamo 编排，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型的代价为 0.012 美元/百万 token，而 H100 + vLLM 为 0.09 美元/百万——差距达 7 倍。该栈由三种浮点精度复合构成：FP8 对 KV 缓存和注意力核函数至关重要，因其具备所需的动态范围；NVFP4（4-bit 微缩放）处理权重与激活值；多 token 预测（MTP）与拆分离 prefill/decode 服务又在此基础上额外提升 2-3 倍吞吐。Day-0 模型支持可直接加载 FP4 权重，无需后训练转换。2026 年工程团队需注意：TRT-LLM 虽为开源，但专用于 NVIDIA——针对 CUDA 和 Blackwell 深度优化——采用它意味着以可移植性换取吞吐量。在做出承诺前，请根据你的模型与硬件组合算清这笔账。

**类型：** 学习
**语言：** Python（标准库，FP8/NVFP4 内存与成本测算玩具脚本）
**前置知识：** 阶段 17 · 04（推理引擎内部原理），阶段 10 · 13（量化）
**用时：** 约 75 分钟

## 学习目标

- 解释即使权重使用 NVFP4，FP8 为何仍是 KV 缓存和注意力机制的关键。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 显存占用，并推导节省来源。
- 列举 TRT-LLM 利用的 Blackwell 专属特性（Day-0 FP4、MTP、拆分离服务、all-to-all 原语）。
- 判断在何种场景下，TRT-LLM 的 NVIDIA 绑定代价相较于 Hopper 上的 vLLM 有 7 倍成本优势时是合理的。

## 问题背景

2026 年推理经济的前沿问题是“每美元能换多少 token”。答案取决于四个叠加决策：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、推理引擎（vLLM vs SGLang vs TRT-LLM）以及编排方式（普通模式 vs 拆分离模式 vs Dynamo）。

在 Hopper 上搭配 vLLM，120B MoE 模型的推理成本约为 0.09 美元/百万 token。在 Blackwell 上搭配 TRT-LLM + Dynamo，同一模型仅需约 0.012 美元——便宜 7 倍。这部分的差距源于硬件（Blackwell 的单机 GPU LLM 吞吐量是 Hopper 的 11-15 倍），另一部分源于技术栈：FP4 权重、MTP 推测草稿、拆分离 prefill/decode 服务，以及用于 MoE 专家间通信的 NVLink 5 all-to-all。

脱离 NVIDIA 技术栈无法复现这一结果。这就是取舍——以可移植性换取经济效益。理解各项栈选择各自贡献了多少差距，是本课的核心目标。

## 核心概念

### 为什么 FP8 仍是 KV 缓存的底线

2026 年的一个常见误区：认为 NVFP4 适用于所有场景。并非如此。KV 缓存需要 FP8（8 位浮点数），因为它存储的注意力键和值跨越了很大的动态范围。将 KV 量化至 FP4 会导致灾难性的精度损失——分布尾部被截断，注意力分数崩溃。FP8 的指数位为 KV 缓存提供了所需的数据范围。

NVFP4（2025-2026）应用于权重和激活值。微缩放技术：每个权重块拥有独立的比例因子，使小块权重能在不同动态范围内运行，而无需承受逐张量缩放的精度损失。对于激活值，FP4 仍能保持稳定，因为单一层内的激活值范围较小。

Blackwell 典型配置如下：

- 权重：NVFP4（4-bit 微缩放）。
- 激活值：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（保障 softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 专属原语

- **Day-0 FP4 权重**：模型提供方直接提供 FP4 权重；TRT-LLM 可直接加载，无需后训练转换。FP4 无需经过 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：理念与 EAGLE（阶段 17 · 05）相同，但已深度集成至 TRT-LLM 编译流程中。
- **拆分离服务**：prefill 与 decode 部署在独立的 GPU 资源池，KV 缓存通过 NVLink 或 InfiniBand 传输。理念与 Dynamo（阶段 17 · 20）一致。
- **All-to-all 通信原语**：NVLink 5 将 MoE 专家通信延迟较 Hopper 降低 3 倍。TRT-LLM 的 MoE 核函数已为此专项调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 硬件加速的比例因子处理。

### 必须牢记的数据

- 通过 TRT-LLM 在 GPT-OSS-120B 上，HGX B200 成本为 0.02 美元/百万 token。
- 通过 Dynamo（编排 TRT-LLM），GB200 NVL72 成本为 0.012 美元/百万 token。
- 同等负载下，H100 + vLLM 约为 0.09 美元/百万 token。
- TRT-LLM 在 2026 年三个月的更新中带来 2.8 倍吞吐提升。
- 单机 GPU LLM 吞吐量：Blackwell 较 Hopper 提升 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在每项提交任务中均占据绝对优势。

### FP4 的精度代价

NVFP4 相当激进。在重度推理负载（思维链、数学、长上下文代码生成）下，FP4 权重的质量衰退肉眼可见。分块校准可缓解但无法彻底消除。上线推理模型的团队通常以 FP8 权重 + FP4 激活值作为折中，或全程在 H200 上使用 FP8。

铁律：在正式采用 NVFP4 权重前，务必在评估集上验证任务质量。

### 为何这是 NVIDIA 绑定决策

TRT-LLM 基于 C++ + CUDA + 闭源核函数。模型需针对特定 GPU SKU 进行编译。不支持 AMD、Intel 或 ARM。若基础设施策略为多供应商路线，TRT-LLM 在服务层完全不可用——你仍可在混合硬件上通过 vLLM 提供服务。若全线押注 NVIDIA，7 倍成本差距足以抵消绑定代价。

### 2026 实战方案

对于年推理账单超 1 亿美元的团队，运行于 Hopper + vLLM 会白白损失 7-10 倍的效率。应将成本主导型负载迁移至 Blackwell + TRT-LLM + Dynamo。保持实验层仍在 H100 + vLLM 上以维持模型迭代速度。在生产环境上线前，对每个经 NVFP4 转换的模型进行质量验证。

### 拆分离的叠加红利

TRT-LLM 的拆分离服务（独立的 prefill 与 decode 资源池）在阶段 17 · 20 中有详尽讲解。在 Blackwell 上，各项加速倍率呈乘积叠加：FP4 权重 × MTP 提速 × 拆分离部署 × 感知缓存的路由。7 倍差值即基于该完整技术栈。

```figure
pipeline-parallel
```

## 实践应用

`code/main.py` 可计算模型在三种技术栈下的 HBM 显存占用、decode 吞吐量（显存带宽受限阶段）及 $/M-token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行脚本即可观察各项变更的叠加效应及其对成本差距的贡献占比。

## 交付成果

本课将生成 `outputs/skill-trtllm-blackwell-advisor.md`。输入负载规模、模型大小与年 token 吞吐量后，它将判断 Blackwell + TRT-LLM 技术栈是否值得承受 NVIDIA 绑定的代价。

## 练习题

1. 运行 `code/main.py`。针对 30% 激活参数的 120B MoE 模型，分别计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 下的显存带宽受限 decode 吞吐量。最大的性能跃升来自哪里？
2. 某客户在 H100 + vLLM 上年投入 200 万美元。在 7 倍经济差距的前提下，他们需要采购多少块 Blackwell GPU 才能在 12 个月内摊销迁移至 TRT-LLM 的成本？
3. NVFP4 权重转换后，MATH 基准准确率下降 3 个百分点。请给出两条恢复路径：一条优先保质量（保留 FP8 权重），一条优先控成本（使用领域数据校准）。
4. 查阅 MLPerf v6.0 推理结果。哪项任务的 Blackwell 对 Hopper 优势最小？原因是什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV 缓存、128k 上下文长度下的 HBM 显存需求。能否单节点运行于 GB200 NVL72？

## 核心术语

| 术语 | 行业通俗说法 | 实际含义 |
|------|----------------|------------------------|
| FP8 | “8位浮点” | 8 位浮点数；因动态范围需求，用于 KV 缓存与注意力机制 |
| NVFP4 | “4位微缩放” | NVIDIA 的 4 位微缩放浮点格式；用于 Blackwell 的权重与激活值 |
| MXFP8 | “MX 8” | 微缩放 FP8 变体；在 Blackwell Tensor Core 上硬件加速 |
| Day-0 FP4 | “直接交付 FP4 权重” | 模型提供方直接发布已量化为 FP4 的权重；无需后训练转换步骤 |
| MTP | “多 token 预测” | TRT-LLM 内置的推测解码草稿生成机制（阶段 17 · 05） |
| Disaggregated serving | “拆分 prefill/decode” | prefill 与 decode 部署于独立 GPU 资源池；KV 缓存通过 NVLink/IB 传输 |
| All-to-all | “MoE 专家通信” | 将 token 路由至专家 GPU 的通信模式；NVLink 5 使延迟降低 3 倍 |
| InferenceX | “SemiAnalysis 推理基准” | 2026 年业界公认的 token 单价基准测试 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) —— 2026 年 4 月 MLPerf 测试结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) —— NVLink 5 all-to-all 通信与 MoE 核函数。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) —— 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) —— TRT-LLM 之上的拆分离编排框架。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) —— 发布 Blackwell 性能数据的基准测试套件。
