# 边缘推理 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 核心边缘约束是内存带宽，而非算力。移动 DRAM 带宽仅 50-90 GB/s；数据中心 HBM3 可达 2-3 TB/s — 差距 30-50 倍。解码是内存带宽瓶颈的，因此这一差距至关重要。2026 年，边缘推理格局分为四个方向：Apple M4/A18 Neural Engine 峰值 38 TOPS（统一内存，无需 CPU↔NPU 拷贝）；Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS；WebGPU + WebLLM 在 M3 Max 上以 Q4 量化运行 Llama 3.1 8B，约 41 tok/s（约为原生的 70-80%）；17.6k GitHub stars，OpenAI 兼容 API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b，约 40 tok/s；Jetson T4000（JetPack 7.1）性能为 AGX Orin 的两倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill — CES 2026 由 Bosch、ThunderSoft、MediaTek 展示。

**类型:** 学习
**语言:** Python (stdlib, 玩具级带宽瓶颈解码模拟器)
**前置知识:** Phase 17 · 04 (服务引擎内部机制), Phase 17 · 09 (生产量化)
**耗时:** 约 60 分钟

## 学习目标

- 解释为什么移动端 LLM 推理是内存带宽瓶颈，而算力是次要因素。
- 列举四种边缘目标平台（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并将每个平台匹配到相应的使用场景。
- 指出 2026 年 WebGPU 的覆盖缺口（Firefox Android 正在追赶）以及 Safari iOS 26 的落地。
- 根据目标平台选择量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题所在

客户需要一个设备端聊天机器人：语音优先、默认隐私、支持离线。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 以约 55 tok/s 运行 — 尚可。但在 iPhone 16 Pro 上，同一模型仅 3 tok/s — 不理想。在中端 Android（Snapdragon 8 Gen 3）上，7 tok/s。通过 Chrome Android v121+ 的 WebGPU 在浏览器中运行，4-8 tok/s（取决于设备）。

吞吐量的差异不是移植问题。这是带宽差距 × 量化格式 × NPU 是否可从用户态访问的结果。2026 年的边缘推理是四个不同的问题，对应四种不同的解决方案。

## 概念解析

### 带宽才是真正的天花板

解码时，每个 token 都需要读取完整的权重集。一个 Q4 量化的 7B 模型约 3.5 GB。在 50 GB/s 带宽下读取 3.5 GB 需要 70 ms — 理论天花板约 14 tok/s。在 90 GB/s（高端移动 DRAM）下，天花板升至约 25 tok/s。低于这个数值，再多算力也无济于事。

数据中心 HBM3 在 3 TB/s 带宽下，读取同样的 3.5 GB 仅需 1.2 ms — 天花板高达 830 tok/s。同一个模型，同样的权重。不同的内存子系统。

### Apple Neural Engine（M4 / A18）

- 峰值 38 TOPS。统一内存（CPU 和 ANE 共享同一内存池）— 无拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 中的 Metal Performance Shaders (MPS) 访问。
- Llama.cpp 的 Metal 后端使用 MPS，而非直接调用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实用路径：Core ML + INT4 权重 + FP16 激活值。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 峰值 45 TOPS。与 SoC 中的 CPU 和 GPU 集成，但拥有独立的内存域。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- Chat templates、Llama 3.2、Phi-3 均在 AI Hub 上作为一等公民提供。

### Intel / AMD NPUs（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件生态落后于 Apple/Qualcomm；OpenVINO 正在改善但仍属小众。
- 最适合 Windows ARM Copilot 应用；在 AMD/Intel 桌面端原生支持本地优先。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型，无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s — 约为通过相同后端运行的原生的 70-80%。
- WebLLM 有 17.6k GitHub stars；OpenAI 兼容的 JS API；Apache 2.0 许可。
- 2026 年覆盖情况：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。整体移动端覆盖率约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：可流畅运行 Llama 3.2 3B、Phi-3。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b，约 40 tok/s。
- Thor / T4000（JetPack 7.1）：性能为 AGX Orin 的两倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、chunked prefill — 将数据中心优化移植到边缘。

### 按目标平台选择量化格式

| 目标平台 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4 权重 + FP16 激活值 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存瓶颈 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 长上下文的陷阱

Llama 3.1 的 128K 上下文是数据中心功能。在 8 GB RAM 的手机上，4 GB 模型 + 32K token 的 2 GB KV cache + 系统开销 = 内存溢出。边缘部署通常将上下文限制在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首 token < 500 ms）。本地推理完全消除网络延迟。结合语音转文本（Whisper Turbo 变体可在边缘运行），边缘推理成为生产级语音闭环的关键。

### 需要记住的数字

- Apple M4 / A18 ANE: 38 TOPS。
- Qualcomm Hexagon SD X Elite: 45 TOPS。
- WebLLM M3 Max: Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin: 通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- 数据中心-边缘带宽差距: 30-50 倍。
- WebGPU 移动端覆盖率: 约 70-75%（Firefox Android 滞后）。

```figure
edge-bandwidth-pipe
```

## 实践操作

`code/main.py` 从带宽瓶颈的数学模型计算各边缘目标平台的理论解码吞吐量天花板。与观测基准对比，并突出显示瓶颈在于带宽而非算力的情况。

## 交付物

本课产出 `outputs/skill-edge-target-picker.md`。根据平台（iOS/Android/浏览器/Jetson）、模型以及延迟/内存预算，选择量化格式和转换管线。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（~77 GB/s 带宽）上的 Q4 量化 7B 模型，计算解码天花板。与观测值 6-8 tok/s 对比 — 运行时效率如何？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计降级方案 — 通过相同的 OpenAI 兼容 API 在服务器端运行。
3. 你的 iOS 应用需要 4K 上下文流式输出。在 iPhone 16 上，哪种模型/格式组合能让活跃内存保持在 4 GB 以下？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能运行 3B 模型。如果产品需要同时支持两者，如何统一推理栈？
5. 论证"WebLLM 在 2026 年是否已具备生产就绪状态"。引用覆盖率、性能以及 Firefox Android 的缺口。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | M 系列和 A 系列设备上的片上 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "browser GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "browser LLM runtime" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS API |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "edge TensorRT" | 2026 年边缘版的 TensorRT-LLM；支持 EAGLE-3 + NVFP4 |
| Unified memory | "shared pool" | CPU 和 NPU 看到相同的 RAM；无拷贝开销 |
| Bandwidth-bound | "memory limited" | 解码受限于每秒读取权重的字节数 |
| Core ML | "Apple conversion" | Apple 框架用于 ANE 原生模型转换 |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 格局与基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘版发布。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计与基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生模型转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 预转换模型。
