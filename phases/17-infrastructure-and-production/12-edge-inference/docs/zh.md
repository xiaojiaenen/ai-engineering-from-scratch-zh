# 边缘推理 —— Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 核心边缘约束是内存带宽，而非算力。移动设备的DRAM带宽为50-90 GB/s；数据中心HBM3可达2-3 TB/s —— 30-50倍的差距。解码是内存瓶颈型任务，因此这个差距是决定性的。到2026年，格局将分为四条路径。Apple M4/A18 Neural Engine峰值达38 TOPS，配备统一内存（无CPU↔NPU拷贝开销）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon可达45 TOPS。WebGPU + WebLLM在M3 Max上以约41 tok/s的速度运行Llama 3.1 8B (Q4)（约为原生性能的70-80%）；GitHub星标17.6k，提供OpenAI兼容API，移动设备覆盖率约70-75%。NVIDIA Jetson Orin Nano Super (8GB)可运行Llama 3.2 3B / Phi-3；AGX Orin通过vLLM运行gpt-oss-20b，速度约40 tok/s；Jetson T4000 (JetPack 7.1)性能是AGX Orin的2倍。TensorRT Edge-LLM支持EAGLE-3、NVFP4、分块预填充 —— 已由Bosch、ThunderSoft、MediaTek在CES 2026展示。

**类型：** 学习
**语言：** Python（标准库，简易带宽瓶颈解码模拟器）
**前提条件：** 阶段17 · 04 (vLLM服务内部原理)，阶段17 · 09 (生产环境量化)
**时间：** 约60分钟

## 学习目标

- 解释为什么移动设备LLM推理受内存带宽限制，而算力是次要因素。
- 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并将每个与使用场景匹配。
- 指出2026年WebGPU的覆盖缺口（Firefox Android正在追赶）以及Safari iOS 26的落地。
- 根据每个目标选择量化格式（ANE用Core ML INT4 + FP16，Hexagon用QNN INT8/INT4，浏览器用WebGPU Q4，Jetson Thor用NVFP4）。

## 问题

一位客户想要一个设备端聊天机器人：语音优先、默认隐私、可离线工作。在MacBook Pro M3 Max上，Llama 3.1 8B Q4以约55 tok/s的速度运行 —— 很好。在iPhone 16 Pro上，同一模型以3 tok/s的速度运行 —— 不好。在搭载Snapdragon 8 Gen 3的中端Android设备上，7 tok/s。在Android Chrome v121+通过WebGPU的浏览器中，根据设备不同为4-8 tok/s。

吞吐量的差异不是移植问题。它是带宽差距乘以量化格式乘以用户空间是否可访问NPU。2026年的边缘推理是四个不同的问题，有四种不同的解决方案。

## 核心概念

### 带宽才是真正的天花板

解码每个token都要读取完整的权重集。一个7B模型的Q4版本大小为3.5 GB。以50 GB/s的速度读取3.5 GB需要70毫秒 —— 理论上限约为14 tok/s。在90 GB/s（高端移动DRAM）下，上限提升至约25 tok/s。在这个数字之下，再多的算力也无济于事。

数据中心HBM3以3 TB/s的速度读取相同的3.5 GB仅需1.2毫秒 —— 上限为830 tok/s。相同的模型，相同的权重。不同的内存子系统。

### Apple Neural Engine (M4 / A18)

- 峰值达38 TOPS。统一内存（CPU和ANE共享同一内存池）—— 无拷贝开销。
- 通过Core ML + `.mlmodel` 编译后的模型访问，或通过Metal Performance Shaders (MPS)在PyTorch中访问。
- Llama.cpp的Metal后端使用MPS，而非直接访问ANE；原生ANE需要Core ML转换。
- 2026年iOS应用的最佳实践路径：Core ML + INT4权重 + FP16激活。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 峰值达45 TOPS。与SoC中的CPU和GPU集成，但内存域独立。
- QNN (Qualcomm Neural Network) SDK和AI Hub提供从PyTorch/ONNX的转换。
- Chat模板、Llama 3.2、Phi-3均在AI Hub上作为一流制品提供。

### Intel / AMD NPUs (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS。软件支持落后于Apple/Qualcomm；OpenVINO在改进但应用较窄。
- 最适合Windows ARM协处理应用；在AMD/Intel桌面端原生支持，适用于本地优先场景。

### WebGPU + WebLLM

- 通过WebGPU计算着色器在浏览器中运行模型；无需安装。
- 在M3 Max上，Llama 3.1 8B Q4以约41 tok/s的速度运行 —— 通过相同后端约为原生性能的70-80%。
- WebLLM在GitHub上获17.6k星标；提供OpenAI兼容的JS API；Apache 2.0许可。
- 2026年覆盖情况：Chrome Android v121+，Safari iOS 26正式版，Firefox Android仍在追赶。总体移动覆盖率约70-75%。

### NVIDIA Jetson家族

- Orin Nano Super (8GB)：可良好运行Llama 3.2 3B、Phi-3。
- AGX Orin：通过vLLM运行gpt-oss-20b，速度约40 tok/s。
- Thor / T4000 (JetPack 7.1)：性能是AGX Orin的2倍，支持EAGLE-3和NVFP4。
- TensorRT Edge-LLM (2026) 支持EAGLE-3推测性解码、NVFP4权重、分块预填充 —— 将数据中心优化移植到边缘端。

### 每个目标的量化选择

| 目标 | 格式 | 备注 |
|------|------|------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译后的 `.wasm`；不支持GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存瓶颈 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘端的长上下文陷阱

Llama 3.1的128K上下文是数据中心特性。在具有8GB RAM的手机上，4GB模型 + 用于32K tokens的2GB KV缓存 + 系统开销 = 内存溢出。边缘部署将上下文保持在4K-8K，除非接受激进的KV量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首token < 500毫秒）。本地推理完全消除了网络延迟。结合语音转文本（Whisper Turbo变体可在边缘运行），边缘推理就能成为生产级的语音闭环。

### 应记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：在Llama 3.1 8B Q4上约41 tok/s。
- AGX Orin：通过vLLM运行gpt-oss-20b约40 tok/s。
- 数据中心-边缘带宽差距：30-50倍。
- WebGPU移动覆盖率：约70-75%（Firefox Android滞后）。

## 实践应用

`code/main.py` 根据带宽瓶颈数学计算各边缘目标的理论解码吞吐量上限。与实测基准进行比较，并突出显示瓶颈在于带宽而非算力的场景。

## 交付产出

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选择量化格式和转换流程。

## 练习

1. 运行 `code/main.py`。对于一个在Snapdragon 8 Gen 3（带宽约77 GB/s）上运行的7B Q4模型，计算解码上限。与观测到的6-8 tok/s比较 —— 运行时是否高效？
2. Android上的WebGPU需要Chrome v121+。为旧浏览器设计一个回退方案 —— 通过相同的OpenAI兼容API进行服务器端处理。
3. 你的iOS应用需要4K上下文流式传输。哪种模型/格式组合能在iPhone 16上将活动内存控制在4GB以内？
4. Jetson AGX Orin以40 tok/s运行gpt-oss-20b。Jetson Nano仅能运行3B模型。如果你的产品同时面向这两者，如何统一推理栈？
5. 论证"WebLLM在2026年是否已具备生产就绪性"。引用覆盖率、性能表现以及Firefox Android的差距。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|----------|------------|
| ANE | "苹果神经引擎" | M系列和A系列中的设备端NPU；统一内存 |
| Hexagon | "高通NPU" | 骁龙NPU；通过QNN SDK访问 |
| WebGPU | "浏览器GPU" | W3C标准化的浏览器GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器LLM运行时" | MLC-LLM项目；Apache 2.0许可；OpenAI兼容JS |
| Jetson | "英伟达边缘" | Orin Nano / AGX / Thor / T4000系列 |
| TRT Edge-LLM | "边缘TensorRT" | 2026年TensorRT-LLM的边缘移植版；EAGLE-3 + NVFP4 |
| 统一内存 | "共享内存池" | CPU和NPU访问相同RAM；无拷贝开销 |
| 带宽瓶颈型 | "内存受限" | 解码受限于读取权重的字节/秒速率 |
| Core ML | "苹果转换" | 苹果用于ANE原生模型的框架 |
| QNN | "高通技术栈" | 高通神经网络SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) —— 现状与基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) —— Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) —— 2026年边缘移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) —— 设计与基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) —— ANE原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) —— 用于Hexagon的预转换模型。