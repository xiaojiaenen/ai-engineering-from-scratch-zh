# 自托管推理引擎选择 — 匹配硬件与规模

> 引擎选择是硬件、规模和生态系统的函数，而非排行榜的阅读结果。2026 年，四种引擎主导自托管推理：llama.cpp、Ollama、vLLM、SGLang，而 TGI 已进入维护模式。**llama.cpp** 在 CPU 上最快——模型支持最广，可完全控制量化和线程。**Ollama** 是开发者笔记本的一键安装方案，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在生产级负载下吞吐差距为 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式**——仅进行 bug 修复，原始吞吐比 vLLM 慢约 10%，但历史上拥有顶尖的可观测性和 HF 生态系统集成。该维护状态使其成为高风险的长期选择——新项目更安全的默认值是 SGLang 或 vLLM。**vLLM** 是通用生产默认引擎——v0.15.1（2026 年 2 月）增加了 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是面向智能体多轮对话/前缀密集型场景的专家级引擎——已在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 等生产环境中部署超过 400,000 张 GPU。硬件约束：CPU 优先 → llama.cpp。AMD / 非 NVIDIA → vLLM 是支持最强的路径（TRT-LLM 仅支持 NVIDIA）。2026 年流水线模式：开发 = Ollama， staging = llama.cpp，生产 = vLLM 或 SGLang。各引擎采用不同的权重格式——llama.cpp 系列使用 GGUF，GPU 引擎使用 HF safetensors——因此阶段之间可能需要进行格式转换。

**类型：** 学习
**语言：** Python（stdlib，引擎决策树遍历器）
**前置条件：** 第 17 阶段所有关于引擎的课程（04、06、07、09、18）
**时间：** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用对话 / 智能体 / 长上下文）选择引擎。
- 说出 2026 年 TGI 的维护模式状态（2025 年 12 月 11 日）以及为何新项目倾向于 vLLM 或 SGLang。
- 描述开发/ staging / 生产流水线，包括 GGUF 到 safetensors 的格式转换在各阶段之间的位置。
- 解释为何"CPU 优先"指向 llama.cpp，而"AMD"排除了 TRT-LLM。

## 问题

你的团队开始一个新的自托管 LLM 项目。一位工程师说用 Ollama，另一位说用 vLLM，第三位说"TGI 不也能开箱即用吗？"这三种说法在不同场景下都是对的。但没有一种对所有场景都正确。

在 2026 年，决策树的优先级很重要：硬件第一，规模第二，工作负载第三。而一个特定的 2025 年事件——TGI 于 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 概念

### 五大引擎

| 引擎 | 适用场景 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘 / 依赖最小 / 模型支持最广 | CPU 上最快，完全可控 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐量差距 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广泛的生产默认值；v0.15.1 2026 年 2 月 |
| **SGLang** | 智能体多轮对话、前缀密集型工作负载 | 生产环境部署 400,000+ GPU |

### 硬件优先决策

**CPU 优先** → llama.cpp。Ollama 也可用，但速度更慢。没有其他引擎能在 CPU 上与之竞争。

**AMD GPU** → vLLM 是支持最强的路径（AMD ROCm 支持）。SGLang 也兼容。TRT-LLM 仅限 NVIDIA，因此排除。

**NVIDIA Hopper（H100 / H200）** → vLLM 或 SGLang 或 TRT-LLM。三者均为顶级选择。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐领导者（第 17 阶段 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 封装了此路径。

### 规模次优决策

**1 用户 / 本地开发** → Ollama。一条命令，秒级首 token。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10,000 用户 / 生产环境** → vLLM 生产栈（第 17 阶段 · 18）或 SGLang。

**10,000+ 用户 / 企业级** → vLLM 生产栈 + 解耦部署（第 17 阶段 · 17）+ LMCache（第 17 阶段 · 18）。

### 工作负载第三决策

**通用对话 / 问答** → vLLM 在广泛默认场景中胜出。

**智能体多轮对话（工具、规划、记忆）** → SGLang 的 RadixAttention（第 17 阶段 · 06）占主导地位。

**重度前缀复用的 RAG** → SGLang。

**代码生成** → vLLM 可用；SGLang 在缓存方面略优。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——此后仅进行 bug 修复。历史表现：顶级可观测性、业界最佳的 HF 生态集成（模型卡片、安全工具），原始吞吐量略逊于 vLLM。

对于 2026 年的新项目：默认远离 TGI。现有 TGI 部署可以继续运行，但应最终迁移。SGLang 和 vLLM 是更安全的默认选择。

### 流水线模式

开发（Ollama）→ staging（llama.cpp）→ 生产（vLLM）。各引擎采用不同的权重格式——GGUF 用于 llama.cpp 系列，HF safetensors 用于 GPU 引擎——因此阶段之间可能需要进行格式转换。工程师在笔记本上快速迭代；staging 镜像生产量化；生产是最终的服务目标。

### Ollama 注意事项

Ollama 非常适合开发。但它不适合共享生产环境：Go HTTP 序列化增加了开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。在 Ollama 擅长的场景使用它——单用户、一条命令——然后在共享场景中切换到 vLLM。

### 自托管与托管是独立决策

第 17 阶段 · 01（托管超大规模云）、· 02（推理平台）覆盖托管方案。本课假设你已决定自托管。自托管原因：数据驻留、自定义微调、规模化总拥有成本、托管服务中不可用的领域模型。

### 你需要记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产规模：400,000+ GPU。
- Ollama 与 llama.cpp 的吞吐差距：慢 15-30%；生产负载下差距 3 倍。

```figure
data-parallel
```

## 使用它

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择引擎并解释原因。

## 交付它

本课产出 `outputs/skill-engine-picker.md`。给定约束条件，选择引擎并编写迁移计划。

## 练习

1. 使用你的硬件 / 规模 / 工作负载运行 `code/main.py`。输出是否符合你的直觉？
2. 你的基础设施是 12 张 H100 和 8 张 MI300X AMD。选择什么引擎？为什么 TRT-LLM 不在考虑范围内？
3. 一个团队希望在 2026 年使用 TGI，因为"这是我们熟悉的"。请论证迁移的理由。
4. Ollama 开发到 vLLM 生产：量化、配置和可观测性方面有哪些变化？
5. RAG 产品，P99 前缀长度 8K，租户间高复用率。选择引擎并结合第 17 阶段 · 11 + 18 构建方案。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | "CPU 那个" | 模型支持最广，CPU 上最快 |
| Ollama | "笔记本那个" | 一键安装，开发级吞吐量 |
| TGI | "HF 的推理服务" | 自 2025 年 12 月起处于维护模式 |
| vLLM | "默认选择" | 2026 年广泛生产基线 |
| SGLang | "智能体专用" | 前缀密集型，RadixAttention |
| TRT-LLM | "NVIDIA 锁定" | Blackwell 吞吐领导者，仅 NVIDIA |
| GGUF | "llama.cpp 格式" | 捆绑多种 K-quant 变体 |
| 生产栈 | "vLLM K8s" | 第 17 阶段 · 18 参考部署 |
| 流水线模式 | "开发→staging→生产" | Ollama → llama.cpp → vLLM；每引擎权重格式不同 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
