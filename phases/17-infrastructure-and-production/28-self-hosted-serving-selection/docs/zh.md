# 自托管推理引擎选择指南 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 在2026年，四大引擎主导了自托管推理领域。请根据硬件、规模和生态系统进行选择。**llama.cpp** 在CPU上速度最快 — 支持最广泛的模型，完全控制量化和线程配置。**Ollama** 是开发笔记本上一键安装的方案，比llama.cpp慢约15-30%（Go + CGo + HTTP序列化开销），在类生产负载下吞吐量差距达3倍。**TGI于2025年12月11日进入维护模式** — 仅进行错误修复，原始吞吐量比vLLM慢约10%，但历史上具有出色的可观测性和HF生态系统集成。这种维护状态使其成为高风险的长期选择 — SGLang或vLLM对新项目来说是更安全的默认选项。**vLLM** 是通用的生产默认引擎 — v0.15.1（2026年2月）增加了对PyTorch 2.10、RTX Blackwell SM120和H200的优化。**SGLang** 是代理式多轮/前缀密集型场景的专家 — 已在生产环境中部署超过40万块GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件限制：仅CPU → 只能用llama.cpp。AMD / 非NVIDIA → 只能用vLLM（TRT-LLM锁定NVIDIA）。2026年流水线模式：开发 = Ollama，预发布 = llama.cpp，生产 = vLLM或SGLang。全程使用相同的GGUF/HF权重。

**类型：** 学习
**语言：** Python（标准库，引擎决策树遍历）
**先决条件：** 第17阶段所有涵盖引擎的课程（04、06、07、09、18）
**时间：** ~45分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1用户 / 100 / 10,000）和工作负载（通用聊天 / 代理 / 长上下文）选择引擎。
- 阐明2026年TGI的维护模式状态（2025年12月11日起）及其对新项目偏向vLLM或SGLang的原因。
- 描述开发/预发布/生产流水线模式，全程使用相同的GGUF或HF权重。
- 解释为什么“仅CPU”强制使用llama.cpp，以及“AMD”排除了TRT-LLM。

## 问题背景

你的团队启动了一个新的自托管LLM项目。一位工程师建议使用Ollama，另一位推荐vLLM，第三位则认为“TGI开箱即用不是挺好？” 三人对于不同的使用场景都是对的，但没有一个是适用于所有场景的完美选择。

在2026年，选择树很重要：首先是硬件，其次是规模，第三是工作负载。而2025年的一个特定事件 — TGI于12月11日进入维护模式 — 改变了新项目的默认选项。

## 核心概念

### 五大引擎

| 引擎 | 最佳适用场景 | 备注 |
|------|--------------|------|
| **llama.cpp** | CPU / 边缘设备 / 依赖最小化 / 模型支持最广 | CPU上最快，控制完全 |
| **Ollama** | 开发笔记本，单用户，一键安装 | 比llama.cpp慢15-30%；生产吞吐量差3倍 |
| **TGI** | HF生态系统，受监管行业 | **2025年12月11日起进入维护模式** |
| **vLLM** | 通用生产环境，100+用户 | 2026年广泛的生产默认选项；v0.15.1于2026年2月发布 |
| **SGLang** | 代理式多轮交互，前缀密集型工作负载 | 生产环境部署超40万块GPU |

### 硬件优先决策

**仅CPU** → llama.cpp。Ollama也能工作但更慢。其他引擎在CPU上不具竞争力。

**AMD GPU** → vLLM（支持AMD ROCm）。SGLang也可以。TRT-LLM锁定NVIDIA，因此排除。

**NVIDIA Hopper (H100 / H200)** → vLLM、SGLang或TRT-LLM。三者均为顶级。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM是吞吐量领先者（第17阶段 · 07）。vLLM和SGLang紧随其后。

**Apple Silicon (M系列)** → llama.cpp（Metal加速）。Ollama封装了此功能。

### 规模次选决策

**1用户 / 本地开发** → Ollama。一条命令，几秒内生成首个token。

**10-100用户 / 小团队** → vLLM单GPU方案。

**100-10k用户 / 生产环境** → vLLM生产栈（第17阶段 · 18）或SGLang。

**10k+用户 / 企业级** → vLLM生产栈 + 分解式部署（第17阶段 · 17） + LMCache（第17阶段 · 18）。

### 工作负载第三决策

**通用聊天 / 问答** → vLLM凭借广泛的默认支持胜出。

**代理式多轮交互（工具、规划、记忆）** → SGLang的RadixAttention（第17阶段 · 06）占据主导。

**前缀复用率高的RAG** → SGLang。

**代码生成** → vLLM表现良好；SGLang在缓存方面略有优势。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分级KV。

### TGI维护模式陷阱

Hugging Face TGI于2025年12月11日进入维护模式 — 今后仅进行错误修复。历史上：一流的可观测性，最佳的HF生态系统集成（模型卡片、安全工具），原始吞吐量略逊于vLLM。

对于2026年的新项目：默认应避开TGI。现有的TGI部署可以继续，但最终应迁移。SGLang和vLLM是更安全的默认选择。

### 流水线模式

开发（Ollama）→ 预发布（llama.cpp）→ 生产（vLLM）。全程使用相同的GGUF或HF权重。工程师在笔记本电脑上快速迭代；预发布环境镜像生产环境的量化配置；生产环境是服务目标。

### Ollama注意事项

Ollama非常适合开发。但它不适合共享生产环境：Go的HTTP序列化带来额外开销，并发管理比vLLM简单，OpenTelemetry支持滞后。在它擅长的场景使用它 — 单用户、一条命令 — 然后为共享场景切换到vLLM。

### 自托管 vs. 托管服务是独立的决策

第17阶段 · 01（托管超大规模云）、· 02（推理平台）涵盖托管服务。本课程假设你已经决定自托管。自托管的原因包括：数据驻留要求、自定义微调、规模化后的总拥有成本、主机托管平台不提供所需的领域模型。

### 需要牢记的数字

- TGI维护模式起始日：2025年12月11日。
- vLLM v0.15.1：2026年2月；PyTorch 2.10；支持Blackwell SM120。
- SGLang生产环境规模：40万+块GPU。
- Ollama与llama.cpp的吞吐量差距：慢15-30%；在生产负载下差距达3倍。

## 实践应用

`code/main.py` 是一个决策树遍历器：给定硬件、规模和工作负载，选择一个引擎并解释原因。

## 交付成果

本课程将产出 `outputs/skill-engine-picker.md`。给定约束条件，选择一个引擎并编写迁移计划。

## 练习

1. 使用你的硬件/规模/工作负载运行 `code/main.py`。输出结果是否符合你的直觉？
2. 你的基础设施是12块H100和8块MI300X AMD。应该选择哪个引擎？为什么TRT-LLM不在考虑范围内？
3. 一个团队因为“这是我们熟悉的技术”想在2026年使用TGI。请论证迁移的理由。
4. 从Ollama开发环境切换到vLLM生产环境：量化、配置和可观测性方面会有哪些变化？
5. 一个RAG产品，P99前缀长度为8K，且跨租户复用率高。选择一个引擎，并结合第17阶段 · 11 + 18 的技术栈。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| llama.cpp | “那个CPU版本” | 模型支持最广，CPU上最快 |
| Ollama | “笔记本版本” | 一键安装，开发级吞吐量 |
| TGI | “HF的推理服务” | 自2025年12月起进入维护模式 |
| vLLM | “默认选项” | 2026年广泛的生产基线 |
| SGLang | “那个代理版本” | 前缀密集型，RadixAttention |
| TRT-LLM | “NVIDIA锁定” | Blackwell吞吐量领先者，仅限NVIDIA |
| GGUF | “llama.cpp格式” | 捆绑K量化变体 |
| 生产栈 | “vLLM K8s” | 第17阶段 · 18 参考部署 |
| 流水线模式 | “开发→预发布→生产” | 在相同权重上从Ollama → llama.cpp → vLLM |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — 综合LLM推理引擎对比](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 2026年10大vLLM替代方案](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI维护公告](https://github.com/huggingface/text-generation-inference) — 版本说明。
- [vLLM v0.15.1版本说明](https://github.com/vllm-project/vllm/releases)