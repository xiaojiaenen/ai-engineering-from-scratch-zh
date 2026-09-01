# Capstone 07 — 端到端微调流水线（数据 → SFT → DPO → 服务）

> 在自有数据上训练的 8B 模型，经过自有偏好 DPO 对齐、量化、投机解码、并部署为可度量 $/1M tokens 成本的服务端点。2026 年开源工具栈为 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代、GPTQ/AWQ/GGUF 用于量化、vLLM 0.7 + EAGLE-3 用于服务。本 Capstone 要求以可复现方式运行完整流水线——YAML 输入，服务接口输出——并在 2026 Model Openness Framework 下发布模型卡。

**类型：** Capstone  
**语言：** Python（流水线）、YAML（配置）、Bash（脚本）  
**前置条件：** Phase 2 (ML)、Phase 3 (DL)、Phase 7 (transformers)、Phase 10 (LLMs from scratch)、Phase 11 (LLM engineering)、Phase 17 (infrastructure)、Phase 18 (safety)  
**涉及的阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**时间：** 35 小时

## 问题背景

2026 年每一个严肃的 AI 团队都会把微调流水线备好。不是因为他们会量产 frontier 基座模型，而是因为下游适配——领域 SFT、基于标注偏好的 DPO、用于投机解码的蒸馏草稿模型、带 EAGLE-3 的服务部署——才是可衡量收益的来源。Axolotl v0.8 处理多 GPU SFT 配置，TRL 0.15 处理 DPO 和 GRPO，Unsloth 让你快速单 GPU 迭代，vLLM 0.7 + EAGLE-3 可将吞吐提升 2-3 倍且无损质量。工具链已经就位；真正的技巧在于 YAML 写法、数据清洗规范和评测纪律。

你将选择一个 8B 基座模型（Llama 3.3、Qwen3 或 Gemma 3），在任务特定数据上先跑 SFT 再跑 DPO，量化后部署，并在 lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro 上对比衡量提升。最终在 2026 Model Openness Framework 框架下发布模型卡。核心目标是可复现——一条命令即可从端到端重跑整个流水线。

## 概念

流水线共五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 脱敏、与公共基准污染的划分卫生检查。**SFT**：Axolotl YAML，8×H100 上 ZeRO-3，余弦调度，序列打包，2-3 个 epoch。**DPO/GRPO**：TRL 配置，1 个 epoch，偏好对来自人工标注或模型裁判，调优 beta 参数。**量化**：GPTQ + AWQ + GGUF，便于多种部署场景。**服务**：vLLM 0.7 + EAGLE-3 投机头（或 SGLang + SpecForge），K8s 部署，HPA 基于队列等待指标。

交付物中的消融实验：在同一组任务特定基准上对比 SFT-only vs SFT+DPO vs SFT+GRPO。服务指标：batch size 1/8/32 下的 tokens/s、EAGLE-3 接受率、$1M tokens 成本。安全评测：Llama Guard 4 通过率。模型卡：偏见评估、可复现种子、数据授权信息。

## 架构

```
原始数据 (HF datasets + 内部数据)
    |
    v
Datatrove 去重 + Nemotron-CC 质量分类器 + PII 脱敏
    |
    v
划分卫生检查（MMLU-Pro 污染检测）
    |
    v
Axolotl SFT 配置 (YAML)  ---> 8×H100，ZeRO-3
    |
    v
TRL DPO / GRPO 配置       ---> 4×H100，1 个 epoch
    |
    v
GPTQ + AWQ + GGUF 量化
    |
    v
vLLM 0.7 + EAGLE-3 投机解码
    |
    v
K8s 部署，HPA 基于队列等待指标
    |
    v
lm-evaluation-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
模型卡 (2026 MOF) + 安全评测 (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 去重、Nemotron-CC 分类器质量评估、Presidio PII 脱敏
- 基座：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8 + ZeRO-3、Flash Attention 3、序列打包
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ (Marlin)、AWQ、GGUF via llama.cpp
- 服务：vLLM 0.7 + EAGLE-3 投机解码（或 SGLang 0.4 + SpecForge）
- 评测：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评测：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin，HPA 基于队列等待指标
- 可观测性：W&B 训练监控、Langfuse 推理监控

```figure
ce-finetune-stages
```

## 构建步骤

1. **数据流水线。** 在原始语料上运行 Datatrove 去重。应用 Nemotron-CC 风格质量分类器。Presidio 脱敏 PII。写入 train/val 划分并记录明确种子。

2. **污染检测。** 对每个验证划分，与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集计算 MinHash。任何重叠均视为不可接受。

3. **Axolotl SFT。** YAML 配置含 ZeRO-3、FA3、序列打包。在 8×H100 上跑 2-3 个 epoch。日志推送至 W&B。

4. **TRL DPO / GRPO。** 取 SFT checkpoint，在偏好对上跑 1 个 epoch DPO（或在数学/代码任务上用 GRPO + 可验证奖励）。进行 beta 参数扫描。

5. **量化。** 产出三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M for llama.cpp。记录模型体积和理论吞吐量。

6. **投机解码服务。** vLLM 0.7 配置 + EAGLE-3 草稿头（通过 Red Hat Speculators 训练）。在 batch size 1/8/32 下测量接受率和尾部延迟。对比同评测下的 $/1M tokens 与 Anthropic / OpenAI。

7. **评测矩阵。** 在 base、SFT-only、SFT+DPO、SFT+GRPO 上分别运行 lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。产出对比表格。

8. **安全评测。** Llama Guard 4 在开发集上的通过率。ShieldGemma-2 输出过滤器。

9. **模型卡。** 使用 MOF 2026 模板：数据、训练、评测、安全、授权、含 YAML 和 commit SHA 的可复现章节。

## 运行示例

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k 去重后，12k 过滤后，280k 接受 (seed=7)
[SFT]     3 epochs, 8×H100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4×H100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 接受率 0.74, p99 延迟 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md 已按 2026 MOF 规范生成
```

## 交付说明

`outputs/skill-finetuning-pipeline.md` 描述交付物。一条命令即可将数据流经 SFT → DPO → 量化 → 服务 → 评测，最终产出模型卡 + 服务接口。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 相对于基座的评测增益 | 目标任务上的提升（MMLU-Pro、MT-Bench-v2、任务特定指标） |
| 20 | 流水线可复现性 | 一条命令配合相同种子可端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 脱敏覆盖率、污染检测通过 |
| 20 | 服务效率 | bs=1/8/32 下的 tokens/s、EAGLE-3 接受率、$1M tokens 成本 |
| 15 | 模型卡 + 安全评测 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习题

1. 在同一任务特定基准上分别运行 SFT-only、SFT+DPO、SFT+GRPO。报告哪种偏好方法胜出及其幅度。

2. 将 Llama 3.3 8B 替换为 Qwen3 14B。在同等质量水平下测量 $/1M tokens 成本。

3. 分别测量领域数据和通用 ShareGPT 数据上的 EAGLE-3 接受率。报告差异及其对延迟预算的含义。

4. 注入 1% 的污染（将 MMLU-Pro 答案泄漏进训练数据）并重跑评测。观察 MMLU-Pro 准确率异常飙升。构建一个能在 CI 中拦截此类污染的污染检测门禁。

5. 添加 LoRA SFT 作为全量微调的替代方案。在内存降低 10 倍的条件下衡量质量差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | "SFT 训练器" | 统一 YAML 驱动的训练器，支持 SFT、DPO 和蒸馏 |
| TRL | "偏好调优库" | Hugging Face 的 DPO、GRPO、PPO 实现库 |
| GRPO | "群体相对策略优化" | DeepSeek R1 的 RL 配方，使用可验证奖励 |
| EAGLE-3 | "投机解码草稿" | 可预测 N 个未来 token 的草稿头；vLLM 用目标模型验证 |
| MOF | "模型开放框架" | 2026 年对模型发布在数据、代码、授权维度进行评级的标准 |
| 污染检测 | "划分卫生" | 基于 MinHash 的检测训练集向测试集泄漏的机制 |
| 接受率 | "EAGLE / MTP 指标" | 目标模型接受的草稿 token 占比 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — SFT / DPO 训练器参考
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选投机解码训练器
- [Model Openness Framework 2026](https://isocpp.org/) — 开源发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评测运行器
