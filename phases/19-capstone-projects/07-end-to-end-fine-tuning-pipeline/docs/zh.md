# Capstone 07 — 端到端微调流程（从数据到SFT到DPO到服务）

> 一个在您自己的数据上训练的8B模型，根据您的偏好进行DPO对齐，量化，并进行投机解码，最后以可测量的 $/1M token 成本提供服务。2026年的开源技术栈包括 Axolotl v0.8、TRL 0.15、用于迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF、以及用于服务的 vLLM 0.7 配合 EAGLE-3。毕业设计的目标是可复现地运行整个流程 — 输入 YAML，输出一个可服务的端点 — 并在2026模型开放框架下发布一个模型卡。

**类型：** 毕业设计
**语言：** Python（流程）、YAML（配置）、Bash（脚本）
**先决条件：** 阶段2（机器学习）、阶段3（深度学习）、阶段7（Transformers）、阶段10（从零构建LLMs）、阶段11（LLM工程）、阶段17（基础设施）、阶段18（安全性）
**应用阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35小时

## 问题

2026年，每个严肃的AI团队都会保留一个微调流程。不是因为他们发布了一个前沿的基础模型，而是因为下游适配——领域SFT、针对标注偏好的DPO、用于投机解码的蒸馏草案、使用EAGLE-3进行服务——才是可量化收益所在。Axolotl v0.8 处理多GPU的SFT配置。TRL 0.15 处理DPO和GRPO。Unsloth 帮助您实现快速的单GPU迭代。vLLM 0.7 配合EAGLE-3 可将解码吞吐量提升2-3倍，且不损失质量。工具已经就位；技巧在于YAML配置、数据卫生和评估纪律。

您将运行一个8B基础模型（Llama 3.3、Qwen3或Gemma 3），先进行SFT，然后在特定任务数据上进行DPO，为服务进行量化，并针对lm-evaluation-harness、RewardBench-2、MT-Bench-v2和MMLU-Pro评估收益。您将根据2026模型开放框架（MOF）制作一个模型卡。重点在于可复现性——一个命令就能端到端地重跑整个流程。

## 概念

该流程有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（类似Nemotron-CC的分类器）、PII清除、基于公共基准污染的数据集划分卫生检查。**SFT**：Axolotl YAML，在8xH100上使用ZeRO-3，余弦学习率调度，序列打包，2-3个周期。**DPO或GRPO**：TRL配置，1个周期，偏好对可以是人工标注或模型判断的，调整beta参数。**量化**：GPTQ + AWQ + GGUF，提供部署灵活性。**服务**：vLLM 0.7 配合 EAGLE-3 投机头（或 SGLang 配合 SpecForge），Kubernetes部署，基于队列等待指标的HPA。

消融实验是交付成果：在三个特定任务基准上，对比仅SFT、SFT+DPO与SFT+GRPO。服务指标：batch大小为1/8/32时的tokens/s、EAGLE-3接受率、$/1M tokens。安全评估：Llama Guard 4通过率。模型卡：偏差评估、可复现种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove用于去重，Nemotron-CC分类器用于质量，Presidio用于PII清除
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，使用ZeRO-3、Flash Attention 3、序列打包
- 偏好调优：TRL 0.15 用于DPO或GRPO；Unsloth 用于单GPU迭代
- 量化：GPTQ（Marlin）、AWQ、通过llama.cpp实现GGUF
- 服务：vLLM 0.7 配合 EAGLE-3 投机解码（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA设备插件，基于队列等待指标的HPA
- 可观测性：W&B用于训练，Langfuse用于推理

## 构建它

1. **数据流程。** 对原始语料库运行Datatrove去重。应用类似Nemotron-CC的质量分类器。使用Presidio清除PII。使用明确种子划分训练集/验证集。

2. **污染检查。** 对于每个验证集划分，计算其相对于MMLU-Pro、MT-Bench-v2、RewardBench-2测试集的MinHash。拒绝任何重叠。

3. **Axolotl SFT。** 使用ZeRO-3、FA3、序列打包的YAML配置。在8xH100上运行2-3个周期。记录到W&B。

4. **TRL DPO / GRPO。** 取SFT检查点，运行一个周期的DPO（使用偏好对）或GRPO（在数学/代码任务上使用可验证奖励）。调整beta参数。

5. **量化。** 生成三种量化模型：用于llama.cpp的GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M。记录大小和名义吞吐量。

6. **使用投机解码进行服务。** 使用通过Red Hat Speculators训练的EAGLE-3草稿头配置vLLM 0.7。测量batch大小为1/8/32时的接受率和尾部延迟。报告相对于Anthropic / OpenAI在相同评估上的 $/1M token成本。

7. **评估矩阵。** 在基础模型、仅SFT、SFT+DPO、SFT+GRPO上运行lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **安全评估。** 在开发集上测试Llama Guard 4通过率。使用ShieldGemma-2进行输出过滤。

9. **模型卡。** MOF 2026模板：数据、训练、评估、安全、许可、包含YAML和commit SHA的可复现性部分。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付它

`outputs/skill-finetuning-pipeline.md` 描述了交付成果。一个命令就能运行从数据到SFT到DPO到量化到服务到评估的整个过程，并输出一个模型卡 + 可服务的端点。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 与基线的评估增量 | 在目标任务（MMLU-Pro、MT-Bench-v2、特定任务）上测得的提升 |
| 20 | 流程可复现性 | 一个命令端到端重跑，使用相同种子 |
| 20 | 数据卫生 | 去重率、PII清除覆盖率、污染检查通过 |
| 20 | 服务效率 | batch大小为1/8/32时的tokens/s、EAGLE-3接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF完整性 + Llama Guard 4通过率 |
| **100** | | |

## 练习

1. 在相同的特定任务基准上运行仅SFT、SFT+DPO、SFT+GRPO。报告哪种偏好方法获胜以及胜出幅度。

2. 将Llama 3.3 8B换为Qwen3 14B。在匹配质量下测量 $/1M token成本。

3. 在领域数据与通用ShareGPT上测量EAGLE-3接受率。报告差异及其对延迟预算的意义。

4. 注入1%的污染（将MMLU-Pro答案泄露到训练数据中）并重新运行评估。观察MMLU-Pro准确性不切实地跳升。构建一个能捕获此情况的污染检查CI门禁。

5. 添加LoRA SFT作为全微调的替代方案。在内存降低10倍的情况下测量质量差距。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|------------------------|
| Axolotl | "SFT训练器" | 统一的、基于YAML的SFT、DPO和蒸馏训练器 |
| TRL | "偏好调优器" | Hugging Face库，用于在LLMs上进行DPO、GRPO、PPO |
| GRPO | "群组相对策略优化" | DeepSeek R1的强化学习方法，使用可验证奖励 |
| EAGLE-3 | "投机解码草稿" | 提前预测N个token的草稿头；vLLM使用目标模型进行验证 |
| MOF | "模型开放框架" | 2026年标准，根据数据、代码、许可对模型发布进行评级 |
| 污染检查 | "数据集划分卫生" | 基于MinHash的测试集泄露到训练集的检测 |
| 接受率 | "EAGLE / MTP指标" | 目标模型接受的草稿token所占比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单GPU迭代参考
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务技术栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 替代投机解码训练器
- [模型开放框架 2026](https://isocpp.org/) — 开源发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 权威评估运行器