# 音频语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026年的音频语言模型能够对语音、环境声和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 表现相当。Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开源模型与闭源模型之间的差距已基本消除——但在多音频任务上除外，所有模型的表现都接近随机水平。

**类型：** 学习
**语言：** Python
**前置课程：** 阶段 6 · 04 (ASR)，阶段 12 · 03 (视觉语言模型)，阶段 7 · 10 (音频转换器)
**时长：** ~45 分钟

## 问题

你有一段5秒的音频：狗叫声，有人喊“停下！”，然后是寂静。有用的问题涵盖多个维度：

- **转录。** “说了什么？”——这是 ASR 的领域。
- **语义推理。** “那个人有危险吗？”——需要联合理解吠叫、喊叫和寂静。
- **音乐推理。** “旋律由哪些乐器演奏？”
- **长音频检索。** “在这90分钟的讲座中，导师在哪里解释了梯度下降？”

一个能用一个提示回答所有这些问题的模型就是**音频语言模型**。它与纯 ASR 不同：LALM 生成自由形式的自然语言答案，而不仅仅是转录文本。

## 概念

![音频语言模型：音频编码器 + 投影器 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

2026年每个 LALM 都具有相同的骨架：

1.  **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或模型专用的自定义编码器。
2.  **投影器。** 线性层或 MLP，将音频编码器特征桥接到 LLM 的 token 嵌入空间。
3.  **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本和音频 token；生成文本。

训练：

-   **第一阶段。** 冻结编码器和 LLM；仅在 ASR/描述数据上训练投影器。
-   **第二阶段。** 在指令遵循的音频任务（问答、推理、音乐理解）上进行全参数或 LoRA 微调。
-   **第三阶段（可选）。** 语音输入/语音输出添加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 这样做了。

### 2026 年模型图谱

| 模型 | 基础模型 | 音频编码器 | 输出模态 | 获取方式 |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商用 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商用 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro (闭源) | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio (闭源) | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准现实检查 (2026)

**MMAU-Pro。** 1800 个问答对，涵盖语音/声音/音乐/混合。包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频列对所有人都是致命的。** 四选一多选题的随机准确率 = 25%；大多数模型得分都在此附近。LALM 在比较两个音频片段方面仍然很吃力。

### LALM 在 2026 年有用的场景

-   **客服中心录音合规审计。** “客服人员是否提到了必需的免责声明？”
-   **无障碍。** 向听障用户描述声音事件（不仅仅是转录）。
-   **内容审核。** 检测暴力语言、威胁性语调和背景上下文。
-   **播客/会议分章。** 语义摘要，而不仅仅是说话人轮次。
-   **音乐目录分析。** “找到所有 B 段转调的曲目。”

### 尚未（还）有用的场景

-   细粒度音乐理论（和弦级别以下）。
-   长对话中基于说话人的推理（超过 10 分钟后性能下降）。
-   多音频比较（22-26% 仅略高于随机水平）。
-   实时流式推理（大多数是离线批量推理）。

## 实现

### 第 1 步：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第 2 步：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常是 1-3 个线性层。在 ASR 对（音频 → 转录文本）上训练它是第一阶段的前置任务。

### 第 3 步：在 MMAU / LongAudioBench 上进行基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告每个类别（语音/声音/音乐/多音频）的结果。聚合数字会掩盖模型的失败之处。

## 应用

| 任务 | 2026 年首选 |
| :--- | :--- |
| 开放式音频问答 (开放) | Qwen2.5-Omni-7B |
| 开放模型中长音频表现最佳 | Audio Flamingo Next |
| 最佳闭源模型 | Gemini 2.5 Pro |
| 语音输入/语音输出代理 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2 (音乐特化的 AF-CLAP) |
| 客服中心审计 | Gemini 2.5 Pro (通过 API)，结合 RAG 应用于策略文档 |

## 陷阱

-   **过度信任多音频能力。** 如果你的任务需要“哪个片段具有 X”，那么接近随机水平的性能是真实存在的。
-   **长音频性能下降。** 超过 10 分钟，大多数模型的说话人归属会失效。先进行说话人分离（第6课），然后进行总结。
-   **在静默时产生幻觉。** 使用 Whisper 编码器的 LALM 继承了相同的 Whisper 风格问题。使用 VAD 门控。
-   **基准选择性报告。** 供应商的博客文章会突出最好的类别。自己运行 MMAU-Pro 多音频子集。

## 部署

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM + 基准子集 + 输出模态（文本 vs 语音）。

## 练习

1.  **简单。** 运行 `code/main.py` 查看一个简单的投影器模式 + 模拟的 LALM 路由（音频嵌入，文本 token） → 输出 token。
2.  **中等。** 在 100 个 MMAU-Pro 语音项目上评估 Qwen2.5-Omni-7B。与论文报告的数字进行比较。
3.  **困难。** 构建一个最小的音频描述基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。在 Clotho-AQA 上与 SALMONN 进行比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
| :--- | :--- | :--- |
| LALM | 音频 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| 投影器 | 适配器 | 将音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | 那个基准 | 1 万对音频问答对，涵盖语音、声音、音乐。 |
| MMAU-Pro | 更难的 MMAU | 1800 个多音频/重推理的问题。 |
| LongAudioBench | 长格式评估 | 多分钟片段，带有语义查询。 |
| 语音输入/语音输出 | 语音原生 | 模型摄入语音并发出语音，无需文本中转。 |

## 延伸阅读

-   [Chu 等 (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
-   [阿里巴巴 (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出。
-   [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开放长音频领导者。
-   [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
-   [Tang 等 (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
-   [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/) — 2026 年实时排名。