# 音频-语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频-语言模型能够基于语音、环境声和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 持平；Audio Flamingo Next 在 LongAudioBench 上超越了 Gemini 2.5 Pro。开放模型与闭源模型之间的差距已基本抹平——除了多音频任务，那几乎所有模型都在随机水平徘徊。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04（ASR）、Phase 12 · 03（视觉-语言模型）、Phase 7 · 10（音频 Transformer）
**时间：** 约 45 分钟

## 问题所在

你有一段 5 秒的音频：狗叫声、有人喊"stop!"、然后是静音。 Useful 的问题横跨多个维度：

- **转录。** "说了什么？"——属于 ASR 范畴。
- **语义推理。** "此人是否处于危险之中？"——需要联合理解狗叫、喊声和静音。
- **音乐推理。** "哪些乐器在演奏主旋律？"
- **长音频检索。** "在那 90 分钟的课程中，讲师在何处解释了梯度下降？"

一个能用单一 prompt 回答所有这些问题的模型就是**音频-语言模型**（LALM / ALM）。它与纯 ASR 不同：LALM 产出的是自由形式的自然语言回答，而不仅仅是转录文本。

## 概念

![音频-语言模型：音频编码器 + 投影层 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年的 LALM 都有相同的骨架：

1. **音频编码器。** Whisper encoder · BEATs · CLAP · WavLM · 或每模型自定义编码器。
2. **投影层（Projector）。** 线性层或 MLP，将音频编码器特征桥接到 LLM 的词元嵌入空间。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本 + 音频词元；生成文本。

训练：

- **阶段 1。** 冻结编码器 + LLM；仅在有 ASR / 字幕数据上训练投影层。
- **阶段 2。** 在全参数或 LoRA 级别对指令跟随音频任务（QA、推理、音乐理解）进行微调。
- **阶段 3（可选）。** 语音输入/语音输出会加入一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 都做了这一步。

### 2026 年模型地图

| 模型 | 主干 | 音频编码器 | 输出模态 | 访问方式 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准评测的残酷现实（2026）

**MMAU-Pro。** 涵盖语音/声音/音乐/混合的 1800 对 QA。包含多音频子集。

| 模型 | 综合 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频这一列对所有人都是致命的。** 四选一选择题的随机水平 = 25%；大多数模型就在那附近。LALM 仍然难以比较两段音频。

### 2026 年 LALM 适用的场景

- **呼叫中心录音合规审计。** "客服是否提及了必需的披露条款？"
- **无障碍辅助。** 为听障用户描述声音事件（而不只是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **播客/会议章节划分。** 语义摘要，而非仅识别说话人轮次。
- **音乐曲目录分析。** "找出所有在 B 段发生转调的曲目。"

### 目前尚不适用的场景

- 细粒度音乐理论（低于和弦层面）。
- 长对话中的说话人归属推理（超过 10 分钟后性能急剧下降）。
- 多音频比较（22-26% 仅略高于随机）。
- 实时流式推理（大多数仍是离线批量推理）。

```figure
v4-alm-tokens
```

## 动手构建

### 步骤 1：调用 Qwen2.5-Omni

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

### 步骤 2：投影层模式

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

就是这样。投影层通常只有 1-3 层线性网络。在 ASR 对（音频 → 转录文本）上训练它就是阶段 1 的预训练任务。

### 步骤 3：MMAU / LongAudioBench 基准评测

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

按类别分别报告（语音/声音/音乐/多音频）。聚合数字会掩盖模型在哪方面失败。

## 如何使用

| 任务 | 2026 年首选 |
|------|-----------|
| 自由形式音频 QA（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源 | Audio Flamingo Next |
| 最佳闭源 | Gemini 2.5 Pro |
| 语音输入/输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用的 AF-CLAP） |
| 呼叫中心审计 | 通过 API 使用 Gemini 2.5 Pro，并结合策略文档的 RAG |

## 陷阱

- **不要过度信任多音频结果。** 如果你的任务需要"哪段音频包含 X"，随机水平的性能是真实存在的。
- **长音频性能衰减。** 超过 10 分钟后，大多数模型的说话人归属能力会崩溃。先做说话人分离（Lesson 6），再做摘要。
- **对静音的幻觉。** 这与使用 Whisper 编码器的 LALM 继承的 Whisper 风格问题相同。用 VAD 门控。
- **基准 cherry-picking。** 厂商博客文章只会突出最佳类别。自己去跑一遍 MMAU-Pro 的多音频子集。

## 交付

保存为 `outputs/skill-alm-picker.md`。针对给定的音频理解任务，选定 LALM + 基准子集 + 输出模态（文本或语音）。

## 练习

1. **简单。** 运行 `code/main.py` 查看玩具投影层模式 + 模拟 LALM 如何将 (音频嵌入, 文本词元) 路由到输出词元。
2. **中等。** 在 100 条 MMAU-Pro 语音样本上评测 Qwen2.5-Omni-7B。与论文报告数值对比。
3. **困难。** 构建一个最小音频字幕基线：BEATs 编码器 + 2 层投影层 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影层。与 SALMONN 在 Clotho-AQA 上的表现对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | 音频 ChatGPT | 音频编码器 + 投影层 + LLM 解码器。 |
| Projector | Adapter | 小型 MLP，将音频特征映射到 LLM 嵌入空间。 |
| MMAU | 那个基准 | 涵盖语音、声音、音乐的 10k 音频 QA 对。 |
| MMAU-Pro | 更难的 MMAU | 1800 道多音频/重度推理题。 |
| LongAudioBench | 长音频评测 | 多分钟级片段上的语义查询。 |
| Voice-in / voice-out | 语音原生 | 模型直接处理语音并输出语音，不经过文本中转。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入/语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领跑者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 年实时排名。
