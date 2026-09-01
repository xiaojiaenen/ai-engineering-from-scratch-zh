# Whisper — 架构与微调

> Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器模型，在 68 万小时多语言弱监督音频-文本对上进行训练。一套架构，多种任务，在 99 种语言上均表现稳健。2026 年参考级 ASR 模型。

**类型：** Build
**语言：** Python
**前置知识：** Phase 6 · 04（ASR）、Phase 5 · 10（注意力机制）、Phase 7 · 05（完整 Transformer）
**预计时间：** 约 75 分钟

## 问题背景

Whisper 由 OpenAI 于 2022 年 9 月发布，是最早以"开箱即用"方式发布的 ASR 模型之一：粘贴音频，得到文本，支持 99 种语言，抗噪性强，可在笔记本上运行。到 2024 年，OpenAI 已推出 Large-v3 和 Turbo 变体；到了 2026 年，Whisper 已成为从播客转录、语音助手到 YouTube 字幕的各种应用的事实基线。

但 Whisper 并非一个可以永远当作黑盒使用的工具。领域偏移会损害其性能——技术术语、说话人口音、专有名词、短音频片段、静音等场景都会导致效果下降。你需要了解：

1. 它的内部原理是什么。
2. 如何正确地为它提供分块、流式或长语音输入。
3. 何时以及如何对它进行微调。

## 概念解析

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准 Transformer 编码器-解码器结构。

- 输入：30 秒对数梅尔谱图，80 个 mel 滤波器组，10ms 帧移 → 3000 帧。较短的片段进行零填充，较长的片段进行分块处理。
- 编码器：卷积下采样（步幅为 2）+ `N` 个 Transformer 块。Large-v3 使用 32 层、1280 维隐层、20 个注意力头。
- 解码器：`N` 个 Transformer 块，包含因果自注意力 + 对编码器输出的交叉注意力。尺寸与编码器相同。
- 输出：基于 51,865 token 词表的 BPE token。

Large-v3 包含 15.5 亿参数。Turbo 使用 4 层解码器（从 32 层缩减），在 WER 仅增加不到 1% 的情况下将延迟降低 8 倍。

**Prompt 格式。** Whisper 是一个多任务模型，通过解码器 prompt 中的特殊 token 来引导：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.
```

- `<|en|>` — 语言标签；强制控制翻译与转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 将任意语言输入翻译为英语，或忠实转录原文。
- `<|notimestamps|>` — 跳过词级时间戳（速度更快）。

正是这种 prompt 机制让一个模型能够完成多种任务。将 `<|en|>` 改为 `<|fr|>` 即可转录法语。

**30 秒窗口。** 所有内容都固定在 30 秒。更长的音频需要分块，更短的音频需要填充。窗口本身不原生支持流式处理——这也是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔归一化。** 计算方式为 `(log_mel - mean) / std`，其中统计量来自 Whisper 自身的训练语料。你 *必须* 使用 Whisper 的预处理逻辑（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的各变体

| 变体 | 参数量 | 延迟（A100） | WER（LibriSpeech-clean） |
|------|--------|-------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming（2024） | 1.55B | 流式 | 2.0% |

### 微调

2026 年的标准工作流：

1. 收集 10–100 小时目标域音频，附带对齐的转录文本。
2. 使用 `transformers.Seq2SeqTrainer` 配合 `generate_with_loss` 回调进行训练。
3. 参数高效微调：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上添加 LoRA，可将 GPU 显存占用降低 4 倍，且 WER 损失不到 0.3%。
4. 如果数据少于 10 小时，冻结编码器，仅微调解码器。
5. 使用 Whisper 自带的 tokenizer 和 prompt 格式；切勿替换 tokenizer。

社区实测结果：在 20 小时医学听写数据上微调 Medium 模型，医学词汇的 WER 从 12% 降至 4.5%。在 4 小时冰岛语数据上微调 Turbo，WER 从 18% 降至 6%。

```figure
sp-asr-attention
```

## 动手实现

### 步骤 1：直接运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止重复输出失控
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认参数：`temperature=0.0`（采样默认值会从 0.0 → 0.2 → 0.4 逐步回退）、`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 步骤 2：长语音分块处理

```python
# whisperx 是 2026 年长语音词级时间戳的首选方案
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 额外提供了：(1) Silero VAD 门控，(2) 通过 wav2vec 2.0 实现词级对齐，(3) 通过 `pyannote.audio` 实现说话人分离。是 2026 年生产级转录的工作主力。

### 步骤 3：使用 LoRA 进行微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> 可训练参数约 3M / 总计 809M
```

然后使用标准的 Trainer 训练循环。每 1000 步保存一次检查点，在预留测试集上以 WER 评估。

### 步骤 4：检查各层学到了什么

```python
# 在解码过程中提取交叉注意力权重，观察解码器关注编码器的哪些位置。
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: 层 × 头 × 时间步 × 源序列长度
```

用热力图可视化——你会看到对角线对齐模式，解码器逐步扫描编码器帧时注意力沿对角线移动。这条对角线就是 Whisper 所代表的词级时间戳。

## 使用指南

2026 年技术栈选型：

| 场景 | 推荐方案 |
|------|---------|
| 通用英语、离线环境 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动端 / 边缘设备 | 量化（int8）的 Whisper-Tiny 或 Moonshine |
| 多语言长语音 | 通过 `whisperx` 使用 Large-v3 + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式处理（延迟 2 秒） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，且输出完全一致。

## 2026 年仍常见的坑

- **静音时的幻觉文本。** Whisper 在字幕数据上训练过，会输出"感谢观看！"、"订阅！"、歌词等内容。调用前务必先做 VAD 门控。
- **`condition_on_previous_text` 级联效应。** 一次幻觉会污染后续窗口的结果。除非你需要在分块间保持语义连贯性，否则设为 `False`。
- **短片段填充问题。** 2 秒的音频被填充到 30 秒后，尾部静音区域容易产生幻觉。使用 `pad=False` 或先做 VAD 门控。
- **错误的 mel 统计量。** 使用 librosa 的 mel 谱而非 Whisper 的会产出近乎随机的结果。请使用 `whisper.audio.log_mel_spectrogram`。

## 交付成果

保存为 `outputs/skill-whisper-tuner.md`。针对给定领域设计一套 Whisper 微调或推理流水线。

## 练习题

1. **简单。** 运行 `code/main.py`。它会 tokenizes 一个 Whisper 风格的 prompt，计算解码后的形状预算，并打印一段 10 分钟音频的分块调度计划。
2. **中等。** 安装 `faster-whisper`，转录一段 10 分钟的播客，并与人工转录稿对比 WER。尝试 `language="auto"` 与强制指定 `language="en"` 的效果差异。
3. **困难。** 使用 HF `datasets`，选择一个 Whisper 表现不佳的语言（如乌尔都语），在 2 小时数据上用 LoRA 对 Medium 模型微调 2 个 epoch，并报告 WER 变化量。

## 核心术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|-------------|---------|
| 30 秒窗口 | Whisper 的限制 | 硬编码的输入上限；更长的音频需分块。 |
| SOT | 转录起始标记 | `<\|startoftranscript\|>` 启动解码器 prompt。 |
| 时间戳 token | 时间对齐 | 每个 0.02 秒偏移在 51k 词表中对应一个特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，速度提升 8 倍，WER 劣化不到 1%。 |
| WhisperX | 长语音封装 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效调优 | 在注意力层添加低秩适配器；仅训练约 0.3% 的参数。 |
| 幻觉 | 隐性故障 | Whisper 从噪声或静音中生成通顺的英文文本。 |

## 延伸阅读

- [Radford et al. (2022). Whisper 论文](https://arxiv.org/abs/2212.04356) — 原始架构与训练方法。
- [OpenAI (2024). Whisper Large-v3-turbo 发布说明](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长语音、词级对齐、说话人分离。
- [Systran — faster-whisper 仓库](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2，速度快 4 倍。
- [HuggingFace — Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper) — 标准的 LoRA / 全参数微调指南。
