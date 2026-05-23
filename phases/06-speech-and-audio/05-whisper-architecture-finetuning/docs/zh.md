# Whisper — 架构与微调

> Whisper 是一个30秒窗口的Transformer编码器-解码器，基于680k小时的多语言弱监督音频-文本对训练。单一架构，支持多种任务，在99种语言中表现稳健。2026年的参考自动语音识别系统。

**类型:** 构建
**语言:** Python
**先决条件:** 阶段 6 · 04（自动语音识别），阶段 5 · 10（注意力），阶段 7 · 05（完整Transformer）
**时间:** ~75分钟

## 问题所在

Whisper 由 OpenAI 于 2022 年 9 月发布，是首个作为通用产品交付的自动语音识别模型：粘贴音频，获取文本，支持 99 种语言，对噪声稳健，可在笔记本电脑上运行。到 2024 年，OpenAI 已发布了 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转录到语音助手，再到 YouTube 字幕等一切领域的默认基准。

但 Whisper 并非一个可以永远当作黑盒使用的流水线。领域迁移会摧毁它——技术术语、说话者口音、专有名词、短音频片段、静音。你需要了解：

1.  它内部实际是什么。
2.  如何正确处理分块、流式或长格式音频。
3.  何时以及如何进行微调。

## 概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准的 Transformer 编码器-解码器。

-   输入：30秒的对数梅尔频谱图，80个梅尔滤波器，10毫秒帧移 → 3000帧。较短的片段会进行零填充，较长的片段会进行分块。
-   编码器：卷积下采样（步长2）+ `N` 个Transformer块。对于 Large-v3：32层，1280维，20个注意力头。
-   解码器：`N` 个Transformer块，包含因果自注意力机制以及对编码器输出的交叉注意力机制。大小与编码器相同。
-   输出：基于 51,865 个词表的 BPE token。

Large-v3 有 15.5 亿参数。Turbo 使用 4 层解码器（从 32 层缩减），在词错率（WER）损失 <1% 的情况下将延迟降低 8 倍。

**提示格式。** Whisper 是一个多任务模型，由解码器提示中的特殊 token 驱动：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

-   `<|en|>` — 语言标签；强制执行翻译或转录行为。
-   `<|transcribe|>` 或 `<|translate|>` — 从任何语言的输入翻译成英文输出，或逐字转录。
-   `<|notimestamps|>` — 跳过单词级时间戳（更快）。

提示格式使得单一模型能够执行多种任务。将 `<|en|>` 更改为 `<|fr|>`，它就会转录法语。

**30秒窗口。** 一切都固定在 30 秒。较长的片段需要分块；较短的片段需要填充。窗口并非原生流式——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔归一化。** `(log_mel - mean) / std` 其中统计数据来自 Whisper 自身的训练语料库。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的变体

| 变体 | 参数量 | 延迟 (A100) | 词错率 (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 3900万 | 1倍实时 | 5.4% |
| Base | 7400万 | 1倍 | 4.1% |
| Small | 2.44亿 | 1倍 | 3.0% |
| Medium | 7.69亿 | 1倍 | 2.7% |
| Large-v3 | 15.5亿 | 2倍 | 1.8% |
| Large-v3-turbo | 8.09亿 | 8倍 | 1.58% |
| Whisper-Streaming (2024) | 15.5亿 | 流式 | 2.0% |

### 微调

2026 年的标准工作流程：

1.  收集 10-100 小时目标领域的音频及对齐的转录文本。
2.  运行 `transformers.Seq2SeqTrainer`，使用 `generate_with_loss` 回调。
3.  参数高效：对注意力层的 `q_proj`、`k_proj`、`v_proj` 应用 LoRA，可将 GPU 显存减少 4 倍，词错率代价 <0.3%。
4.  如果数据少于 10 小时，请冻结编码器。只微调解码器。
5.  使用 Whisper 自己的分词器和提示格式；切勿更换分词器。

社区结果：在医学听写上微调 Medium 模型 20 小时，医学词汇的词错率从 12% 降至 4.5%。在冰岛语上微调 Turbo 模型 4 小时，词错率从 18% 降至 6%。

## 构建它

### 第 1 步：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认为 0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 第 2 步：分块长音频

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了 (1) Silero VAD 门控，(2) 通过 wav2vec 2.0 进行单词级对齐，(3) 通过 `pyannote.audio` 进行说话人分离。2026 年生产环境转录的主力工具。

### 第 3 步：使用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

然后是标准的 Trainer 循环。每 1000 步保存一个检查点。使用保留数据集上的词错率进行评估。

### 第 4 步：检查每一层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

使用热图进行可视化——你会看到解码器步进扫描编码器帧时产生的对角线对齐。这个对角线就是 Whisper 对单词时间戳的理解。

## 使用它

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语，离线 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动/边缘设备 | Whisper-Tiny 量化 (int8) 或 Moonshine |
| 多语言长音频 | 通过 `whisperx` 使用 Large-v3 + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式（2秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 单词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原始版本快 4 倍，输出完全相同。

## 2026 年仍然存在的陷阱

-   **静音时产生幻觉文本。** Whisper 基于字幕训练，会包含“感谢观看！”、“订阅！”、歌词等内容。调用前务必进行 VAD 门控。
-   **`condition_on_previous_text` 级联。** 一次幻觉会污染后续窗口。除非你需要跨分块保持流畅性，否则请设置 `False`。
-   **短片段填充。** 一个 2 秒的片段被填充到 30 秒时，可能会在末尾的静音中产生幻觉。使用 `pad=False` 或 VAD 门控。
-   **错误的梅尔统计数据。** 使用 librosa 的梅尔频谱而非 Whisper 的，会导致近乎随机的输出。使用 `whisper.audio.log_mel_spectrogram`。

## 部署它

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计一个 Whisper 微调或推理流水线。

## 练习

1.  **简单。** 运行 `code/main.py`。它会对一个 Whisper 风格的提示进行分词，计算解码的形状预算，并打印出一个 10 分钟音频片段的分块计划。
2.  **中等。** 安装 `faster-whisper`，转录一个 10 分钟的播客，与人工转录稿比较词错率。尝试使用 `language="auto"` 与强制 `language="en"`。
3.  **困难。** 使用 HuggingFace `datasets`，选择一种 Whisper 处理得不好的语言（例如乌尔都语），用 LoRA 在 2 小时数据上微调 Medium 模型 2 个周期，并报告词错率的变化。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 30秒窗口 | Whisper 的限制 | 硬性输入上限；需对更长的音频进行分块。 |
| SOT | 转录开始 | `<|startoftranscript|>` 启动解码器提示。 |
| 时间戳 token | 时间对齐 | 51k 词表中的每个 0.02 秒偏移量都是一个特殊 token。 |
| Turbo | 快速变体 | 4层解码器，速度快 8 倍，词错率回归 <1%。 |
| WhisperX | 长音频封装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效微调 | 在注意力机制中添加低秩适配器；训练约 0.3% 的参数。 |
| 幻觉 | 无声的失败 | Whisper 从噪声/静音中产生流畅的英文文本。 |

## 延伸阅读

-   [Radford 等 (2022). Whisper 论文](https://arxiv.org/abs/2212.04356) — 原始架构与训练方法。
-   [OpenAI (2024). Whisper Large-v3-turbo 发布](https://github.com/openai/whisper/discussions/2363) — 4层解码器，速度提升 8 倍。
-   [Bain 等 (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频、单词对齐、说话人分离。
-   [Systran — faster-whisper 代码库](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2，速度快 4 倍。
-   [HuggingFace — Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper) — 标准的 LoRA / 全参数微调指南。