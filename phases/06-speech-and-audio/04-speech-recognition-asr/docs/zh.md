# 语音识别 (ASR) — CTC, RNN-T, 注意力机制

> 语音识别是每个时间步的音频分类，通过一个理解英语和静音的序列模型粘合在一起。CTC、RNN-T 和注意力机制是实现这一目标的三种方法。选择其中一个并理解其原理。

**类型:** 构建
**语言:** Python
**先修课程:** 第 6 阶段 · 02（频谱图与梅尔频率倒谱系数），第 5 阶段 · 08（用于文本的 CNN 和 RNN），第 5 阶段 · 10（注意力机制）
**时间:** ~45 分钟

## 问题描述

你有一段 10 秒 16 kHz 的音频片段。你想要一个字符串：“turn on the kitchen lights”。挑战在于结构性：音频帧与字符并非一一对应。单词 "okay" 可能占据 200 毫秒或 1200 毫秒。静音分隔了语句。某些音素比其他音素更长。输出 token 的数量是未知的。

三种解决方案：

1.  **CTC (连接时序分类)。** 在每一帧发射 token 概率，包括一个特殊的*空白* token。在解码时合并重复项和空白。非自回归，速度快。用于 wav2vec 2.0, MMS。
2.  **RNN-T (循环神经网络转换器)。** 联合网络根据编码器帧和之前的 token 预测下一个 token。可流式处理。用于谷歌的设备端 ASR、NVIDIA Parakeet。
3.  **注意力编码器-解码器。** 编码器将音频压缩为隐藏状态，解码器通过交叉注意力自回归地生成 token。用于 Whisper、SeamlessM4T。

截至 2026 年，LibriSpeech test-clean 上的最先进 WER 为 1.4% (Parakeet-TDT-1.1B, NVIDIA) 和 1.58% (Whisper-Large-v3-turbo)。差异很小；但部署差异巨大。

## 核心概念

![三种 ASR 公式：CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 帧级分布，覆盖 `V+1` 个 token（V 个字符 + 空白）。对于一个长度为 `U < T` 的目标字符串 `y`，任何能合并为 `y` 的帧对齐都算作有效。CTC 损失函数对所有此类对齐求和。推理：逐帧取 argmax，合并重复项，移除空白。

优点：非自回归、可流式、零前瞻。缺点：*条件独立假设* — 每一帧的预测独立于其他帧，因此没有内置的语言模型。可以通过波束搜索或浅层融合添加外部语言模型来弥补。

**RNN-T 直觉。** 增加了一个*预测器*网络来嵌入 token 历史，以及一个*连接器*，将预测器状态与编码器帧组合成 `V+1` 上的联合分布（其中 `+1` 表示空/不发射）。明确建模了 CTC 忽略的条件依赖性。可流式，因为每一步只依赖于过去的帧和过去的 token。

优点：可流式 + 内置语言模型。缺点：训练更复杂且内存消耗大（三维损失晶格）；RNN-T 损失函数内核本身就是一个完整的库类别。

**注意力编码器-解码器。** 编码器（6-32 层 Transformer）处理对数梅尔频谱帧。解码器（6-32 层 Transformer）通过交叉注意力关注编码器输出，自回归地生成 token。没有对齐约束——注意力可以关注音频中的任何位置。除非限制注意力（分块 Whisper-Streaming，2024），否则不可流式。

优点：离线 ASR 质量最高，易于使用标准 seq2seq 工具训练。缺点：自回归延迟与输出长度成正比；未经工程改造则无法流式处理。

### WER：唯一的数字指标

**词错误率** = `(S + D + I) / N`，其中 S=替换错误，D=删除错误，I=插入错误，N=参考词数。在词级别上匹配 Levenshtein 编辑距离。越低越好。高于 20% 的 WER 通常不可用；低于 5% 对于朗读语音已达到人类水平。2026 年在标准基准上的数据：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 参数量 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是基于编码器-解码器或 RNN-T 的系统。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 构建它

### 步骤 1：贪婪 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：合并连续重复项，丢弃空白。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：波束搜索 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带语言模型融合的前缀树波束搜索；这里是概念骨架。

### 步骤 3：计算 WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：使用 Whisper 进行推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的单行代码。在 24 GB GPU 上以约 20 倍实时速度运行。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 进行流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和状态传递；使用支持该功能的库（对于 Parakeet 使用 NeMo，`transformers` 管线配合 `chunk_length_s`）。

## 使用它

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语，离线，最高质量 | Whisper-large-v3-turbo |
| 多语言，鲁棒性强 | SeamlessM4T v2 |
| 流式，低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘设备，移动，延迟 <500 毫秒 | Whisper-Tiny 量化版或 Moonshine (2024) |
| 长音频 | 结合 VAD 分块的 Whisper (WhisperX) |
| 特定领域（医疗、法律） | 微调 wav2vec 2.0 + 领域语言模型融合 |

## 2026 年仍然会犯的错误

- **没有 VAD。** 在静音上运行 Whisper 会产生幻觉（“Thanks for watching!”）。务必使用 VAD 进行门控。
- **字符级、词级还是子词级 WER。** 报告*归一化后*（小写，去除标点）的词级 WER。
- **语言标识漂移。** Whisper 的自动语言标识会将嘈杂片段误判为日语或威尔士语；当已知语言时，强制指定 `language="en"`。
- **长片段未分块。** Whisper 有 30 秒的窗口。对于更长的音频，使用 `chunk_length_s=30, stride=5`。

## 部署它

保存为 `outputs/skill-asr-picker.md`。根据给定的部署目标，选择模型、解码策略、分块和语言模型融合。

## 练习

1.  **简单。** 运行 `code/main.py`。它贪婪地解码一个手工制作的 CTC 输出，并与参考文本计算 WER。
2.  **中等。** 正确实现步骤 2 中的前缀树波束搜索（考虑空白合并规则）。在一个包含 10 个示例的合成数据集上与贪婪解码进行比较。
3.  **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。对前 100 条话语计算 WER。与公布的数字进行比较。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|-----------------------|
| CTC | 空白 token 损失 | 对所有帧到 token 对齐的边际化；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一个 token 预测器；处理词序。 |
| 注意力编码器-解码器 | Whisper 风格 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 你报告的那个数字 | `(S+D+I)/N` 在词级别上的值。 |
| 空白 | 空 | CTC 中的特殊 token，表示“此帧无发射”。 |
| LM 融合 | 外部语言模型 | 在波束搜索期间添加加权的语言模型对数概率。 |
| VAD | 静音门控 | 语音活动检测器；修剪非语音部分。 |

## 扩展阅读

- [Graves 等人 (2006). 连接时序分类](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). 使用 RNN 进行序列转导](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford 等人 / OpenAI (2022). Whisper: 通过大规模弱监督实现鲁棒语音识别](https://arxiv.org/abs/2212.04356) — 2022 年经典论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT 卡片](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年开源 ASR 排行榜榜首。
- [Hugging Face — 开源 ASR 排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 覆盖 25+ 个模型的实时基准测试。