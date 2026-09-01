# 语音识别 (ASR) — CTC, RNN-T, Attention

> 语音识别是在每个时间步进行音频分类，并通过一个懂英语和静音的序列模型将其串联起来。CTC、RNN-T 和 Attention 是三种实现方式。选一种并理解为什么。

**类型:** Build
**语言:** Python
**前置知识:** Phase 6 · 02 (频谱图与 Mel), Phase 5 · 08 (用于文本的 CNN 和 RNN), Phase 5 · 10 (Attention)
**时间:** ~45 分钟

## 问题

你有一段 10 秒的 16 kHz 音频。你想要一个字符串："turn on the kitchen lights"。挑战在于结构性：音频帧与字符并不是一一对应的关系。单词 "okay" 可能持续 200 毫秒或 1200 毫秒。静音贯穿整个语音。有些音素比其他的更长。输出 token 的数量在事前是未知的。

三种方式可以解决这个问题：

1. **CTC (连接时序分类)。** 每帧输出 token 概率，包括一个特殊的 *空白 (blank)*。在解码时折叠重复项和空白。非自回归，速度快。被 wav2vec 2.0、MMS 使用。
2. **RNN-T (循环神经网络转导器)。** 联合网络根据编码器帧和之前的 token 预测下一个 token。可流式传输。被 Google 的设备端 ASR、NVIDIA Parakeet 使用。
3. **Attention 编码器-解码器。** 编码器将音频压缩为隐状态，解码器通过交叉注意力自回归生成 token。被 Whisper、SeamlessM4T 使用。

在 2026 年，LibriSpeech test-clean 上的 SOTA WER 为 1.4% (Parakeet-TDT-1.1B, NVIDIA) 和 1.58% (Whisper-Large-v3-turbo)。差异很小；部署差异却巨大。

## 概念

![三种 ASR 公式：CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器在 `V+1` 个 token (V 个字符 + 空白) 上输出 `T` 帧级别分布。对于长度为 `U < T` 的目标字符串 `y`，任何折叠后得到 `y` 的帧对齐都算作有效。CTC 损失对所有此类对齐求和。推理：每帧 argmax，折叠重复项，移除空白。

优点：非自回归，可流式，零前瞻。缺点：*条件独立性假设* — 每个帧预测独立于其他帧，因此没有内部语言模型。通过 beam search 或浅层融合 (shallow fusion) 使用外部 LM 来解决。

**RNN-T 直觉。** 增加一个嵌入 token 历史的 *预测器 (predictor)* 网络和一个将预测器状态与编码器帧结合以生成 `V+1` 上联合分布的 *联合器 (joiner)* (`+1` 是空值 / 不发射)。显式建模了 CTC 忽略的条件依赖。可流式是因为每一步仅依赖于过去的帧和过去的 token。

优点：可流式 + 内部 LM。缺点：训练更复杂且内存占用高 (3D 损失格)；RNN-T 损失核本身就是一个完整的库类别。

**Attention 编码器-解码器。** 在 log-mel 帧上的编码器 (6-32 个 transformer 层)。解码器 (6-32 个 transformer 层) 对编码器输出进行交叉注意力以自回归生成 token。没有对齐约束 — 注意力可以查看音频的任何位置。除非限制注意力 (chunked Whisper-Streaming, 2024)，否则不可流式。

优点：离线 ASR 最高质量，使用标准 seq2seq 工具链易于训练。缺点：自回归延迟与输出长度成正比；不进行工程改造无法流式传输。

### WER：那个唯一的数字

**词错误率 (Word Error Rate)** = `(S + D + I) / N`，其中 S=替换，D=删除，I=插入，N=参考词数。匹配词级别的 Levenshtein 编辑距离。越低越好。WER 高于 20% 通常不可用；低于 5% 对于朗读语音已达到人类水平。2026 年标准基准上的数字：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 大小 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是基于 encoder-decoder 或 RNN-T 的。纯 CTC 系统 (wav2vec 2.0) 在 test-clean 上约为 1.8–2.1%。

```figure
ctc-collapse
```

## 构建它

### 步骤 1：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧概率向量列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复项，丢弃空白。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：beam-search CTC

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

生产环境使用前缀树 beam search 配合 LM 融合；这是概念骨架。

### 步骤 3：WER

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

### 步骤 4：针对 Whisper 的推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20× 实时速度运行。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 进行流式传输

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力并携带状态；使用支持此功能的库 (Parakeet 用 NeMo，`transformers` pipeline 用 `chunk_length_s`)。

## 使用它

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语，离线，最高质量 | Whisper-large-v3-turbo |
| 多语言，鲁棒 | SeamlessM4T v2 |
| 流式，低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘，移动设备，延迟 <500 ms | 量化 Whisper-Tiny 或 Moonshine (2024) |
| 长音频 | 带 VAD 分块的 Whisper (WhisperX) |
| 特定领域 (医疗、法律) | 微调 wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍会踩的坑

- **没有 VAD。** 在静音上运行 Whisper 会产生幻觉 ("Thanks for watching!")。始终用 VAD 门控。
- **词 vs 字 vs 子词 WER。** 报告经过归一化 (小写、去除标点) 后的词级 WER。
- **语言 ID 漂移。** Whisper 的自动 LID 会将嘈杂片段误路由到日语或威尔士语；当你确定时强制 `language="en"`。
- **不分块的长音频。** Whisper 有 30 秒窗口。超过此长度请使用 `chunk_length_s=30, stride=5`。

## 交付它

保存为 `outputs/skill-asr-picker.md`。针对给定的部署目标选择模型、解码策略、分块和 LM 融合。

## 练习

1. **简单。** 运行 `code/main.py`。它对人工构造的 CTC 输出进行贪心解码并计算相对于参考的 WER。
2. **中等。** 正确实现步骤 2 中的前缀树 beam search (考虑空白合并规则)。在 10 个示例的合成数据集上与贪心算法进行比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 个语句的 WER。与已发布的数字进行比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 空白 token 损失 | 对所有帧到 token 对齐的边缘分布；非自回归 (Non-AR)。 |
| RNN-T | 流式损失 | CTC + 下一个 token 预测器；处理词序。 |
| Attention enc-dec | Whisper 风格 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 你报告的数字 | 词级的 `(S+D+I)/N`。 |
| Blank | 空无 | CTC 中的特殊 token，表示"此帧无发射"。 |
| LM fusion | 外部语言模型 | 在 beam search 期间添加加权 LM 对数概率。 |
| VAD | 静音门 | 语音活动检测器；修剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年的经典论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年开放 ASR 排行榜领先者。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 跨 25+ 个模型的实时基准测试。
