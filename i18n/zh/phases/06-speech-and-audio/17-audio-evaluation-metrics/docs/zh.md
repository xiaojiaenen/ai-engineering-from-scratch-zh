# 音频评估 — WER、MOS、UTMOS、MMAU、FAD 以及公开排行榜

> 你无法交付你无法衡量的东西。本课列出 2026 年所有音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、音频语言模型（MMAU、LongAudioBench）、音乐（FAD、CLAP）以及说话人（EER）。还有你可以在其中比较的排行榜。

**类型：** 学习
**语言：** Python
**先修：** Phase 6 · 04、06、07、09、10；Phase 2 · 09（模型评估）
**时间：** 约 60 分钟

## 问题所在

每个音频任务都有多个指标，每个指标衡量不同的维度。使用错误的指标会导致模型在你的仪表盘上看起来很棒，但在生产环境中表现糟糕。2026 年的标准列表如下：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 首 token 延迟 |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| 声音克隆 | SECS (ECAPA 余弦相似度) | MOS · CER |
| 说话人验证 | EER | minDCF · 工作点处的 FAR / FRR |
| 说话人分离 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | 宏平均 F1 · 各类别召回率 |
| 音乐生成 | FAD | CLAP · 听评小组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 概念

![音频评估矩阵 — 指标 vs 任务 vs 2026 排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率）。** `(S + D + I) / N`。评分前需小写、去标点、归一化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。低于 5% = 类人朗读水平。

**CER（字符错误率）。** 公式相同，但在字符级别计算。用于声调语言（普通话、粤语），因为分词存在歧义。

**RTFx（实时因子的倒数）。** 每墙钟秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**首 token 延迟。** 从音频输入到第一个转录 token 的墙钟时间。对流场景至关重要。Deepgram Nova-3：约 150 ms。

### TTS 指标

**MOS（平均意见得分）。** 1-5 分的人类评分。黄金标准但耗时。每个样本收集 20+ 位听评员，每个模型 100+ 个样本。

**UTMOS（2022-2026）。** 学习得到的 MOS 预测器。在标准基准上与人类 MOS 相关系数约 0.9。F5-TTS：UTMOS 3.95；真实语音：4.08。

**SECS（说话人编码器余弦相似度）。** 用于声音克隆。参考音频与克隆输出的 ECAPA 嵌入余弦相似度。大于 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip。** 用 Whisper 对 TTS 输出进行转录，计算与输入文本的 WER。可发现清晰度退化。2026 SOTA：CER < 2%。

**TTFA（首音频到达时间）。** 墙钟延迟。Kokoro-82M：约 100 ms；F5-TTS：约 1 秒。

### 声音克隆专用指标

**SECS + MOS + CER** 作为三重评估。克隆得分 SECS 高但 MOS 低，意味着音色正确但不自然；反之则声音自然但说话人不匹配。

### 说话人验证

**EER（等错误率）。** 假接受率等于假拒绝率时的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF（最小检测代价）。** 在选定工作点（通常为 FAR=0.01）的加权代价。比 EER 更具生产相关性。

### 说话人分离

**DER（分离错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。遗漏语音 + 虚警语音 + 说话人混淆，各占总比例的分数。AMI 会议：DER 约 10-20% 是现实的。pyannote 3.1 + Precision-2 商业版：良好录制音频上 DER < 10%。

**JER（Jaccard 错误率）。** DER 的替代方案，对短片段偏差更鲁棒。

### 音频分类

多标签：**mAP（平均精度均值）** 在所有类别上。AudioSet：BEATs-iter3 达到 0.548 mAP。

多分类互斥：**top-1、top-5 准确率**。Speech Commands v2：99.0% top-1（Audio-MAE）。

不平衡数据：**宏平均 F1** + **各类别召回率**。报告各类别 — 聚合准确率会掩盖哪些类别表现差。

### 音乐生成

**FAD（Fréchet 音频距离）。** 真实音频与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP 分数。** 使用 CLAP 嵌入的文本-音频对齐分数。大于 0.3 = 合理对齐。

**听评小组 MOS。** 消费级音乐的最后评判。Suno v5 在 TTS Arena 上 ELO 1293（基于成对人类偏好）。

### 音频语言模型基准

**MMAU（大规模多音频理解）。** 1 万个音频-QA 对。

**MMAU-Pro。** 1800 个难题，四个类别：speech / sound / music / multi-audio。4 选 1 随机猜测为 25%。Gemini 2.5 Pro 总体约 60%；multi-audio 各类模型均约 22%。

**LongAudioBench。** 带语义查询的多分钟片段。Audio Flamingo Next 超过 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 字幕生成基准。SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户语音结束到第一个可听响应的墙钟时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

**输出端的 WER / MOS。**

**插话响应性。** 从用户打断到助手静音的时间。目标 < 150 ms。

### 2026 排行榜

| 排行榜 | 赛道 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 英语 + 多语言 + 长段落 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，基于成对投票的 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU music subset | 音乐 LALM | （在 MMAU 内） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

```figure
sp-wer-align
```

## 动手实践

### 步骤 1：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤 2：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：声音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：说话人验证的 EER（与第 6 课相同代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用建议

每次部署时配一个固定的评估框架，在每个模型更新时运行。三条核心原则：

1. **评分前先归一化。** 小写、去标点、展开数字。报告归一化规则。
2. **报告分布而非平均值。** 延迟用 P50/P95/P99。分类用各类别召回率。MMAU 用各类别分开报告。
3. **运行一个标准的公开基准。** 即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告也让评审者可以公平比较。

## 陷阱

- **UTMOS 外推风险。** 在 VCTK 式干净语音上训练；对噪声/克隆/情感音频评分不佳。
- **MOS 小组偏差。** 20 名 Amazon Mechanical Turk 工人 ≠ 20 名目标用户。如果 stakes 高，请聘请领域专家小组。
- **FAD 依赖参考集。** 在不同模型间比较时使用相同的参考分布。
- **聚合 WER 的欺骗性。** 整体 5% WER 可能掩盖口音语音的 30% WER。按人口统计切片报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上已接近天花板。构建一个反映你真实流量的内部保留集。

## 交付

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **中等。** 构建 TTS 往返 WER 评估框架。将你的 Kokoro 或 F5-TTS 输出通过 Whisper。在 50 个提示上计算 WER。标记 WER > 10% 的提示。
3. **困难。** 在第 10 课的 LALM 选择上，用 MMAU-Pro 的 speech + multi-audio 子集（各 50 项）评分。报告各类别准确率并与已发布数字比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR 分数 | 归一化后词级别的 `(S+D+I)/N`。 |
| CER | 字符级 WER | 用于声调语言或字符级系统。 |
| MOS | 人工意见 | 1-5 分评分；20+ 听评员 × 100 样本。 |
| UTMOS | ML MOS 预测器 | 学习到的模型；与人工 MOS 相关系数约 0.9。 |
| SECS | 声音克隆相似度 | 参考音频与克隆之间的 ECAPA 余弦相似度。 |
| EER | 说话人验证分数 | FAR = FRR 时的阈值。 |
| DER | 分离评分 | (FA + Miss + Confusion) / total。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每墙钟秒处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带归一化工具的 WER/CER 库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学习得到的 MOS 预测器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成的标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人工投票的 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) — 音频 SSL 基准。
