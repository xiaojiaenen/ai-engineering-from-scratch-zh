# 音频评估 — WER、MOS、UTMOS、MMAU、FAD及开放排行榜

> 无法衡量，即无法发布。本课程列举了2026年各音频任务的核心评估指标：语音识别（WER、CER、RTFx）、文本转语音（MOS、UTMOS、SECS、ASR往返WER）、音频语言理解（MMAU、LongAudioBench）、音乐生成（FAD、CLAP）及说话人验证（EER）。附对比评估用排行榜。

**类型：** 学习
**语言：** Python
**前置课程：** 第6阶段 · 04、06、07、09、10；第2阶段 · 09（模型评估）
**时长：** ~60分钟

## 问题背景

每个音频任务都有多个指标，分别衡量不同维度。使用错误指标会导致模型在监控面板上表现优异，实际应用却效果糟糕。2026年标准指标清单如下：

| 任务 | 主要指标 | 次要指标 |
|------|----------|----------|
| 语音识别 | 词错误率 | 字符错误率 · 实时率 · 首字延迟 |
| 文本转语音 | 平均意见分 / UTMOS | 说话人编码余弦相似度 · ASR往返WER · 字符错误率 · 首音频延迟 |
| 语音克隆 | 说话人编码余弦相似度 | 平均意见分 · 字符错误率 |
| 说话人验证 | 等错误率 | 最小检测代价 · 工作点误拒率/误识率 |
| 说话人分离 | 分离错误率 | Jaccard错误率 · 说话人混淆率 |
| 音频分类 | top-1准确率 · 平均精度均值 | 宏观F1分数 · 各类别召回率 |
| 音乐生成 | Fréchet音频距离 | CLAP得分 · 听众评分MMAU-Pro |
| 音频语言模型 | 长音频基准 | AudioCaps FENSE |
| 流式语音到语音 | 延迟P50/P95 | 词错误率 · 平均意见分 |

## 核心概念

![音频评估矩阵——指标vs任务vs2026年排行榜](../assets/eval-landscape.svg)

### 语音识别指标

**WER（词错误率）。** `(S + D + I) / N`。评分前需转为小写、去除标点、数字规范化。使用`jiwer`或OpenAI的`whisper_normalizer`。＜5%即达到人类朗读语音水平。

**CER（字符错误率）。** 公式相同，但基于字符计算。用于汉语、粤语等分词模糊的声调语言。

**RTFx（实时率倒数）。** 每秒处理的音频时长。数值越高越好。Parakeet-TDT达3380×；Whisper-large-v3约30×。

**首字延迟。** 从音频输入到首个识别文本生成的时间。对流式应用至关重要。Deepgram Nova-3：约150毫秒。

### 文本转语音指标

**MOS（平均意见分）。** 1-5分人工评分。黄金标准但耗时长。每段音频需20+听众评分，每个模型需100+样本。

**UTMOS（2022-2026）。** 学习型MOS预测器。在标准基准上与人工MOS相关性达~0.9。F5-TTS：UTMOS 3.95；基准真值：4.08。

**SECS（说话人编码余弦相似度）。** 用于语音克隆。计算参考音频与克隆音频的ECAPA嵌入余弦相似度。＞0.75即为可识别克隆。

**ASR往返WER。** 用Whisper识别TTS输出，计算与原始文本的WER。检测可懂度退化。2026年最先进水平：＜2%字符错误率。

**TTFA（首音频延迟）。** 端到端延迟。Kokoro-82M：约100毫秒；F5-TTS：约1秒。

### 语音克隆专项指标

**SECS+MOS+CER三联评估。** SECS高但MOS低表示音色准但不自然；反之则表示自然但不像目标说话人。

### 说话人验证指标

**EER（等错误率）。** 误拒率等于误识率的阈值点。ECAPA在VoxCeleb1-O数据集：0.87%。

**minDCF（最小检测代价）。** 指定工作点（通常FAR=0.01）的加权代价。比EER更贴近生产需求。

### 说话人分离指标

**DER（分离错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。包含漏检语音、虚警语音、说话人混淆三部分错误。AMI会议数据集：DER 10-20%属合理范围。pyannote 3.1+商业方案在高质量音频上可实现＜10% DER。

**JER（Jaccard错误率）。** DER的替代指标，对短片段偏见更鲁棒。

### 音频分类指标

多标签任务：使用**mAP（平均精度均值）** 全类评估。AudioSet数据集：BEATs-iter3达0.548 mAP。

多分类任务：使用**top-1、top-5准确率**。Speech Commands v2数据集：99.0% top-1（Audio-MAE）。

类别不平衡：需报告**宏观F1分数**及**各类别召回率**。聚合准确率会掩盖特定类别的失败。

### 音乐生成指标

**FAD（Fréchet音频距离）。** 比较真实音频与生成音频的VGGish嵌入分布距离。MusicGen-small在MusicCaps数据集：4.5；MusicLM：4.0。数值越低越好。

**CLAP得分。** 使用CLAP嵌入计算的文本-音频对齐度。＞0.3即表示合理对齐。

**听众评分MMAU。** 消费级音乐的最终评判标准。Suno v5在TTS Arena的ELO评分1293（基于成对人工偏好）。

### 音频语言基准

**MMAU（大规模多模态音频理解）。** 包含10k组音频问答对。

**MMAU-Pro。** 1800个高难度项目，分四类：语音/声音/音乐/混合音频。四选一随机概率25%。Gemini 2.5 Pro总体~60%；混合音频类所有模型均~22%。

**LongAudioBench。** 包含数分钟片段的语义查询任务。Audio Flamingo Next超越Gemini 2.5 Pro。

**AudioCaps/Clotho。** 音频描述基准。采用SPICE、CIDEr、FENSE指标。

### 流式语音到语音

**延迟P50/P95/P99。** 从用户结束发言到首个可听回复的时间。Moshi：200毫秒；GPT-4o实时：300毫秒。

**输出WER/MOS。**

**抢话响应。** 从用户打断到助手静音的时间。目标＜150毫秒。

### 2026年排行榜

| 排行榜 | 覆盖领域 | 网址 |
|--------|----------|------|
| Open ASR排行榜（HF） | 英语+多语种+长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英文TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis语音 | TTS+STT，基于成对投票的ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | 音频语言模型推理 | `mmaubenchmark.github.io` |
| SpeakerBench/VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU音乐子集 | 音乐音频语言模型 | （在MMAU内） |
| HEAR基准 | 自监督音频 | `hearbenchmark.com` |

## 实操指南

### 步骤1：带规范化的WER计算

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

### 步骤2：TTS往返WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤3：语音克隆SECS评估

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤4：音乐生成FAD评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤5：说话人验证EER（代码同课程6）

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

## 应用原则

每次部署必须配套固定评估套件，随每次模型更新执行。三条黄金准则：

1. **评分前规范化。** 转小写、去标点、数字展开。明确说明所用规范规则。
2. **报告分布而非均值。** 延迟指标用P50/P95/P99。分类任务报告各类别召回率。MMAU按类别分析。
3. **运行至少一个公开标准基准。** 即使生产数据不同，在Open ASR/TTS Arena/MMAU上的评估能让对比更公平。

## 常见陷阱

- **UTMOS外推局限。** 基于VCTK风格干净语音训练，对噪声/克隆/情感音频评分不准。
- **MOS听众偏差。** 20名众包工人≠20名目标用户。关键场景应采用领域专家评分。
- **FAD依赖参考集。** 不同模型必须使用相同参考音频分布计算。
- **聚合WER误导。** 总体5%WER可能掩盖带口音语音30%的错误率。应按人口统计分组报告。
- **公开基准饱和。** 前沿模型在标准基准上已近天花板。应构建反映真实流量的内部测试集。

## 发布规范

保存为`outputs/skill-audio-evaluator.md`。为任意音频模型发布选择评估指标、基准及报告格式。

## 练习任务

1. **初级。** 运行`code/main.py`。对示例输入计算WER/CER/EER/SECS类/FAD类/MMAU类指标。
2. **中级。** 构建TTS往返WER评估工具。用Whisper识别你的Kokoro或F5-TTS输出。对50条提示计算WER，标记WER＞10%的条目。
3. **高级。** 在MMAU-Pro语音+混合音频子集（各50项）上评估你在课程10中选择的音频语言模型。报告各类别准确率并与已发表数据对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| WER | ASR评分 | 规范化后`(S+D+I)/N`在词级别的错误率 |
| CER | 字符级WER | 用于声调语言或字符级系统 |
| MOS | 人工评分 | 1-5分评级；需20+听众×100样本 |
| UTMOS | 机器MOS预测 | 学习模型；与人工MOS相关性~0.9 |
| SECS | 语音克隆相似度 | 参考音频与克隆音频的ECAPA余弦相似度 |
| EER | 说话人验证评分 | FAR=FRR时的阈值点 |
| DER | 分离评分 | (虚警+漏检+混淆)/总时长 |
| FAD | 音乐生成质量 | 基于VGGish嵌入的Fréchet距离 |
| RTFx | 吞吐量 | 每秒处理的音频时长 |

## 扩展阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带规范化工具的WER/CER计算库
- [UTMOS (Saeki等 2022)](https://arxiv.org/abs/2204.02152) — 学习型MOS预测器
- [Fréchet音频距离 (Kilgour等 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成标准指标
- [Open ASR排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026实时排名
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人工投票TTS排行榜
- [MMAU-Pro基准](https://mmaubenchmark.github.io/) — 音频语言模型推理排行榜
- [HEAR基准](https://hearbenchmark.com/) — 自监督音频基准集