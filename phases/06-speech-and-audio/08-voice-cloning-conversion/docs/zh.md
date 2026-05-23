# 语音克隆与语音转换

> 语音克隆用他人的声音朗读你的文本。语音转换在保留原意的前提下，将你的声音重写为他人的声音。二者共同依赖一个基本原理：将说话人身份与语音内容分离。

**类型：** 构建项目
**语言：** Python
**前置知识：** 第六阶段 · 06（说话人识别），第六阶段 · 07（TTS）
**所需时间：** 约75分钟

## 问题背景

2026年，仅需5秒音频片段，配合消费级GPU即可生成任何人声的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox均已实现零样本或少样本克隆。这项技术既是福音（无障碍TTS、配音、辅助语音）也是武器（诈骗电话、政治深度伪造、知识产权侵权）。

两项密切相关的任务：
- **语音克隆（TTS端）：** 文本 + 5秒参考语音 → 该声音合成的音频。
- **语音转换（语音端）：** 源音频（人物A说X） + 人物B的参考语音 → 人物B说X的音频。

两者都将声波分解为（内容、说话人、韵律）成分，并将来自不同来源的内容与说话人重新组合。

2026年需遵守的关键约束：**水印和同意机制在欧盟（《人工智能法案》，2026年8月强制执行）和加利福尼亚州（AB 2905法案，2025年生效）具有法律强制性**。您的处理流程必须嵌入不可听水印并拒绝未经同意的克隆。

## 核心概念

![语音克隆与转换对比：分解、替换说话人、重新组合](../assets/voice-cloning.svg)

**零样本克隆。** 将5秒音频片段输入经过数千说话人训练的模型。说话人编码器将片段映射为说话人嵌入向量；TTS解码器基于该嵌入向量和文本进行条件生成。

使用者：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制目标声音5-30分钟音频。通过LoRA微调基础模型约一小时。质量从“尚可”跃升至“难以区分”。Coqui和ElevenLabs均支持此模式；社区将其用于F5-TTS。

**语音转换（VC）两类方法：**
- **识别-合成法。** 运行类似ASR的模型提取内容表征（如软音素后验概率、PPG），然后用目标说话人嵌入重新合成。对语言和口音具有鲁棒性。使用者：KNN-VC（2023）、Diff-HierVC（2023）。
- **解耦法。** 训练自编码器，在瓶颈层分离潜空间中的内容、说话人和韵律。推理时替换说话人嵌入向量。质量较低但速度更快。使用者：AutoVC（2019）、VITS-VC变体。

**神经编解码器克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox —— 将音频视为来自SoundStream/EnCodec的离散token，训练大型自回归或流匹配模型处理编解码器token。短提示词下的质量可与ElevenLabs媲美。

### 伦理要素，非附加功能

**水印技术。** PerTh（Perth）和SilentCipher（2024）可在音频中不可察觉地嵌入16-32位标识符。能承受重新编码、流媒体传输和常规编辑。生产就绪的开源方案。

**同意机制。** 必须将每个克隆输出与可验证的同意记录配对。“本人Rohit于2026年4月22日授权此声音用于X用途。”存储于防篡改日志。

**检测工具。** AASIST、RawNet2和Wav2Vec2-AASIST作为检测器发布。ASVspoof 2025挑战赛显示，最先进检测器针对ElevenLabs、VALL-E 2和Bark输出的等错误率为0.8-2.3%。

### 数据指标（2026年）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 3.35亿 |
| XTTS v2 | 是 | 0.65 | 3.5% | 4.70亿 |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 2.20亿 |
| VALL-E 2 | 是 | 0.77 | 2.4% | 3.70亿 |
| VoiceBox | 是 | 0.78 | 2.1% | 3.30亿 |

SECS > 0.70时对大多数听者而言通常与目标声音无法区分。

## 动手实现

### 步骤1：通过识别-合成法分解（main.py中的纯代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念简单；实现复杂度集中在`tts_model`和说话人编码器。

### 步骤2：使用F5-TTS进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考文本必须与音频精确匹配；不匹配会破坏对齐。

### 步骤3：使用KNN-VC进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC运行WavLM提取源音频和目标语音池的逐帧嵌入向量，然后用目标池中的最近邻帧替换每个源帧。非参数方法，仅需一分钟目标语音即可工作。

### 步骤4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约32位有效载荷，经MP3重编码和轻度噪声处理后仍可检测。

### 步骤5：设置同意机制

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 应用指南

2026年技术栈：

| 场景 | 推荐方案 |
|-----------|------|
| 5秒零样本克隆，开源方案 | F5-TTS或OpenVoice v2 |
| 商业生产级克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写） | KNN-VC或Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2或VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 常见陷阱

- **参考文本不匹配。** F5-TTS等模型要求参考文本与参考音频完全匹配，包括标点符号。
- **混响过重的参考音频。** 回声会破坏克隆效果。请在干燥环境中近麦录音。
- **情感不匹配。** 情感“欢快”的参考音频会生成所有内容的欢快克隆。需匹配参考情感与目标用途。
- **语言泄露。** 克隆英语说话人后要求模型说法语时，通常仍会带有口音；请使用跨语言模型（XTTS、VALL-E X）。
- **无水印。** 自2026年8月起在欧盟法律上不可发布。

## 部署建议

保存为`outputs/skill-voice-cloner.md`。设计包含同意机制 + 水印 + 质量目标的克隆或转换流程。

## 练习任务

1. **入门级。** 运行`code/main.py`。通过计算交换前后两个“说话人”的余弦相似度，演示说话人嵌入向量的替换。
2. **进阶级。** 使用OpenVoice v2克隆自己的声音。测量参考音频与克隆音频间的SECS。通过Whisper测量字符错误率。
3. **挑战级。** 对20个克隆样本应用SilentCipher水印，经过128 kbps MP3编码+解码处理后检测有效载荷。报告比特准确率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆 | 5秒足够 | 预训练模型 + 说话人嵌入向量；无需训练。 |
| PPG | 音素后验图 | 逐帧ASR后验概率，用作语言无关的内容表征。 |
| KNN-VC | 最近邻转换 | 用目标语音池中最近邻帧替换每个源帧。 |
| 神经编解码器TTS | VALL-E风格 | 基于EnCodec/SoundStream token的自回归模型。 |
| 水印 | 不可听签名 | 嵌入音频中的比特信息，能承受重编码。 |
| SECS | 克隆保真度 | 目标与克隆说话人嵌入向量间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 扩展阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源最先进的零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码器TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的语音转换。
- [SilentCipher (2024) — 音频水印](https://github.com/sony/silentcipher) — 生产就绪的32位音频水印。
- [ASVspoof 2025结果](https://www.asvspoof.org/) — 检测器与合成器军备竞赛，2026年更新。