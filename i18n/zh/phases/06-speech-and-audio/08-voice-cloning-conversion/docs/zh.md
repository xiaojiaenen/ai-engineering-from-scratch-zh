# 语音克隆与语音转换

> 语音克隆是用他人的声音朗读你的文本。语音转换是将你的声音改写为他人的声音，同时保留你所说的内容。两者都基于同一个分解思路：将说话人身份与内容分离。

**类型：** 构建
**语言：** Python
**前置条件：** 第6阶段 · 06（说话人识别），第6阶段 · 07（TTS）
**时间：** 约75分钟

## 问题描述

在2026年，一段5秒的音频片段就足以用消费级GPU生成任何人声音的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 等都提供零样本或少样本克隆。这项技术既是一种祝福（无障碍TTS、配音、辅助语音），也是一种武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两个密切相关的任务：

- **语音克隆（TTS侧）：** 文本 + 5秒参考语音 → 该声音的音频。
- **语音转换（语音侧）：** 源音频（A说X）+ B的参考语音 → B说X的音频。

两者都将波形分解为（内容、说话人、韵律），然后将从一个来源的内容与从另一个来源的说话人重新组合。

你现在在2026年交付时必须遵守的关键约束：**水印和同意验证在法律上是欧盟（AI法案，2026年8月生效）和加利福尼亚州（AB 2905，2025年生效）强制要求的。** 你的管道必须生成不可听水印并拒绝非自愿克隆。

## 概念

![语音克隆与转换：分解、交换说话人、重组](../assets/voice-cloning.svg)

**零样本克隆。** 向模型提供5秒片段，该模型已在数千名说话人上训练。说话人编码器将片段映射为说话人嵌入；TTS解码器以该嵌入加上文本为条件。

被以下模型使用：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**少样本微调。** 录制目标声音5-30分钟。对基础模型进行LoRA微调一小时。质量从"还行"跃升到"难以区分"。Coqui和ElevenLabs都支持这种模式；社区也将其用于F5-TTS。

**语音转换（VC）。** 两大类：

- **识别-合成。** 运行类似ASR的模型来提取内容表示（例如，软音素后验、PPG），然后用目标说话人嵌入重新合成。对语言和口音具有鲁棒性。被以下模型使用：KNN-VC (2023)、Diff-HierVC (2023)。
- **解耦。** 训练一个自编码器，在瓶颈处的潜空间中将内容、说话人和韵律分离。在推理时交换说话人嵌入。质量较低但更快。被以下模型使用：AutoVC (2019)、VITS-VC变体。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox —— 将音频视为来自SoundStream / EnCodec的离散token，在codec token上训练大型自回归或流匹配模型。质量与ElevenLabs在短提示上相当。

### 伦理部分，不是附加功能

**水印。** PerTh (Perth) 和 SilentCipher (2024) 在音频中嵌入约16-32位ID，不可察觉。能经受重新编码、流式传输和常见编辑。已可用于生产环境。

**同意门控。** 必须将每条克隆输出与可验证的同意记录配对。"我，Rohit，于2026-04-22授权此语音用于X目的。" 存储在防篡改日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发布。ASVspoof 2025挑战赛针对ElevenLabs、VALL-E 2和Bark输出，发布了EER为0.8–2.3%的最先进检测器结果。

### 数据（2026年）

| 模型 | 零样本？ | SECS（目标相似度） | WER（智力测试） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众来说通常与目标难以区分。

```figure
sp-voice-factorize
```

## 构建它

### 步骤1：用识别-合成分解（main.py中的纯代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    # 从参考语音提取说话人嵌入
    speaker_emb = target_embedder.encode(ref_audio)
    # 用目标说话人嵌入和文本生成mel频谱
    mel = tts_model(text, speaker=speaker_emb)
    # 通过声码器转换为波形
    return vocoder(mel)
```

概念简单；实现复杂度在于 `tts_model` 和说话人编码器。

### 步骤2：用F5-TTS进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须与音频完全匹配；不匹配会破坏对齐。

### 步骤3：用KNN-VC进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023模型，https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC运行WavLM为源和目标池提取逐帧嵌入，然后用池中最近邻替换每个源帧。非参数化方法，只需一分钟目标语音即可工作。

### 步骤4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
# 嵌入包含同意ID和时间戳的消息
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
# 检测水印以验证完整性
detected = sc.detect(watermarked, sr=24000)   # 返回载荷字节
```

约32位载荷，经过MP3重新编码和轻微噪声后仍可检测。

### 步骤5：同意门控

```python
def cloned_inference(text, ref_audio, consent_record):
    # 验证签名确保同意有效
    assert verify_signature(consent_record), "需要签名同意"
    # 确保说话人与引用一致
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    # 嵌入水印以防止滥用
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用它

2026年技术栈：

| 场景 | 选择 |
|-----------|------|
| 5秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **参考转录未对齐。** F5-TTS 和类似模型要求参考文本与参考音频完全匹配，包括标点符号。
- **有混响的参考。** 回声会破坏克隆。录制干声，近距离拾音。
- **情感不匹配。** 训练参考"欢快"会产生对一切内容的欢快克隆。使参考情感与目标用途匹配。
- **语言泄漏。** 克隆英语说话人后要求模型说法语，通常仍会带有口音；使用跨语言模型（XTTS、VALL-E X）。
- **无水印。** 从2026年8月起在欧盟无法合法发货。

## 交付

保存为 `outputs/skill-voice-cloner.md`。设计具有同意门控 + 水印 + 质量目标的克隆或转换管道。

## 练习

1. **简单。** 运行 `code/main.py`。通过计算交换前后两个"说话人"之间的余弦相似度来演示说话人嵌入交换。
2. **中等。** 使用OpenVoice v2克隆你自己的声音。测量参考与克隆之间的SECS。通过Whisper测量CER。
3. **困难。** 对20个克隆应用SilentCipher水印，将它们通过128 kbps MP3编解码，检测载荷。报告比特准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆 | 5秒就足够 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | 音素后验图 | 逐帧ASR后验，用作语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 用目标池最近帧替换每个源帧。 |
| 神经编解码器TTS | VALL-E风格 | 在EnCodec/SoundStream token上的AR模型。 |
| 水印 | 不可听签名 | 嵌入音频中的比特，能经受重新编码。 |
| SECS | 克隆保真度 | 目标与克隆说话人嵌入之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen 等 (2024)。F5-TTS](https://arxiv.org/abs/2410.06885) —— 开源SOTA零样本克隆。
- [Baevski 等 / 微软 (2023)。VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) —— 神经编解码器TTS。
- [Qian 等 (2019)。AutoVC](https://arxiv.org/abs/1905.05879) —— 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023)。KNN-VC](https://arxiv.org/abs/2305.18975) —— 基于检索的VC。
- [SilentCipher (2024) —— 音频水印](https://github.com/sony/silentcipher) —— 可生产使用的32位音频水印。
- [ASVspoof 2025 结果](https://www.asvspoof.org/) —— 检测器与合成器军备竞赛，2026年更新。
