# 构建语音助手管道 —— 第6阶段毕业设计项目

> 综合第1-11课全部内容。构建一个能听、能思考、能回应的语音助手。在2026年，这是一个已解决的工程问题，而非研究问题——但集成细节决定它能否成功上线。

**类型：** 构建练习
**语言：** Python
**前置知识：** 第6阶段 · 04、05、06、07、11；第11阶段 · 09（函数调用）；第14阶段 · 01（Agent循环）
**预计时长：** ~120分钟

## 问题描述

构建端到端的语音助手系统：

1. 捕获麦克风输入（16 kHz单声道）。
2. 检测用户语音的起止。
3. 流式转录语音。
4. 将转录文本传给可调用工具的LLM（定时器、天气、日历）。
5. 将LLM生成的文本流式传给TTS。
6. 播放音频回复给用户。
7. 如果用户在回答过程中打断，立即停止回复。

延迟目标：在笔记本电脑CPU上，用户说完话后800ms内输出第一个TTS音频字节。质量目标：不漏字、沉默时不产生幻觉字幕、无声音克隆泄露、不受提示词注入攻击。

## 概念框架

![语音助手管道：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个核心组件

1. **音频采集。** 麦克风 → 16 kHz单声道 → 20ms分块。Python中通常使用`sounddevice`，生产环境中使用原生AudioUnit/ALSA/WASAPI。
2. **VAD（第11课）。** Silero VAD，阈值0.5，最小语音长度250ms，静音延续500ms。用于标记语音"开始"和"结束"。
3. **流式STT（第4-5课）。** Whisper-streaming、Parakeet-TDT或Deepgram Nova-3（API）。输出部分转录和最终转录。
4. **带工具调用的LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。使用JSON schema定义工具。流式输出token。
5. **流式TTS（第7课）。** Kokoro-82M（最快开源方案）或Cartesia Sonic（商业方案）。收到20个LLM token后开始TTS合成。
6. **音频播放。** 扬声器输出；低带宽网络使用opus编码。
7. **中断处理。** 如果TTS播放期间VAD触发，立即停止播放、取消LLM生成、重启STT。

### 你会遇到的三种故障模式

1. **首词截断。** VAD触发延迟了一拍。用户的"嘿"字被丢失。将阈值设为0.3而非0.5。
2. **响应中打断混淆。** 用户打断后LLM仍在生成；助手与用户重叠说话。需要将VAD信号连接到LLM取消逻辑。
3. **静音幻觉。** Whisper在静音预热帧上输出"感谢观看"。必须始终通过VAD门控。

### 2026年生产环境参考方案

| 方案 | 延迟 | 许可证 | 说明 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业API | 2026年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms |  mostly开源 | 适合DIY |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，见第15课 |
| Vapi / Retell（托管服务） | 300-500 ms | 商业 | 上线最快；自定义受限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私/边缘场景 |

```figure
v4-voice-latency
```

## 动手实现

### 步骤1：带分块的麦克风采集（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤2：VAD门控的对话轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤3：流式STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤4：LLM循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤5：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用方法

参见 `code/main.py`，其中包含一个可运行的模拟程序，将所有七个组件用存根模型连接起来，即使没有硬件也能看到管道形状。对于真实实现，请替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（gpt-4o）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于I/O

## 常见陷阱

- **永久记录PII。** 完整轮次音频在大多数司法管辖区属于个人身份信息。保留30天，静态加密。
- **不支持打断。** 用户会打断。你的助手必须能够停止说话。
- **阻塞式TTS。** 同步TTS会阻塞事件循环。使用异步或单独线程。
- **缺少工具调用错误处理。** 工具会失败。LLM必须收到错误信息并重试一次，然后优雅降级。
- **过度激进的幻觉过滤。** 过滤过度会导致助手重复"我帮不了你"。过滤不足会导致胡言乱语。在预留测试集上调校。
- **缺少唤醒词选项。** 始终监听是隐私隐患。添加唤醒词门控（Porcupine或openWakeWord）。

## 交付物

保存为 `outputs/skill-voice-assistant-architect.md`。根据预算、规模、语言和合规约束，输出完整的技术栈规格说明。

## 练习题

1. **简单。** 运行 `code/main.py`。它模拟一个完整的端到端轮次，使用存根模块并打印各阶段延迟。
2. **中等。** 用预录制的`.wav`文件替换STT存根，使用真实的Whisper模型。测量WER和端到端延迟。
3. **困难。** 添加工具调用：实现`get_weather`（任意API）和`set_timer`。让LLM通过工具路由，验证当用户说"设定5分钟计时器"时，正确的函数被触发且语音回复确认了这一操作。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 轮次（Turn） | 用户+助手往返 | 一次VAD界定的用户语音 + 一次LLM-TTS回复 |
| 打断（Barge-in） | 用户插话 | 用户在助手说话时开口；助手停止 |
| 唤醒词（Wake word） | "Hey assistant" | 短关键词检测器；Porcupine、Snowboy、openWakeWord |
| 终点检测（End-pointing） | 轮次结束 | VAD + 最小静音判断用户已说完 |
| 预滚动（Pre-roll） | 语音前缓冲区 | 保留VAD触发前200-400ms的音频以避免首词截断 |
| 工具调用（Tool call） | 函数调用 | LLM输出JSON；运行时分发；结果反馈到循环中 |

## 延伸阅读

- [LiveKit — 语音Agent快速入门](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — 语音Agent示例](https://github.com/pipecat-ai/pipecat) — 适合DIY的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管式原生语音方案。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第15课）。
- [Porcupine 唤醒词](https://picovoice.ai/products/porcupine/) — 唤醒词门控。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM函数调用。
