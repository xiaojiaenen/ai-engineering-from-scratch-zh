# 构建语音助手管道 —— 第六阶段顶石项目

> 将第01-11课的所有内容拼接整合。构建一个能听、能思考、能回应的语音助手。在2026年，这已是一个工程实现问题，而非研究课题——但集成细节决定了它能否成功部署。

**类型：** 构建项目
**语言：** Python
**前置要求：** 第六阶段 · 04、05、06、07、11课；第十一阶段 · 09课（函数调用）；第十四阶段 · 01课（智能体循环）
**时间：** 约120分钟

## 问题描述

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始与结束。
3. 进行流式转录。
4. 将转录文本传给可调用工具（计时器、天气、日历）的大语言模型。
5. 将大语言模型输出的文本流式传输给文本转语音系统。
6. 将音频回放给用户。
7. 若用户在助手回应中途打断，则停止当前回应。

延迟目标：在笔记本CPU上，用户说完话后800毫秒内输出首个文本转语音音频字节。质量目标：无漏词、无静默时产生幻听字幕、无语音克隆泄露、无提示注入成功。

## 核心概念

![语音助手管道：麦克风 → 语音活动检测 → 语音转文本 → 大语言模型+工具 → 文本转语音 → 扬声器](../assets/voice-assistant.svg)

### 七大组件

1.  **音频采集。** 麦克风 → 16 kHz 单声道 → 20 毫秒音频块。生产环境通常使用Python的 `sounddevice` 或原生的 AudioUnit/ALSA/WASAPI。
2.  **语音活动检测（第11课）。** Silero VAD，阈值 0.5，最小语音时长 250 毫秒，静默持续时间 500 毫秒。发出“开始”和“结束”信号。
3.  **流式语音转文本（第4-5课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。输出部分转录和最终转录。
4.  **具备工具调用能力的大语言模型。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具使用JSON架构。流式输出token。
5.  **流式文本转语音（第7课）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在大语言模型输出20个token后开始文本转语音。
6.  **音频播放。** 扬声器输出；为低带宽网络进行opus编码。
7.  **打断处理器。** 若语音活动检测在文本转语音播放期间触发，则停止播放，取消大语言模型生成，并重启语音转文本。

### 你将遇到的三种失败模式

1.  **首词截断。** 语音活动检测开始得稍晚。用户的“hey”被丢失。初始阈值应设为0.3，而非0.5。
2.  **回应中打断混乱。** 用户打断后，大语言模型继续生成；助手在用户说话时发声。需连接语音活动检测至“取消大语言模型”。
3.  **静默幻听。** Whisper在静默的预热帧上输出“Thanks for watching”。务必用语音活动检测进行门控。

### 2026年生产环境参考技术栈

| 技术栈 | 延迟 | 许可证 | 备注 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500毫秒 | 商业API | 2026年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800毫秒 | 大部分开源 | DIY友好 |
| Moshi（全双工） | 200-300毫秒 | CC-BY 4.0 | 单模型；不同架构，见第15课 |
| Vapi / Retell（托管式） | 300-500毫秒 | 商业 | 启动最快；自定义受限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私/边缘计算 |

## 动手构建

### 第1步：带分块的麦克风采集（伪代码）

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

### 第2步：语音活动检测门控的轮次捕获

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

### 第3步：流式语音转文本 → 大语言模型 → 文本转语音

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第4步：在大语言模型循环内进行工具调用

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

### 第5步：打断处理

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

## 使用指南

查看 `code/main.py` 获取一个可运行的模拟示例，它将所有七个组件用存根模型连接起来，这样即使没有硬件也能看到管道的形态。对于真实实现，请用以下组件替换存根：

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 或 `openai-whisper`
- `openai` (`gpt-4o`) 或 `anthropic`
- `kokoro` 或 `cartesia`
- 用于输入/输出的 `sounddevice`

## 注意事项

- **永久记录个人身份信息。** 完整轮次的音频在大多数司法管辖区属于个人身份信息。需保留30天，静态加密。
- **无法打断。** 用户会打断。你的助手必须停止说话。
- **阻塞的文本转语音。** 同步的文本转语音会阻塞事件循环。使用异步或单独线程。
- **缺乏工具调用错误处理。** 工具会失败。大语言模型必须收到错误信息并重试一次，然后优雅降级。
- **过度激进的幻听过滤器。** 过度过滤导致助手重复“我无法帮助处理这个。”过滤不足则它会胡言乱语。在保留集上进行校准。
- **缺少唤醒词选项。** 始终监听存在隐私风险。添加唤醒词门控（Porcupine或openWakeWord）。

## 部署建议

保存为 `outputs/skill-voice-assistant-architect.md`。根据预算、规模、语言和合规性约束，制定完整的技术栈规范。

## 练习

1.  **简单。** 运行 `code/main.py`。它使用存根模块模拟一个完整的端到端轮次，并打印各阶段的延迟。
2.  **中等。** 用真实的Whisper模型替换语音转文本存根，处理预录制的 `.wav`。测量词错误率和端到端延迟。
3.  **困难。** 添加工具调用：实现 `get_weather`（任意API）和 `set_timer`。让大语言模型通过工具路由，并验证当用户说“设定一个5分钟计时器”时，正确的函数被触发且回复语音确认了这一点。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|----------|
| 轮次 | 一次用户+助手的往返 | 一次由语音活动检测界定的用户语音 + 一次大语言模型-文本转语音的回应。 |
| 打断 | 插话 | 用户在助手说话时发言；助手停止。 |
| 唤醒词 | “嘿，助手” | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测 | 轮次结束 | 语音活动检测 + 最小静默时长，判断用户是否说完。 |
| 预卷 | 语音前缓冲 | 在语音活动检测触发前保留200-400毫秒音频，以避免首词截断。 |
| 工具调用 | 函数调用 | 大语言模型发出JSON；运行时调度；结果在循环内反馈。 |

## 延伸阅读

- [LiveKit — 语音智能体快速入门](https://docs.livekit.io/agents/) — 生产级参考实现。
- [Pipecat — 语音智能体示例](https://github.com/pipecat-ai/pipecat) — DIY友好框架。
- [OpenAI 实时API](https://platform.openai.com/docs/guides/realtime) — 托管的语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第15课）。
- [Porcupine 唤醒词](https://picovoice.ai/products/porcupine/) — 唤醒词门控。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — 大语言模型函数调用。