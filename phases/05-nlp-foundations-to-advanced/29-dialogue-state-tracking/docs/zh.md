# 对话状态跟踪

> “我想要一家北边的便宜餐厅……其实改成中等价位吧……再加上意大利菜。”三轮对话，三次状态更新。DST确保槽位-值字典同步，使预订流程顺利进行。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 5 · 17（聊天机器人），阶段 5 · 20（结构化输出）  
**时间：** 约 75 分钟

## 问题所在

在任务导向型对话系统中，用户的目标被编码为一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每轮用户交互都可能增加、更改或删除一个槽位。系统必须读取整个对话并正确输出当前状态。

一旦弄错一个槽位，系统可能会预订错误的餐厅、安排错误的航班或扣错信用卡。DST是连接用户所言与后端执行的关键枢纽。

尽管大语言模型（LLMs）发展至今，DST在2026年仍然重要，原因在于：

- 合规敏感领域（银行、医疗、航班预订）需要确定性的槽位值，而非自由形式的生成。
- 工具调用代理在调用API之前仍然需要槽位解析。
- 多轮修正比看起来更难：“其实不，改成周四。”

现代流程：经典DST概念 + LLM提取器 + 结构化输出约束。

## 核心概念

![DST: 对话历史 → 槽位-值状态](../assets/dst.svg)

**任务结构。** 模式定义了领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以是空的，用一个封闭集中的值填充（价格：{便宜、中等、昂贵}），或是一个自由形式的值（名称：“The Copper Kettle”）。

**两种DST形式化方法。**

- **分类。** 对于每个（槽位，候选值）对，预测是/否。适用于封闭词汇槽位。2020年前的标准方法。
- **生成。** 给定对话，以自由文本形式生成槽位值。适用于开放词汇槽位。现代默认方法。

**评估指标。** 联合目标准确率——所有槽位都正确的轮次比例。要么全对，要么全错。MultiWOZ 2.4排行榜在2026年大约达到83%。

**架构。**

1. **基于规则（槽位正则表达式+关键词）。** 窄领域的强基线。易于调试。
2. **TripPy / BERT-DST。** 基于BERT编码的复制式生成。LLM前的标准。
3. **LDST（LLaMA + LoRA）。** 使用领域-槽位提示进行指令微调的LLM。在MultiWOZ 2.4上达到ChatGPT级别质量。
4. **无本体（2024-26）。** 跳过模式；直接生成槽位名称和值。处理开放领域。
5. **提示+结构化输出（2024-26）。** 使用Pydantic模式+约束解码的LLM。仅需5行代码，即可投入生产。

### 经典失败模式

- **跨轮次共指。** “我们还是选第一个选项吧。”需要解析指的是哪个选项。
- **覆盖 vs 追加。** 用户说“加上意大利菜”。你是替换菜系还是追加？
- **隐式确认。** “好的，酷”——这是接受了提议的预订吗？
- **修正。** “其实改成下午7点。”必须更新时间而不清除其他槽位。
- **指向先前系统话语的共指。** “对，就是那个。”哪个“那个”？

## 开始构建

### 步骤 1：基于规则的槽位提取器

见 `code/main.py`。正则表达式+同义词词典可覆盖窄领域中70%的规范表达：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词汇表之外则很脆弱。适用于确定性槽位确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三条不变式：

- 绝不重置用户未触及的槽位。
- 显式否定（“算了，不要菜系了”）必须清除槽位。
- 用户修正（“其实…”）必须覆盖，而非追加。

### 步骤 3：使用结构化输出的LLM驱动DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证了有效的状态对象。无需正则表达式，无模式不匹配，无幻觉槽位。

### 步骤 4：JGA评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统正确获得*所有*槽位的轮次比例是多少？对于MultiWOZ 2.4，2026年顶级系统：80-83%。你的领域内系统若使用窄词汇应超过该值，否则LLM基线会击败你。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最后更新的槽位而非追加。没有LLM帮助很难做好。现代模式：让LLM根据历史始终重新生成整个状态，而不是增量更新——这自然地处理了修正。

## 注意事项

- **全历史重新生成成本。** 让LLM每轮重新生成状态，总token成本为O(n²)。限制历史长度或总结早期轮次。
- **模式漂移。** 事后添加新槽位会破坏旧的训练数据。对你的模式进行版本控制。
- **大小写敏感性。** “Italian” vs “italian” vs “ITALIAN”——在所有地方进行规范化。
- **隐式继承。** 如果用户先前指定了“4人”，针对不同时段的新请求不应清除“人数”。始终传递完整历史。
- **自由形式 vs 封闭集。** 名称、时间和地址需要自由形式槽位；菜系和区域是封闭的。在模式中混合使用两者。

## 使用场景

2026年的技术栈：

| 场景 | 方法 |
|------|------|
| 窄领域（一个或两个意图） | 基于规则 + 正则表达式 |
| 宽领域，有标注数据 | LDST（在MultiWOZ风格数据上微调的LLaMA + LoRA） |
| 宽领域，无标注，需投入生产 | LLM + Instructor + Pydantic模式 |
| 语音对话 | ASR + 规范化器 + LLM-DST |
| 多领域预订流程 | 模式引导的LLM，每个领域使用Pydantic模型 |
| 合规敏感 | 基于规则为主，LLM作为后备并配合确认流程 |

## 部署使用

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中为3个槽位（菜系、区域、价格）构建基于规则的状态跟踪器。在10个手写对话上进行测试。测量JGA。
2. **中等。** 使用Instructor + Pydantic + 一个小型LLM处理相同数据集。比较JGA。检查最困难的轮次。
3. **困难。** 实现两者并进行路由：基于规则的为主，当规则系统输出置信度<2个槽位时使用LLM后备。测量组合JGA和每轮推理成本。

## 关键术语

| 术语 | 人们通常指 | 实际含义 |
|------|------------|----------|
| DST | 对话状态跟踪 | 在对话轮次中维护槽位-值字典。 |
| 槽位 | 用户意图的单位 | 后端需要的命名参数（如菜系、日期）。 |
| 领域 | 任务区域 | 餐厅、酒店、出租车——槽位的集合。 |
| JGA | 联合目标准确率 | 所有槽位都正确的轮次比例。非此即彼。 |
| MultiWOZ | 基准数据集 | 多领域WOZ数据集；标准DST评估基准。 |
| 无本体DST | 无模式 | 直接生成槽位名称和值，无需固定列表。 |
| 修正 | “其实…” | 覆盖先前已填充槽位的轮次。 |

## 延伸阅读

- [Budzianowski 等 (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典基准数据集。
- [Feng 等 (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — 用于DST的LLaMA + LoRA指令微调。
- [Heck 等 (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的DST主力模型。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于EM的无监督任务导向型对话。
- [MultiWOZ 排行榜](https://github.com/budzianowski/multiwoz) — 权威DST结果。