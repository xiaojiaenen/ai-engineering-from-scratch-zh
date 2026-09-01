# 对话状态追踪（DST）

> "我要找一家便宜的餐厅，在北边……算了改成中等价位……再加个意大利菜。"三个回合，三次状态更新。DST 确保槽值字典保持同步，让预订能顺利完成。

**类型：** 实战练习
**语言：** Python
**前置知识：** 阶段 5 · 17（聊天机器人）、阶段 5 · 20（结构化输出）
**时间：** 约 75 分钟

## 问题所在

在面向任务的对话系统中，用户目标被编码为一组槽值对：`{cuisine: italian, area: north, price: moderate}`。每个用户回合都可能新增、修改或删除槽位。系统必须阅读完整对话并正确输出当前状态。

一个槽值出错，系统就会订错餐厅、排错航班或扣错卡。DST 是连接"用户说了什么"和"后端执行了什么"之间的枢纽。

为什么 2026 年了仍然重要，尽管有 LLM：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽值，而非自由生成。
- 工具使用型智能体仍需在调用 API 前完成槽值解析。
- 多轮修正远比看起来复杂："算了，改成周四吧。"

现代管线：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 概念

![DST：对话历史 → 槽值状态](../assets/dst.svg)

**任务结构。** 模式定义领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以为空、填入闭集候选值（如 price: {cheap, moderate, expensive}），或填入自由文本值（如 name: "The Copper Kettle"）。

**两种 DST 形式化方法。**

- **分类法。** 对每个 (槽位, 候选值) 对预测是/否。适用于闭集词表的槽位。2020 年以前的标准做法。
- **生成法。** 给定对话，生成槽值作为自由文本。适用于开放词表的槽位。现代默认方案。

**评估指标。** 联合目标准确率（Joint Goal Accuracy, JGA）—— 每个槽位都正确的回合占比。全有或全无。MultiWOZ 2.4 榜单在 2026 年顶尖成绩约为 83%。

**架构类型。**

1. **基于规则（槽位正则 + 关键词）。** 窄领域下的强基线。可调试。
2. **TripPy / BERT-DST。** 基于复制的生成 + BERT 编码。LLM 出现前的标准方案。
3. **LDST（LLaMA + LoRA）。** 指令微调 LLM + 领域-槽位提示。在 MultiWOZ 2.4 上达到 ChatGPT 级别质量。
4. **本体自由（2024–2026）。** 跳过模式；直接生成槽名和槽值。适用于开放领域。
5. **提示 + 结构化输出（2024–2026）。** LLM + Pydantic 模式 + 约束解码。5 行代码，可直接投入生产。

### 经典失败模式

- **跨回合指代。** "就按第一个选项来。" 需要解析是哪个选项。
- **覆盖 vs 追加。** 用户说"加个意大利菜"。是替换菜系还是追加？
- **隐式确认。** "好的，没问题" —— 这是否接受了提供的预订？
- **修正。** "其实改成晚上 7 点吧。" 必须更新时间而不清空其他槽位。
- **指向前序系统 utterance。** "对，就是那个。" 哪个"那个"？

```figure
n5-slot-tracker
```

## 动手实现

### 步骤 1：基于规则的槽位提取器

见 `code/main.py`。正则表达式 + 同义词词典可覆盖窄领域中 70% 的标准表达：

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

超出标准词表后鲁棒性较差。适用于确定性的槽位确认场景。

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

- 不要重置用户未触碰的槽位。
- 显式否定（"不用在意菜系了"）必须清空。
- 用户修正（"其实……"）必须覆盖，而非追加。

### 步骤 3：基于 LLM 的结构化输出 DST

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

Instructor + Pydantic 保证生成有效的状态对象。无需正则、无模式不匹配、无幻觉槽位。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准标准：系统有多少比例的回合能**所有**槽位全部正确？对于 MultiWOZ 2.4，2026 年顶尖系统为 80-83%。你的领域内系统应在有限词表上超越该成绩，否则 LLM 基线会胜过你。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最近更新的槽位而非追加。没有 LLM 辅助很难做对。现代做法：让 LLM 始终基于完整历史重新生成整个状态，而非增量更新 —— 这样自然处理修正场景。

## 注意事项

- **全历史重新生成的成本。** 每轮让 LLM 重新生成状态会产生 O(n²) 的总 token 消耗。限制历史长度或对早期回合进行摘要。
- **模式漂移。** 事后新增槽位会破坏旧训练数据。对模式进行版本管理。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN" —— 统一规范化。
- **隐式继承。** 如果用户之前指定了"4 人"，新请求改变时间不应清空人数。始终传递完整历史。
- **自由文本 vs 闭集。** 名称、时间、地址需要自由文本槽位；菜系和区域是闭集。在模式中混合两者。

## 使用指南

2026 技术栈：

| 场景 | 方案 |
|------|------|
| 窄领域（一到两个意图） | 基于规则 + 正则 |
| 宽领域，有标注数据 | LDST（在 MultiWOZ 风格数据上微调 LLaMA + LoRA） |
| 宽领域，无标注，需生产可用 | LLM + Instructor + Pydantic 模式 |
| 语音 /  spoken | ASR + 归一化 + LLM-DST |
| 多领域预订流程 | 领域级 Pydantic 模式的模式引导 LLM |
| 合规敏感 | 基于规则为主，LLM 为辅并搭配确认流程 |

## 交付物

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计对话状态追踪器 —— 模式、提取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

根据用例（领域、语言、词表开放性、合规需求），输出：

1. 模式。领域列表、各领域的槽位、每槽位的开放/闭集词表。
2. 提取器。基于规则 / seq2seq / LLM+Pydantic。说明理由。
3. 更新策略。全量重新生成 / 增量更新；修正处理；否定处理。
4. 评估。在预留对话集上的联合目标准确率、槽位级精确率/召回率、最难槽位的混淆分析。
5. 确认流程。何时应主动向用户请求确认（破坏性操作、低置信度提取）。

拒绝在合规敏感槽位上仅使用 LLM-DST 而不带基于规则的二次校验。拒绝任何无法在用户修正时回滚槽位的 DST。标注缺少版本标签的模式。
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个槽位（cuisine、area、price）构建基于规则的状态追踪器。在 10 条手工构造的对话上测试。测量 JGA。
2. **中等。** 同一数据集，使用 Instructor + Pydantic + 小型 LLM。对比 JGA。检查最难处理的回合。
3. **困难。** 实现两种方案并做路由：基于规则为主，当规则提取槽位数 < 2 且置信度低时回退到 LLM。测量综合 JGA 和每回合推理开销。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| DST | 对话状态追踪 | 在多轮对话中维护槽值字典。 |
| Slot | 用户意图单元 | 后端需要的命名参数（如菜系、日期）。 |
| Domain | 任务领域 | 餐厅、酒店、出租车 —— 一组槽位。 |
| JGA | 联合目标准确率 | 每个槽位都正确的回合占比。全有或全无。 |
| MultiWOZ | 基准数据集 | 多领域 WOZ 数据集；标准 DST 评估数据集。 |
| Ontology-free DST | 无本体 DST | 直接生成槽名和槽值，无需固定列表。 |
| Correction | "其实……" | 覆盖先前已填槽位的回合。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) —— 标准基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) —— 用于 DST 的 LLaMA + LoRA 指令微调。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) —— 基于复制的 DST 主力方案。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) —— 基于 EM 的无监督面向任务对话。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) —— 标准 DST 成绩榜单。
