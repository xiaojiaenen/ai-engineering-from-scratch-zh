# 基准测试：WebArena 和 OSWorld

> WebArena 在四个自托管应用上测试网页代理能力。OSWorld 在 Ubuntu、Windows、macOS 上测试桌面代理能力。在发布时（2023–2024），这两项基准都显示出顶级代理与人类之间存在巨大差距。这个差距正在缩小，但故障模式没有改变。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 19（SWE-bench、GAIA）
**时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用，以及为何基于执行的评估很重要。
- 解释为什么 OSWorld 使用真实操作系统截图而不是辅助功能 API。
- 列举 OSWorld 的两个主要故障模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准之上增加了什么。

## 问题

通用代理可以调用工具。但它们能否驱动浏览器完成 20 次点击来完成一次购物结账？能否仅用键盘和鼠标配置一台 Linux 机器？WebArena 和 OSWorld 回答的就是这些问题。

## 概念

### WebArena（Zhou 等，ICLR 2024）

- 四个自托管 Web 应用上的 812 个长程任务：一个购物网站、一个论坛、一个类 GitLab 的开发工具、一个企业 CMS。
- 附加工具：地图、计算器、便签。
- 通过 gym API 进行基于执行的评估——订单是否已提交？问题是否已关闭？CMS 页面是否已更新？
- 发布时：最佳 GPT-4 代理成功率 14.41%，人类 78.24%。

自托管的设定很关键——因为目标应用被固定下来，基准不会出现漂移。

### 扩展

- **VisualWebArena** — 视觉接地任务，成功取决于对图像的解释（截图作为一等公民的观察输入）。
- **TheAgentCompany**（2024 年 12 月）— 增加了终端和编程；更像真实的远程工作环境。

### OSWorld（Xie 等，NeurIPS 2024）

- 跨 Ubuntu、Windows、macOS 的 369 个真实计算机任务。
- 对真实应用程序的自由键盘和鼠标控制。
- 以 1920×1080 截图作为观察输入。
- 发布时：最佳模型 12.24%，人类 72.36%。

### 主要故障模式

1. **GUI 定位。** 像素 → 元素映射。模型难以在 1920×1080 分辨率中可靠地定位 UI 元素。
2. **操作知识。** 哪个菜单有该设置、哪个快捷键、哪个偏好面板。这些知识是人类经年累月积累的尾部知识。

### 后续研究

- **OSWorld-G** — 564 样本定位套件 + Jedi 训练集。将定位从规划中分离，以便分别测量两者。
- **OSWorld-Human** — 手动整理的专家级动作轨迹。显示顶级代理比必要步骤多出 1.4–2.7 倍（轨迹效率差距）。

### 为什么这很重要

Claude Computer Use、OpenAI CUA、Gemini 2.5 Computer Use（课程 21）都在由 WebArena 和 OSWorld 塑造的工作负载上进行训练。这些基准是靶心，生产模型是射出的箭。

### 基准测试容易出错的地方

- **仅基于截图的评估。** OSWorld 是截图驱动的；在 OSWorld 上评估使用 DOM 或辅助功能 API 的代理，会错过定位挑战。
- **忽略轨迹长度。** 只评分成功率会遗漏 OSWorld-Human 揭示的 1.4–2.7 倍步骤低效问题。
- **过时的自托管应用。** WebArena 的应用锁定在特定版本；若更新而不重新整理标注，可比性就会断裂。

```figure
ae-agent-human-gap
```

## 动手实践

`code/main.py` 实现了一个玩具级网页代理框架：

- 一个极简的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个尝试完成每个任务的脚本化代理。
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤数 vs 黄金轨迹）。

运行方式：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，与 OSWorld-Human 的方法论一致。

## 使用它

- **WebArena Verified** — 在内部集群上自托管，用于持续评估。
- **OSWorld** — 在虚拟机集群中运行桌面代理。
- **Computer-use 代理**（课程 21）— Claude、OpenAI CUA、Gemini 都在这类工作负载上进行训练。
- **你自己的产品流程** — 为你最重要的 20 个任务录制黄金轨迹；每周运行代理与它们对比。

## 交付物

`outputs/skill-web-desktop-harness.md` 构建了一个带有基于执行评估和轨迹效率指标的网页/桌面代理框架。

## 练习

1. 用第二个应用（一个论坛）扩展玩具框架。编写 3 个任务及黄金轨迹。
2. 为每个任务添加轨迹效率报告。在你的玩具框架中，代理是 1x、2x 还是 3x 高于黄金轨迹？
3. 实现一个"干扰"工具——黄金轨迹从未使用过的工具。脚本化代理会被诱惑吗？
4. 阅读 OSWorld-G。你会如何在自己的评估中将定位失败与规划失败区分开？
5. 阅读 WebArena 各应用的 README。当你升级其中一个锁定版本的应用时会发生什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| WebArena | "网页代理基准" | 跨 4 个自托管应用的 812 个任务；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉接地的 WebArena；截图即观察 |
| OSWorld | "桌面代理基准" | 369 个任务，覆盖真实 Ubuntu/Windows/macOS |
| GUI 定位 | "像素到元素映射" | 模型在 1920×1080 中定位 UI 元素 |
| 操作知识 | "OS 经验" | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | "定位套件" | 564 个仅定位样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 手动专家动作序列，用于衡量效率 |
| 轨迹效率 | "相对于黄金的步数比" | 代理步数除以人类最小步数 |

## 延伸阅读

- [Zhou 等，WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用网页基准
- [Xie 等，OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨操作系统桌面基准
- [Anthropic，Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准塑造能力
- [OpenAI，Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数据
