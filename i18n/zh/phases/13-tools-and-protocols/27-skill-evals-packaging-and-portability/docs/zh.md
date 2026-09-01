# 技能评估、打包与可移植性

> 一个技能完成的标准是：其包能通过静态检查、在正确的请求上触发、改善可衡量的任务、遵守安全策略，并在另一主机上如实展现降级行为。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 13 阶段 · 22、24、25、26
**时间：** 约 150 分钟

## 学习目标

- 通过将判断、确定性计算、参考资源和输出契约分离，把一个专家工作流转化为技能。
- 将包结构、触发路由、任务行为、脚本正确性、安全性和可移植性作为独立层次进行测试。
- 使用真阳性、明确负例和近似误例来度量触发精度与召回率。
- 在多次重复运行中比较启用与不启用技能的性能差异。
- 构建并强制执行跨运行时能力矩阵，并为完整技能包设立发布门控。

## 问题所在

一个技能在演示中表现正常：用户恰好问了描述中使用的原句，作者知道打开哪份参考资源，脚本收到干净输入，目标主机认识所有自定义字段。

然后真实使用开始了。

- 模型为一个相近但不同的任务调用了它。
- 合法请求使用了不熟悉的措辞，导致模型遗漏。
- 工作体告诉代理做什么，但没有说明能证明完成的产物。
- 脚本在含空格的输入、重复执行或部分状态失败。
- 包安装器复制了 `SKILL.md` 却把其参考资源留了下来。
- 另一个运行时忽略了调用标记和工具授权。
- 一次运行成功，三次等价运行却走向不同分支。

"Markdown 看起来不错" 并不能捕获这些失败。技能是带有概率性路由和执行层的小型软件包。它们需要与其他生产接口相同的关注点分离。

## 概念

### 从真实工作流出发，而非主题

"创建一个 Kubernetes 技能" 不是一个可用的范围。Kubernetes 包含数百个任务，各有不同的工具、风险与输出。

"诊断为什么某个部署未进入 Available 状态，收集证据而不修改集群，并生成一份排序的事故报告" 才是一个技能候选。它具有：

- 触发边界；
- 稳定的证据收集步骤序列；
- 需要判断的决策点；
- 可转化为窄脚本或工具的命令；
- 定义明确的产物；
- 安全边界：只读诊断。

使用以下提取访谈：

1. 什么具体事件会让专家开始这个工作流？
2. 哪些相似请求不应触发它？
3. 专家首先收集哪些证据？
4. 哪些决策依赖于这些证据？
5. 哪些步骤足够确定可以被脚本化？
6. 哪些领域规则值得作为参考资源？
7. 哪些操作需要批准或必须保持在范围外？
8. 什么产物能证明工作流已完成？
9. 独立评审者如何检查它？
10. 哪些步骤依赖单一运行时？

答案将构成包架构和评估数据集。

### 将判断与确定性工作分离

```figure
skill-workflow-extraction
```

用模型判断处理分类、优先级排序、综合和歧义。用脚本或工具处理解析、计数、验证、转换、查询类型化 API 以及强制不变量。

包含 80 行手工模拟解析的工作体是脆弱的。试图做出主观架构决策的脚本是不透明的。将每种行为放在最能测试它的地方。

### 按依赖顺序编写包

不要从润色文字开始。从可观察的契约向内构建。

1. **产物契约：** 定义必需的文件、字段或决策。
2. **验证：** 定义如何检查每个要求。
3. **证据工具：** 实现确定性收集器和验证器。
4. **决策映射：** 将证据状态连接到分支。
5. **参考资源：** 在需要它的分支处提供领域细节。
6. **工作体：** 解释工作流、边界、失败情况和输出。
7. **描述：** 声明能力与触发边界。
8. **运行时适配器：** 分别添加调用或上下文扩展。
9. **评估：** 运行结构、路由、行为、安全性和可移植性层次。
10. **打包：** 安装完整目录并从目标位置测试。

这个顺序让文字服务于一个可测试的系统，而不是在演示成功后才发明成功标准。

### 六个评估层次

```figure
skill-eval-layers
```

每个层次回答不同的问题。通过一个层次不能替代另一个层次。

## 第一层：包结构

静态检查应验证不需要模型的事实：

- `SKILL.md` 存在于包根目录；
- 前端元数据可安全解析；
- `name` 与父目录匹配；
- 必需字段存在且在限制范围内；
- 每个非核心前端字段都出现在发布策略的运行时扩展白名单中；
- 每个直接引用都能在包内解析；
- 参考资源、脚本、资产和评估夹具使用发布策略允许的扩展名，且不超过其字节限制；
- 不存在禁止的符号链接或特殊文件；
- 工作体字符数在发布策略的预算内；
- 刻意收窄的密钥模式扫描未发现明显的凭据赋值或私钥头；
- 存在非空的 `## Output contract` 和 `## Failure behavior` 章节。

在解析 `SKILL.md`、评估数据、证据、主机夹具或清单之前，执行物理树的前检查。拒绝符号链接根目录、符号链接父目录或入口、缺少的必需常规文件，以及在读取任何内容之前的特殊文件。然后运行内容感知的策略检查。在进入前检查之前解析捆绑包路径会抹除检查所需的根符号链接证据。

课堂实现让这些策略值具体化：10,000 字符的工作体限制、1,000,000 字节的配套文件限制、按目录划分的扩展名白名单，以及由包需求提供的显式运行时扩展名。这些是发布策略示例，并非通用 Agent Skills 限制。密钥模式扫描是对明显错误的护栏，而非证明包不含敏感数据的依据。

检查报告应使用稳定的问题代码。CI 可以阻止 `E_*` 错误，同时允许已审查的 `W_*` 设计警告。

静态检查证明包形状。它不能证明模型会选择或遵循该技能。

## 第二层：触发路由

在反复编辑描述之前创建带标签的用例。

| 用例类型 | 目的 | 发布就绪示例 |
|---|---|---|
| 正例 | 衡量预期覆盖范围 | "版本 3.1.0 可以发布吗？" |
| 改述正例 | 避免短语记忆 | "在我们发布之前审计这个标签" |
| 明确负例 | 捕获严重过度路由 | "解释批量归一化" |
| 近似误例 | 定义相邻边界 | "包构建为什么失败了？" |
| 竞争技能 | 在可行条目间测试选择 | "起草发布说明" |
| 对抗性措辞 | 测试关键词堆砌和注入名称 | "不要用发布就绪；解释这个堆栈跟踪" |

将用例分为开发与验证集。在开发用例上调整描述。用验证用例决定修改后的描述是否具有泛化能力。如果发布决策足够重要，保留一个最终隔离集。

对于二元调用：

```text
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * precision * recall / (precision + recall)
```

报告原始计数与比率。十个中十个和一百个中一百个都是 100%，但提供的证据不同。

对于目录，还须度量 top-one 技能准确率、拒绝质量，以及相邻技能之间的混淆。一个在选对技能之前先选错三个技能的路由器是不健康的。

### 路由评估必须使用目标运行时

词法模拟器适用于解释指标和捕获明显重叠。它无法证明模型驱动的生产路由器的行为。在声称运行时质量之前，通过实际主机、模型、目录序列化和策略配置运行带标签数据集。

## 第三层：指令与产物行为

正确触发只是入口。技能必须改善任务。

创建带有以下内容的夹具任务：

- 输入文件与环境假设；
- 允许的工具与边界；
- 预期产物路径；
- 确定性检查；
- 需要判断的量规项；
- 最大时间、调用次数或成本；
- 失败情况及预期终止行为。

运行配对条件：

```text
baseline: 相同模型 + 相同工具 + 相同任务，无技能
treatment: 相同模型 + 相同工具 + 相同任务，可用技能
```

保持模型、温度或采样策略、工具集、任务夹具和预算不变。否则你无法将差异归因于技能。

有用的结果维度包括：

| 维度 | 示例度量 |
|---|---|
| 正确性 | 必需测试与不变量通过 |
| 完整性 | 每个产物契约字段都存在 |
| 效率 | 工具调用、 elapsed 时间、令牌或成本 |
| 证据 | 声明指向有效文件或观察结果 |
| 范围 | 禁止的文件和操作保持未碰触 |
| 恢复 | 中断的运行无需重复副作用即可恢复 |
| 人工工作量 | 评审者修正的数量与严重程度 |

不要只优化更少的令牌数。一次遗漏必需安全检查的更短运行更差。

### 产物契约使行为可执行

产物契约是一份独立可检查属性的列表：

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

Schema 验证检查结构。领域检查验证候选修订和证据路径。人工或校准的评审者可评估推荐是否基于证据。

## 第四层：脚本正确性

在模型运行之外像普通软件一样测试技能脚本。

最小用例：

- 正常输入；
- 空输入；
- 格式错误的输入；
- Unicode、空白和路径边缘情况；
- 重复执行；
- 超时或依赖失败；
- 上一次运行的部分输出；
- 输出大小限制；
- 试运行行为；
- 结构化退出与错误契约。

使用固定夹具。不要为单元测试要求实时网络。将网络集成测试放在显式标记之后，并记录其依赖的远程契约。

如果脚本执行副作用，则将计划与提交分开测试。对重试的外部写入要求幂等性或补偿。

## 第五层：安全性与授权

安全评估询问包是否停留于被赋予的权限之内。

至少测试：

- 超出技能范围的用户请求；
- 引用输入中的恶意指令；
- 逃逸包的资源路径；
- 逃逸允许根目录的工作区符号链接；
- 对未声明网络目标的请求；
- 需要环境凭据的命令；
- 未经批准的破坏性或外部操作；
- 超大的输出或无限进程；
- 技能到技能循环；
- 可能重复副作用的恢复。

记录控制是纯指令、工具策略、审批、沙箱还是验证。仅靠指令的防御不应被报告为已强制实施的隔离。

## 第六层：打包与可移植性

### 将整个目录作为一个单元安装

发布测试应安装到干净的目的地，然后对已安装的副本运行验证。

```figure
skill-package-install
```

仅测试源码树会错过安装器 bug、丢失的执行位、展平的引用、重写的名称，以及旧版本遗留的过时文件。

清单可包含：

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

保留 `assets/manifest.json` 作为清单元数据，并将其从自身的 `files` 映射中排除。一个文件不能在其内部携带自身完整当前内容的稳定哈希。验证其他每个打包文件，并通过外部可信通道（如签名发布或可信注册表记录）建立清单的真实性。交付的包裹只接受 `manifestVersion: 1` 和 `algorithm: "sha256"`；未知值将失败关闭。清单键必须是规范化的相对 POSIX 路径，因此 `./SKILL.md`、反斜杠、绝对路径和父级段应被拒绝而非规范化。课堂实现直接使用内部的 path-to-digest 映射，而两条路径都拒绝该映射内的保留清单路径。

哈希检测漂移。版本号传达兼容性。两者都不能认证清单，也不能取代升级前的完整差异与评估运行。

### 可移植性是一个能力矩阵

不要问主机是否"支持技能"这样一个布尔值。问它支持哪些行为。

| 能力 | 可移植包依赖 | 缺失时的回退 |
|---|---|---|
| 必需 `name` 和 `description` | 核心 | 包不能参与目录 |
| 工作体激活 | 核心客户端行为 | 显式文件加载适配器 |
| 参考资源、脚本、资产 | 核心包形状 | 主机需要文件与进程工具 |
| 显式人工调用 | 主机 UI 或提示约定 | 在普通文本中命名技能 |
| 隐式模型调用 | 主机路由器 | 应用显式激活 |
| 人工/模型 2x2 策略 | 主机扩展或应用策略 | 全局禁用隐式选择 |
| 参数绑定 | 主机解析器 | 激活后询问值 |
| 预批准工具 | 实验性或主机特定 | 普通权限提示 |
| 委托上下文 | 主机特定 | 在当前上下文或应用子代理中运行 |
| 生命周期钩子 | 主机特定 | 外部自动化或无钩子 |
| 上下文保留 | 主机特定 | 持久化状态并显式重新进入 |

对于每个必需能力，选择一个结果：

- 支持并已测试；
- 通过适配器支持；
- 降级并有文档记录的回退；
- 不支持，安装必须失败。

静默降级是需要避免的可移植性 bug。

### 可移植性测试需要主机夹具

能力声明应指向测试或当前官方契约。主机行为会变化。在兼容性报告中保留适配器版本与测试日期。

测试：

1. 从预期作用域的发现；
2. 重复名称的行为；
3. 显式调用；
4. 隐式调用或其禁用状态；
5. 参数处理；
6. 参考资源与脚本访问；
7. 权限提示与批准；
8. 委托或当前上下文执行；
9. 上下文压缩或重启后的恢复；
10. 卸载与升级行为。

### 规模数据不是质量证据

GitSkills 数据集论文报道了 2026 年 7 月的一次爬取，包含来自 282,200 个仓库的 3,797,117 个技能类文件，具有 1,877,981 种不同的字节内容。约 50.5% 的匹配文件在该论文的字节级度量下是逐字副本。

这些数字表明技能工件以仓库规模存在，且重复对数据集构建、搜索、溯源和升级分析很重要。它们并不表明一半的技能好坏、技能能改善任务性能、任何调用字段是通用的，或任何沙箱设计是安全的。该论文是数据集研究，而非有效性或安全基准。

使用生态数量来激励去重与溯源。使用自己的评估来做出质量声明。

## 重复运行与不确定性

模型和路由行为可能变化。在生产采样策略下对每个行为用例运行多次。

对于 `n` 次等价运行和 `k` 次通过：

```text
observed_pass_rate = k / n
```

保留单次轨迹。70% 通过率可能意味着一个一致失败类或几个无关失败。聚合速率指导比较；轨迹指导修复。溯源信息绑定到每次原始预测，而不仅是第零次运行和聚合速率。不同的预测顺序可能具有相同的首值与通过率，但代表不同的运行时行为。

按任务比较基线与处理，而不仅仅是作为池化平均值。即使平均值改善也要报告退化。高影响任务可能要求所有安全用例通过，而非接受平均阈值。

## 发布门控

一个实用的发布门控可要求：

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

阈值取决于风险与样本量。重要的属性是在查看最终结果之前就已声明。

失败应指明层次与证据。不要将路由、行为和安全合并为一个分数，使得强文案质量能抵消权限违规。

### 区分夹具成功、本地完整性与生产就绪

确定性课堂夹具可证明门控机制工作正常。它不能证明目标运行时实际选择了技能、生成了比较的产物、运行了脚本，或停留在测试的权限边界内。

保持三个边界：

- `fixturePassed`：使用声明的确定性触发、产物、证据和主机能力夹具模式通过所有层次；
- `localEvidenceReady`：所有四个捕获模式标签具有非空来源，且其 SHA-256 摘要与完整本地触发观察、产物、脚本和安全证据及非空主机矩阵匹配；
- `productionReady`：所有层次和本地完整性检查通过，且可信外部证实绑定评估器的完整 `evidenceRoot`。

总体发布字段 `passed` 遵循 `productionReady`，而非 `fixturePassed` 或 `localEvidenceReady`。本地哈希检测不匹配。它们不能证明捕获，因为任何能编辑包的人都可以重新标记夹具、虚构来源字符串并重新计算每个本地摘要。

交付的评估器对完整的触发、产物、证据、主机和清单配置对象计算一个 SHA-256 `evidenceRoot`。生产调用在包外提供证实文件：

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

它还通过 `--trusted-attestation-sha256` 提供这些证实字节的精确 SHA-256。该预期摘要必须来自带外可信策略、CI 密钥、签名发布记录或注册表决策。将其存储在同一个包内会使检查退化为另一个本地可重新计算的哈希。评估器拒绝缺失的、包内的、符号链接的、格式错误的、不匹配的或不支持版本的证实。

## 构建它

`code/main.py` 实现了迷你赛道的发布夹具。

它提供：

- 交付的评估器在读取任何配置之前执行物理树前检查；
- `lint_package(root)` 用于静态包检查；
- `TriggerCase`、`repeated_run_observations(...)` 和 `evaluate_triggers(...)` 用于带标签路由用例和完整原始轨迹；
- `classification_metrics(...)` 用于精度、召回率、准确率和原始计数；
- `repeated_run_rates(...)` 用于每个用例的重复行为结果；
- `ArtifactContract` 和 `evaluate_artifact(...)` 用于输出检查；
- `EvidenceCheck` 和 `evaluate_evidence_checks(...)` 用于显式脚本与安全证据；
- `EvaluationProvenance`、本地完整性摘要、完整证据根摘要，以及单独的夹具、本地完整性、信任锚点和生产裁决；
- `build_manifest(...)` 和 `verify_manifest(...)` 用于源码与干净安装树完整性；
- `HostCapabilities` 和 `portability_matrix(...)` 用于显式支持与回退状态；
- `run_release_gate(...)` 用于保持层次分离的最终裁决。

运行顶点实验：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

此块需要本地克隆，并从克隆内任意工作目录解析仓库根。

演示评估了捆绑的顶点技能、带标签触发集、重复结果、一个产物契约、显式脚本与安全检查、清单验证的干净副本，以及多个模拟主机配置。它打印一个 JSON 发布报告，其中 `checks_passed` 和 `fixture_passed` 为 true，而 `local_evidence_ready`、`trust_anchor_valid`、`production_ready` 和 `passed` 仍为 false。替换夹具并重新计算本地摘要可建立本地完整性，但生产仍需要外部可信证实。

### 按层次阅读报告

从硬性安全与包失败开始。然后检查路由混淆。然后与基线比较行为。只有在正确性和范围通过后，效率才有意义。

将报告与包修订版和评估夹具版本一起存储。来自旧模型、主机或技能树的通过是历史证据，而非当前组合的证明。

## 使用它

对每次技能修订使用以下编写循环：

```figure
skill-authoring-loop
```

更改负责失败的层次。当真正问题是丢弃引用的安装器或暴露主目录的沙箱时，不要往 `SKILL.md` 中塞入更多文字。

## 真实主机可移植性检查点

确定性夹具证明了发布门控机制。此检查点证明一个实际主机发现、加载、允许和移除什么。在描述包可移植之前完成它。

此检查点需要本地克隆、Node.js、`npx`、Python 3、一个选定的技能主机，以及一个可写的项目或用户技能作用域。验证 `node --version`、`npx --version` 和 `python3 --version`，然后在选择主机与作用域后再继续。如果该前检查不可用，则概念性地跟踪检查点并将每个主机观察标记为待定。网站或手册阅读不能建立可移植性。

### 1. 建立本地夹具边界

从本地克隆内任意位置运行。保留 `TARGET_ROOT` 为从原始仓库工作空间解析的课程目录：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

报告应显示 `checksPassed` 和 `fixturePassed` 为 true，而 `productionReady` 和 `passed` 仍为 false。在笔记中保留该区分。夹具通过不是主机结果。

### 2. 将完整包安装到第一个主机

从同一目录运行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

记录主机、可见的主机版本、作用域、安装路径和日期。在探测行为之前开始新会话或重新扫描目录。

将 `SKILL_ROOT` 设置为安装器报告的绝对安装目录。它必须包含已安装的 `SKILL.md`：

```bash
# 将占位符替换为安装器打印的目标路径。
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. 探测发现、路由、参考与脚本

使用第一个主机支持的显式语法：

| 主机 | 显式调用 |
|---|---|
| Codex | `skill-release-gate`，或从 `/skills` 选择，然后提供评估请求 |
| Claude Code | `/skill-release-gate` 后跟评估请求 |
| 可移植回退 | `Use skill-release-gate to evaluate the target bundle.` |

将这些作为单独的代理轮次运行，将每个占位符替换为上面打印的绝对值：

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

第一个提示检查显式调用。第二个检查隐式选择。第三个是近似误例，不应激活包评估。如果主机不暴露它选择了哪个技能，请将两个路由结果标记为未验证，而非从流畅响应推断。

对于显式运行，验证主机能否读取 `references/eval-contract.md` 并从已安装包执行 `scripts/evaluate_skill.py`。确切解析的命令必须具有此形状：

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

仅基于入口文件的响应不能证明完整包支持。记录解析的脚本路径、解析的目标包、cwd、确切 argv 和退出码。如果主机不能暴露某个字段，将该字段标记为未验证。

### 4. 探测批准行为

使用另一个请求：

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

预期行为：不发生发布。技能必须保留夹具与生产边界并在发布前停止。记录控制是来自技能指令、主机批准、缺失工具还是沙箱策略。不要称四个控制等价。

### 5. 使用第二个主机或声明回退

当有第二个兼容主机时，在第二个主机中重复步骤 2 至 4。如果不可用，则在主机矩阵中添加 `unverified` 或 `unsupported` 行并命名回退，如显式文件加载或显式调用。一个测试的主机永远不能证明通用可移植性。

你的证据表应包含：

| 检查项 | 主机 1 | 主机 2 或回退 |
|---|---|---|
| 发现与安装路径 | 观察值 | 观察值或未验证 |
| 显式调用 | 带证据的通过或失败 | 通过、失败或回退 |
| 隐式与近似误例路由 | 观察或未验证 | 观察或未验证 |
| 参考访问 | 观察路径或失败 | 观察路径或回退 |
| 脚本执行 | 命令与退出结果 | 命令与退出结果或不支持 |
| 批准行为 | 控制层次 | 控制层次或不支持 |

### 6. 练习升级与卸载

在用于安装的同一作用域中运行：

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

记录升级是否报告了变更或已是最新包。卸载后，开始新会话或重新扫描，并重复显式调用。主机不应再发现 `skill-release-gate`。过期的目录条目是值得记录的卸载失败。

## 交付它

本课程产生 `skill-release-gate`，一个完整的顶点包，包含 `SKILL.md`、一份参考、一个只读评估脚本、主机夹具、带标签触发用例和产物契约。从本地克隆内任意位置，解析仓库根并对绝对目标包运行已安装或源码评估器，以验证包含的课堂夹具而不声称发布。

对于生产，替换所有夹具为捕获值，重建保留清单，通过独立发布基础设施获取证实及其可信摘要，然后运行：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

仅当六层层门、本地证据完整性和外部信任锚都通过时，命令才会成功退出。重新标记和本地重新哈希的夹具在没有该锚时仍为非生产。

课程安装器复制完整包树。目录和网站指向其 `SKILL.md` 条目同时保留嵌套资源。这是扁平单文件工件中缺失的具体可移植性测试。

## 练习

1. 为你常用的技能编写十个正例、十个明确负例和十个近似误例。在编辑描述前拆分它们。
2. 运行五次运行的基线与处理比较。即使平均值改善也报告每个任务的退化。
3. 添加一个需要人工判断的量规维度。在使用它作为门控之前在五个样例上调校它。
4. 添加一个主机能力并定义支持、适配、降级和不支持的结果。
5. 在清单创建后修改已安装的参考。在安装前证明包验证失败。
6. 创建一个工作体通过检查但其脚本违反产物契约的技能。指出哪个发布层次阻止它。
7. 添加一个升级评估，比较两个包版本之间的调用策略和必需能力。
8. 发布一份兼容性报告，列出测试的主机版本、日期、回退和未验证行为，不使用任何"可移植"徽章。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| 触发评估 | "技能是否触发？" | 路由边界处选择、拒绝与混淆的带标签度量 |
| 行为评估 | "它能用吗？" | 针对产物、质量、范围和效率契约衡量的任务执行 |
| 基线 | "没有技能时" | 比较条件下相同的模型、工具、任务和预算 |
| 产物契约 | "预期输出" | 完成所需的独立可检查属性 |
| 能力矩阵 | "支持的运行时" | 按主机的原生支持、适配器、降级和不兼容性核算 |
| 发布门控 | "所有测试通过" | 分层阈值在不隐藏失败类的情况下阻止包 |
| 静默降级 | "被忽略的元数据" | 主机会丢失必需行为而不警告安装器或用户 |

## 延伸阅读

- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills) 关于触发评估、输出评估、重复运行和基线。
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices) 关于连贯的范围和资源架构。
- [Using scripts in skills](https://agentskills.io/skill-creation/using-scripts) 关于确定性辅助和结构化接口。
- [Client implementation guide](https://agentskills.io/client-implementation/adding-skills-support) 关于发现、激活、上下文、信任和生命周期行为。
- [GitSkills: A Dataset of Agent Skills from GitHub](https://arxiv.org/abs/2608.10906) 关于生态规模数据集及其声明的测量限制。
