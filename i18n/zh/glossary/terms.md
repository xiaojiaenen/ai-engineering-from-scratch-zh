# AI Engineering Glossary

使用本术语表时，当课程、论文、模型卡或代码审查引入术语的速度快于解释的速度。按精确术语或别名搜索，先阅读直接定义，然后使用实践说明将其与你可构建的系统联系起来。

每个词条属于一个学习类别。`Related terms` 提供下一个有用的概念，而不强制固定的学习路径。定义描述的是通用工程含义，但特定提供商的行为可能有所不同。当 API 合约或模型卡与通用定义不一致时，以当前官方文档为准。

这十二个类别是：Math & training；Models & inference；Data & representations；Retrieval & generation；Prompting & context；Agents & tools；Evaluation & safety；AI-native development；Infrastructure & serving；Reliability & operations；Security & governance；Multimodal systems。

## A

### Activation Checkpointing
- **Category:** Math & training
- **What it actually means:** 一种训练内存优化技术，仅保存选定的前向传播激活值，并在反向传播时重新计算被跳过的激活值。
- **Why it matters:** 它通过以额外计算换取消活存储，使你在固定内存预算内训练更大的模型或序列。
- **In practice:** 对内存密集型的 transformer 模块进行 checkpoint，测量额外的步骤时间，并将恢复 checkpoint 与激活重计算设置分开管理。
- **Common confusion:** Activation checkpointing 并非持久化的训练 checkpoint。它帮助单次前向和反向传播适应内存，但无法从崩溃的运行中恢复。
- **Related terms:** Autograd, Backpropagation, Checkpoint, Mixed Precision
- **Sources:** [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)

### Activation Function
- **Category:** Math & training
- **What people say:** 层之间的非线性运算。
- **What it actually means:** 应用于线性或仿射层之后的函数，引入非线性。如果没有它，具有权重和偏置的层组合会退化为单个仿射变换。ReLU、GELU 和 SiLU 是常见选择。选择直接影响训练过程中梯度能否流动。
- **Learn it:** [Activation Functions](../phases/03-deep-learning-core/04-activation-functions/)
- **Related terms:** ReLU, Gradient, Backpropagation

### Adam (Optimizer)
- **Category:** Math & training
- **What people say:** 不假思索就使用的优化器。
- **What it actually means:** 自适应矩估计（Adaptive Moment Estimation）。它将梯度的指数加权平均与梯度的指数加权平方的平均相结合，应用偏差校正，并针对每个参数自适应调整更新幅度。它是一个有用的基线，但仍需要合适的学习率和调度策略。
- **Common confusion:** Adam 是一个强大的基线，并非通用的最优优化器。
- **Sources:** [Adam paper](https://arxiv.org/abs/1412.6980)
- **Related terms:** AdamW, Optimizer, Learning Rate

### AdamW
- **Category:** Math & training
- **What people say:** 修复了权重衰减的 Adam。
- **What it actually means:** 一种 Adam 变体，将权重衰减与基于梯度的参数更新解耦。这使得收缩行为比在 Adam 自适应缩放的梯度内部添加 L2 惩罚更容易推理。
- **Common confusion:** 解耦的权重衰减并不意味着 AdamW 是通用的最优选择。模型、数据和训练规模仍然决定最优优化器和调度策略。
- **Sources:** [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- **Related terms:** Adam (Optimizer), Weight Decay, Optimizer

### Admission Control
- **Category:** Reliability & operations
- **What it actually means:** 一个预先接受检查门，根据系统当前的容量、优先级和政策，决定是否允许请求进入有界队列或服务。
- **Why it matters:** 在可控边界拒绝超额工作，可以保护已接受的请求免受队列增长、超时级联和资源耗尽的影响。
- **In practice:** 估计请求的成本，检查租户和系统容量，原子性地预留所需预算，并在拒绝时标识过载的范围。仅当条件是瞬态的且调用方的重试预算允许再次尝试时，才提供重试指导。
- **Common confusion:** Admission control 在接受之前起作用。Load shedding 可以在入口、队列、依赖项或其他过载边界处拒绝或移除工作。
- **Related terms:** Load Shedding, Backpressure, Rate Limit, Saturation
- **Sources:** [Google SRE: Handling Overload](https://sre.google/sre-book/handling-overload/)

### Agent
- **Category:** Agents & tools
- **What people say:** 一个独立思考行动的智能模型。
- **What it actually means:** 一个软件系统，允许模型为达成目标而选择动作，观察工具或环境的反馈，并在编排策略下继续执行。Agent 可能使用循环、状态机、工作流引擎或人工审批。模型只是其中一个组件，而非整个系统。
- **Why it matters:** 可靠性来自围绕模型的 harness、工具合约、状态、权限和验证机制。
- **In practice:** 一个 coding agent 读取仓库上下文，提出补丁，在沙箱中运行测试，并在部署前暂停等待审批。
- **Common confusion:** 自主性是一种委托权力的程度，而非每个 agent 的必需属性。
- **Learn it:** [The Agent Loop](../phases/14-agent-engineering/01-the-agent-loop/)
- **Related terms:** Agent Harness, Agent State, Tool Contract, Human-in-the-Loop (HITL)

### Agent Harness
- **Category:** Agents & tools
- **What it actually means:** 围绕模型的运行时环境，负责组装上下文、暴露工具、管理状态、执行限制、记录追踪，并决定 agent 何时继续、重试、询问或停止。
- **Why it matters:** 两个使用相同模型的系统可能表现迥异，因为它们的 harness 提供了不同的上下文、工具、反馈和安全边界。
- **In practice:** 你的 harness 可以限制 agent 进行五次工具调用，在每个已接受补丁后持久化 checkpoint，并在完成前要求测试命令通过。
- **Common confusion:** Harness 的范围比提示模板更广，但比完整产品更窄。
- **Learn it:** [Minimal Agent Workbench](../phases/14-agent-engineering/32-minimal-agent-workbench/)
- **Related terms:** Agent, Tool Contract, Agent State, Verification Gate, Sandbox

### Agent Memory
- **Category:** Agents & tools
- **What it actually means:** 存储在模型外部、并在后续 agent 步骤中按需选取使用的信息，如先前的决策、用户偏好、任务片段或已验证的事实。
- **Why it matters:** 它赋予 agent 超越单个上下文窗口的连续性，而无需将每个过去事件都塞入每次提示。
- **In practice:** 存储带有来源证明的紧凑任务结果，仅在相关时检索，并让用户检查或纠正持久的个人信息。
- **Common confusion:** Agent memory 与 agent state 不同。State 跟踪当前运行；memory 保留选定信息供未来可能的运行使用。
- **Related terms:** Agent State, Context Engineering, Checkpoint, Semantic Cache
- **Sources:** [Generative Agents](https://arxiv.org/abs/2304.03442)

### Agent State
- **Category:** Agents & tools
- **What it actually means:** agent 跨步骤携带的显式数据，如当前目标、已完成动作、工具结果、待解决问题、预算、审批和工件引用。
- **Why it matters:** 显式状态使长任务可恢复、可检查，并减少对模型从转录文本重建进展的依赖。
- **In practice:** 将选中的问题、更改的文件、最新测试结果和剩余检查项存储在类型化对象中，并在每次动作后更新。
- **Common confusion:** State 不等于对话历史。转录文本是证据；state 是用于决定下一步操作的紧凑操作记录。
- **Learn it:** [Repository Memory and State](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **Related terms:** Checkpoint, Durable Execution, Context Engineering, Handoff

### Agent Skill
- **Category:** Agents & tools
- **What it actually means:** 一个可发现的过程指令目录，入口点是 `SKILL.md`，包含可选的引用、脚本和资产，兼容的运行时可按阶段加载。
- **Why it matters:** 它将可复用的任务知识打包，与单次对话分离，同时保持深层上下文和确定性辅助工具按需可用。
- **In practice:** 发布紧凑的名称和路由描述，仅在激活后加载工作流，并在任务到达时读取分支特定的引用。
- **Common confusion:** 激活 skill 会提供上下文，但它不会暴露工具、授予权限、创建沙箱，也无法保证生成结果的正确性。
- **Learn it:** [Agent Skills: Portable Contract and Runtime Boundary](../phases/13-tools-and-protocols/22-skills-and-agent-sdks/)
- **Related terms:** Skill Bundle, Skill Catalog, Skill Invocation, Progressive Disclosure, MCP (Model Context Protocol)
- **Sources:** [Agent Skills specification](https://agentskills.io/specification)

### AI Risk Assessment
- **Category:** Security & governance
- **What it actually means:** 一份文件化的分析，说明 AI 系统如何影响人员、组织和环境，包括上下文、危害、可能性、影响、控制措施、残余风险及监控职责。
- **Why it matters:** 仅靠模型能力并不能决定风险。部署上下文、受影响群体、人类授权、数据和系统集成都会改变危害和所需控制措施。
- **In practice:** 定义预期用途和受影响方，识别可信的故障和滥用场景，为控制措施分配负责人，记录残余风险，并针对重大变更设置审查触发条件。
- **Common confusion:** 风险评估是在既定假设下支持决策的。它不是一次性的安全证书，也无法证明已发现所有危害。
- **Related terms:** Threat Model, Guardrails, Human-in-the-Loop (HITL), Data Classification
- **Sources:** [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

### Alignment
- **Category:** Evaluation & safety
- **What people say:** 让 AI 安全。
- **What it actually means:** 使模型或 AI 系统在预期和对抗性情境中都能以符合预期目标、约束和人类偏好的方式行事的一系列努力。
- **Why it matters:** 系统可能优化声明的指标却违反用户的真实意图，因此 alignment 需要评估、监督与系统控制，而不仅是模型训练。
- **Related terms:** Guardrails, Evaluation (Eval), Human-in-the-Loop (HITL)

### Approval Gate
- **Category:** Agents & tools
- **What it actually means:** 一个控制点，在授权人员或策略批准之前阻止关键操作。
- **Why it matters:** 它在保留自动化处理可逆工作的同时，限制不确定模型决策的影响范围。
- **In practice:** 允许 agent 起草数据库迁移并在临时数据库上运行，但要求所有者批准任何生产执行。
- **Common confusion:** Approval gate 询问操作是否被授权。Verification gate 询问证据是否显示操作是正确的。
- **Learn it:** [Verification Gates](../phases/14-agent-engineering/38-verification-gates/)
- **Related terms:** Human-in-the-Loop (HITL), Verification Gate, Least Privilege

### Approximate Nearest Neighbor (ANN)
- **Category:** Retrieval & generation
- **What it actually means:** 一种搜索方法，返回可能与查询向量最近的向量，而无需穷举比较查询与每个存储向量。
- **Why it matters:** 近似搜索使大规模向量索引变得可行，但引入了可度量的搜索速度、内存与检索召回率之间的权衡。
- **In practice:** 针对预留的查询集调整索引和搜索参数，然后在报告延迟的同时报告 Recall@K，而非假设每个真实近邻都被找到。
- **Common confusion:** ANN 描述的是一种搜索目标和权衡，而 HNSW 是实现它的一种特定索引算法。
- **Related terms:** Vector Database, HNSW, Cosine Similarity, Recall@K
- **Sources:** [Efficient and Robust Approximate Nearest Neighbor Search Using HNSW](https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473)

### Attention
- **Category:** Models & inference
- **What people say:** 模型如何关注重要 token。
- **What it actually means:** 一种通过比较查询向量和键向量、对结果分数进行归一化，并使用它们组合值向量来形成上下文表示的机制。掩码、位置规则或稀疏模式可以限制参与的位置。
- **Why it matters:** Attention 让模型能够在序列位置间路由信息，但它本身并不能解释或证明模型理解了什么。
- **Common confusion:** Attention 权重是计算系数，而非模型推理的忠实解释。
- **Learn it:** [Self-Attention from Scratch](../phases/07-transformers-deep-dive/02-self-attention-from-scratch/)
- **Sources:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Related terms:** Self-Attention, Transformer, KV Cache

### Audio Token
- **Category:** Multimodal systems
- **What it actually means:** 由音频编解码器或分词器为音频信号的短片段或特征产生的离散标识符，有时跨多个 codebook。
- **Why it matters:** 离散音频表示使序列模型能够使用基于 token 的架构处理、预测、存储或生成声音。
- **In practice:** 将编解码器与模型一同版本化，保留采样率和 codebook 元数据，测量重构质量，并区分语义音频 token 与波形压缩 token。
- **Common confusion:** Audio token 不具有固定的时长、音素或词汇含义。其含义和时间跨度取决于分词器和 codebook 设计。
- **Learn it:** [Neural Audio Codecs](../phases/06-speech-and-audio/13-neural-audio-codecs/)
- **Related terms:** Token, Embedding, Automatic Speech Recognition (ASR), Multimodal Model
- **Sources:** [SoundStream](https://arxiv.org/abs/2107.03312)

### Audit Log
- **Category:** Security & governance
- **What it actually means:** 一份持久化、受访问控制的记录，记载与安全性或问责制相关的事件，包括谁或什么实施了操作、什么发生了变更、何时发生以及结果状态。
- **Why it matters:** 关键性的 agent 操作需要证据，以支持调查、政策审查和超出性能调试的责任追溯。
- **In practice:** 记录工具授权、审批决策、外部写入、策略版本和工件标识符，同时脱敏敏感字段并限制日志访问。
- **Common confusion:** Trace 有助于诊断单次执行路径。Audit log 保留跨执行和时间推移所需的问责事件。
- **Related terms:** Trace, Observability, Approval Gate, Provenance Attestation
- **Sources:** [NIST SP 800-92](https://csrc.nist.gov/pubs/sp/800/92/final)

### Autograd
- **Category:** Math & training
- **What people say:** 自动梯度。
- **What it actually means:** 一种记录或变换张量操作的系统，以便计算导数，通常使用反向模式自动微分。你编写前向计算，框架推导出反向传播所需的梯度。
- **Learn it:** [Chain Rule and Automatic Differentiation](../phases/01-math-foundations/05-chain-rule-and-autodiff/)
- **Related terms:** Backpropagation, Gradient, Tensor

### Automatic Speech Recognition (ASR)
- **Category:** Multimodal systems
- **What it actually means:** 将语音信号映射为转录文本的任务和系统流水线，通常包含可选的 token 或片段时间戳及置信度信息。
- **Why it matters:** 语音界面不仅依赖语言建模。声学变化、分段、解码、词汇表和领域条件都会影响最终转录结果。
- **In practice:** 按语言、说话人、噪声和领域评估词错率或字错率，当下游接地需要时保留时间戳，并在生产中测试实际使用的音频预处理。
- **Common confusion:** ASR 转录的是所说的内容。确定谁在说话需要说话人分离或识别，而翻译和意图理解是独立的任务。
- **Learn it:** [Speech Recognition and ASR](../phases/06-speech-and-audio/04-speech-recognition-asr/)
- **Related terms:** Audio Token, Encoder, Tokenization, Multimodal Model
- **Sources:** [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

### Autoregressive
- **Category:** Models & inference
- **What people say:** 模型一次生成一个词。
- **What it actually means:** 一种因子分解方式，每个输出 token 由 preceding tokens 预测。在生成过程中，选中的 token 被追加到序列中，成为下一次预测的上下文的一部分。
- **Common confusion:** 单位是 token，不一定是词，且生成可以使用除始终选择最高概率 token 之外的解码方法。
- **Related terms:** Token, Temperature, KV Cache

### Autoscaling
- **Category:** Infrastructure & serving
- **What it actually means:** 一个控制环路，根据观察到的需求、资源使用率或应用指标在配置的边界内改变 serving workers 的数量或容量。
- **Why it matters:** AI 工作负载的变化可能快于手动调配，但扩展决策必须考虑模型加载时间、加速器可用性、排队和请求成本。
- **In practice:** 从与有效工作相关的需求信号进行扩展，设置最小预热容量，限制缩容抖动，并验证新副本通过就绪检查后再接收流量。
- **Common confusion:** Autoscaling 增加或移除容量。它不能使过载的依赖项变快，也无法保证在时限内获得足够的硬件。
- **Learn it:** [GPU Autoscaling on Kubernetes](../phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes/)
- **Related terms:** Model Serving, Saturation, Readiness Probe, Backpressure
- **Sources:** [Kubernetes Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

### Availability
- **Category:** Reliability & operations
- **What it actually means:** 在规定的测量边界内，用户可获得定义的可接受服务的eligible service interactions 或时间窗口的比例。
- **Why it matters:** 服务可能正在运行，但用户仍无法完成有效请求，因此 availability 必须与用户可见的成功率相关联，而非仅依赖进程正常运行时间。
- **In practice:** 定义 eligible events 和 acceptable outcomes，仅排除已记录的情况，在固定窗口内计算指标，并调查完全失败和长期部分降级。
- **Common confusion:** Availability 是可靠性的一种结果。它不描述延迟、正确性、安全性或每个用户分段的体验。
- **Related terms:** Service Level Indicator (SLI), Service Level Objective (SLO), Error Budget, Incident Response
- **Sources:** [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)

## B

### Backpressure
- **Category:** AI-native development
- **What it actually means:** 一种流量控制机制，当下游组件无法在当前速率下安全处理时，减缓或拒绝上游工作。
- **Why it matters:** 没有 backpressure 时，队列中的 agent 运行、工具调用或流式事件可能耗尽内存、超出速率限制并放大重试。
- **In practice:** 当 evaluator 队列达到限制时，暂停新的 agent 作业或返回可重试响应，而非接受无界工作。
- **Common confusion:** Backpressure 在故障前保护容量。Circuit breaker 在故障表明依赖项不健康后停止调用。
- **Related terms:** Rate Limit, Retry with Backoff, Circuit Breaker

### Backpropagation
- **Category:** Math & training
- **What people say:** 神经网络如何学习。
- **What it actually means:** 链式法则的高效应用，将标量 loss 的导数沿计算图向后传播。它计算梯度；优化器使用这些梯度来更新参数。
- **Common confusion:** Backpropagation 计算梯度。它不选择更新规则或学习率。
- **Why it's called that:** 导数信息从 loss 向早期操作反向传播。
- **Learn it:** [Backpropagation from Scratch](../phases/03-deep-learning-core/03-backpropagation/)
- **Related terms:** Autograd, Gradient, Optimizer

### Batch Size
- **Category:** Math & training
- **What people say:** 一次处理多少个样本。
- **What it actually means:** 在优化器更新之前，其 loss 贡献于单次梯度估计的样本数量。较大的 batch 可以提高硬件利用率和减少梯度噪声，但需要更多内存，并可能需要不同的学习率或调度策略。
- **Common confusion:** 不存在通用的 batch size 范围，也没有每条 batch 增加都应产生相同学习率增加的规则。
- **Related terms:** Learning Rate, Gradient, Optimizer

### Benchmark Contamination
- **Category:** Evaluation & safety
- **What it actually means:** 评估样本与用于预训练、微调、提示、选择或以其他方式改进被评估系统的数据之间存在的重叠或信息泄露。
- **Why it matters:** 污染可能使 benchmark 分数反映的是先前接触，而非泛化到未见任务的能力。
- **In practice:** 跟踪数据集来源，搜索训练源中的精确和近似重复，保留私有测试用例，并用新撰写的样本刷新公共 evals。
- **Common confusion:** 污染比精确复制更广泛。改写、答案键、benchmark 元数据和重复的 prompt tuning 也可能泄露评估信息。
- **Related terms:** Data Leakage, Data Deduplication, Eval Set, Exact Match (EM)
- **Sources:** [Investigating Data Contamination in Modern Benchmarks for Large Language Models](https://arxiv.org/abs/2311.09783)

### BM25
- **Category:** Retrieval & generation
- **What it actually means:** 一种词汇排序函数，根据查询词匹配对文档评分，同时考虑词频稀有度、重复出现和文档长度。
- **Why it matters:** 它是一种强大的精确词检索基线，可与 dense retrieval 互补，适用于标识符、罕见词和领域特定短语。
- **In practice:** 用 BM25 和密集搜索检索候选项，合并它们的排名，然后在添加更昂贵的 reranker 之前评估合并结果。
- **Common confusion:** BM25 不直接理解语义相似性，其分数在不同查询或索引配置之间没有通用含义。
- **Related terms:** Hybrid Retrieval, Dense Retrieval, Reranker, RAG (Retrieval-Augmented Generation)
- **Sources:** [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019)

### Byte Pair Encoding (BPE)
- **Category:** Data & representations
- **What it actually means:** 一种 subword 分词方法，反复合并频繁的相邻单元，从训练文本中构建固定词汇表。
- **Why it matters:** 它在词汇表大小与将罕见或未见词表示为较小单元的能力之间取得平衡。
- **In practice:** 仅在批准的语料库拆分上训练分词器，将合并规则与模型一同版本化，并检查它如何分割代码、多语言文本和空白字符。
- **Common confusion:** BPE 是分词器家族之一，而非描述每个模型如何创建 token 的通用说法。
- **Related terms:** Tokenization, Vocabulary, Token, Embedding
- **Sources:** [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

## C

### Calibration
- **Category:** Evaluation & safety
- **What it actually means:** 系统声明的置信度与在该置信度下预测正确的观察频率之间的一致性。
- **Why it matters:** 系统可能在平均意义上准确，但在人们依赖其分数的情形下过于自信，从而带来风险。
- **In practice:** 按置信度对预测分桶，比较置信度与经验准确率，当差距不可接受时重新校准或拒绝预测。
- **Common confusion:** Calibration 衡量置信度可靠性，而非整体准确率、事实性或推理质量。
- **Related terms:** Softmax, Evaluation (Eval), Precision & Recall, Logits
- **Sources:** [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

### Canary Release
- **Category:** Reliability & operations
- **What it actually means:** 一种部署策略，在扩大发布范围之前，先将新版本暴露给有限的流量或基础设施切片。
- **Why it matters:** 它限制缺陷的影响，并在新模型、提示、agent 或服务触及所有人之前提供生产证据。
- **In practice:** 将一小批 eligible 用户路由到发布版本，与对照组比较质量和运营指标，并在预定义失败时停止或回滚。
- **Common confusion:** Canary release 限制暴露范围；它不能替代部署前测试、审批或回滚准备。
- **Related terms:** Evaluation (Eval), Observability, Rollback, Verification Gate
- **Sources:** [Kubernetes Deployments: Canary Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#canary-deployment)

### Chain of Thought (CoT)
- **Category:** Prompting & context
- **What people say:** 要求模型展示其思考的每一步。
- **What it actually means:** 用于在产出答案之前分解任务的中间推理。提示可以请求可见的推理过程，而某些系统使用不返回给用户的内部推理。
- **Why it matters:** 分解有助于多步骤任务，但流畅的推理过程并不能证明答案是正确的或文本忠实地代表了模型的内部计算。
- **In practice:** 请求简洁的计划，独立检查结果，并要求可验证的计算或引用，而非依赖冗长的推理转录文本。
- **Common confusion:** Chain of thought 不能替代工具、测试或外部验证。
- **Learn it:** [Few-Shot and Chain of Thought](../phases/11-llm-engineering/02-few-shot-cot/)
- **Related terms:** Prompt Engineering, Verification Gate, Evaluation (Eval)

### Checkpoint
- **Category:** Agents & tools
- **What it actually means:** 用于从已知边界恢复的持久化快照。在工作流中，它存储操作状态和工件引用。在模型训练中，它可以存储参数、优化器状态、调度器状态和训练位置。
- **Why it matters:** 长运行的工作流和训练运行可以从中断中恢复，而无需重放已完成工作或丢失昂贵的进展。
- **In practice:** 在验证步骤后保存 agent 已接受的补丁和测试证据，或在关闭前保存训练运行的权重、优化器状态、随机状态和数据位置。
- **Common confusion:** 工作流 checkpoint 和模型训练 checkpoint 服务于相同的恢复目标，但保留不同的状态。它们都不是单纯的转录文本或无恢复元数据的权重文件。
- **Learn it:** [Checkpoint Save and Resume](../phases/19-capstone-projects/47-checkpoint-save-resume/); [Repository Memory and State](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **Related terms:** Agent State, Durable Execution, Parameter, Optimizer

### Chunked Prefill
- **Category:** Infrastructure & serving
- **What it actually means:** 一种 serving 技术，将长 prompt 的 prefill 工作划分为更小的可调度单元，使 prompt 处理可以与来自其他请求的 decode 工作交错执行。
- **Why it matters:** 一个长 prompt 否则会占用加速器并延迟活跃生成，即使在总吞吐量看起来健康时也会产生较差的尾部延迟。
- **In practice:** 从测量工作负载中选择 chunk 策略，考虑调度开销，并在混合 prompt 长度下比较 prefill 完成、decode 延迟和 goodput。
- **Common confusion:** Chunked prefill 改变了 prompt 计算的调度方式。它不会将用户上下文拆分为独立的语义块，也不会改变模型的上下文窗口。
- **Learn it:** [vLLM Serving Internals](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **Related terms:** Prefill, Decode Phase, Dynamic Batching, Tail Latency
- **Sources:** [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

### Chunking
- **Category:** Retrieval & generation
- **What people say:** 将文档拆分为片段。
- **What it actually means:** 在索引前将源材料划分为可检索的单元。chunk 边界、重叠、元数据和文档结构决定了检索是否能返回足够的上下文而不淹没提示。
- **Why it matters:** 合适的 chunking 策略取决于文档形状、查询类型、embedding 模型和评估结果。没有通用的 token 大小或重叠百分比。
- **In practice:** 保持标题和代码块完整，附加来源元数据，然后在调整大小之前在真实问题上测量检索质量。
- **Related terms:** RAG (Retrieval-Augmented Generation), Reranker, Grounding

### Circuit Breaker
- **Category:** AI-native development
- **What it actually means:** 一种可靠性控制，在失败超过阈值后临时停止对依赖项的调用，然后探测依赖项是否已恢复。
- **Why it matters:** 它防止重复的模型或工具失败消耗系统的延迟、预算和容量。
- **In practice:** 在重复的提供商超时时打开断路器，故障转移或返回受控响应，然后在冷却期后允许有限的关键性探测。
- **Common confusion:** Circuit breaker 响应依赖项健康状况。Rate limit 控制允许的请求量。
- **Related terms:** Retry with Backoff, Rate Limit, Model Router, Backpressure

### CNN (Convolutional Neural Network)
- **Category:** Models & inference
- **What people say:** 用于图像的神经网络。
- **What it actually means:** 一种使用卷积操作（在输入上滑动的滤波器）检测局部模式的神经网络。堆叠卷积检测越来越复杂的特征：边缘、纹理、物体。
- **Common confusion:** 卷积也适用于音频、时间序列和其他网格状数据。
- **Related terms:** Feature, Inductive Bias, Activation Function

### Coding Agent
- **Category:** AI-native development
- **What it actually means:** 专为软件工程任务设计的 agent，能够检查仓库、编辑文件、运行开发工具，并利用其输出来推进 scoped engineering 任务。
- **Why it matters:** 其价值取决于仓库上下文、工具权限、审查边界和验证，而不仅仅是代码生成质量。
- **In practice:** 给予 agent 一个问题、范围合约、仓库说明和测试命令；在接受补丁前审查结果补丁和证据。
- **Common confusion:** 仅建议文本的 coding assistant 不一定是 agent。Agent 通过工具行动并观察结果。
- **Learn it:** [Skill Discovery and Progressive Disclosure](../phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/)
- **Related terms:** Agent Harness, Repository Map, Patch, Scope Contract, Reviewer Agent

### Compensating Action
- **Category:** Agents & tools
- **What it actually means:** 当原始操作无法原子回滚时，一种有意的操作，以语义方式抵消已完成的副作用。
- **Why it matters:** 多步骤 agent 工作流跨越数据库和外部服务，后续失败无法通过单个事务撤消早期写入。
- **In practice:** 如果预订工作流扣款成功但预订失败，应发出跟踪的退款并保留两个事件，而非删除历史记录。
- **Common confusion:** 补偿是一种新的业务操作，而非时光倒流。它可能失败，因此需要幂等性、监控和升级机制。
- **Related terms:** Durable Execution, Idempotency, Checkpoint, Approval Gate
- **Sources:** [Sagas](https://dl.acm.org/doi/10.1145/38713.38742)

### Content Provenance
- **Category:** Security & governance
- **What it actually means:** 关于媒体或其他数字内容的来源和编辑历史的可验证信息，包括与其关联的行为者、工具、转换和声明。
- **Why it matters:** 生成系统使来源声明难以仅从外观推断，因此消费者和平台需要可检查的证据来了解内容是如何生产的。
- **In practice:** 将 provenance 声明绑定到内容上，使用受控身份签名，保留转换历史，并在证据缺失或无法验证时明确提示。
- **Common confusion:** Provenance 可以确立谁声明了历史和记录是否被修改。它不能证明所描绘的事件为真或内容无害。
- **Learn it:** [Watermarking, SynthID, Stable Signature, and C2PA](../phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/)
- **Related terms:** Data Provenance, Provenance Attestation, Audit Log, Grounding
- **Sources:** [C2PA Technical Specification](https://c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html)

### Context Compression
- **Category:** Prompting & context
- **What it actually means:** 在尝试保留后续模型决策所需信息的同时，减少源材料的 token 占用。
- **Why it matters:** 压缩可以使长任务适应预算，但每个被省略的细节都会造成模型丢失证据、约束或未决状态的风险。
- **In practice:** 逐字保留权威事实和标识符，总结冗余历史，附加来源指针，并在代表性任务上测试压缩上下文。
- **Common confusion:** 除非保留完整原文，否则压缩是有损的。较短的摘要并不自动等同于等效上下文。
- **Related terms:** Token Budget, Context Engineering, Progressive Disclosure, Handoff
- **Sources:** [LLMLingua](https://arxiv.org/abs/2310.05736)

### Context Engineering
- **Category:** Prompting & context
- **What it actually means:** 设计在每个步骤提供给模型的完整信息环境，包括指令、选定文件、检索证据、工具结果、示例、状态和输出约束。
- **Why it matters:** 模型性能失败往往因为相关证据缺失、过时、顺序错误或被噪声淹没。
- **In practice:** 构建包含目标、仓库规则、相关接口、最近工具输出和未决决策的紧凑任务包，然后随状态变化进行更新。
- **Common confusion:** Prompt engineering 侧重于指令措辞。Context engineering 还决定哪些证据和状态进入模型的工作上下文。
- **Learn it:** [Context Engineering](../phases/11-llm-engineering/05-context-engineering/)
- **Related terms:** Context Window, Progressive Disclosure, Agent State, Repository Map

### Context Window
- **Category:** Prompting & context
- **What people say:** 模型能记住多少。
- **What it actually means:** 在特定模型和 API 合约下，单次模型推理可用的最大 token 容量。该容量可能包括系统指令、消息、检索内容、工具交互和生成输出，并带有提供商特定的计费和输出限制。
- **Why it matters:** 仅当应用程序发送或重建时，对话历史才可用。大上下文窗口不保证每个包含的细节都能被可靠使用。
- **Common confusion:** 上下文是推理的临时输入。持久化记忆存储在模型外部，并选择性地回填到后续上下文中。
- **Learn it:** [Context Engineering](../phases/11-llm-engineering/05-context-engineering/)
- **Related terms:** Token Budget, Context Engineering, Prompt Cache, Agent State

### Continuous Batching
- **Category:** Infrastructure & serving
- **What it actually means:** 一种 serving 调度器，在迭代边界处添加和移除生成请求，而非等待固定批次中的所有请求完成。
- **Why it matters:** 自回归请求产生不同的输出长度，因此 continuous batching 可以在不迫使短请求等待最长请求的情况下保持加速器利用率。
- **In practice:** 当容量可用时接纳新请求，跟踪每个请求的延迟，并在 live batch 或 KV-cache 预算已满时施加 backpressure。
- **Common confusion:** Continuous batching 是一种推理调度策略，而非梯度累积或训练 batch size 技术。
- **Related terms:** Dynamic Batching, Decode Phase, Backpressure, Rate Limit
- **Sources:** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)

### Contrastive Learning
- **Category:** Math & training
- **What people say:** 通过比较学习。
- **What it actually means:** 通过在嵌入空间中拉近相似对、推远不相似对来进行训练。CLIP 使用此方法：匹配图像-文本对与非匹配对。
- **Related terms:** Embedding, Cosine Similarity, Loss Function

### Cosine Similarity
- **Category:** Data & representations
- **What people say:** 两个向量有多相似。
- **What it actually means:** 两个向量的归一化点积。它比较方向而非幅度，对于实值向量的范围是 -1 到 1。
- **Common confusion:** 高余弦相似性仅相对于 embedding 模型和数据分布有意义。它不能证明事实或语义等价。
- **Related terms:** Embedding, Semantic Search, Reranker

### Cost per Successful Task
- **Category:** AI-native development
- **What it actually means:** 总系统成本除以满足定义成功标准任务的数量，包括重试、失败运行、工具使用和评估开销。
- **Why it matters:** 一次便宜的模型调用如果经常失败或需要重复人工修正，可能导致昂贵的工作流。
- **In practice:** 在 100 个仓库任务上测量提供商费用和基础设施成本，然后除以补丁通过测试和审查的任务数量。
- **Common confusion:** Cost per token 衡量使用量。Cost per successful task 衡量有效产出。
- **Related terms:** Evaluation (Eval), Retry with Backoff, Model Router, Verification Gate

### Cross-Attention
- **Category:** Multimodal systems
- **What it actually means:** 查询表示来自一个序列或表示，而键和值来自另一个序列或表示的注意力机制。
- **Why it matters:** 它让一个流能以可学习的方式从另一个流检索信息，例如语言 token 关注视觉特征。
- **In practice:** 明确哪个流提供查询、键和值，对缺失或无效位置应用掩码，并检查当一种模态被消融时模型是否仍能正常工作。
- **Common confusion:** Cross-attention 并非本质上是多模态的。它可以连接两个文本序列或其他表示；而 self-attention 的查询、键和值均来自同一序列表示。
- **Related terms:** Attention, Self-Attention, Vision-Language Model (VLM), Multimodal Fusion
- **Sources:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Cross-Entropy
- **Category:** Math & training
- **What people say:** 分类 loss。
- **What it actually means:** 一种基于分配给目标结果的负对数概率的 loss。在 next-token 训练中，当模型对观测到的下一个 token 赋予低概率时会受到惩罚。
- **Common confusion:** 仅当平均和对数底数定义一致时，Perplexity 才是平均交叉熵的指数。
- **Related terms:** Loss Function, Softmax, Perplexity

### CUDA
- **Category:** Models & inference
- **What people say:** GPU 编程。
- **What it actually means:** NVIDIA 的通用计算平台和编程模型，适用于兼容的 GPU。深度学习框架使用 CUDA 库和内核并行执行许多张量操作。
- **Common confusion:** GPU 加速不等同于 CUDA；还存在其他硬件和软件栈。
- **Related terms:** Tensor, Mixed Precision, JAX

## D

### Data Augmentation
- **Category:** Math & training
- **What people say:** 制造更多训练数据。
- **What it actually means:** 创建修改后的样本，如变换的图像、扰动的音频或改写的文本，以增加训练多样性，而无需收集全新的源数据。当转换保留任务信号时，它可以减少过拟合。
- **Common confusion:** Augmentation 必须保留你希望模型学习的目标标签或行为。
- **Related terms:** Overfitting, Epoch, Eval Set

### Data Classification
- **Category:** Security & governance
- **What it actually means:** 将数据分配到文档化的敏感性或影响类别，以便处理、访问、保留、共享和事件响应规则遵循披露或丢失的后果。
- **Why it matters:** 如果源文档、提示、追踪记录和生成工件被视为同等敏感，AI 流水线无法应用相称的控制措施。
- **In practice:** 在摄入时分类数据，在派生工件中传递标签，按类别限制工具和目标，并定义转换或聚合后标签如何变化。
- **Common confusion:** Data classification 描述保护需求。它与机器学习分类任务或数据准确性声明不同。
- **Related terms:** Data Minimization, Trust Boundary, Least Privilege, Audit Log
- **Sources:** [NIST SP 1800-39 Initial Public Draft: Data Classification Practices](https://www.nccoe.nist.gov/sites/default/files/2026-02/nist-sp-1800-39-ipd.pdf); [NIST FIPS 199: Federal Information and Information System Categorization](https://csrc.nist.gov/pubs/fips/199/final)

### Data Deduplication
- **Category:** Data & representations
- **What it actually means:** 检测和删除数据集内或跨数据集的精确和近似重复样本。
- **Why it matters:** 重复会扭曲训练分布、增加记忆化、泄露测试材料并使评估结果看起来比实际更好。
- **In practice:** 规范化内容，使用精确哈希和相似度方法，审查边界集群，并记录每个样本的版本和删除规则。
- **Common confusion:** Deduplication 不是普通的数据清洗。两个独立记录可能合法地共享文本，而两个改写版本仍可能携带相同的泄露信息。
- **Related terms:** Data Provenance, Benchmark Contamination, Dataset Split, Overfitting
- **Sources:** [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

### Data Exfiltration
- **Category:** Security & governance
- **What it actually means:** 受保护数据未经授权的转移，从系统或信任区域到不允许接收它的人员、工具、服务或存储位置。
- **Why it matters:** 即使原始数据存储完好，agent 也可能通过生成文本、工具参数、URL、日志或副作用暴露机密。
- **In practice:** 最小化可读数据，允许目标白名单，检查出站工具调用，脱敏敏感字段，并对跨信任边界的异常传输发出警报。
- **Common confusion:** Exfiltration 关注未经授权的移动或披露。授权组件的普通数据检索不是 exfiltration，尽管后续使用可能构成。
- **Learn it:** [EchoLeak and CVEs for AI](../phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai/)
- **Related terms:** Trust Boundary, Least Privilege, Indirect Prompt Injection, Audit Log
- **Sources:** [NIST SP 800-53 Rev. 5: AC-4 Information Flow Enforcement](https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/sp800-53r5-controls.xlsx)

### Data Leakage
- **Category:** Data & representations
- **What it actually means:** 在训练或特征构建期间无意中使用了在真实预测点不可用或属于预留评估边界的信息。
- **Why it matters:** 泄漏会产生乐观的指标，但当系统遇到真正未见过的输入时会崩溃。
- **In practice:** 在拟合预处理器之前拆分数据，将未来信息排除在历史特征之外，并将测试标签和 benchmark 答案与提示和微调循环隔离。
- **Common confusion:** 泄漏不仅限于重复行。全局归一化统计、时间戳、目标派生特征和重复的测试驱动提示编辑都可能导致信息泄漏。
- **Related terms:** Dataset Split, Benchmark Contamination, Eval Set, Data Provenance
- **Sources:** [scikit-learn: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### Data Lineage
- **Category:** Security & governance
- **What it actually means:** 记录数据工件如何跨来源、转换、连接、过滤、版本和下游用途派生的轨迹。
- **Why it matters:** 当来源被修正、撤销或发现不安全时，lineage 能识别哪些数据集、embeddings、评估和模型工件可能受到影响。
- **In practice:** 为输入和输出提供稳定标识符，记录每个转换和版本，保留父子关系，并测试受影响来源是否能追溯到每个派生工件。
- **Common confusion:** Data provenance 广泛解释来源和保管。Lineage 强调转换路径和数据工件之间的依赖关系。
- **Related terms:** Data Provenance, Datasheet for Datasets, Audit Log, Content Provenance
- **Sources:** [W3C PROV-O](https://www.w3.org/TR/prov-o/)

### Data Minimization
- **Category:** Security & governance
- **What it actually means:** 对于个人数据，将收集、处理、暴露和保留的内容限制为指定目的所必需的。团队可以将同样的纪律应用于敏感的非个人数据，作为工程控制。
- **Why it matters:** 放入提示、追踪、缓存或工具调用的每个不必要字段都会增加隐私暴露和滥用或泄露的潜在影响。
- **In practice:** 在收集前定义必需字段，在最早边界脱敏或聚合，设置保留限制，并在保留前验证可选上下文是否能改善测量的任务结果。
- **Common confusion:** Minimization 并不意味着不保留数据。它意味着能够针对声明的目的证明每个数据元素、用途、接收者和保留期限的合理性。
- **Related terms:** Purpose Limitation, Data Classification, Least Privilege, Context Engineering
- **Sources:** [General Data Protection Regulation, Article 5(1)(c)](https://eur-lex.europa.eu/eli/reg/2016/679/oj)

### Data Provenance
- **Category:** Data & representations
- **What it actually means:** 关于数据起源、谁或什么转换了它、使用了哪些版本以及派生工件如何与其来源关联的可追溯信息。
- **Why it matters:** 你需要 provenance 来重现结果、遵守使用约束、调查污染，并在来源变更时移除受影响的数据。
- **In practice:** 分配不可变的数据集版本，记录转换任务和来源标识符，并将 lineage 元数据带入 embeddings、eval 案例和模型工件。
- **Common confusion:** 来源 URL 只是 provenance 的一部分；它不描述收集时间、许可、过滤、转换或下游用途。
- **Related terms:** Dataset Split, Data Deduplication, Provenance Attestation, Grounding
- **Sources:** [W3C PROV Overview](https://www.w3.org/TR/prov-overview/)

### Dataset Split
- **Category:** Data & representations
- **What it actually means:** 将样本文档化分区为用于拟合、开发决策和最终评估的独立子集。
- **Why it matters:** 分离防止用于选择系统的证据同时作为已选系统泛化能力的独立证明。
- **In practice:** 按真实部署单元（如用户、仓库、组织或时间）拆分，而非随机划分相关行。
- **Common confusion:** 随机拆分不一定独立。近似重复、未来观察或来自同一实体的记录可能跨越边界。
- **Related terms:** Eval Set, Overfitting, Data Leakage, Distribution Shift
- **Sources:** [Datasheets for Datasets](https://cacm.acm.org/research/datasheets-for-datasets/)

### Datasheet for Datasets
- **Category:** Security & governance
- **What it actually means:** 数据集的动机、组成、收集过程、预处理、用途、分布、维护和已知限制的结构化文档。
- **Why it matters:** 数据集仅因可用并不意味安全或适用。下游构建者需要关于其创建方式和假设失效点的证据。
- **In practice:** 随版本化数据集发布 datasheet，指定可回答问题的人，记录排除的群体和转换，并在数据集变更时更新文档。
- **Common confusion:** Datasheet 记录证据和预期用途。它不是许可证、质量担保或部署特定评估的替代品。
- **Learn it:** [Model, System, and Dataset Cards](../phases/18-ethics-safety-alignment/26-model-system-dataset-cards/)
- **Related terms:** Data Lineage, Data Provenance, Model Card, Dataset Split
- **Sources:** [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

### Deadline Propagation
- **Category:** Reliability & operations
- **What it actually means:** 将剩余的端到端时间预算传递给下游调用，使每个依赖项知道原始请求还能有效等待多长时间。
- **Why it matters:** 独立的超时可能超过用户截止时间，并使已放弃的工作在结果不再有用后继续消耗容量。
- **In practice:** 在入口设置单次请求截止时间，为每个下游调用减去已过时间，取消过期工作，并记录哪个边界耗尽了预算。
- **Common confusion:** Deadline 是绝对或剩余的完成边界。Retry delay 控制另一次尝试的开始时间，必须适应同一预算。
- **Related terms:** Retry with Backoff, Retry Budget, Tail Latency, Service Level Objective (SLO)
- **Sources:** [gRPC Deadlines](https://grpc.io/docs/guides/deadlines/)

### Decode Phase
- **Category:** Infrastructure & serving
- **What it actually means:** 自回归推理的迭代阶段，在输入前缀被处理后逐步生成新 token。
- **Why it matters:** Decode 工作与 prefill 的计算、内存和调度行为不同，因此单一聚合延迟数字可能隐藏实际的 serving 瓶颈。
- **In practice:** 分别测量 token 间延迟和输出吞吐量，考虑 KV-cache 占用，并测试活跃 decode 与新 prefill 共享容量的混合工作负载。
- **Common confusion:** Decode phase 不是 encoder-decoder 模型的 decoder 组件。它命名的是运行时生成阶段。
- **Learn it:** [Disaggregated Prefill and Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **Related terms:** Prefill, Autoregressive, KV Cache, Time per Output Token (TPOT)
- **Sources:** [DistServe](https://arxiv.org/abs/2401.09670)

### Decoder
- **Category:** Models & inference
- **What people say:** 模型的输出端。
- **What it actually means:** 将表示映射到输出的组件。在 encoder-decoder transformer 中，decoder 使用掩码 self-attention 和 cross-attention 生成输出。Decoder-only 语言模型则从单个因果堆栈生成。
- **Related terms:** Encoder, Transformer, Autoregressive

### Decoding Strategy
- **Category:** Models & inference
- **What it actually means:** 将模型的 next-token 分数序列转换为选中 token 和完成输出的算法。
- **Why it matters:** 贪婪选择、采样、截断和搜索可以从相同的 logits 产生不同的质量、多样性、延迟和可重复性。
- **In practice:** 在 eval 配置中定义任务的解码设置、停止规则和种子行为，以便公平比较结果。
- **Common confusion:** Decoding 改变输出选择方式；它不改变模型的训练参数或添加知识。
- **Related terms:** Autoregressive, Temperature, Top-k Sampling, Nucleus Sampling (Top-p)
- **Sources:** [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

### Defense in Depth
- **Category:** Security & governance
- **What it actually means:** 在多个系统边界使用独立的预防、检测和纠正控制，使单个控制的失效不决定结果。
- **Why it matters:** AI 系统结合了概率模型、不受信任的内容、工具和外部服务，使得任何单一过滤器或提示都不足以作为安全边界。
- **In practice:** 将指令控制与窄权限、沙箱、schema 验证、关键操作审批、监控和经过测试的恢复路径配对。
- **Common confusion:** 更多控制不自动更好。各层应针对不同的失效模式并保持可测试性，而非重复相同的假设。
- **Related terms:** Guardrails, Sandbox, Least Privilege, Trust Boundary
- **Sources:** [NIST Glossary: Defense in Depth](https://csrc.nist.gov/glossary/term/defense_in_depth)

### Delegation
- **Category:** Agents & tools
- **What it actually means:** 将受限的子任务分配给另一人或 agent，并附上所需的上下文、权限、输出合约和返回条件。
- **Why it matters:** 显式委派 enables specialization and parallel work，同时不丧失所有权、范围或将结果集成的能力。
- **In practice:** 给予 reviewer agent 精确的文件、评分标准、证据和截止时间，并要求其返回 findings 而非静默修改主工件。
- **Common confusion:** 向另一 agent 发送模糊消息不是可靠的委派。接收者需要范围合约和定义的交接。
- **Related terms:** Scope Contract, Handoff, Reviewer Agent, Orchestration
- **Sources:** [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

### Dense Retrieval
- **Category:** Retrieval & generation
- **What it actually means:** 将查询和候选项嵌入向量表示，并通过相似度函数对候选项排序的首阶段检索。
- **Why it matters:** 它可以检索共享少量精确词的改写和语义匹配，补充 BM25 等词汇方法。
- **In practice:** 为领域训练或选择 embedding 模型，索引候选向量，并在连接结果到生成之前评估检索召回率。
- **Common confusion:** Dense retrieval 不是 reranker。它搜索整个集合，而 reranker 对较小的候选集重新评分。
- **Related terms:** Embedding, Semantic Search, BM25, Hybrid Retrieval
- **Sources:** [Dense Passage Retrieval](https://aclanthology.org/2020.emnlp-main.550/)

### Diffusion Model
- **Category:** Models & inference
- **What people say:** 从噪声生成图像的模型。
- **What it actually means:** 一种围绕渐进加噪过程和学习的去噪过程训练的生成模型。采样通常从噪声开始并应用重复去噪步骤，有时在学习的潜在空间中进行。
- **Common confusion:** Diffusion 是通用生成框架，而非仅用于图像的技术。
- **Related terms:** Latent Space, VAE (Variational Autoencoder), Inference

### Disaggregated Serving
- **Category:** Infrastructure & serving
- **What it actually means:** 一种 serving 架构，在单独配置的 worker 池中运行 prefill 和 decode 工作，并在它们之间转移所需的注意力状态。
- **Why it matters:** Prefill 和 decode 对硬件的压力不同，因此独立池可以根据各自瓶颈进行 sizing 和调度，而非在单一队列中竞争。
- **In practice:** 测量状态转移成本，通过兼容的模型版本路由请求，从各自的需求信号扩展每个池，并测试阶段间的故障恢复。
- **Common confusion:** Disaggregation 分离运行时阶段。它不会将一个模型拆分为阶段内的 tensor 或 pipeline 并行分片。
- **Learn it:** [Disaggregated Prefill and Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **Related terms:** Prefill, Decode Phase, Model Serving, Goodput
- **Sources:** [DistServe](https://arxiv.org/abs/2401.09670)

### Distribution Shift
- **Category:** Evaluation & safety
- **What it actually means:** 构建或评估系统所使用的数据分布与部署后遇到的分布之间的差异。
- **Why it matters:** 模型可能通过预留测试，但当用户、任务、语言、工具或操作条件变化时会失败。
- **In practice:** 定义预期的部署切片，按切片监控性能和输入特征，并将新失败添加到版本化的 eval set 中。
- **Common confusion:** Distribution shift 不总是 model drift。模型可能未变，而其环境或用户群体发生变化。
- **Related terms:** Dataset Split, Eval Set, Overfitting, Model Card
- **Sources:** [WILDS](https://proceedings.mlr.press/v139/koh21a.html)

### DPO (Direct Preference Optimization)
- **Category:** Math & training
- **What people say:** 无需独立 reward model 阶段的偏好训练。
- **What it actually means:** 一种偏好优化目标，直接从相对于参考策略的偏好和拒绝响应对训练策略。它在此阶段避免运行显式 reward model 和强化学习循环。
- **Common confusion:** DPO 仍然依赖偏好数据的质量和覆盖范围，并不能消除评估或 alignment 风险。
- **Learn it:** [Direct Preference Optimization](../phases/10-llms-from-scratch/08-dpo/)
- **Sources:** [Direct Preference Optimization paper](https://arxiv.org/abs/2305.18290)
- **Related terms:** RLHF (Reinforcement Learning from Human Feedback), SFT (Supervised Fine-Tuning), Alignment

### Dropout
- **Category:** Math & training
- **What people say:** 随机关闭激活。
- **What it actually means:** 在训练期间，随机将部分激活设为零，鼓励网络不依赖单一激活路径。通常在标准推理中禁用，尽管 Monte Carlo dropout 故意保持激活以估计不确定性。
- **Related terms:** Overfitting, Weight Decay, Activation Function

### Durable Execution
- **Category:** Agents & tools
- **What it actually means:** 运行工作流，使其状态和已完成步骤在进程崩溃、重启或长时间等待后仍能存活，而无需重做已确认的副作用。
- **Why it matters:** Agent 任务通常跨模型调用、工具、审批和外部系统。瞬态进程不应是唯一进展记录。
- **In practice:** 持久化每个工作流转换，对外部写入使用幂等键，并在 worker 重启后从最新 checkpoint 恢复。
- **Common confusion:** Durable execution 不会自动使每个操作安全。副作用仍需要幂等性和补偿规则。
- **Related terms:** Checkpoint, Agent State, Idempotency, Approval Gate

### Dynamic Batching
- **Category:** Infrastructure & serving
- **What it actually means:** 一种运行时策略，根据兼容形状、最大大小、优先级和允许的队列延迟从排队请求中形成推理批次。
- **Why it matters:** 分组请求可以提高硬件利用率，但当流量稀疏或请求差异显著时，等待批次可能使延迟恶化。
- **In practice:** 从测量的延迟目标设置队列延迟和批次限制，分离不兼容的请求形状，并在真实到达率下比较吞吐量与尾部延迟。
- **Common confusion:** Dynamic batching 从排队工作中组装批次。Continuous batching 在自回归生成运行时改变成员。
- **Learn it:** [vLLM Serving Internals](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **Related terms:** Admission Control, Continuous Batching, Saturation, Tail Latency
- **Sources:** [NVIDIA Triton: Models and Schedulers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#scheduling-and-batching)

## E

### Early Fusion
- **Category:** Multimodal systems
- **What it actually means:** 在大多数任务特定建模发生之前，将来自多个模态的原始或低级表示组合。
- **Why it matters:** 早期交互可以揭示细粒度的跨模态关系，但也需要兼容的表示并小心处理对齐和缺失输入。
- **In practice:** 将每个模态转换为声明的 token 或特征表示，保留来源和位置标记，在共享骨干网络前融合它们，并与单模态和 late fusion 基线比较。
- **Common confusion:** Early fusion 描述的是架构中流的组合位置。它不保证模型学习到有用的对齐。
- **Learn it:** [Chameleon Early-Fusion Tokens](../phases/12-multimodal-ai/11-chameleon-early-fusion-tokens/)
- **Related terms:** Late Fusion, Multimodal Fusion, Modality Alignment, Token
- **Sources:** [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818); [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

### Eigenvalue
- **Category:** Math & training
- **What people say:** 用于 PCA 的矩阵属性。
- **What it actually means:** 描述线性变换如何缩放相应非零特征向量而不改变其方向的标量。在协方差矩阵 PCA 中，较大的特征值对应方差较大的方向。
- **Related terms:** Tensor, Feature, Latent Space

### Embedding
- **Category:** Data & representations
- **What people say:** 表示意义的向量。
- **What it actually means:** 从离散项（词、图像、用户）到连续空间中稠密向量的学习映射，相似项最终靠近在一起。
- **Common confusion:** 相似性取决于模型、训练目标和度量。一个 embedding 空间中的距离不会转移到另一个空间。
- **Why it's called that:** 项被放置或嵌入到几何表示空间中。
- **Learn it:** [Embeddings](../phases/11-llm-engineering/04-embeddings/)
- **Related terms:** Cosine Similarity, Semantic Search, Vector Database

### Encoder
- **Category:** Models & inference
- **What people say:** 模型的输入端。
- **What it actually means:** 将输入转换为表示的组件。Transformer encoder 通常使用受掩码约束的非因果 self-attention，使每个位置可以融入来自输入跨度的上下文。
- **Common confusion:** Encoder-only 模型可以通过任务头产生输出，尽管它们通常不用于自回归文本生成。
- **Related terms:** Decoder, Transformer, Embedding

### Epoch
- **Category:** Math & training
- **What people say:** 遍历训练数据一次。
- **What it actually means:** 对定义的训练数据集的一次遍历。在分布式或采样训练中，epoch 的确切实现取决于数据加载器和采样策略。
- **Common confusion:** 更多 epoch 不保证更好的泛化；应在预留数据上评估。
- **Related terms:** Batch Size, Overfitting, Eval Set

### Error Budget
- **Category:** Reliability & operations
- **What it actually means:** 在服务级别目标在其测量窗口内耗尽之前，由 SLO 允许的不成功服务量。
- **Why it matters:** 它为可靠性和产品工作提供了共享决策边界：团队可以在剩余预算内支出变更，而在用户可见故障消耗预算时减缓风险。
- **In practice:** 从 SLO 推导预算，按原因和用户分段跟踪消耗率，在耗尽前定义发布行动，并在事件后避免重置会计。
- **Common confusion:** Error budget 不是引发事件配额。它是从用户可见可靠性目标派生的操作策略。
- **Related terms:** Service Level Objective (SLO), Service Level Indicator (SLI), Availability, Incident Response
- **Sources:** [Google SRE Workbook: Error Budget Policy](https://sre.google/workbook/error-budget-policy/)

### Eval Set
- **Category:** Evaluation & safety
- **Aliases:** Evaluation set
- **What it actually means:** 用于测量 AI 系统在定义能力或风险上的版本化输入、预期属性、评分规则和元数据集合。
- **Why it matters:** 可重复的集合将模糊的质量声明转化为可比证据，并在提示、模型、工具或检索变更后捕获回归。
- **In practice:** 在审查过的数据集中保留代表性支持问题、对抗性指令、预期引用和失败标签，与开发示例分离。
- **Common confusion:** 开发 eval 指导迭代，最终预留测试估计固定选择后的性能，标准化 benchmark 支持共享协议下的比较。针对任何预留集的重复调整会泄露测试信息并膨胀结果。
- **Learn it:** [Eval-Driven Agent Development](../phases/14-agent-engineering/30-eval-driven-agent-development/)
- **Related terms:** Evaluation (Eval), Regression Test, LLM-as-a-Judge, Verification Gate

### Evaluation (Eval)
- **Category:** Evaluation & safety
- **Aliases:** Eval
- **What it actually means:** 使用明确成功标准、数据、评分器和审查程序，在代表性任务上测量模型或系统行为的定义过程。
- **Why it matters:** 如果成功仅是几个演示的主观印象，你就无法改进可靠性。
- **In practice:** 在更改检索前后运行相同的客户支持场景，评分正确性和引用支持，并按类别检查失败。
- **Common confusion:** Benchmark 分数仅是评估结果之一，而非生产质量的完整描述。
- **Learn it:** [LLM Evaluation](../phases/11-llm-engineering/10-evaluation/)
- **Related terms:** Eval Set, LLM-as-a-Judge, Cost per Successful Task, Regression Test

### Exact Match (EM)
- **Category:** Evaluation & safety
- **What it actually means:** 一种度量，仅在输出的规范化表示与已接受参考答案完全相等时计为正确。
- **Why it matters:** 对于具有单一规范答案的任务，它是确定性的且易于审计，但不提供部分得分。
- **In practice:** 在评估前定义规范化方式和所有 accepted references，然后在多种输出可能有效时与任务特定检查配对。
- **Common confusion:** 低 exact-match 分数可能反映无害的格式差异，而匹配的字符串在上下文中仍可能不受支持或不安全。
- **Related terms:** ROUGE, Eval Set, Structured Output, Pass@k
- **Sources:** [SQuAD](https://aclanthology.org/D16-1264/)

### Expert Parallelism
- **Category:** Infrastructure & serving
- **What it actually means:** 将 mixture-of-experts 子网络跨设备分布，并将每个 token 的激活路由到托管其选中 experts 的设备。
- **Why it matters:** Sparse experts 增加模型容量而无需对每个 token 执行所有 experts，但路由引入了通信、负载均衡和放置约束。
- **In practice:** 按 expert 测量 token 分布，预留通信带宽，有意识地限制或路由溢出，并在流量产生不均匀 expert 需求时测试质量。
- **Common confusion:** Expert parallelism 分区由路由器选择的 experts。Tensor parallelism 分区层内的张量操作。
- **Learn it:** [Mixture of Experts](../phases/07-transformers-deep-dive/11-mixture-of-experts/)
- **Related terms:** MoE (Mixture of Experts), Tensor Parallelism, Pipeline Parallelism, Model Serving
- **Sources:** [GShard](https://arxiv.org/abs/2006.16668)

## F

### Feature
- **Category:** Data & representations
- **What people say:** 数据集中的列。
- **What it actually means:** 数据的单个可测量属性。在经典 ML 中，你手工工程特征。在深度学习中，网络从原始数据自动学习特征。
- **Common confusion:** 存储的列可能包含多个有用特征，而学习到的表示可能包含无简单人类标签的特征。
- **Related terms:** Embedding, Latent Space, Inductive Bias

### Few-Shot
- **Category:** Prompting & context
- **What people say:** 在提示中给模型几个示例。
- **What it actually means:** 在目标输入前包含一小组演示的 in-context learning，使模型能够推断期望的任务、格式或决策边界。
- **Why it matters:** 示例质量和覆盖范围比通用示例数量更重要。差或矛盾的演示会降低可靠性。
- **Related terms:** Zero-Shot, In-Context Learning, Prompt Engineering, Context Window

### Fine-tuning
- **Category:** Math & training
- **What people say:** 在你的数据上训练模型。
- **What it actually means:** 从预训练参数开始在更窄数据集或目标上继续训练。根据方法，你可以更新所有参数、选定参数或添加的 adapter 参数。
- **Why it matters:** Fine-tuning 可以适配行为、风格、格式或任务性能，但不是保持事实新鲜或可追溯的可靠替代方案。
- **Common confusion:** Fine-tuning 可以影响编码知识，但它不会简单地将记录追加到模型内的可搜索数据库。
- **Learn it:** [Fine-Tuning and LoRA](../phases/11-llm-engineering/08-fine-tuning-lora/)
- **Related terms:** SFT (Supervised Fine-Tuning), LoRA (Low-Rank Adaptation), QLoRA, RAG (Retrieval-Augmented Generation)

### Flaky Test
- **Category:** AI-native development
- **What it actually means:** 在没有相关代码变更或预期测试环境变更的情况下，可在等效运行中通过或失败的测试。
- **Why it matters:** 不稳定性削弱验证门，并可能训练人员或 agent 忽略真实失败或重试直到获得虚假通过。
- **In practice:** 保留失败的 seed 和环境，仅在有负责人和截止时间的情况下隔离，然后修复不受控制的时间、并发、网络、顺序或共享状态依赖。
- **Common confusion:** 始终暴露间歇性产品 bug 的测试是有价值的证据，不一定是 flaky test。
- **Related terms:** Regression Test, Test Oracle, Retry with Backoff, Verification Gate
- **Sources:** [De-Flake Your Tests](https://conferences.computer.org/icsme/pdfs/ICSME2020-1oOutvkGTwF4GyVvNtr3Mm/561900a736/561900a736.pdf)

### FlashAttention
- **Category:** Infrastructure & serving
- **What it actually means:** 一种精确注意力算法，通过分块计算减少加速器内存层级间的传输，同时避免在高带宽内存中物化完整注意力矩阵。
- **Why it matters:** 注意力可能受限于内存移动而非算术，尤其对于长序列，因此 IO 感知内核可以改善可用速度和内存效率。
- **In practice:** 使用模型形状、掩码、dtype 和硬件支持的 kernel，验证数值容差，并对端到端延迟进行基准测试，而非将论文结果引用为固定倍数。
- **Common confusion:** FlashAttention 改变注意力计算方式，而非其目标的数学注意力结果。它与 KV caching 和量化分开。
- **Learn it:** [KV Cache and Flash Attention](../phases/07-transformers-deep-dive/12-kv-cache-flash-attention/)
- **Related terms:** Attention, Self-Attention, KV Cache, Mixed Precision
- **Sources:** [FlashAttention](https://arxiv.org/abs/2205.14135)

### Function Calling
- **Category:** Agents & tools
- **What people say:** 模型使用工具。
- **What it actually means:** 一种提供商或应用程序接口，模型通过它发出命名工具和参数的结构化请求。应用程序代码验证请求、执行操作，并可将结果返回给下一轮模型步骤。
- **Common confusion:** 模型请求函数调用；你的可信代码决定是否及如何执行它。Function calling 本身不是完整 agent。
- **Learn it:** [Function Calling](../phases/11-llm-engineering/09-function-calling/)
- **Related terms:** Structured Output, Tool Contract, Agent, MCP (Model Context Protocol)

## G

### GAN (Generative Adversarial Network)
- **Category:** Models & inference
- **What people say:** 两个神经网络在训练期间竞争。
- **What it actually means:** 生成器网络试图创建逼真数据，判别器网络试图区分真假。它们共同训练：生成器变得更擅长欺骗判别器，判别器变得更擅长检测伪造。
- **Related terms:** Loss Function, Latent Space, Diffusion Model

### Goodput
- **Category:** Infrastructure & serving
- **What it actually means:** 在声明工作负载下，
