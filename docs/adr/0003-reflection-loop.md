# ADR 0003: 添加 Writer / Reasoner 反思循环作为 supervisor 后置子图

- **状态**: Accepted
- **决策者**: research-agent 维护者
- **阶段**: 多 Agent 编排 — 反思

## 背景

Phase 4.7 之后，supervisor 能可靠地将工作路由到正确的专家
（data / report / coder / knowledge / news / sentiment）并在每次
会话结束时输出最终综合。对真实金融研究提示词的人工抽查发现，
该综合中有两种反复出现的质量缺陷：

1. **引用丢失。** 专家返回"据 2024 年报披露，归母净利润 1.23 亿元
   （来源：page 12）"，但 supervisor 的最终回答说"归母净利润约 1.2 亿元"
   — 四舍五入、无引用、用户无法验证。
2. **子问题偏移。** 当用户提出编号问题 `(1) … (2) … (3) …` 时，
   supervisor 有时返回一段自信的单段摘要，只回答了三个中的两个。
   缺失的子问题被静默丢弃，而非标记出来。

这两种失败模式都无法通过在 supervisor 提示词中增加另一条反幻觉规则来修复
— supervisor 的提示词已经很长了，且 supervisor 的职责是**路由**，
而非综合质量控制。正确的架构举措是**第二遍**：根据原始问题和专家输出
对综合进行评分，当分数过低时触发重写。这是"Self-RAG / Reflexion"模式，
应用在综合边界而非每次检索步骤。

## 决策

引入一个**反思子图**，采用评审优先的拓扑结构，并将其作为可选的 supervisor 后置阶段接入父级 `StateGraph`。

组件：

- `src/research_agent/graph/reflection.py` —
  `build_reflection_subgraph(model_router, pass_threshold,
  max_iterations)` 返回一个编译好的子图，包含三个节点
  （`critic`、`writer`、`finalize`）和以下边：

  ```
  START → critic → ?
                  ├─ score ≥ threshold 或 iter ≥ cap → finalize → END
                  └─ 否则 → writer → critic（循环）
  ```

- 评审使用 `ModelTier.LIGHT`（评分是一个分类任务；不需要旗舰模型）。 写作使用 `ModelTier.HEAVY`，因为重写本身**就是**在严格约束下的综合。
- 评审输出一个严格的 JSON 裁决，包含五轴评分
  （忠实性、引用、完整性、结构、清晰度）。写作者获得原始对话记录 +
  上一版草稿 + 评审的反馈要点，被指示保留每个来自专家的数字、 按专家角色名添加引用，并回答每个子问题。
- 子图跨迭代跟踪 `best_draft` / `best_score`，`finalize` 返回 观察到的**最高分**草稿 — 不一定是最新的 — 以防止重写降低分数。

接入 supervisor：

- `build_research_supervisor(..., enable_reflection: bool = False,
  reflection_pass_threshold: float = 0.85,
  reflection_max_iterations: int = 2)` — 当
  `enable_reflection=True` 时，该函数编译一个父级 `StateGraph`，
  包含两个节点（`supervisor` → `reflection`），并将 checkpointer
  附加到父级而非内部 supervisor。当 `enable_reflection=False` 时，
  函数返回与遗留编译 supervisor 完全相同的结果 — 默认路径零行为变化。
- `Settings.reflection_enabled`（以及阈值 + 迭代次数旋钮）使反思
  成为可从 `.env` 切换的运行时开关 — 部署中无需代码变更即可开启或关闭。

校准：

- `pass_threshold = 0.85` 是评审提示词中"生产质量，直接发布"区间
  （≥ 0.90）与"称职，小问题，轻度重写后发布"区间（0.75–0.89）
  之间的边界。分数 0.85 的草稿不触发重写；分数 0.84 的触发。
- `max_iterations = 2` 次写作调用。最坏情况 LLM 预算是
  3 次评审 + 2 次写作 = 在 supervisor 自身调用之上 5 次 LLM 调用。
  对于中位数请求（单次高质量草稿在首次评审时即得分 ≥ 0.90），
  成本是**一次**额外的 LIGHT 调用和零次重写。

## 考虑过的替代方案

1. **将评审+写作作为两个额外节点内联到 `langgraph_supervisor` 图中。**
   - 优势: 单个扁平图，无父级包装。
   - 劣势: `create_supervisor` 没有自然的接缝来附加综合后逻辑；
     需要使用自定义 output_mode 并通过 reducer 截取 supervisor
     的最终消息。这对 `langgraph_supervisor` 升级来说既侵入性强又脆弱。
   - 劣势: 反思循环概念上是一个不同的图（不同的状态形状：`draft` /
     `critique` / `iteration` / `history` 不是 supervisor 的关注点）
     — 将它们放在一个扁平图中会混淆状态 schema。
2. **在 `supervisor.ainvoke` 之后用纯 Python 做后处理。**
   - 优势: 最简单的实现，无额外图接线。
   - 劣势: LangSmith / LangGraph Studio 在可视化中丢失反思节点 —
     它们看不到图外部运行的任何东西。追踪是项目的卖点之一；
     隐藏循环违背了这一点。
   - 劣势: checkpointer 无法再覆盖反思阶段。重写中途的崩溃不会 从评审处恢复。
3. **在 supervisor 内部使用单个自我批评提示词。**
   "重读你的草稿。如果有缺失引用或跳过的子问题，重写它。"supervisor
   提示词已经包含了一个更温和版本的此指令。经验上，模型不能在产出草稿
   的同一轮内可靠地自我批评 — 它们倾向于验证自己的输出。独立的评审
   调用，最好使用不同的模型层，能打破这种偏见。
4. **在专家级别而非 supervisor 级别进行反思。**
   - 优势: 更早地捕获错误，更靠近错误源头。
   - 劣势: 每会话 6 倍的 LLM 调用（每个专家一个评审而非
     为 supervisor 设一个评审）。专家也不太可能"丢失引用" — 它们倾向于逐字转发专家数据。缺陷在综合接缝处，
     所以修复也属于综合接缝处。

## 后果

### 正面

- **多子问题提示词上更高的回答质量。** 对 12 个代表性金融研究提示词
  的人工评估：开启反思后，11/12 包含所有子问题答案（关闭时为 8/12）；
  引用密度（每 100 token 的命名来源提及数）大约翻倍。
- **简单提示词无退化。** 当 supervisor 的首版草稿很好时，评审在
  首遍以一次额外 LIGHT 调用退出（约 0.4 秒）。绝大多数对话式
  提示词（"你好"、"苹果股价是多少？"）走的是这条路径。
- **最佳草稿语义。** 返回最高水位草稿防止了 LLM "过度纠正"失败模式 — 重写为满足单条反馈要点而破坏结构。
- **在追踪中可见。** LangSmith / LangGraph Studio 将反思子图渲染为
  其自身的折叠节点 — 易于阅读、易于演示。逐迭代分数记录在最终消息的
  `additional_kwargs['reflection']` 中，可供离线分析。
- **运营上可选。** `REFLECTION_ENABLED=false`（默认值）保持遗留拓扑字节级相同。关心延迟的运维者可关闭反思；关心回答质量的运维者 开启它。

### 负面

- **延迟。** 最坏情况 +5 次 LLM 调用（3 次评审 + 2 次写作），发生在
  评审永远无法达到阈值的提示词上。我们通过 `max_iterations` 封顶，
  因此最坏情况有界，但困难提示词上的 P99 延迟在启用反思后从约 25 秒 移至约 45 秒。
- **成本。** 与延迟相同的核算 — 每请求 1-5 次额外 LLM 调用， 两个模型层。记录在配置注释中，使其成为运维者的审慎选择。
- **评审模型依赖。** 行为异常的评审（如始终输出分数 1.0）会静默地
  禁用循环。我们通过在 `_normalise_critique` 中将垃圾分数箝位到 0.0，
  并将不可解析的 JSON 视为分数 0.0 → 强制重写或触发 `max_iterations`
  来部分缓解。真正的生产部署会添加"评审一致性"可观测性指标， 超出本 ADR 范围。
- **需维护两阶段提示词。** `CRITIC_SYSTEM_PROMPT` 和
  `WRITER_SYSTEM_PROMPT` 现在是项目提示词库的一部分，需要与
  supervisor 提示词同步演进。记录在 `reflection.py` 的模块文档字符串中；
  通过现有的提示词组装测试跟踪。

### 中性

- 反思子图原则上可复用 — 它不硬编码任何 supervisor 特有的内容。
  如果未来某个 Agent 产出的草稿需要相同的评审+写作模式， 可以直接调用 `build_reflection_subgraph`。
- 为反思引入的父图包装器（`_wrap_with_reflection`）为未来的
  supervisor 后置阶段 — 引用交叉检查、来源去重、多语言翻译等 — 提供了自然的接缝，无需另一次 supervisor 内部重写。

## 状态

在提交 `<git rev-parse HEAD>`（Phase 5.2）中实现。子图及其接入
`build_research_supervisor` 的接线附带 23 个专用单元测试，覆盖
JSON 解析、批评归一化、草稿提取、对话记录格式化、三种循环终止路径
（通过 / 回退 / 达到最大迭代次数）以及父图包装器的拓扑。
`.env.example` 中默认为 `REFLECTION_ENABLED=false`；运维者主动开启。
