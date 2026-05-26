"""研究 supervisor — 金融研究工作流。

为什么要与 ``minimal_supervisor.py`` 分离成独立图？
``minimal_supervisor.py`` 是一个教学脚手架：三个简单的``@tool`` 支持的专家加上一个可选的 MCP 编程专家。它的存在是为了在隔离环境中、无需网络依赖地演示 supervisor 模式本身。

本图是真正的产品。它编排了多个 MCP 交付的专家，共享智能体在生产环境中使用的 A 股 / 巨潮资讯数据接口：

              ┌─────────────────────────────────────────────────────┐
              │              research_supervisor                    │ ← HEAVY 级 LLM
              └─┬────────┬───────────┬─────────┬───────┬──────────┬─┘
                │        │           │         │       │          │
                ▼        ▼           ▼         ▼       ▼          ▼
          data_expert report_expert coder  news_expert knowledge sentiment
          (fin_data)  (pdf_report)  _expert (news_srv) _expert   _expert
                                   (code)              (knowledge)(sentiment)

"分析 宁德时代 2023 年业绩 + ESG 披露中提到的碳中和承诺"的典型流程：

  1. supervisor → data_expert：拉取财务摘要 + 财务指标
  2. supervisor → report_expert：定位 2023 年报 PDF 并提取经营情况 / 风险因素章节
  3. supervisor → coder_expert：计算衍生比率或交叉验证前两个输出中的数字
  4. supervisor → knowledge_expert：在用户之前导入的 ESG 知识库中搜索"碳中和"相关内容（纠正式 RAG：智能体读取每次调用的``quality`` 信号，如果命中结果较弱则重写查询，最多 3 次尝试）
  5. supervisor 撰写最终综合报告。

重要的设计选择
-----------------------------------------------------------
- 每个专家拥有不相交的工具集。无重叠意味着 supervisor 的路由选择是明确的 — 这是朴素"一个智能体拥有所有工具"设计中的常见失败模式。
- 专家运行在 :class:`ModelTier.MEDIUM`（通过:attr:`AgentName.ANALYST`），只做定向工具调用。；supervisor 运行在 HEAVY。Supervisor 负责困难的推理（规划 + 综合）。
- supervisor 提示词中精确地列出每个专家拥有的工具。supervisor 通过阅读自己的系统提示词来决定路由，而非窥探专家的工具列表。
- ``output_mode="last_message"`` 使共享状态保持紧凑。仅在调试路由循环时切换到 ``full_history``。
- 专家的构建是惰性的 — 如果调用者未提供某专家的 MCP 工具，该专家将被排除在团队之外，supervisor 提示词也会相应裁剪。这使得单元试可以编译仅含 1-2 个专家的图，而无需启动完整的子进程集群。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from loguru import logger

from research_agent.agents.specialists import (
    build_coder_expert,
    build_data_expert,
    build_knowledge_expert,
    build_news_expert,
    build_report_expert,
    build_sentiment_expert,
)
from research_agent.graph.reflection import build_reflection_subgraph
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import ModelTier


SUPERVISOR_PROMPT_BASE = """\
你是金融研究 Supervisor（主管）。你协调一个小型专家团队，为用户提供
关于 A 股上市公司的简明、有引用来源的回答。你的默认语言跟随用户 —
如果用户使用中文，则用中文回答。

团队成员：
"""

SUPERVISOR_PROMPT_DATA = """\
  - data_expert   ：通过 akshare MCP 获取 A 股市场数据和基本面。
    工具集（工具名可能带有 MCP 服务器前缀）：
        * fin_search_stock_by_name    — 名称 → 6 位股票代码查询
        * fin_get_stock_basic_info    — 公司简介 / 最新股价
        * fin_get_stock_price_history — OHLCV 行情 + 汇总统计
        * fin_get_financial_abstract  — 营收 / 利润 / 现金流
        * fin_get_financial_indicators — ROE / 利润率 / 杠杆率
    当用户询问以下内容时委派给该专家：最新股价 / 市值 / 行业分类、OHLCV 历史行情、季度/年度财务数据、关键比率、或名称→代码解析。
"""

SUPERVISOR_PROMPT_REPORT = """\
  - report_expert ：巨潮资讯披露 PDF。
      工具集：
        * pdf_search_announcements    — 列出指定日期范围内的年报 / 季报 / 公告文件
        * pdf_download_pdf            — 带缓存的文件获取
        * pdf_extract_pdf_metadata    — 页数 / 标题 / 作者
        * pdf_parse_pdf_pages         — 按页窗口提取文本（每次最多 20 页）
      当用户询问以下内容时委派给该专家：年报/季报内容、经营情况讨论与分析、风险因素、业绩预告 / 预增 / 预减、或巨潮资讯 PDF 中的任何摘录。
"""

SUPERVISOR_PROMPT_CODER = """\
  - coder_expert  ：通过 MCP 的沙箱化 Python 执行环境。仅可使用安全内置库（math / statistics / json / collections 已预导入）。
      当用户需要衍生指标（均值 / 标准差 / 增长率）、对返回行进行排序 /筛选、或其他专家未预先计算的任何运算时，委派给该专家。
"""

SUPERVISOR_PROMPT_NEWS = """\
  - news_expert   ：通过东方财富 / 财联社 / 百度财经 / 雪球获取 A 股新闻与舆情。
    工具集：
        * news_get_stock_news        — 获取指定 6 位代码个股的近期新闻（东方财富个股资讯）
        * news_get_market_telegraph  — 财联社实时市场快讯（筛选类别：仅支持"全部"或"重点"）
        * news_get_hot_keywords      — 与某只个股共现的热门主题 / 关键词
        * news_get_economic_news     — 每日宏观 / 政策摘要（百度财经早晚报）
        * news_get_xueqiu_discussion_hot_rank — 雪球讨论热度个股榜
                                       （``ranking``：``"最热门"`` 或``"本周新增"``；封装 ``stock_hot_tweet_xq`` — 行为个股维度，非帖子维度）
    当用户询问以下内容时委派给该专家："<公司>最近的新闻 / 舆情 / 热度"、
    "今天 A 股有什么大事 / 重要快讯"、"市场对 <公司> 的情绪如何"、"最近的宏观 / 政策 / 央行新闻"、"雪球讨论榜 / 雪球最热标的"。
    不要将原始数值 / 基本面数据请求路由到此专家，也不要将年报 / 公告等官方披露文件路由到此 — 它们属于其他专家。
"""

SUPERVISOR_PROMPT_KNOWLEDGE = """\
  - knowledge_expert ：用户的私有 PDF 知识库，使用持久化 FAISS 向量存储，支持混合检索（向量 + BM25 + 交叉编码器重排序）。
    工具集：
        * knowledge_list_collections — 列举用户的知识集合
        * knowledge_ingest_pdf       — 将本地 PDF 分块 + 嵌入到某个集合中
        * knowledge_search           — 带重排序的混合检索；返回``quality`` 标签和每条结果的``rerank_score``，专家据此执行内部纠正式 RAG 循环
        * knowledge_delete_collection
    当用户询问只有其个人上传文档才能回答的问题时委派给该专家：
    "我之前上传的 ESG 报告里关于碳中和怎么写的"、"我那份招股说明书里的募投项目"、或追问"把这份报告灌进我的知识库并按xx检索"。
    不要将通用 A 股市场或公开披露文件的问题路由到此 —该专家只能看到用户个人上传的内容。
"""

SUPERVISOR_PROMPT_SENTIMENT = """\
  - sentiment_expert ：结构化新闻情感量化分析（SnowNLP + 金融关键词词典，确定性模型，不走大模型打分）。
    工具集：
        * sentiment_get_stock_sentiment_report — 一站式个股舆情报告：
            拉东财新闻 → 逐条打分 → 聚合。返回每条新闻的``sentiment_score ∈ [-1, 1]``、标签（正面/中性/负面）、
            命中关键词、文本指纹 + 聚合统计（正/负/中性比例、均分、样本量）+ 审计元数据（模型版本 + 时间戳）。
        * sentiment_analyze_text_sentiment — 纯文本批量打分。传入任意中文文本列表，返回逐条分数 + 聚合。
            可用于对其他专家返回的文本做二次情感标注。
    当用户询问以下内容时委派给该专家：个股舆情量化（"宁德时代最近舆情如何 / 市场情绪"）、新闻情感打分（"帮我分析这几条新闻的情绪"）、批量文本情感标注。
    与 news_expert 的区别：news_expert获取原始新闻文本，sentiment_expert 对文本做可复现的量化评分。二者配合使用效果最佳。
"""

# 注意：这些规则在所有团队组合中不变。它们绝不能按名称提及某个特定专家，
# 因为团队在运行时动态组装，缺席的专家否则会作为幽灵路由目标泄露进提示词 — 导致 ``transfer_to_<missing>`` 工具调用失败。
# 针对特定专家的指导写在上面的 ``*_PROMPT_*`` 部分中。
SUPERVISOR_PROMPT_RULES = """\
你的职责
--------
1. 仔细阅读用户的请求。识别其中包含的每个独立子问题
   （例如"基本资料 + 最近披露 + 算均值"是三个子问题，不是一个）。
   如果用户请求使用了编号步骤 (1) (2) (3) ... 或项目符号，说明用户已明确给出了分解 — 每个编号步骤都算作一个独立子问题，且必须有对应的独立移交。
2. 规划一个最小化的移交序列。对于每个子问题，选择工具集（如上所述）
   最匹配的那一位专家。如果用户给了公司名称但后续步骤需要 6 位股票代码，请先通过拥有名称查询工具的专家解析代码。
3. 每次只移交一个子任务，调用对应的 ``transfer_to_<name>`` 工具。
   等待该专家的结果后再路由下一个子任务。不要同时发起两个移交 —共享状态假定串行轮次。
4. 只有在所有子问题都已委派并得到回答后，才自己撰写最终回答。
   在生成最终回答前做一个有用的自检：重新阅读用户的原始请求，
   验证每个编号 / 项目符号子任务是否都通过实际的``transfer_to_<name>`` 移交处理过（而非由你自己处理）。
   如果有任何子任务被跳过，现在就路由它。
   多步骤研究请求的必需结构：
     - ### 核心发现 / Key findings  （3-5 个要点，包含具体数据，并在相关处附上 PDF 的简短引文）
     - ### 数据来源 / Sources（列出调用了哪些专家，以及每个专家贡献了什么）
5. 不要编造数据或引文。如果某专家返回的字典中包含 ``"error"`` 键，请如实说明，不要捏造替代内容。
6. 不要自己调用专家工具。你无法直接访问 ``fin_*``、``pdf_*``、 ``code_*``、``news_*``、``knowledge_*`` 或 ``sentiment_*`` —你只能使用 ``transfer_to_*`` 移交工具。

关键反幻觉规则
---------------------------------
A. 绝不声称某个工具、专家或功能"不可用"、"受工具限制"、"无法访问"、"由于工具限制"、"暂不支持"或任何等价表述。
   上面团队名单中列举的每个专家此刻都是可用的。如果你发现自己在写这类措辞，请停下来 —重新阅读名单并发起正确的 ``transfer_to_<name>`` 移交。
B. 绝不用你自己的知识替代专家的输出。如果用户要求 PDF 中的内容、用户知识库中的内容、最新股价或数值计算，
   你必须路由到拥有该能力的专家 — 即使你"自己也能回答"。路由本身就是交付物；专家的输出才是用户需要的。
C. 当团队中有 coder 专家时，绝不自己做算术 / 统计 / 数据转换。
   即使是简单的均值和标准差也要通过 ``transfer_to_<coder>`` 移交给coder — 这是保证可复现性的方式。
D. 如果某个子任务应该有移交但你发现自己在自行生成文本，那就是 bug —
   在为该子任务撰写任何文字之前，先发起缺失的 ``transfer_to_<name>``调用来修复它。
"""


def _build_supervisor_prompt(
    *,
    has_data: bool,
    has_report: bool,
    has_coder: bool,
    has_knowledge: bool,
    has_news: bool,
    has_sentiment: bool,
) -> str:
    """组装 supervisor 提示词，使其与实际团队成员匹配。

    如果在图中列出不存在的专家，将导致运行时 ``transfer_to_<missing>``工具调用失败。因此提示词仅列举实际编译的专家。
    """
    parts = [SUPERVISOR_PROMPT_BASE]
    if has_data:
        parts.append(SUPERVISOR_PROMPT_DATA)
    if has_report:
        parts.append(SUPERVISOR_PROMPT_REPORT)
    if has_coder:
        parts.append(SUPERVISOR_PROMPT_CODER)
    if has_news:
        parts.append(SUPERVISOR_PROMPT_NEWS)
    if has_knowledge:
        parts.append(SUPERVISOR_PROMPT_KNOWLEDGE)
    if has_sentiment:
        parts.append(SUPERVISOR_PROMPT_SENTIMENT)
    parts.append("\n" + SUPERVISOR_PROMPT_RULES)
    return "".join(parts)


class _ResearchState(TypedDict, total=False):
    """supervisor + 反思包装器的父图状态。

    唯一重要的字段：消息流。
    ``add_messages`` 归约器按消息 id 去重，因此当内部 supervisor 返回完整对话记录（输入 + 新消息）时，
    只有新产生的消息会真正追加到父状态中 — 这正是实现干净 SSE 流式传输所需的行为。
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# HITL 人工审核节点
# ---------------------------------------------------------------------------

def _build_human_review_node():
    """创建一个暂停执行以进行人工审核的图节点。

    该节点从消息流中提取 supervisor 的草稿并调用 ``interrupt()`` — LangGraph 将图状态持久化到 checkpointer 并暂停执行。
    SSE 层检测到暂停后发出 ``review_requested`` 事件。

    当审核者调用 ``/approve`` 或 ``/resume`` 时，图通过``Command(resume=value)`` 恢复执行。
    ``interrupt()`` 调用返回该值：

    * ``{"action": "approve", ...}`` — 节点直接通过；草稿原样进入反思 / END。
    * ``{"action": "revise", "feedback": "..."}`` — 节点将反馈注入为 ``HumanMessage``，以便下游节点（反思或可能的 supervisor 重新运行）可以采纳。
    """

    async def human_review_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        messages = state.get("messages", [])
        draft = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    draft = content
                    break

        decision = interrupt({
            "draft": draft,
            "action_required": "approve_or_revise",
        })

        if isinstance(decision, dict) and decision.get("action") == "revise":
            feedback = decision.get("feedback", "")
            if feedback:
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[REVIEWER FEEDBACK]\n{feedback}"
                        )
                    ]
                }

        return {"messages": []}

    return human_review_node


def _wrap_with_hitl_only(
    supervisor: CompiledStateGraph,
    *,
    checkpointer: BaseCheckpointSaver | None,
) -> CompiledStateGraph:
    """用人工审核中断包装 supervisor（不含反思）。"""

    async def supervisor_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        result = await supervisor.ainvoke(
            {"messages": state.get("messages", [])},
        )
        return {"messages": result.get("messages", [])}

    parent: StateGraph = StateGraph(_ResearchState)
    parent.add_node("supervisor", supervisor_node)
    parent.add_node("human_review", _build_human_review_node())
    parent.add_edge(START, "supervisor")
    parent.add_edge("supervisor", "human_review")
    parent.add_edge("human_review", END)
    compiled = parent.compile(checkpointer=checkpointer)
    logger.info("Research supervisor wrapped with HITL review (no reflection).")
    return compiled


def _wrap_with_reflection(
    supervisor: CompiledStateGraph,
    *,
    model_router: ModelRouter,
    pass_threshold: float,
    max_iterations: int,
    checkpointer: BaseCheckpointSaver | None,
    enable_hitl: bool = False,
) -> CompiledStateGraph:
    """将已编译的 supervisor 包装在运行反思的父图中。

    为什么用父图而不是内联后处理？
    -----------------------------------------------------
    可以简单地调用 ``supervisor.ainvoke``，然后对其输出运行 ``reflection.ainvoke``。没有这样做的原因是：

      1. 如果流水线的一部分在图外运行，LangGraph 的追踪 / LangSmith集成会丢失逐节点的可视化。将反思作为图节点保留可使完整 DAG 在 studio 中可见。
      2. checkpointer 附加在外层图上，因此 supervisor + 反思在崩溃恢复方面是原子性的：在反思过程中崩溃的线程会从 critic 节点恢复，而不是重新运行整个专家团队。
      3. 日后添加更多 supervisor 后置阶段（如基于引用索引的事实核查）只需编辑一个图节点，而非重写编排代码。
    """
    reflection = build_reflection_subgraph(
        model_router=model_router,
        pass_threshold=pass_threshold,
        max_iterations=max_iterations,
    )

    async def supervisor_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        """运行内部 supervisor 图并将输出向上传递。"""
        result = await supervisor.ainvoke(
            {"messages": state.get("messages", [])},
        )
        return {"messages": result.get("messages", [])}

    async def reflection_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        """在 supervisor 输出上运行反思子图。"""
        result = await reflection.ainvoke(
            {"messages": state.get("messages", [])},
        )
        return {"messages": result.get("messages", [])}

    parent: StateGraph = StateGraph(_ResearchState)
    parent.add_node("supervisor", supervisor_node)
    parent.add_node("reflection", reflection_node)

    if enable_hitl:
        parent.add_node("human_review", _build_human_review_node())
        parent.add_edge(START, "supervisor")
        parent.add_edge("supervisor", "human_review")
        parent.add_edge("human_review", "reflection")
        parent.add_edge("reflection", END)
    else:
        parent.add_edge(START, "supervisor")
        parent.add_edge("supervisor", "reflection")
        parent.add_edge("reflection", END)

    compiled = parent.compile(checkpointer=checkpointer)
    logger.info(
        "Research supervisor wrapped with reflection: "
        "pass_threshold={:.2f} max_iterations={} hitl={}",
        pass_threshold,
        max_iterations,
        enable_hitl,
    )
    return compiled


def build_research_supervisor(
    *,
    model_router: ModelRouter,
    data_tools: Sequence[BaseTool] | None = None,
    report_tools: Sequence[BaseTool] | None = None,
    coder_tools: Sequence[BaseTool] | None = None,
    knowledge_tools: Sequence[BaseTool] | None = None,
    news_tools: Sequence[BaseTool] | None = None,
    sentiment_tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    supervisor_tier: ModelTier = ModelTier.HEAVY,
    enable_reflection: bool = False,
    reflection_pass_threshold: float = 0.85,
    reflection_max_iterations: int = 2,
    enable_hitl: bool = False,
) -> CompiledStateGraph:
    """金融研究 supervisor 图。

    图中仅包含工具列表非空的 specialist。
    零 specialist 的 supervisor 毫无用处，因此至少须有一个工具列表非空 — 否则显式失败。

    Args:
        model_router: 共享路由器（supervisor 使用 ``supervisor_tier``；specialist 通过其构建器内的 ``ANALYST`` / ``RETRIEVER`` Agent 名称映射使用 MEDIUM）。
        data_tools: ``fin_*`` 工具。省略/空 → 无 ``data_expert``。
        report_tools: ``pdf_*`` 工具。省略/空 → 无 ``report_expert``。
        coder_tools: ``code_*`` 工具。省略/空 → 无 ``coder_expert``。
        knowledge_tools: ``knowledge_*`` 工具。省略/空 → 无 ``knowledge_expert``。
        news_tools: ``news_*`` 工具。省略/空 → 无 ``news_expert``。
        sentiment_tools: ``sentiment_*`` 工具。省略/空 → 无 ``sentiment_expert``。
        checkpointer: 可选的 LangGraph checkpointer。
        supervisor_tier: 默认 HEAVY。
        enable_reflection: 为 True 时，将 supervisor 包装在父图中，对其最终综合运行批评者+写作者反思循环。循环在批评者评分达到或
        超过 ``reflection_pass_threshold`` 或 ``reflection_max_iterations``次改写后终止（以先到者为准）。
        包装器保持 supervisor 的 ``ainvoke`` / ``astream`` 契约；调用者可见的唯一区别是对话记录末尾多了一条 ``AIMessage``，其 ``additional_kwargs['reflection']``携带审计轨迹。
        reflection_pass_threshold: 传递给 ``build_reflection_subgraph``。
        reflection_max_iterations: 传递给 ``build_reflection_subgraph``。
        enable_hitl: 为 True 时，插入一个 ``human_review`` 节点，在 supervisor 草稿生成后调用 ``interrupt()``。图暂停等待人工审批；调用者通过 ``Command(resume=...)`` 恢复。

    Returns:
        A compiled LangGraph app.

    Raises:
        ValueError: If every tool list was empty.
    """
    has_data = bool(data_tools)
    has_report = bool(report_tools)
    has_coder = bool(coder_tools)
    has_knowledge = bool(knowledge_tools)
    has_news = bool(news_tools)
    has_sentiment = bool(sentiment_tools)

    if not (has_data or has_report or has_coder or has_knowledge or has_news or has_sentiment):
        raise ValueError(
            "build_research_supervisor 至少需要一个专家的工具列表非空，"
            "但六组工具全部为空。"
        )

    agents: list = []
    roster: list[str] = []

    if has_data:
        agents.append(build_data_expert(model_router, data_tools or []))
        roster.append("data_expert")
    if has_report:
        agents.append(build_report_expert(model_router, report_tools or []))
        roster.append("report_expert")
    if has_coder:
        agents.append(build_coder_expert(model_router, coder_tools or []))
        roster.append("coder_expert")
    if has_news:
        agents.append(build_news_expert(model_router, news_tools or []))
        roster.append("news_expert")
    if has_knowledge:
        agents.append(build_knowledge_expert(model_router, knowledge_tools or []))
        roster.append("knowledge_expert")
    if has_sentiment:
        agents.append(build_sentiment_expert(model_router, sentiment_tools or []))
        roster.append("sentiment_expert")

    supervisor_model = model_router.get_model(supervisor_tier)
    prompt = _build_supervisor_prompt(
        has_data=has_data,
        has_report=has_report,
        has_coder=has_coder,
        has_knowledge=has_knowledge,
        has_news=has_news,
        has_sentiment=has_sentiment,
    )

    workflow = create_supervisor(
        agents=agents,
        model=supervisor_model,
        prompt=prompt,
        output_mode="last_message",
    )

    # 启用反思时，父（包装器）图持有 checkpointer —— 内部 supervisor 无状态编译，避免两层争夺同一 thread_id。
    # 反思关闭时，supervisor 自身持有 checkpointer，行为与之前完全相同。
    if enable_reflection:
        inner = workflow.compile()
        compiled = _wrap_with_reflection(
            inner,
            model_router=model_router,
            pass_threshold=reflection_pass_threshold,
            max_iterations=reflection_max_iterations,
            checkpointer=checkpointer,
            enable_hitl=enable_hitl,
        )
    elif enable_hitl:
        inner = workflow.compile()
        compiled = _wrap_with_hitl_only(inner, checkpointer=checkpointer)
    else:
        compiled = workflow.compile(checkpointer=checkpointer)

    logger.info(
        "Research supervisor compiled: tier={} specialists={} reflection={} hitl={}",
        supervisor_tier.value,
        roster,
        enable_reflection,
        enable_hitl,
    )
    return compiled


__all__ = [
    "build_research_supervisor",
    "SUPERVISOR_PROMPT_BASE",
    "SUPERVISOR_PROMPT_DATA",
    "SUPERVISOR_PROMPT_REPORT",
    "SUPERVISOR_PROMPT_CODER",
    "SUPERVISOR_PROMPT_NEWS",
    "SUPERVISOR_PROMPT_KNOWLEDGE",
    "SUPERVISOR_PROMPT_SENTIMENT",
    "SUPERVISOR_PROMPT_RULES",
]
