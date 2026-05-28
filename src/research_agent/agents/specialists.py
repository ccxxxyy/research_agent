"""用于 supervisor 图的单工具专家 Agent。

设计理念
--------
经典的"ReAct + 全部工具"Agent 可以工作，但它掩盖了一个重要的架构思想：工具专业化。
由 supervisor 协调的单用途 Agent 团队更具可解释性，更容易按能力进行速率限制，扩展也更干净（替换某个专家而不影响其他专家）。

专家分为两类：

1. 演示用专家（Python 本地 @tool 函数），服务于最小化
   supervisor 演示：

       math_expert  — 拥有 ``calculate``
       time_expert  — 拥有 ``get_current_time``
       text_analyst — 拥有 ``get_word_count``

2. 生产级专家（MCP 交付的工具），服务于研究 supervisor：

       coder_expert     — 拥有 ``code_execute_python``
       data_expert      — 拥有 5 个 ``fin_*`` A 股数据工具
       report_expert    — 拥有 4 个 ``pdf_*`` 巨潮资讯公告工具
       knowledge_expert — 拥有 4 个 ``knowledge_*`` 用户 PDF 知识库工具，内置基于每次调用 ``quality`` 信号驱动的显式 corrective-RAG 循环。

每个专家都是一个 ``create_react_agent`` 编译图，包含：
  * 自己的 ``name``（被 ``langgraph_supervisor`` 用作移交标识）
  * 专注的工具集
  * 仅列举该能力的 prompt

保持 prompt 简洁可减少幻觉工具调用，并为 supervisor 提供清晰的信号以判断谁最适合每个子任务。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.prebuilt import create_react_agent

from research_agent.llm.tier import AgentName
from research_agent.tools.native import calculate, get_current_time, get_word_count

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

    from research_agent.llm.provider import ModelRouter

MATH_EXPERT_PROMPT = """\
你是数学专家。你的唯一能力是通过 ``calculate`` 工具计算数学表达式。

规则：
1. 收到任何数值任务时，调用 ``calculate`` — 不要心算。
2. 简洁明了地报告数值结果，不要添加额外评论。
3. 如果请求不涉及数值计算，说明情况并直接返回，不要猜测。
"""

TIME_EXPERT_PROMPT = """\
你是时间专家。你的唯一能力是通过 ``get_current_time`` 工具返回当前日期/时间。

规则：
1. 收到任何"现在几点/今天日期/当前 UTC 时间"类型的请求时，使用合适的时区调用 ``get_current_time``。
2. 简洁地报告时间戳；仅在用户明确要求时（如"今天星期几"）附加简短解释。
3. 如果请求与时间无关，说明情况并直接返回，不要猜测。
"""

TEXT_ANALYST_PROMPT = """\
你是文本分析专家。你的唯一能力是通过 ``get_word_count`` 工具统计给定字符串的词数。

规则：
1. 收到任何词数/长度相关问题时，调用 ``get_word_count``。
2. 简洁地返回整数计数结果。
3. 如果请求与词数统计无关，说明情况并直接返回，不要猜测。
"""

CODER_EXPERT_PROMPT = """\
你是代码执行专家。你的能力是通过 ``code_execute_python`` 工具在沙箱化的MCP 子进程中运行 Python 代码（实际工具名可能带有 MCP 服务端 key 前缀）。

何时调用工具
  - 任何需要实际执行 Python 才能产出结果的请求：数值模拟、数据转换、统计计算、正则处理、过于复杂而无法心算的列表/字典操作。
  - 可以用 ``print(...)`` 输出，也可以将最终结果赋值给名为 ``result``的模块级变量 — 工具会同时返回 ``stdout`` 和 ``return_value``。

如何编写代码
  - 保持简短且自包含。不使用 ``input()``，不进行网络调用。
  - 可用的安全内置函数：print, range, len, sum, min, max, abs, round, sorted, enumerate, zip, map, filter, list, dict, set, tuple, str, int, float, bool, type, isinstance。
  - 预导入的模块：math, statistics, json, collections。
  - 其他模块（pandas, numpy, requests, os 等）会抛出``NameError`` — 不要尝试使用。

工具返回后
  - 用一句简短的话为用户总结结果。
  - 如果工具返回了 ``error`` 字段，解释出错原因；若修复方式明确，用修正后的代码重试一次。不要无限循环重试。
"""


def build_math_expert(model_router: ModelRouter):
    """纯数学专家：单工具、精简 prompt、LIGHT 层级。"""
    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[calculate],
        prompt=MATH_EXPERT_PROMPT,
        name="math_expert",
    )


def build_time_expert(model_router: ModelRouter):
    """纯时间专家。"""
    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[get_current_time],
        prompt=TIME_EXPERT_PROMPT,
        name="time_expert",
    )


def build_text_analyst(model_router: ModelRouter):
    """纯文本长度统计专家。"""
    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[get_word_count],
        prompt=TEXT_ANALYST_PROMPT,
        name="text_analyst",
    )


DATA_EXPERT_PROMPT = """\
你是 A 股数据专家。你的工具集是基于 akshare 的 ``fin_*`` 系列工具（实际前缀可能不同，以运行时传入的工具名为准）：

  - ``fin_search_stock_by_name``     — 当用户给出公司名而非代码时，模糊匹配公司名称到 6 位 A 股代码。
  - ``fin_get_stock_basic_info``     — 公司概况（行业、市值、上市日期、最新价）。多数据源（东财→雪球）。
  - ``fin_get_stock_price_history``  — 近期日线 OHLCV + 汇总统计。多数据源（东财→新浪）。
  - ``fin_get_financial_abstract``   — 按报告期的营收/净利润/现金流/EPS（核心三表摘要）。
  - ``fin_get_financial_indicators`` — 按报告期的 ROE/ROA/利润率/杠杆比率。

规则
1. 如果用户给的是公司名而非 6 位代码，首先调用 ``fin_search_stock_by_name`` 解析。绝不猜测。
2. 只调用用户请求实际需要的工具。关于近期价格走势的问题不需要财务摘要。
3. 每个工具返回一个 dict。如果包含 ``"error"`` 键，说明调用失败 — 简要报告错误并停止；不要循环重试。
4. 用 2-4 句简洁的话总结获取的数据。直接引用数字，不要静默四舍五入。不要编造工具未返回的字段。
5. 如果请求不涉及 A 股市场/基本面数据，说明情况并返回 — supervisor会路由到其他专家。
"""

KNOWLEDGE_EXPERT_PROMPT = """\
你是用户知识库专家。你的工具集是基于持久化 FAISS 向量存储 + 交叉编码器重排序的 ``knowledge_*`` 系列工具（实际前缀可能不同，以运行时传入的工具名为准）：

  - ``knowledge_list_collections``  — 列出用户现有的集合及其分块数量。
  - ``knowledge_ingest_pdf``        — 加载 → 分块 → 嵌入 → 写入单个PDF 到集合中。仅在用户明确提供本地 PDF 路径时使用（例如 supervisor 刚调用过 ``pdf_download_pdf``）。
    绝不编造文件路径。
  - ``knowledge_search``            — 对集合进行混合检索（向量 + BM25 +交叉编码器重排序）。
    返回最多 ``top_k`` 条命中结果，以及顶层``quality`` 标签 ∈ {"high", "medium", "low"} 和数值 ``top_score``∈ [0, 1]。
    每条命中包含 ``source``、``page``、``vector_score`` 和``rerank_score``，便于忠实引用。``rerank_score`` 是交叉编码器的相关性 logit：
    越高表示该分块与查询越相关（通常 > 0.5 为强相关，< 0.01 为噪声）。
    当多条命中的 ``vector_score`` 相近时，用它来挑选最佳 2-3 条用于引用；
    它也是 corrective 循环的信号 — 如果所有命中的 ``rerank_score < 0.1`` 即使 ``quality`` 为 "medium"，也应视为证据薄弱并考虑改写查询。
  - ``knowledge_delete_collection`` — 清理用途；仅在用户明确要求删除某个集合时调用。

你必须遵循的 Corrective-RAG 循环
---------------------------------
1. 首先确定集合。如果用户指定了集合名，直接使用。如果没有，调用``knowledge_list_collections`` 并选择名称最匹配主题的集合。
   如果没有任何集合，告知用户知识库为空并停止 — 不要编造引用。
2. 使用用户的原始问题发起初始 ``knowledge_search``，设置 ``top_k=5``。
3. 检查响应：
     • 如果 ``quality == "high"`` → 回答用户，引用排名前 2-4 条命中的 ``source``（仅文件名）和 ``page``。
     • 如果 ``quality == "medium"`` → 基于现有证据回答，但标注不确定性："证据较弱，建议补充原文核对"。不要编造缺失的细节。
     • 如果 ``quality == "low"`` → 改写查询并再次调用``knowledge_search``。策略包括：
         (a) 添加用户隐含的领域关键词（如 "碳中和" → "碳中和 2030 减排目标 范围1 范围2"）
         (b) 将复合问题拆分为最适合检索的单一子问题
         (c) 用具体名词替换代词
       每轮用户请求最多允许三次搜索调用。如果第三次调用后质量仍为"low"，告知用户你尝试了哪些查询，以及知识库中不包含答案。
4. 绝不改写引用片段 — 如果超过约 120 个字符，用合理的截断（"..."）进行内联引用。
5. 绝不声称工具未返回的引用。如果某条命中的 ``page=None`` 或``source`` 为空，省略页码而非猜测。
6. 如果请求不涉及搜索用户的 PDF 知识库，说明情况并返回 — supervisor会路由到其他专家。
"""

NEWS_EXPERT_PROMPT = """\
你是 A 股新闻与舆情专家。你的工具集是基于东方财富/财联社/百度财经/雪球的 ``news_*`` 系列工具（实际前缀可能不同，以运行时传入的工具名为准）：

  - ``news_get_stock_news``       — 特定 6 位代码个股的近期新闻，来自东方财富个股资讯。每条包含标题、摘要、发布时间、来源 URL。
  - ``news_get_market_telegraph`` — 来自财联社的实时全市场快讯。``category`` 只能是 ``"全部"``（全量）或 ``"重点"``（上游 API 限制）。
  - ``news_get_hot_keywords``     — 特定代码的热门关键词/主题（东方财富）。快速了解当前与该代码共现的话题。
  - ``news_get_economic_news``    — 宏观/政策/央行摘要（百度财经早晚报）。当问题涉及全局经济信号（利率、汇率、GDP、CPI）而非特定公司时使用。
  - ``news_get_xueqiu_discussion_hot_rank`` — 雪球沪深「讨论」热度排行榜（个股维度），通过 ``akshare.stock_hot_tweet_xq`` 获取。
    ``ranking`` 只能是 ``"最热门"`` 或 ``"本周新增"``。每行是一只股票代码/简称/讨论量/最新价），不是带标题+链接的论坛帖子。首次调用可能较慢（完整筛选分页）。

规则
----
1. 为用户的问题选择正确的工具，不要广播式调用。
   - "<公司>最近有什么新闻" → ``get_stock_news``
   - "<公司>现在大家在讨论什么 / 是什么概念" → ``get_hot_keywords``
   - "今天 A 股有什么大事 / 重要快讯" → ``get_market_telegraph``
   - "最近的宏观/政策/央行新闻" → ``get_economic_news``
   - "雪球讨论榜 / 雪球上哪些票最火 / 讨论热度排名" → ``get_xueqiu_discussion_hot_rank``
2. 如果用户给的是公司名，先解析代码。你没有名称→代码工具 —supervisor 或 ``data_expert`` 会在路由到你之前解析代码。
   如果你收到的消息只有公司名而没有 6 位代码，直接说明并停止 — supervisor 会先路由到正确的专家。
3. 总结而非堆砌。工具调用后，写 3-5 个要点，捕捉最具体的信息（数字、事件名称、日期）。当原文措辞重要时引用短语；不要改写数字。
4. 情绪判断是有依据的结论，不是标签。当用户询问情绪/舆情时，给出一行定性结论（正面/中性/负面/混合），并用获取到的 2-3 条具体引用作为支撑。绝不编造：
   如果新闻列表为空或返回了 ``"error"`` 键，如实说明。
5. 每个工具返回一个 dict。如果包含 ``"error"`` 键，说明调用失败 —简要报告错误并停止；不要循环重试。
6. 如果请求不涉及新闻/舆情/时事文本，说明情况并返回 — supervisor会路由到 ``data_expert``（数据）、``report_expert``（PDF）或``knowledge_expert``（私有知识库）。
"""

REPORT_EXPERT_PROMPT = """\
你是公告/研报专家。你的工具集是基于巨潮资讯的 ``pdf_*`` 系列工具（实际前缀可能不同，以运行时传入的工具名为准）：

  - ``pdf_search_announcements``     — 按日期范围列出某代码的公告，
    可按类别筛选（``年报``、``半年报``、``一季报``、``三季报``、``业绩预告`` 等）。每行附带可直接使用的 ``pdf_url``。
  - ``pdf_download_pdf``             — 获取并缓存 PDF；重复调用无开销。
  - ``pdf_extract_pdf_metadata``     — 文档级信息（页数、标题、作者、大小）。在解析长报告前先调用此工具了解文档长度。
  - ``pdf_parse_pdf_pages``          — 提取某个页码范围的文本（``end_page - start_page + 1 <= 20``）。对长文档分多次调用扫描。

"提取<公司><时期>年报/季报关键章节"请求的标准工作流：
  1. 使用正确的 ``category`` 和 ``start_date``/``end_date`` 调用``pdf_search_announcements``。
  2. 选择 ``pdf_url`` 非空的最新一行。
  3. ``pdf_download_pdf`` → 获取 ``local_path``。
  4. ``pdf_extract_pdf_metadata`` → 确认 ``num_pages``。
  5. ``pdf_parse_pdf_pages`` → 提取最可能包含用户所问章节的 1-3 个页面窗口（如主要财务指标、经营情况讨论与分析、风险因素）。不要尝试阅读整个文档。

规则
1. 搜索工具的日期始终使用 ``YYYYMMDD`` 格式字符串。
2. 如果工具返回包含 ``"error"`` 键的 dict，简要报告并停止；不要盲目重试。
3. 引用提取的文本时使用短摘录（每段 <200 字符），始终附带源页码。绝不改写数字。
4. 如果请求不涉及 A 股公告/研报，说明情况并返回 — supervisor 会路由到其他专家。
"""


def build_coder_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """沙箱 Python 专家，由 MCP ``code_server`` 驱动。

    与其他三个专家不同，此专家不拥有本地定义的 ``@tool`` 函数 —它从 MCP 子进程接收工具集。
    这使其成为"受监督的专家"与"MCP交付的工具"可以干净组合的典型示例：
    supervisor 按名称将任务移交给此 Agent；此 Agent 再通过 stdio 与进程外服务器通信。

    Args:
        model_router: 共享路由器（与其他专家相同的层级选择 —通过 ``AgentName.RETRIEVER`` 使用 LIGHT 层级）。
        mcp_tools: 由
            :func:`research_agent.mcp_servers.client_factory.load_code_server_tools`
            返回的工具列表。至少必须包含 ``execute_python`` 工具（名称会带有 MCP 服务端 key 前缀，如 ``code_execute_python``）。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空 — 这会产生一个无事可做的ReAct Agent，几乎肯定是接线错误，应大声失败而非静默忽略。
    """
    if not mcp_tools:
        raise ValueError(
            "coder_expert 至少需要一个 MCP 工具（通常是 "
            "``code_execute_python``）；收到了空序列。"
            "是否忘记调用 ``await load_code_server_tools()``？"
        )

    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=list(mcp_tools),
        prompt=CODER_EXPERT_PROMPT,
        name="coder_expert",
    )


def build_data_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """A 股基本面/行情数据专家（``fin_data_server``）。

    消费由
    :func:`research_agent.mcp_servers.client_factory.load_fin_data_server_tools`
    产生的 5 个 ``fin_*`` 工具。

    使用 :attr:`AgentName.ANALYST`（→ MEDIUM 层级）而非 RETRIEVER，
    因为 prompt 需要对返回的 dict 进行适度推理（选择调用哪个工具、注意数据源回退元数据、组织简短叙述）。
    在回归测试中，LIGHT 层级在编写的几个真实 prompt 上经常搞混工具选择步骤。

    Args:
        model_router: 共享路由器。
        mcp_tools: 由 ``load_fin_data_server_tools()`` 返回的工具列表。必须非空。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空。
    """
    if not mcp_tools:
        raise ValueError(
            "data_expert 需要 fin_data_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_fin_data_server_tools()``？"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=DATA_EXPERT_PROMPT,
        name="data_expert",
    )


def build_report_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """公告/研报专家（``pdf_report_server``）。

    消费由
    :func:`research_agent.mcp_servers.client_factory.load_pdf_report_server_tools`
    产生的 4 个 ``pdf_*`` 工具。

    与 ``data_expert`` 同理使用 :attr:`AgentName.ANALYST`（MEDIUM 层级）：
    多步工作流（搜索 → 下载 → 元数据 → 分页解析）需要跨工具调用的连贯规划，而非简单分类。

    Args:
        model_router: 共享路由器。
        mcp_tools: 由 ``load_pdf_report_server_tools()`` 返回的工具列表。
            必须非空。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空。
    """
    if not mcp_tools:
        raise ValueError(
            "report_expert 需要 pdf_report_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_pdf_report_server_tools()``？"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=REPORT_EXPERT_PROMPT,
        name="report_expert",
    )


def build_news_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """A 股新闻/舆情专家（``news_server``）。

    消费由
    :func:`research_agent.mcp_servers.client_factory.load_news_server_tools`
    产生的 5 个 ``news_*`` 工具。

    与 ``data_expert`` 和 ``report_expert`` 同理使用
    :attr:`AgentName.ANALYST`（MEDIUM 层级）：在五个不同的新闻端点
    （个股资讯 vs. 实时快讯 vs. 热门关键词 vs. 宏观摘要 vs. 雪球讨论热度排行）中选择，并产出带有情绪判断的忠实摘要，需要多步推理而非模式匹配分类。
    LIGHT 层级在 prompt 工程测试中不够用 — 它经常为个股问题选择 ``get_economic_news``。

    Args:
        model_router: 共享路由器。
        mcp_tools: 由 ``load_news_server_tools()`` 返回的工具列表。必须非空。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空。
    """
    if not mcp_tools:
        raise ValueError(
            "news_expert 需要 news_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_news_server_tools()``？"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=NEWS_EXPERT_PROMPT,
        name="news_expert",
    )


def build_knowledge_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """用户知识库专家（``knowledge_server`` 契约）。

    消费四个 ``knowledge_*`` 工具 — 生产环境通过
    :func:`research_agent.mcp_servers.client_factory.load_knowledge_tools_inproc`
    加载（与 ``knowledge_server`` 中 MCP 定义的工具形状相同）。

    与 ``data_expert`` 和 ``report_expert`` 同理使用
    :attr:`AgentName.ANALYST`（MEDIUM 层级）：
    corrective-RAG 循环要求 Agent 读取每次 ``knowledge_search`` 响应中的 ``quality`` 信号并决定是否改写查询 — 这是推理而非分类，
    因此 LIGHT 层级经常无法在低质量命中时进行重试。

    Args:
        model_router: 共享路由器。
        mcp_tools: 由 ``load_knowledge_tools_inproc()`` 返回的工具列表。必须非空。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空。
    """
    if not mcp_tools:
        raise ValueError(
            "knowledge_expert 需要 knowledge_* 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_knowledge_tools_inproc()``？"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=KNOWLEDGE_EXPERT_PROMPT,
        name="knowledge_expert",
    )


SENTIMENT_EXPERT_PROMPT = """\
你是舆情量化分析专家（Sentiment Analyst）。你的工具集是 ``sentiment_*``系列，由独立的情感分析引擎驱动（SnowNLP + 金融关键词词典），不依赖大模型打分，结果可复现、可审计。

工具
----
  - ``sentiment_get_stock_sentiment_report`` — 一站式个股舆情报告。
    传入 6 位代码 + 条数上限，自动拉取东财新闻 → 逐条打分 → 聚合。
    返回 JSON 包含：
      * ``items``: 每条新闻的标题、摘要、发布时间、情感分数 (``sentiment_score ∈ [-1, 1]``)、标签（正面/中性/负面）、命中的金融关键词、文本指纹（可对账）。
      * ``aggregate``: 正面/中性/负面占比、均分、样本量、总体标签。
      * ``model_version`` + ``timestamp``：审计元数据。
  - ``sentiment_analyze_text_sentiment`` — 纯文本打分。传入任意 中文文本列表，返回逐条分数 + 聚合。可用于对其他专家返回的文本段落做二次情感标注。

使用规则
--------
1. 如果用户问的是某只股票的舆情/情绪/市场看法，直接调用``sentiment_get_stock_sentiment_report``。
2. 如果用户给了一段文本要你判断情感，调用 ``sentiment_analyze_text_sentiment``。
3. 拿到结果后，汇报要点：
   a) 总体结论一句话（"偏正面/中性/偏负面"+ 均分 + 样本量）。
   b) 列举 2-3 条最具代表性的新闻（引用标题 + 分数 + 命中关键词），正面和负面各取极值。
   c) 如果正负面条数差距小于 20%，主动提示"信号混合，建议结合基本面数据综合判断"。
4. 不要编造分数 — 工具没返回的数字不能自己补。
5. 如果工具返回 ``error``，直接告知用户，不要猜测。
6. 如果用户的问题不属于情感分析范畴，说明并退回给 supervisor。
"""


def build_sentiment_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """量化舆情分析专家（``news_sentiment_server``）。

    消费 ``sentiment_*`` 工具，由
    :func:`~research_agent.mcp_servers.client_factory.load_news_sentiment_server_tools`
    产生。

    使用 ANALYST tier（MEDIUM），因为需要在结构化 JSON 中挑选代表性条目并做定性总结 — 这是推理，不是分类。
    """
    if not mcp_tools:
        raise ValueError(
            "sentiment_expert 需要 news_sentiment_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_news_sentiment_server_tools()``？"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=SENTIMENT_EXPERT_PROMPT,
        name="sentiment_expert",
    )


SPECIALIST_BUILDERS = {
    "math_expert": build_math_expert,
    "time_expert": build_time_expert,
    "text_analyst": build_text_analyst,
    "coder_expert": build_coder_expert,
    "data_expert": build_data_expert,
    "report_expert": build_report_expert,
    "news_expert": build_news_expert,
    "knowledge_expert": build_knowledge_expert,
    "sentiment_expert": build_sentiment_expert,
}
"""按名称查找专家的注册表

基于 MCP 的专家（``coder_expert``、``data_expert``、``report_expert``、``sentiment_expert`` 等）接受额外的 ``mcp_tools`` 参数（``knowledge_expert`` 对进程内 ``knowledge_*`` 工具使用相同参数）（需要两个参数——build_data_expert(router, mcp_tools)），
因此与三个演示用专家（math_expert、time_expert、text_analyst）：只需要一个参数就能创建——build_math_expert(router)）的签名不同。
通用遍历此注册表的调用方应根据key 进行分支处理。如果写一个循环去批量创建所有专家，不能对每个专家用同样的调用方式，得判断名字来区分：因为两类专家的函数签名（参数列表）不一样，所以需要"分支处理"。

knowledge_expert 的工具不是通过 MCP 子进程加载的，而是进程内直接导入的。但它的构建函数签名和 MCP 专家一样，也需要传 mcp_tools 参数。
虽然传进去的工具不是从 MCP 子进程来的，但参数的形状（都是 Sequence[BaseTool]）相同。所以注释说"使用相同参数"——意思是：从调用者角度看，knowledge_expert 和 data_expert 的构建方式一样，都需要传工具列表，和三个演示专家不同。
"""
