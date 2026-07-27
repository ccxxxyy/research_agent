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
       data_expert      — 拥有 ``fin_*`` A 股数据工具
       us_data_expert   — 拥有 ``us_*`` 美股（股票/指数/ETF）数据工具（yfinance）
       us_filing_expert — 拥有 ``us_filing_*`` SEC EDGAR 披露工具
       report_expert    — 拥有 4 个 ``pdf_*`` 巨潮资讯公告工具
       knowledge_expert — 拥有 4 个 ``knowledge_*`` 用户 PDF 知识库工具，内置基于每次调用 ``quality`` 信号驱动的显式 corrective-RAG 循环。

每个专家都是一个 ``create_agent`` 编译图，包含：
  * 自己的 ``name``（被 ``langgraph_supervisor`` 用作移交标识）
  * 专注的工具集
  * 仅列举该能力的 prompt

保持 prompt 简洁可减少幻觉工具调用，并为 supervisor 提供清晰的信号以判断谁最适合每个子任务。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents import create_agent

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
    return create_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[calculate],
        system_prompt=MATH_EXPERT_PROMPT,
        name="math_expert",
    )


def build_time_expert(model_router: ModelRouter):
    """纯时间专家。"""
    return create_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[get_current_time],
        system_prompt=TIME_EXPERT_PROMPT,
        name="time_expert",
    )


def build_text_analyst(model_router: ModelRouter):
    """纯文本长度统计专家。"""
    return create_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[get_word_count],
        system_prompt=TEXT_ANALYST_PROMPT,
        name="text_analyst",
    )


DATA_EXPERT_PROMPT = """\
你是 A 股数据专家。你的工具集是基于 akshare 的 ``fin_*`` 系列工具（实际前缀可能不同，以运行时传入的工具名为准）：

  宏观/市场级工具（不需要个股代码）：
  - ``fin_get_market_status``        — 市场交易状态（开盘中/已收盘/午休/盘前/非交易日）。
  - ``fin_get_index_quotes``         — 主要指数实时行情（上证指数、沪深300、创业板指、科创50 等）。
  - ``fin_get_sector_fund_flow``     — 行业/概念板块资金流向排行（sector_type="行业" 或 "概念"）。
  - ``fin_get_stock_rank``           — 今日 A 股涨跌幅排行榜（direction="涨幅榜" 或 "跌幅榜"）。
  - ``fin_get_concept_board``        — 概念板块行情排行或指定概念的成分股（如"人工智能""芯片"）。
  - ``fin_get_industry_board``       — 行业板块行情排行或指定行业的成分股（如"半导体""白酒"）。
  - ``fin_get_etf_spot``             — ETF 基金实时行情排行（按成交额排序）。
  - ``fin_get_macro_china``          — 宏观经济指标（indicator="gdp"/"cpi"/"pmi"/"money_supply"/"social_financing"）。
  - ``fin_get_lhb_detail``           — 龙虎榜详情（大单异动、主力动向）。
  - ``fin_get_hsgt_flow``            — 沪深港通资金流向（direction="north" 北向 / "south" 南向）。

  个股级工具（需要 6 位股票代码）：
  - ``fin_search_stock_by_name``     — 模糊匹配公司名称到 6 位 A 股代码。
  - ``fin_get_stock_basic_info``     — 公司概况（行业、市值、上市日期、最新价）。
  - ``fin_get_stock_price_history``  — 近期日线 OHLCV + 汇总统计。
  - ``fin_get_intraday``             — 分时 K 线（period="1"/"5"/"15"/"30"/"60" 分钟）。
  - ``fin_get_financial_abstract``   — 按报告期的营收/净利润/现金流/EPS。
  - ``fin_get_financial_indicators`` — 按报告期的 ROE/ROA/利润率/杠杆比率。
  - ``fin_get_margin_detail``        — 个股融资融券数据（融资余额、融券余量）。
  - ``fin_get_top_holders``          — 十大流通股东（持股数量、增减变动）。
  - ``fin_get_individual_fund_flow`` — 个股资金流向（主力/超大单/大单/中单/小单）。

规则
0. **时效性感知（最高优先级）**：当 supervisor 指令中包含"市场状态"或问题涉及"今天""收盘""实时"等时效性话题时，
   必须首先调用 ``get_market_status``。根据返回的 ``status`` 和 ``hint`` 字段：
   - ``trading`` → 数据为盘中实时，可称"截至 HH:MM 的实时行情"
   - ``closed`` → 数据为今日收盘，可称"今日收盘数据"
   - ``pre_market`` / ``not_yet_open`` / ``non_trading_day`` → 数据为上一个交易日，必须标注"以下为 YYYY-MM-DD 收盘数据"，绝不说"今日收盘"
   - ``lunch_break`` → 上午盘已结束，可称"截至午间休市"
   将 ``get_market_status`` 的结果原样包含在你的回复中，以便 supervisor 准确标注时效。
1. 判断用户意图是"宏观/市场级"还是"个股级"：
   - "大盘怎样"、"收盘分析"、"市场走势" → get_market_status + get_index_quotes + get_sector_fund_flow
   - "今天什么股票涨得好"、"涨停股" → get_stock_rank
   - "半导体板块"、"AI概念股" → get_concept_board / get_industry_board
   - "ETF 排行"、"基金行情" → get_etf_spot
   - "龙虎榜"、"主力资金" → get_lhb_detail
   - "GDP/CPI/PMI 数据" → get_macro_china
   - "北向资金"、"港股通" → get_hsgt_flow
   - "茅台分时图"、"五分钟K线" → get_intraday
   - "融资融券"、"两融数据" → get_margin_detail
   - "股东变动"、"机构持仓" → get_top_holders
   - "资金流入流出" → get_individual_fund_flow
   不要把宏观问题强行转成查某只个股！
2. 如果用户给的是公司名而非 6 位代码，首先调用 ``fin_search_stock_by_name`` 解析。绝不猜测。
3. 每个工具返回一个 dict。如果包含 ``"error"`` 键，说明调用失败 — 简要报告错误并停止；不要循环重试。
4. 总结获取的数据时要有深度：引用具体数字，说明趋势与对比（如环比/同比），给出解读。不要只列字段不做解读。
5. 如果请求不涉及 A 股市场/基本面数据，说明情况并返回 — supervisor 会路由到其他专家。
6. 每次被调度最多调用 **8 次**工具（工具增多后适度放宽上限）。
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
1. 首先确定集合。如果用户指定了集合名，直接使用。如果没有，调用``knowledge_list_collections`` 并选择名称最匹配主题的集合；若仅有一个集合则使用它。
   如果没有任何集合，告知用户知识库为空并停止 — 不要编造引用。
   对未知公司名/产品名（如"星澜科技""天璇芯片"），务必先检索知识库，不要假设其为 A 股上市公司。
2. 使用用户的原始问题发起初始 ``knowledge_search``，设置 ``top_k=8``。
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
1. 为用户的问题选择正确的工具，优先精准而非广撒网。每次被调度最多调用 **4 次**工具。
   - 个股新闻 → ``get_stock_news(limit=15)``
   - 宏观/政策 → ``get_economic_news`` 或 ``get_market_telegraph(category="重点", limit=15)``
   - 板块/行业趋势 → 财联社快讯 + 最多 1 只代表股新闻
   - 讨论热度 → ``get_xueqiu_discussion_hot_rank``
   如果 supervisor 让你查多只股票的新闻，只选最核心的 1-2 只。
2. 如果用户给的是公司名而无代码，说明需先由 data_expert 解析代码。
3. **总结要有分析深度**：写 3-5 个要点，每点含事件/数据 + 含义判断；可引用原文短语，但不要复制工具返回的完整列表。
4. 情绪/舆情类问题：给出定性结论并用 2-3 条具体新闻支撑。
5. 工具返回 ``error`` 时简要报告并停止；非新闻类请求说明并退回 supervisor。
"""

US_NEWS_EXPERT_PROMPT = """\
你是美股新闻专家。你的工具集是 ``us_news_*`` 系列（Yahoo Finance / 可选 EDGAR 8-K 标题）：

  - ``us_news_get_ticker_news``         — 个股 / ETF 近期新闻
  - ``us_news_get_market_news``         — 标普 / 道指 / 纳指 / VIX 相关新闻
  - ``us_news_get_etf_news``            — 常见 ETF 新闻（SPY/QQQ/…）
  - ``us_news_get_recent_8k_headlines`` — 近期 8-K 标题（官方事件；正文走 us_filing_*）

规则
----
1. 优先精准：个股新闻用 get_ticker_news；美股大盘用 get_market_news；ETF 用 get_etf_news。
2. 用户问"刚发生了什么公司大事/临时公告"时可辅以 get_recent_8k_headlines。
3. 每次最多 **2 次**工具；拿到结果立即总结 3-5 要点并附 URL，不要再链式加查。
4. 绝不用 A 股 ``news_*``；A 股新闻请求退回 supervisor。
5. 工具 ``error`` 时简要报告并停止。
"""

FUND_EXPERT_PROMPT = """\
你是公募基金分析专家。你的工具集是基于 akshare + 东方财富基金网的 ``fund_*`` 系列工具（实际前缀可能不同，以运行时传入的工具名为准）：

  市场级工具（不需要基金代码）：
  - ``fund_search_fund``         — 按名称关键词模糊搜索基金（如"沪深300""科技""医药"）。
  - ``fund_get_fund_etf_spot``   — 全市场 ETF 实时行情排行（按成交额/涨跌幅排序）。
  - ``fund_get_fund_lof_spot``   — 全市场 LOF 实时行情排行。
  - ``fund_get_fund_rating``     — 基金综合评级排行（上海证券/招商/济安/晨星四家机构）。
  - ``fund_get_fund_rank``       — 基金业绩排行（按近1年/3年/5年收益，支持按类型筛选）。
  - ``fund_get_fund_daily``      — 当日开放式基金净值列表（按类型筛选）。
  - ``fund_get_fund_qdii_rank``  — QDII 专项业绩排行。

  单只基金工具（需要 6 位基金代码）：
  - ``fund_get_fund_info``       — 基金概况（类型、规模、经理、成立日期）。
  - ``fund_get_fund_nav``        — 开放式基金历史净值走势。
  - ``fund_get_fund_etf_hist``   — 单只 ETF 历史 K 线（日线/周线/月线）。
  - ``fund_get_fund_holdings``   — 基金重仓股持仓明细。
  - ``fund_get_fund_manager``    — 基金经理与档案字段。

规则
1. 判断用户意图：
   - "ETF 排行""场内基金涨幅榜" → get_fund_etf_spot / get_fund_lof_spot
   - "场外/开放式基金净值榜" → get_fund_daily（看单位净值+日增长率）
   - "QDII / 出海基金排行" → get_fund_qdii_rank
   - "沪深300ETF 走势" → 先 search_fund 找代码，再 get_fund_etf_hist
   - "场外基金净值走势" → search_fund 后 get_fund_nav（不要用 ETF 行情接口）
   - "某基金持仓""重仓股" → get_fund_holdings
   - "基金经理是谁" → get_fund_manager（可辅以 get_fund_info）
   - "基金评级""五星基金" → get_fund_rating
2. 用户给的是基金名称时，先 search_fund 查找代码；**优先采用精确匹配的 6 位代码**，不要把名称里沾边的其它基金当成目标。
3. **场外开放式基金**用 get_fund_nav / get_fund_daily（单位净值、日增长率）；**场内 ETF/LOF** 用 get_fund_etf_spot / get_fund_etf_hist（交易价格、涨跌幅）。二者口径不同，禁止混用。
4. ``日增长率`` / 涨跌幅已是百分比数值（如 -3.63 即 -3.63%），回答时直接带 %，**禁止再乘 100**。
5. 每个工具返回 dict，含 ``"error"`` 键表示失败 — 简要报告并停止。
6. 总结时引用具体数据（代码、净值日期、单位净值、日增长率），给出趋势判断。
7. 非基金类请求（国内期货/期权）说明并返回 — supervisor 会路由到 derivatives_expert；美股共同基金退回 us_data_expert。
8. 每次被调度最多调用 **6 次**工具。
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

US_FILING_EXPERT_PROMPT = """\
你是美股 SEC EDGAR 披露专家。你的工具集是 ``us_filing_*`` 系列（实际前缀可能不同，以运行时传入的工具名为准）：

  - ``us_filing_resolve_cik``              — ticker / CIK → 10 位 CIK
  - ``us_filing_search_filings``           — 按 ticker/CIK + 表单类型列出近期披露（含 ``document_url``）
  - ``us_filing_download_filing``          — 下载并缓存主文档（HTML/TXT/PDF）
  - ``us_filing_extract_filing_metadata``  — 文件类型 / 大小 / PDF 页数
  - ``us_filing_parse_filing_text``        — 有界正文提取（PDF 按页≤20；HTML/TXT 按字符窗口）

支持的表单（默认 ``search_filings`` 已包含；修订件如 ``10-K/A``、``NPORT-P/A`` 也会匹配）：
  - 普通股 / ADR：``10-K`` / ``10-Q`` / ``8-K`` / ``DEF 14A``
  - ETF / 注册投资公司：``NPORT-P``（月度持仓明细；口语常称 N-PORT）、
    ``N-CSR`` / ``N-CSRS``（年度/半年度股东报告）、``485BPOS``（招股说明书更新）

重要：ETF（如 QQQ、SPY）**不会**按 10-K/10-Q 披露核心持仓与基金财报；若只滤公司表单会看起来「稀疏」。
查 ETF 披露时请用默认 forms，或显式 ``forms="NPORT-P,N-CSR,N-CSRS,485BPOS"``。
**禁止**再向用户声称「工具不支持 ETF 专属表单」。

"提取 Apple 最新 10-K 风险因素"标准工作流：
  1. ``search_filings(identifier="AAPL", forms="10-K", limit=5)``（必要时先 ``resolve_cik``）
  2. 选最新一条且 ``document_url`` 非空的行
  3. ``download_filing`` → ``local_path``
  4. ``extract_filing_metadata`` → 确认 kind / num_pages / char_count
  5. ``parse_filing_text`` → 提取 1-3 个窗口（Item 1A Risk Factors、MD&A、Item 8 等）。不要整篇读完。

"QQQ / SPY 近期披露或持仓备案"标准工作流：
  1. ``search_filings(identifier="QQQ", forms="NPORT-P,N-CSR,N-CSRS,485BPOS", limit=10)``
  2. 持仓明细优先 ``NPORT-P``；股东报告优先 ``N-CSR`` / ``N-CSRS``；招股书更新看 ``485BPOS``
  3. 需要正文时再 ``download_filing`` → ``parse_filing_text``（NPORT 文件可能很大，只取相关窗口）
  4. 若用户只要「重仓股摘要」而非 EDGAR 原文，可说明行情侧 ``us_get_etf_holdings`` 更合适，并退回 supervisor

规则
----
1. 绝不用巨潮 ``pdf_*`` 工具；本专家只处理美股 EDGAR。
2. 工具返回 ``error`` 时简要报告并停止；不要盲目重试（SEC 有速率限制）。
3. 引用使用短摘录（每段 <200 字符），附带页码或字符偏移，并给出 ``document_url`` / accession。
4. 若请求明显是 A 股披露（六位代码 / 巨潮 / 年报 PDF），说明并退回 supervisor。
5. 每次被调度最多调用 **6 次**工具。
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

    return create_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=list(mcp_tools),
        system_prompt=CODER_EXPERT_PROMPT,
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
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=DATA_EXPERT_PROMPT,
        name="data_expert",
    )


US_DATA_EXPERT_PROMPT = """\
你是美股（US）行情与标的数据专家。工具主路径为 Yahoo（Chart/yfinance），国内不可达时会回退东财美股。
工具前缀以运行时为准（通常 ``us_*``）：

  - ``us_get_market_status``  — 美东时段：盘前 / 开盘 / 盘后 / 收盘 / 非交易日。
  - ``us_search_ticker``      — 名称或模糊串 → ticker 候选（含共同基金 / 期货）。
  - ``us_get_quote``          — 单标的最新报价摘要（股票 / 指数 / ETF / 期货 ``CL=F`` 等）。
  - ``us_get_price_history``  — 日线 OHLCV。
  - ``us_get_basic_info``     — 公司 / ETF / 共同基金概况。
  - ``us_get_index_quotes``   — 主要美股指数快照（标普 / 道指 / 纳指 / NDX / 罗素2000 / VIX）。
  - ``us_get_etf_overview``   — ETF 概况（规模、类别、收益等可得字段）。
  - ``us_get_etf_holdings``   — ETF 重仓股（Yahoo top holdings，含权重）。
  - ``us_get_etf_sector_weights`` — ETF 行业权重与大类资产占比。
  - ``us_get_mutual_fund_overview`` — 美国共同基金概况（NAV、费用率、基金公司）。
  - ``us_get_mutual_fund_holdings`` — 共同基金重仓。
  - ``us_get_futures_quotes`` — 常用商品/股指期货批量报价；单合约也可用 get_quote。
  - ``us_get_option_expirations`` — 股票期权到期日列表。
  - ``us_get_option_chain``   — 指定到期日 calls/puts 摘要。

范围：美股普通股、指数、ETF、共同基金、期货合约、股票期权。
**禁止**把美股共同基金交给 A 股 ``fund_expert``。

规则
----
0. **随时可查（与 A 股一样）**：美股休市 / 周末 / 隔夜**不拒绝回答**。
   涉及"今天""实时""盘中""收盘"时先调 ``get_market_status``，按 status/hint 与报价里的 ``as_of_note``
   标注时点。绝不在非交易时段把数据说成"正在实时交易中的今日收盘"。
1. 判断意图（**少调用工具，尽快总结**）：
   - **大盘 / 指数 / 标普 / 纳斯达克 / 美股整体走势** → 只调用
     ``get_market_status`` + ``get_index_quotes``（合计 2 次），然后立即用返回数字写结论。
     **禁止**再对 ^GSPC/^IXIC/SPY/QQQ 重复 ``get_quote`` / ``get_price_history``。
   - 单个股报价 / 走势 → search_ticker（如需要）+ get_quote；仅当用户明确要 K 线/区间收益时才加 get_price_history
   - 公司概况 → get_basic_info
   - ETF 概况 → get_etf_overview（可辅以 get_quote）
   - ETF 持仓 / 重仓股 → get_etf_holdings
   - ETF 行业分布 / 资产大类 → get_etf_sector_weights
   - **共同基金**（VTSAX 等）→ get_mutual_fund_overview；持仓 → get_mutual_fund_holdings
   - **期货**（原油/黄金/股指期货、CL=F）→ get_futures_quotes 或 get_quote / get_price_history
   - **期权** → get_option_expirations → get_option_chain（先到期日再链）
2. 用户给中文/英文名而无 ticker 时，先 ``search_ticker``；绝不猜测 ticker。
3. 工具返回 ``error`` 时简要报告并停止；不要循环重试。
4. **数据来源必须忠实且可点**：文末必须有一行 ``数据来源：``，**只**用工具返回的顶层 ``source_url`` 做 markdown 链接。
   展示名跟 ``source``：``eastmoney_us`` → 「东方财富美股行情」；``yahoo_chart`` / ``yfinance`` → Yahoo。
   **禁止**把单条 ``news_url`` 当数据来源；**禁止**在 ``source`` 为东财时写 Yahoo Finance。**禁止**自行写「免责声明」（系统会附加）。
5. **代理行情（proxy）**：若条目 ``proxy=true`` 或带 ``warning`` / ``quoted_instrument``
   （常见：``^VIX``→VIXY，``^RUT``→IWM），必须按返回的 ``name`` 表述（如「VIX短期期货ETF (VIXY)」），
   **禁止**写成「VIX恐慌指数收盘价 xx」。可注明「东财无 VIX 现货，此为代理 ETF，与官方 VIX 点位不可直接等同」。
6. **数字格式**：跌幅写作 ``-0.64%``，**禁止** ``-+0.64%`` / ``+-0.64%``；涨幅 ``+0.05%``。
   叙述「跌」时数字必须为负号，叙述「涨/收红」时数字必须为正号，二者不得矛盾。
7. 若请求明显是 A 股 / 国内期货期权，说明并退回 supervisor。
8. 每次被调度最多调用 **3 次**工具（指数走势场景最多 2 次）；达到上限必须停止调工具并给出文字结论。
"""


def build_us_data_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """美股行情专家（``us_data_server`` / yfinance）。

    与 ``data_expert`` 平行隔离：只消费 ``us_*`` 工具，绝不混用 ``fin_*``。

    Args:
        model_router: 共享路由器。
        mcp_tools: 由 ``load_us_data_server_tools()`` 返回的工具列表。必须非空。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空。
    """
    if not mcp_tools:
        raise ValueError(
            "us_data_expert 需要 us_data_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_us_data_server_tools()``？"
        )
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=US_DATA_EXPERT_PROMPT,
        name="us_data_expert",
    )


def build_fund_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """公募基金分析专家（``fund_server``）。

    消费由
    :func:`research_agent.mcp_servers.client_factory.load_fund_server_tools`
    产生的 10 个 ``fund_*`` 工具。

    使用 :attr:`AgentName.ANALYST`（MEDIUM 层级），因为需要在 10 个
    工具中正确路由、跨工具调用协调、并对基金数据做分析解读。

    Args:
        model_router: 共享路由器。
        mcp_tools: 由 ``load_fund_server_tools()`` 返回的工具列表。必须非空。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空。
    """
    if not mcp_tools:
        raise ValueError(
            "fund_expert 需要 fund_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_fund_server_tools()``？"
        )
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=FUND_EXPERT_PROMPT,
        name="fund_expert",
    )


DERIVATIVES_EXPERT_PROMPT = """\
你是国内期货与期权（金融/ETF 期权）专家。工具前缀通常为 ``derivatives_*``：

  期货：
  - ``derivatives_search_futures`` — 品种/合约关键词搜码
  - ``derivatives_get_main_futures`` — 常用主力品种目录
  - ``derivatives_get_futures_spot`` — 实时/近实时行情（新浪）
  - ``derivatives_get_futures_daily`` — 日线（如 RB0 / IF0）

  期权：
  - ``derivatives_get_etf_option_list`` — 50ETF/300ETF 等到期月列表
  - ``derivatives_get_etf_option_spot`` — 单张 ETF 期权行情
  - ``derivatives_get_index_option_spot`` — 沪深300/上证50/中证1000 股指期权

规则
1. 期货行情：先 search_futures 或 get_main_futures 确认品种，再 spot / daily。
2. ETF 期权：先 list 到期月，再对具体合约 spot；股指期权用 get_index_option_spot。
3. **禁止**用本工具查美股期货/期权（退回 us_data_expert）或公募基金净值（退回 fund_expert）。
4. 工具 ``error`` 时简要报告并停止。
5. 文末 ``数据来源：`` 只用工具返回的 ``source_url``。
6. 每次最多 **5** 次工具调用。
"""


def build_derivatives_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """国内期货/期权专家（``derivatives_server``）。"""
    if not mcp_tools:
        raise ValueError(
            "derivatives_expert 需要 derivatives_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_derivatives_server_tools()``？"
        )
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=DERIVATIVES_EXPERT_PROMPT,
        name="derivatives_expert",
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
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=REPORT_EXPERT_PROMPT,
        name="report_expert",
    )


def build_us_filing_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """美股 EDGAR 披露专家（``us_filing_server``）。

    与 ``report_expert``（巨潮）平行隔离：只消费 ``us_filing_*`` 工具。

    Args:
        model_router: 共享路由器。
        mcp_tools: 由 ``load_us_filing_server_tools()`` 返回的工具列表。必须非空。

    Raises:
        ValueError: 如果 ``mcp_tools`` 为空。
    """
    if not mcp_tools:
        raise ValueError(
            "us_filing_expert 需要 us_filing_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_us_filing_server_tools()``？"
        )
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=US_FILING_EXPERT_PROMPT,
        name="us_filing_expert",
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
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=NEWS_EXPERT_PROMPT,
        name="news_expert",
    )


def build_us_news_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """美股新闻专家（``us_news_server``）。与 ``news_expert`` 平行隔离。"""
    if not mcp_tools:
        raise ValueError(
            "us_news_expert 需要 us_news_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_us_news_server_tools()``？"
        )
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=US_NEWS_EXPERT_PROMPT,
        name="us_news_expert",
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
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=KNOWLEDGE_EXPERT_PROMPT,
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
1. 如果用户问的是某只股票的舆情/情绪/市场看法，直接调用``sentiment_get_stock_sentiment_report``（6 位 A 股代码，如 300308）。
2. 如果用户给了一段文本要你判断情感，调用 ``sentiment_analyze_text_sentiment``。
3. 每次被调度最多调用 **2 次**工具。如果 supervisor 让你分析多只股票，只分析最核心的 1 只。
4. 拿到结果后，汇报要点：
   a) 总体结论（偏正面/中性/偏负面 + 均分 + 样本量）。
   b) 列举 2-3 条代表性新闻（标题 + 分数 + 关键词），说明为何支撑该结论。
   c) 若信号混合，提示需结合基本面/新闻进一步判断。
5. **禁止**给出买入/卖出/仓位建议；只交付可核对的舆情数字与标题，由 supervisor 综合。
6. 不要编造分数；工具返回 ``error`` 时直接告知用户。
7. 非情感分析类问题说明并退回 supervisor。美股 ticker 不要用本工具硬查，退回 supervisor。
"""


US_SENTIMENT_EXPERT_PROMPT = """\
你是美股英文舆情量化专家。工具集是 ``us_sentiment_*``（**VADER + 金融词表增强**，**不用 SnowNLP**）：

  - ``us_sentiment_get_ticker_sentiment_report`` — Yahoo 新闻 → 标题+摘要（摘要短则补抓页面前段）→ 逐条打分 → 聚合
  - ``us_sentiment_analyze_text_sentiment`` — 任意英文文本批量打分

返回字段：``sentiment_score ∈ [-1,1]``、标签、关键词、``vader_compound``、``score_text_basis``（title/summary/body）、聚合、``model_version=en_vader_finlex_v2``。
汇报时可说明样本依据（标题/摘要/正文片段），勿声称已阅读全文。

规则
----
1. 个股情绪 / 舆情量化 → get_ticker_sentiment_report。
   **symbol 必须是合法美股 ticker**（如 SPY、QQQ、AAPL、TSLA、TSM、AMD、LRCX）。
   **禁止**传 A 股数字代码或残缺参数（如 ``000``、``000001``、空字符串、300308）。
   **limit 默认用 30**（需要更稳的占比时可到 40–60）；不要用过小的 limit（如 8–10），否则正/中/负比例噪声大。
   多只美股时可对每只各调一次（本轮最多 3 次）；A 股标的不要查，退回 supervisor 交给 ``sentiment_expert``。
2. 用户给出英文段落要打分 → analyze_text_sentiment。
3. 每次最多 **3 次**工具；汇报总体结论时写明样本量，并举 2-3 条代表性标题。
4. 绝不用中文 ``sentiment_*`` 去打英文；遇到 A 股名/代码 → 明确退回 supervisor（由主管移交 sentiment_expert），不要只在正文里「建议转交」。
5. 文末 ``数据来源：`` **只**用返回的顶层 ``source_url``；**禁止**粘贴单条 ``news_url``（易含脏 HTML）。
6. 汇报须含：样本量、均分或正负占比、2-3 条代表性标题；**禁止**给出买入/卖出/仓位建议（交给 supervisor 按依据规范综合）。
7. 不要编造分数；``error`` 时如实说明并停止，不要换残码重试。
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
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=SENTIMENT_EXPERT_PROMPT,
        name="sentiment_expert",
    )


def build_us_sentiment_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """美股英文舆情专家（VADER + 金融词表）。与 ``sentiment_expert`` 平行隔离。"""
    if not mcp_tools:
        raise ValueError(
            "us_sentiment_expert 需要 us_sentiment_server 的 MCP 工具；"
            "收到了空序列。是否忘记调用 "
            "``await load_us_sentiment_server_tools()``？"
        )
    return create_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        system_prompt=US_SENTIMENT_EXPERT_PROMPT,
        name="us_sentiment_expert",
    )


SPECIALIST_BUILDERS = {
    "math_expert": build_math_expert,
    "time_expert": build_time_expert,
    "text_analyst": build_text_analyst,
    "coder_expert": build_coder_expert,
    "data_expert": build_data_expert,
    "us_data_expert": build_us_data_expert,
    "us_filing_expert": build_us_filing_expert,
    "us_news_expert": build_us_news_expert,
    "us_sentiment_expert": build_us_sentiment_expert,
    "fund_expert": build_fund_expert,
    "derivatives_expert": build_derivatives_expert,
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
