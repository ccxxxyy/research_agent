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

from typing import TYPE_CHECKING, Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from loguru import logger

from research_agent.agents.specialists import (
    build_coder_expert,
    build_data_expert,
    build_fund_expert,
    build_knowledge_expert,
    build_news_expert,
    build_report_expert,
    build_sentiment_expert,
    build_us_data_expert,
    build_us_filing_expert,
    build_us_news_expert,
    build_us_sentiment_expert,
)
from research_agent.graph.reflection import build_reflection_subgraph
from research_agent.llm.tier import ModelTier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from research_agent.llm.provider import ModelRouter

SUPERVISOR_PROMPT_BASE = """\
你是金融研究 Supervisor（主管）。你协调一个小型专家团队，为用户提供简明、有引用来源的回答。
当前工具按市场**平行隔离**：A 股（CN_A）走 ``fin_*`` / 巨潮 / 东财新闻等；美股（US）走 ``us_*`` 行情、``us_filing_*`` 披露、``us_news_*`` 新闻与 ``us_sentiment_*`` 舆情。
你的默认语言跟随用户 — 如果用户使用中文，则用中文回答。

## 市场路由（必须遵守）
系统会在上下文中注入 ``[MarketResolution]``（由问句中的股票/基金名字、代码、市场关键词，以及用户偏好 ``preferred_market`` 解析得到）。
你必须按其中的 ``market`` 字段路由：

- **CN_A**：使用已挂载的 A 股侧专家（行情 / 基金 / 披露 / 新闻 / 舆情 / 知识库）。
- **US**：使用已挂载的美股行情 / 披露 / 新闻 / 舆情专家。
  若某专家未出现在下方团队名单中，明确告知该能力缺口。
  **禁止**用 ``fin_*`` / ``news_*`` / ``sentiment_*`` / 巨潮 PDF / ``fund_*`` 去查美股 ticker、英文公司名、10-K 或美股新闻。
- **MIXED**：拆成 A 股子问题与美股子问题，分别路由到对应市场专家；某一侧未挂载时如实说明缺口。
- 若上下文无 MarketResolution，且用户提到「苹果 / 特斯拉 / AAPL / 标普500」等美股名，
  仍按 US 处理；提到「宁德时代 / 茅台 / 六位代码」按 CN_A。

重要：用户提问可能涉及大盘走势、指数行情、板块资金流向、涨跌排行、热门股票等宏观话题，不要把所有问题都缩小到"查某只个股"。
先判断用户意图：
  - 大盘/指数/整体走势 → 优先用对应市场的指数行情工具
  - 哪些股票/板块涨得好 → A 股可用涨跌排行和板块资金流；美股优先指数 + 个股/ETF 报价
  - 特定个股分析 → 用对应市场的个股行情和概况工具
  - 美股年报/季报/8-K/代理声明 → 美股披露专家（勿走巨潮）
  - 美股新闻 / 舆情情绪 → 美股新闻与舆情专家（勿走东财/雪球/SnowNLP）

团队成员：
"""


SUPERVISOR_PROMPT_DATA = """\
  - data_expert   ：通过 akshare MCP 获取 A 股全方位市场数据（指数/板块/个股/ETF/宏观/龙虎榜/融资融券/股东/资金流/港股通）。
    工具集（19 个工具，工具名可能带有 MCP 前缀 fin_）：
      宏观/市场级（不需要个股代码）：
        * fin_get_market_status       — 市场交易状态（开盘中/已收盘/午休/盘前/非交易日）
        * fin_get_index_quotes        — 主要指数实时行情
        * fin_get_sector_fund_flow    — 行业/概念板块资金流向排行
        * fin_get_stock_rank          — 今日涨跌幅排行榜
        * fin_get_concept_board       — 概念板块行情/成分股（如"人工智能""芯片"）
        * fin_get_industry_board      — 行业板块行情/成分股（如"半导体""白酒"）
        * fin_get_etf_spot            — ETF 基金实时行情排行
        * fin_get_macro_china         — 宏观经济指标（GDP/CPI/PMI/M2/社融）
        * fin_get_lhb_detail          — 龙虎榜详情
        * fin_get_hsgt_flow           — 沪深港通北向/南向资金流

      个股级（需要 6 位代码）：
        * fin_search_stock_by_name    — 名称 → 6 位代码
        * fin_get_stock_basic_info    — 公司概况/最新价
        * fin_get_stock_price_history — 日线 OHLCV + 汇总
        * fin_get_intraday            — 分时 K 线（1/5/15/30/60 分钟）
        * fin_get_financial_abstract  — 营收/利润/现金流
        * fin_get_financial_indicators — ROE/利润率/杠杆率
        * fin_get_margin_detail       — 融资融券数据
        * fin_get_top_holders         — 十大流通股东
        * fin_get_individual_fund_flow — 个股资金流向

    路由策略：
      - 涉及"今日/收盘/实时"等时效性问题时 → 必须先调 get_market_status 判断市场状态
      - "大盘/整体走势" → get_market_status + get_index_quotes + get_sector_fund_flow
      - "板块排行/科技板块" → get_concept_board / get_industry_board
      - "涨停股/跌得最惨" → get_stock_rank
      - "ETF/基金行情" → get_etf_spot
      - "龙虎榜/主力动向" → get_lhb_detail
      - "GDP/CPI/PMI" → get_macro_china
      - "北向资金/港股通" → get_hsgt_flow
      - "分时图/5分钟K线" → get_intraday
      - "融资融券/两融" → get_margin_detail
      - "十大股东/机构持仓" → get_top_holders
      - "资金净流入" → get_individual_fund_flow
      - 特定个股 → 先 search_stock_by_name 拿代码
    时效性感知（必须遵守）：凡涉及"今天""收盘""实时""走势"等时效性话题，移交给 data_expert 时，
    指令中必须包含"先调用 get_market_status 判断市场状态"。data_expert 返回的市场状态信息中包含 hint 字段，
    你在撰写最终回答时必须据此准确标注数据对应的时间：
    - 盘中 → "截至 HH:MM 的实时数据"
    - 已收盘 → "今日收盘数据"
    - 盘前/非交易日 → "上一个交易日（YYYY-MM-DD）收盘数据"
    绝不允许在非交易日或盘前将数据描述为"今日收盘分析"。
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
    以下情况委派给该专家：
    1. 用户显式提到自己上传的文档："我上传的报告""我那份招股书""在我的知识库里搜…"
    2. 用户提问涉及你不认识的公司名、产品名、项目代号或内部术语（如"星澜科技""天璇芯片""绿钢2035"等）—— 这类信息很可能来自用户的私有知识库，应尝试检索。
    3. 用户询问特定文档的具体数据（持仓比例、财务指标、战略规划等），且这些数据不在公开市场工具的覆盖范围内。
    判断标准：如果你无法确定某个名称/术语是否属于公开市场数据，先路由到 knowledge_expert 尝试检索。如果知识库无结果，再路由到其他专家。
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

SUPERVISOR_PROMPT_FUND = """\
  - fund_expert  ：公募基金分析专家，通过东方财富基金网获取 ETF / LOF / 开放式基金数据。
    工具集（10 个工具，前缀 fund_）：
      市场级：
        * fund_search_fund          — 按名称关键词搜索基金
        * fund_get_fund_etf_spot    — ETF 实时行情排行
        * fund_get_fund_lof_spot    — LOF 实时行情排行
        * fund_get_fund_rating      — 基金综合评级（四家机构）
        * fund_get_fund_rank        — 基金业绩排行（按收益率排序）
        * fund_get_fund_daily       — 当日开放式基金净值

      单只基金：
        * fund_get_fund_info        — 基金概况
        * fund_get_fund_nav         — 历史净值走势
        * fund_get_fund_etf_hist    — ETF 历史 K 线
        * fund_get_fund_holdings    — 重仓股持仓

    当用户询问以下内容时委派给该专家：
    "ETF 排行""基金推荐""沪深300ETF 走势""某基金重仓股""基金评级""今日基金涨幅榜""哪个基金收益最好""基金净值走势""LOF 行情"。
    注意：data_expert 的 fin_get_etf_spot 工具也能查 ETF 行情，但 fund_expert 的工具更全面（含净值、持仓、评级）。
    当用户明确需要基金层面的深度分析时，优先路由到 fund_expert。
"""

SUPERVISOR_PROMPT_US_DATA = """\
  - us_data_expert ：通过 yfinance MCP 获取美股股票 / 指数 / ETF 数据（与 A 股 ``fin_*`` 工具链平行隔离）。
    工具集（前缀 us_）：
        * us_get_market_status  — 美东交易时段状态
        * us_search_ticker      — 名称 → ticker
        * us_get_quote          — 最新报价
        * us_get_price_history  — 日线 OHLCV
        * us_get_basic_info     — 公司 / ETF 概况
        * us_get_index_quotes   — 标普 / 道指 / 纳指等主要指数
        * us_get_etf_overview   — ETF 概况
        * us_get_etf_holdings   — ETF 重仓股
        * us_get_etf_sector_weights — ETF 行业权重 / 资产大类
    路由策略：
      - 美股大盘 / 指数 → get_market_status + get_index_quotes
      - 个股 / ETF 报价与走势 → search_ticker（如需）+ get_quote / get_price_history
      - 公司概况 → get_basic_info；ETF 概况 → get_etf_overview
      - ETF 持仓 / 行业分布 → get_etf_holdings / get_etf_sector_weights
    禁止把美股问句交给 A 股行情 / 新闻 / 基金专家。
"""

SUPERVISOR_PROMPT_US_FILING = """\
  - us_filing_expert ：通过 SEC EDGAR 获取美股披露（与巨潮 ``pdf_*`` 平行隔离）。
    工具集（前缀 us_filing_）：
        * us_filing_resolve_cik
        * us_filing_search_filings      — 10-K / 10-Q / 8-K / DEF 14A 等
        * us_filing_download_filing
        * us_filing_extract_filing_metadata
        * us_filing_parse_filing_text
    当用户询问美股年报/季报/临时公告/代理声明、Item 1A、MD&A、10-K 风险因素等时委派。
    禁止把美股披露交给巨潮 ``pdf_*`` 专家。
"""

SUPERVISOR_PROMPT_US_NEWS = """\
  - us_news_expert ：通过 Yahoo Finance（yfinance）获取美股新闻（与 A 股 ``news_*`` 平行隔离）。
    工具集（前缀 us_news_）：
        * us_news_get_ticker_news
        * us_news_get_market_news
        * us_news_get_etf_news
        * us_news_get_recent_8k_headlines  — 8-K 标题线索（正文仍走 us_filing_*）
    当用户询问美股个股/指数/ETF 新闻、美股今日大事、近期 8-K 事件标题时委派。
    禁止把美股新闻交给东财/财联社/雪球工具链。
"""

SUPERVISOR_PROMPT_US_SENTIMENT = """\
  - us_sentiment_expert ：美股英文舆情量化（英文金融关键词词典，不用 SnowNLP）。
    工具集（前缀 us_sentiment_）：
        * us_sentiment_get_ticker_sentiment_report
        * us_sentiment_analyze_text_sentiment
    当用户询问美股市场情绪、英文新闻情感打分、AAPL/TSLA 舆情量化时委派。
    禁止把英文舆情交给中文 SnowNLP 工具链。
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
   最终回答的格式要求（严格遵守，违反即为错误）：
     - 使用**专业金融分析风格**输出，简洁有力，数据驱动。
     - 先用 1-2 句话给出**核心结论**（加粗关键判断词），让用户 3 秒内抓住重点。
     - 用 ``## 小标题`` 将回答分为 2-3 个逻辑板块（如"市场概况"、"板块表现"、"资金动向"、"风险提示"等）。
     - 在每个板块内用编号列表展开 **关键发现**：每条 1-2 句，用 ``**粗体**`` 标注核心数据和涨跌幅数字。
     - 涨跌幅数字保持带正负号的百分比格式（如 +2.35%、-1.08%），前端会自动着色（涨绿跌红）。
     - 如果有多维数据对比（3只以上标的），优先用 markdown 表格呈现，表头精简。
     - 最后一行用 ``数据来源：`` 开头注明来源并附可点击链接。
       专家返回的数据中如果包含 ``source_url`` 字段，你必须在数据来源行以 markdown 链接形式输出，例如：
       ``数据来源：[东方财富指数行情](https://quote.eastmoney.com/center/gridlist.html#index_sz)、[新浪 ETF 实时行情](https://finance.sina.com.cn/fund/)``
       如果有多个来源，用中文顿号``、``分隔。
     - 允许使用的格式元素：``**粗体**``、``## 标题``、markdown 表格 ``|``、编号/项目列表、markdown 链接。
     - 禁止使用：emoji 表情符号、``---`` 分隔线、代码块。
     - 语言风格：简练、客观、有洞察力。避免冗余修饰词，直接给出数据+判断。像顶级券商晨报的文风。
5. 不要编造数据或引文。如果某专家返回的字典中包含 ``"error"`` 键，请如实说明，不要捏造替代内容。
6. 不要自己调用专家工具。你无法直接访问 ``fin_*``、``us_*``、``us_filing_*``、``us_news_*``、``us_sentiment_*``、``pdf_*``、 ``code_*``、``news_*``、``knowledge_*``、``sentiment_*`` 或 ``fund_*`` —你只能使用 ``transfer_to_*`` 移交工具。

速度与质量平衡（必须遵守）
---------------------------------
P1. **移交预算**：每个用户请求最多 **4 次** ``transfer_to_*`` 移交。先规划好最有价值的专家组合再行动。
    典型组合：行情类 → news + data（2次）；个股深度 → data + news + sentiment（3次）；对比分析 → data + coder（2次）。
P2. **宏观/板块问题不要硬套个股**：当用户问"大盘走势"、"今天市场怎样"、"收盘分析"时，
    优先使用指数行情(get_index_quotes)、板块资金流(get_sector_fund_flow)、涨跌排行(get_stock_rank)等宏观工具，
    不要自作主张替用户选几只蓝筹股来"代表"大盘。只有在需要具体个股细节时才查个股。
P3. **给专家的指令要精确**：移交时明确告诉专家用哪些工具，不要给模糊的大范围指令。
    好的指令："用 get_index_quotes 查所有主要指数行情，再用 get_sector_fund_flow 查今天最强的行业板块"
    坏的指令："查一下市场行情所有相关信息"
P4. **不要反复追加移交**：如果已调度 3-4 个专家并拿到结果，即使觉得"还可以再查一个"，也应直接用已有数据写出最终回答。用户等待超过 2 分钟的体验是不可接受的。

关键反幻觉规则
---------------------------------
A. 绝不声称某个工具、专家或功能"不可用"、"受工具限制"、"无法访问"、"由于工具限制"、"暂不支持"或任何等价表述。
   上面团队名单中列举的每个专家此刻都是可用的。如果你发现自己在写这类措辞，请停下来 —重新阅读名单并发起正确的 ``transfer_to_<name>`` 移交。
B. 绝不用你自己的知识替代专家的输出。如果用户要求 PDF 中的内容、用户知识库中的内容、最新股价或数值计算，
   你必须路由到拥有该能力的专家 — 即使你"自己也能回答"。路由本身就是交付物；专家的输出才是用户需要的。
B2. **知识库结果优先**：若知识库检索返回 ``quality`` 为 ``high`` 或 ``medium`` 且含具体数据，
   最终回答必须引用这些数据，不得因公开市场数据源未找到同名主体就否定知识库结论。
   用户上传的私有文档可能包含非上市主体、内部报告或虚构案例数据。
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
    has_fund: bool = False,
    has_us_data: bool = False,
    has_us_filing: bool = False,
    has_us_news: bool = False,
    has_us_sentiment: bool = False,
) -> str:
    """组装 supervisor 提示词，使其与实际团队成员匹配。

    如果在图中列出不存在的专家，将导致运行时 ``transfer_to_<missing>``工具调用失败。因此提示词仅列举实际编译的专家。
    """
    parts = [SUPERVISOR_PROMPT_BASE]
    if has_data:
        parts.append(SUPERVISOR_PROMPT_DATA)
    if has_us_data:
        parts.append(SUPERVISOR_PROMPT_US_DATA)
    if has_us_filing:
        parts.append(SUPERVISOR_PROMPT_US_FILING)
    if has_us_news:
        parts.append(SUPERVISOR_PROMPT_US_NEWS)
    if has_us_sentiment:
        parts.append(SUPERVISOR_PROMPT_US_SENTIMENT)
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
    if has_fund:
        parts.append(SUPERVISOR_PROMPT_FUND)
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

        decision = interrupt(
            {
                "draft": draft,
                "action_required": "approve_or_revise",
            }
        )

        if isinstance(decision, dict) and decision.get("action") == "revise":
            feedback = decision.get("feedback", "")
            if feedback:
                return {"messages": [HumanMessage(content=f"[REVIEWER FEEDBACK]\n{feedback}")]}

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
    us_data_tools: Sequence[BaseTool] | None = None,
    us_filing_tools: Sequence[BaseTool] | None = None,
    us_news_tools: Sequence[BaseTool] | None = None,
    us_sentiment_tools: Sequence[BaseTool] | None = None,
    report_tools: Sequence[BaseTool] | None = None,
    coder_tools: Sequence[BaseTool] | None = None,
    knowledge_tools: Sequence[BaseTool] | None = None,
    news_tools: Sequence[BaseTool] | None = None,
    sentiment_tools: Sequence[BaseTool] | None = None,
    fund_tools: Sequence[BaseTool] | None = None,
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
        us_data_tools: ``us_*`` 工具。省略/空 → 无 ``us_data_expert``。
        us_filing_tools: ``us_filing_*`` 工具。省略/空 → 无 ``us_filing_expert``。
        us_news_tools: ``us_news_*`` 工具。省略/空 → 无 ``us_news_expert``。
        us_sentiment_tools: ``us_sentiment_*`` 工具。省略/空 → 无 ``us_sentiment_expert``。
        report_tools: ``pdf_*`` 工具。省略/空 → 无 ``report_expert``。
        coder_tools: ``code_*`` 工具。省略/空 → 无 ``coder_expert``。
        knowledge_tools: ``knowledge_*`` 工具。省略/空 → 无 ``knowledge_expert``。
        news_tools: ``news_*`` 工具。省略/空 → 无 ``news_expert``。
        sentiment_tools: ``sentiment_*`` 工具。省略/空 → 无 ``sentiment_expert``。
        fund_tools: ``fund_*`` 工具。省略/空 → 无 ``fund_expert``。
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
    has_us_data = bool(us_data_tools)
    has_us_filing = bool(us_filing_tools)
    has_us_news = bool(us_news_tools)
    has_us_sentiment = bool(us_sentiment_tools)
    has_report = bool(report_tools)
    has_coder = bool(coder_tools)
    has_knowledge = bool(knowledge_tools)
    has_news = bool(news_tools)
    has_sentiment = bool(sentiment_tools)
    has_fund = bool(fund_tools)

    if not (
        has_data
        or has_us_data
        or has_us_filing
        or has_us_news
        or has_us_sentiment
        or has_report
        or has_coder
        or has_knowledge
        or has_news
        or has_sentiment
        or has_fund
    ):
        raise ValueError(
            "build_research_supervisor 至少需要一个专家的工具列表非空，但所有工具组全部为空。"
        )

    agents: list = []
    roster: list[str] = []

    if has_data:
        agents.append(build_data_expert(model_router, data_tools or []))
        roster.append("data_expert")
    if has_us_data:
        agents.append(build_us_data_expert(model_router, us_data_tools or []))
        roster.append("us_data_expert")
    if has_us_filing:
        agents.append(build_us_filing_expert(model_router, us_filing_tools or []))
        roster.append("us_filing_expert")
    if has_us_news:
        agents.append(build_us_news_expert(model_router, us_news_tools or []))
        roster.append("us_news_expert")
    if has_us_sentiment:
        agents.append(build_us_sentiment_expert(model_router, us_sentiment_tools or []))
        roster.append("us_sentiment_expert")
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
    if has_fund:
        agents.append(build_fund_expert(model_router, fund_tools or []))
        roster.append("fund_expert")

    supervisor_model = model_router.get_model(supervisor_tier)
    prompt = _build_supervisor_prompt(
        has_data=has_data,
        has_us_data=has_us_data,
        has_us_filing=has_us_filing,
        has_us_news=has_us_news,
        has_us_sentiment=has_us_sentiment,
        has_report=has_report,
        has_coder=has_coder,
        has_knowledge=has_knowledge,
        has_news=has_news,
        has_sentiment=has_sentiment,
        has_fund=has_fund,
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
    "SUPERVISOR_PROMPT_US_DATA",
    "SUPERVISOR_PROMPT_US_FILING",
    "SUPERVISOR_PROMPT_US_NEWS",
    "SUPERVISOR_PROMPT_US_SENTIMENT",
    "SUPERVISOR_PROMPT_FUND",
    "SUPERVISOR_PROMPT_REPORT",
    "SUPERVISOR_PROMPT_CODER",
    "SUPERVISOR_PROMPT_NEWS",
    "SUPERVISOR_PROMPT_KNOWLEDGE",
    "SUPERVISOR_PROMPT_SENTIMENT",
    "SUPERVISOR_PROMPT_RULES",
]
