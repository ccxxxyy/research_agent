"""MCP（Model Context Protocol）工具服务器。

每个服务器通过 MCP 协议暴露工具，实现协议驱动的工具调用（而非提示驱动的工具调用）。
Agent 通过``langchain_mcp_adapters.client.MultiServerMCPClient`` 发现并调用这些工具。

本包公共 API
============
活跃服务器在首次属性访问时延迟导入。使用方式::

    from research_agent.mcp_servers import code_server  # 或 echo_server

活跃服务器
----------
``echo_server``
    确定性的大写转换 / 长度计算工具。用作 MCP 管道本身的冒烟测试服务器；未接入研究管道。

``code_server``
    沙箱化 Python 执行。接入 minimal 和 research supervisor 的 ``coder_expert`` 专家。
    负责在 akshare DataFrame 上运行 LLM 生成的 pandas / numpy 代码片段，计算衍生指标。

``fin_data_server``
    通过 ``akshare`` 获取真实 A 股行情和基本面数据。五个工具：
    ``get_stock_basic_info``、``get_stock_price_history``、
    ``get_financial_abstract``、``get_financial_indicators``、``search_stock_by_name``。
    行情/报价类工具级联 东方财富 → 雪球 / 新浪 以应对上游宕机。
    接入 research supervisor 的 ``data_expert`` 专家。

``us_data_server``
    通过 ``yfinance`` 获取美股股票 / 指数 / ETF 数据（与 ``fin_data_server`` 平行隔离）。
    工具：``get_market_status``、``search_ticker``、``get_quote``、``get_price_history``、
    ``get_basic_info``、``get_index_quotes``、``get_etf_overview``、``get_etf_holdings``、``get_etf_sector_weights``（运行时前缀 ``us_``）。
    接入 research supervisor 的 ``us_data_expert`` 专家。

``us_filing_server``
    通过 SEC EDGAR 获取美股披露（与 ``pdf_report_server`` 平行隔离）。
    工具：``resolve_cik``、``search_filings``、``download_filing``、
    ``extract_filing_metadata``、``parse_filing_text``（运行时前缀 ``us_filing_``）。
    接入 research supervisor 的 ``us_filing_expert`` 专家。

``us_news_server``
    通过 yfinance（Yahoo）获取美股新闻，并可选返回 EDGAR 8-K 标题线索。
    工具：``get_ticker_news``、``get_market_news``、``get_etf_news``、``get_recent_8k_headlines``（运行时前缀 ``us_news_``）。
    接入 ``us_news_expert``。

``us_sentiment_server``
    美股英文舆情量化（VADER + 金融词表增强，不用 SnowNLP）。
    工具：``analyze_text_sentiment``、``get_ticker_sentiment_report``（前缀 ``us_sentiment_``）。
    接入 ``us_sentiment_expert``。

``pdf_report_server``
    巨潮资讯的公告 / 研报 PDF。四个工具：
    ``search_announcements``、``download_pdf``（基于哈希的磁盘缓存，位于 ``./data/pdf_cache/``）、
    ``parse_pdf_pages``（每次调用限 20 页窗口）、``extract_pdf_metadata``。
    供 ``report_expert`` 专家使用。

``news_server``
    通过 东方财富 / 财联社 / 百度财经 / 雪球 获取 A 股新闻 / 情感数据。五个工具：
    ``get_stock_news``、``get_market_telegraph``、``get_hot_keywords``、``get_economic_news``、
    ``get_xueqiu_discussion_hot_rank``（雪球讨论热度个股榜，通过``stock_hot_tweet_xq``）。
    供 ``news_expert`` 专家使用。

``knowledge_server``
    用户上传的 PDF 知识库，支持混合检索（FAISS + BM25 + 交叉编码器重排序）和纠正式 RAG 质量信号。
    四个工具：``ingest_pdf``、``search``（返回``quality`` ∈ ``{high, medium, low}`` + 每条命中的 ``rerank_score``，以便 Agent 决定是否重写查询并重试）、
    ``list_collections``、``delete_collection``。
    通过 FAISS 持久化到 ``./data/knowledge_db/``。接入 ``knowledge_expert`` 专家。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

ACTIVE_SERVERS: tuple[str, ...] = (
    "echo_server",
    "code_server",
    "fin_data_server",
    "us_data_server",
    "us_filing_server",
    "us_news_server",
    "us_sentiment_server",
    "pdf_report_server",
    "news_server",
    "knowledge_server",
)
"""属于本包公共 API 的子模块名称。"""


def __getattr__(name: str):
    """在首次属性访问时延迟导入活跃子模块。

    为何使用延迟导入而非 ``from research_agent.mcp_servers import code_server``？
    当通过 ``python -m research_agent.mcp_servers.code_server``（MCP stdio 启动路径）生成子进程时，
    急切的顶层导入会导致 Python 两次导入同一模块，一次通过包 ``__init__`` 隐式导入，一次通过 ``runpy``显式导入，
    从而触发 ``RuntimeWarning: 'X' found in sys.modules after import of package``。
    通过 PEP 562 ``__getattr__`` 延迟加载可避免此问题，同时保持 ``from research_agent.mcp_servers import code_server`` 对需要包级别别名的调用者正常工作。

    这是一个"拦截器"。当写 from research_agent.mcp_servers import code_server 时，Python 会来问 __init__.py："你有 code_server 这个东西吗？"
    正常情况下，如果 __init__.py 里没有直接定义 code_server，就会报错。
    但 __getattr__ 拦截了这个"找不到"的动作，说"等一下，让我现在去加载它"，然后动态地导入对应的模块返回。这就是延迟导入的实现方式。
    "为什么不直接在顶部导入"——因为 MCP 服务器启动时会以 python -m research_agent.mcp_servers.code_server 的方式启动子进程。
    如果 __init__.py 在顶部就导入 code_server，Python 会重复导入同一个模块两次，产生警告。延迟导入避免了这个问题。
    """
    if name in ACTIVE_SERVERS:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - 辅助类型检查器和 IDE 自动补全
    from research_agent.mcp_servers import (
        code_server,
        echo_server,
        fin_data_server,
        knowledge_server,
        news_server,
        pdf_report_server,
        us_data_server,
        us_filing_server,
        us_news_server,
        us_sentiment_server,
    )


__all__ = [
    "ACTIVE_SERVERS",
    "code_server",
    "echo_server",
    "fin_data_server",
    "knowledge_server",
    "news_server",
    "pdf_report_server",
    "us_data_server",
    "us_filing_server",
    "us_news_server",
    "us_sentiment_server",
]
