"""情感分析 REST 端点。

将 ``news_sentiment_server`` 的能力以直接 HTTP 端点形式暴露，使前端可以获取结构化情感报告和文本级别评分，而无需通过 LangGraph 主管路由。

端点
----
- ``GET  /api/sentiment/report/{symbol}`` —— 某只股票的完整情感报告
  （多源：东方财富新闻 + 雪球热度 + 热搜关键词 + 高频讨论词 + 话题聚类）。
- ``POST /api/sentiment/analyze`` —— 对任意文本列表进行情感评分。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from research_agent.mcp_servers.news_sentiment_server import (
    analyze_text_sentiment as _analyze,
    get_stock_sentiment_report as _report,
)

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


class TextAnalysisRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


@router.get("/report/{symbol}")
async def stock_sentiment_report(
    symbol: str,
    limit: int = Query(default=30, ge=1, le=50),
) -> dict[str, Any]:
    """给定 A 股股票的多源完整情感报告。

    返回新闻条目及每条的情感分数、汇总统计、高频讨论词、话题聚类、雪球热度排名以及东方财富热搜关键词。
    """
    return await _report(symbol=symbol, limit=limit)


@router.post("/analyze")
async def analyze_texts(body: TextAnalysisRequest) -> dict[str, Any]:
    """对任意文本列表进行金融情感评分。

    返回每条文本的分数 + 汇总 + 高频词。
    """
    return await _analyze(texts=body.texts)
