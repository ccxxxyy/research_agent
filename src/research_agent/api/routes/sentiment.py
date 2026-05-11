"""Sentiment analysis REST endpoints.

Exposes the ``news_sentiment_server`` capabilities as direct HTTP
endpoints so a frontend can fetch structured sentiment reports and
text-level scoring without routing through the LangGraph supervisor.

Endpoints
---------
- ``GET  /api/sentiment/report/{symbol}`` — full sentiment report for
  a stock (multi-source: Eastmoney news + Xueqiu heat + trending keywords
  + high-frequency discussion terms + topic clusters).
- ``POST /api/sentiment/analyze`` — score arbitrary text list.
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
    """Full multi-source sentiment report for a given A-share stock.

    Returns news items with per-item scores, aggregate statistics,
    high-frequency discussion terms, topic clusters, Xueqiu heat
    ranking, and Eastmoney trending keywords.
    """
    return await _report(symbol=symbol, limit=limit)


@router.post("/analyze")
async def analyze_texts(body: TextAnalysisRequest) -> dict[str, Any]:
    """Score a list of arbitrary texts for financial sentiment.

    Returns per-text scores + aggregate + high-frequency terms.
    """
    return await _analyze(texts=body.texts)
