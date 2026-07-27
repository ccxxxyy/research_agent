"""看板自选 API — 按 user_id + market 持久化，支持宽松搜码与批量行情。"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from research_agent.api.dependencies import MemoryDep

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])


def _store(request: Request):
    store = getattr(request.app.state, "watchlist_store", None)
    if store is None:
        raise HTTPException(503, "Watchlist store not initialised")
    return store


class SearchBody(BaseModel):
    market: str = Field(..., description="CN_A or US")
    q: str = Field(..., min_length=1, max_length=64)
    limit: int = Field(default=8, ge=1, le=8)


class AddBody(BaseModel):
    user_id: str = Field(default="anonymous", max_length=64)
    market: str
    symbol: str = Field(..., min_length=1, max_length=32)
    asset_class: str = "unknown"
    display_name: str = ""
    exchange: str = ""


class QuotesBody(BaseModel):
    market: str
    symbols: list[str] = Field(default_factory=list, max_length=40)


async def _sync_memory(memory: MemoryDep, store: Any, user_id: str) -> None:
    if not user_id or user_id == "anonymous":
        return
    try:
        from research_agent.memory.watchlist_store import sync_watchlist_to_memory

        await sync_watchlist_to_memory(memory, store, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("watchlist memory sync failed for {}: {}", user_id, exc)


@router.get("")
async def list_watchlist(
    request: Request,
    user_id: str = Query(default="anonymous"),
    market: str = Query(..., description="CN_A or US"),
) -> dict[str, Any]:
    store = _store(request)
    mkt = market.strip().upper()
    if mkt not in ("CN_A", "US"):
        raise HTTPException(400, "market must be CN_A or US")
    items = await asyncio.to_thread(store.list_items, user_id, mkt)
    return {"user_id": user_id, "market": mkt, "items": items, "count": len(items)}


@router.post("/search")
async def search_watchlist(body: SearchBody) -> dict[str, Any]:
    from research_agent.market.watchlist_resolve import search_watchlist as _search

    mkt = body.market.strip().upper()
    if mkt not in ("CN_A", "US"):
        raise HTTPException(400, "market must be CN_A or US")
    results = await asyncio.to_thread(_search, mkt, body.q, limit=body.limit)
    return {"market": mkt, "q": body.q, "results": results, "count": len(results)}


@router.post("")
async def add_watchlist(
    body: AddBody,
    request: Request,
    memory: MemoryDep,
) -> dict[str, Any]:
    store = _store(request)
    mkt = body.market.strip().upper()
    if mkt not in ("CN_A", "US"):
        raise HTTPException(400, "market must be CN_A or US")
    try:
        item = await asyncio.to_thread(
            store.add_item,
            body.user_id,
            mkt,
            body.symbol.strip(),
            asset_class=body.asset_class or "unknown",
            display_name=body.display_name or "",
            exchange=body.exchange or "",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await _sync_memory(memory, store, body.user_id)
    return {"ok": True, "item": item}


@router.delete("")
async def delete_watchlist(
    request: Request,
    memory: MemoryDep,
    user_id: str = Query(default="anonymous", max_length=64),
    market: str = Query(..., description="CN_A or US"),
    symbol: str = Query(..., min_length=1, max_length=32),
) -> dict[str, Any]:
    """删除一条自选。使用 query 参数（避免部分客户端/中间件丢弃 DELETE body）。"""
    store = _store(request)
    mkt = (market or "").strip().upper()
    sym = (symbol or "").strip()
    if mkt not in ("CN_A", "US"):
        raise HTTPException(400, "market must be CN_A or US")
    if not sym:
        raise HTTPException(400, "empty symbol")
    try:
        ok = await asyncio.to_thread(store.remove_item, user_id, mkt, sym)
    except Exception as exc:  # noqa: BLE001
        logger.exception("watchlist remove failed: {}", exc)
        raise HTTPException(500, f"remove failed: {exc}") from exc
    if not ok:
        raise HTTPException(404, "item not found")
    await _sync_memory(memory, store, user_id)
    return {"ok": True, "symbol": sym, "market": mkt}


@router.post("/quotes")
async def watchlist_quotes(body: QuotesBody) -> dict[str, Any]:
    from research_agent.market.watchlist_resolve import fetch_watchlist_quotes

    mkt = body.market.strip().upper()
    if mkt not in ("CN_A", "US"):
        raise HTTPException(400, "market must be CN_A or US")
    quotes = await asyncio.to_thread(fetch_watchlist_quotes, mkt, body.symbols[:40])
    return {"market": mkt, "quotes": quotes, "count": len(quotes)}
