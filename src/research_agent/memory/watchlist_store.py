"""基于 SQLite 的看板自选股/基持久化（按 user_id + market 隔离）。"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

_DEFAULT_DB = Path("./data/watchlist.db").resolve()
_MAX_PER_MARKET = 40

# 写入 LangGraph 长期记忆的命名空间 key（见 MemoryNamespace.WATCHLIST）
WATCHLIST_MEMORY_KEY = "snapshot"

if TYPE_CHECKING:
    from research_agent.memory.manager import MemoryManager


def format_watchlist_context(
    cn_items: list[dict[str, Any]],
    us_items: list[dict[str, Any]],
) -> str:
    """生成注入研究前导的自选摘要；无标的时返回空串。"""

    def _fmt(items: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for it in items[:_MAX_PER_MARKET]:
            sym = str(it.get("symbol") or "").strip()
            if not sym:
                continue
            name = str(it.get("display_name") or sym).strip()
            ac = str(it.get("asset_class") or "").strip()
            if name and name != sym:
                label = f"{name}({sym}"
                label += f", {ac})" if ac else ")"
            else:
                label = f"{sym}({ac})" if ac else sym
            parts.append(label)
        return ", ".join(parts)

    lines: list[str] = []
    cn = _fmt(cn_items)
    us = _fmt(us_items)
    if cn:
        lines.append(f"- CN_A: {cn}")
    if us:
        lines.append(f"- US: {us}")
    if not lines:
        return ""
    return (
        "User dashboard watchlist (when the user refers to 自选/关注/我的股票 "
        "without naming tickers, prefer these symbols):\n" + "\n".join(lines)
    )


def snapshot_watchlist(store: WatchlistStore, user_id: str) -> dict[str, Any]:
    """拉取用户双市场自选，供记忆同步与前导注入。"""
    uid = (user_id or "anonymous").strip() or "anonymous"
    cn = store.list_items(uid, "CN_A")
    us = store.list_items(uid, "US")
    content = format_watchlist_context(cn, us)
    return {
        "user_id": uid,
        "cn": cn,
        "us": us,
        "content": content,
        "count": len(cn) + len(us),
    }


async def sync_watchlist_to_memory(
    memory: MemoryManager,
    store: WatchlistStore,
    user_id: str,
) -> dict[str, Any]:
    """将看板自选写入 Agent 长期记忆（``user_watchlist`` 命名空间）。"""
    from research_agent.memory.manager import MemoryNamespace

    snap = await asyncio.to_thread(snapshot_watchlist, store, user_id)
    uid = snap["user_id"]
    if uid == "anonymous":
        return snap
    content = snap["content"] or "(empty watchlist)"
    await memory.save_memory(
        user_id=uid,
        namespace=MemoryNamespace.WATCHLIST,
        key=WATCHLIST_MEMORY_KEY,
        value={
            "content": content,
            "cn_count": len(snap["cn"]),
            "us_count": len(snap["us"]),
            "symbols_cn": [i.get("symbol") for i in snap["cn"]],
            "symbols_us": [i.get("symbol") for i in snap["us"]],
        },
    )
    return snap


class WatchlistStore:
    """线程安全的同步 SQLite 自选存储。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or _DEFAULT_DB)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist_items (
                user_id      TEXT NOT NULL,
                market       TEXT NOT NULL,
                symbol       TEXT NOT NULL,
                asset_class  TEXT NOT NULL DEFAULT 'unknown',
                display_name TEXT NOT NULL DEFAULT '',
                exchange     TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL,
                PRIMARY KEY (user_id, market, symbol)
            );
            CREATE INDEX IF NOT EXISTS idx_wl_user_market
                ON watchlist_items(user_id, market, created_at DESC);
        """)
        c.commit()

    def list_items(self, user_id: str, market: str) -> list[dict[str, Any]]:
        uid = (user_id or "anonymous").strip() or "anonymous"
        mkt = (market or "").strip().upper()
        rows = (
            self._conn()
            .execute(
                """SELECT user_id, market, symbol, asset_class, display_name, exchange, created_at
                   FROM watchlist_items
                   WHERE user_id = ? AND market = ?
                   ORDER BY created_at ASC""",
                (uid, mkt),
            )
            .fetchall()
        )
        return [dict(r) for r in rows]

    def count(self, user_id: str, market: str) -> int:
        uid = (user_id or "anonymous").strip() or "anonymous"
        mkt = (market or "").strip().upper()
        row = (
            self._conn()
            .execute(
                "SELECT COUNT(*) AS n FROM watchlist_items WHERE user_id = ? AND market = ?",
                (uid, mkt),
            )
            .fetchone()
        )
        return int(row["n"] if row else 0)

    def add_item(
        self,
        user_id: str,
        market: str,
        symbol: str,
        *,
        asset_class: str = "unknown",
        display_name: str = "",
        exchange: str = "",
    ) -> dict[str, Any]:
        uid = (user_id or "anonymous").strip() or "anonymous"
        mkt = (market or "").strip().upper()
        sym = (symbol or "").strip()
        if not sym:
            raise ValueError("empty symbol")
        if mkt not in ("CN_A", "US"):
            raise ValueError("market must be CN_A or US")

        existing = self.list_items(uid, mkt)
        if any(i["symbol"] == sym for i in existing):
            # update metadata
            c = self._conn()
            c.execute(
                """UPDATE watchlist_items
                   SET asset_class = ?, display_name = COALESCE(NULLIF(?, ''), display_name),
                       exchange = COALESCE(NULLIF(?, ''), exchange)
                   WHERE user_id = ? AND market = ? AND symbol = ?""",
                (asset_class or "unknown", display_name, exchange, uid, mkt, sym),
            )
            c.commit()
            return next(i for i in self.list_items(uid, mkt) if i["symbol"] == sym)

        if len(existing) >= _MAX_PER_MARKET:
            raise ValueError(f"watchlist limit {_MAX_PER_MARKET} reached for {mkt}")

        now = datetime.now(UTC).isoformat()
        c = self._conn()
        c.execute(
            """INSERT INTO watchlist_items
               (user_id, market, symbol, asset_class, display_name, exchange, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uid, mkt, sym, asset_class or "unknown", display_name or sym, exchange or "", now),
        )
        c.commit()
        return {
            "user_id": uid,
            "market": mkt,
            "symbol": sym,
            "asset_class": asset_class or "unknown",
            "display_name": display_name or sym,
            "exchange": exchange or "",
            "created_at": now,
        }

    def remove_item(self, user_id: str, market: str, symbol: str) -> bool:
        uid = (user_id or "anonymous").strip() or "anonymous"
        mkt = (market or "").strip().upper()
        sym = (symbol or "").strip()
        c = self._conn()
        cur = c.execute(
            "DELETE FROM watchlist_items WHERE user_id = ? AND market = ? AND symbol = ?",
            (uid, mkt, sym),
        )
        c.commit()
        return cur.rowcount > 0


__all__ = [
    "WATCHLIST_MEMORY_KEY",
    "WatchlistStore",
    "_MAX_PER_MARKET",
    "format_watchlist_context",
    "snapshot_watchlist",
    "sync_watchlist_to_memory",
]
