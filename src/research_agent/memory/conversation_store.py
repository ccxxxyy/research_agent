"""基于 SQLite 的会话历史持久化存储。

按用户隔离存储会话元数据（标题、时间戳）和逐条消息（角色、内容），
使前端能够列出历史会话并在页面刷新后恢复完整的对话记录。
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path("./data/conversations.db").resolve()


class ConversationStore:
    """线程安全的同步 SQLite 会话存储。

    在应用生命周期中仅实例化一次，通过 ``asyncio.to_thread``在异步请求处理器之间共享使用。
    """

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
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id  TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                pinned     INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_conv_user
                ON conversations(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  TEXT NOT NULL REFERENCES conversations(thread_id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                metadata   TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_thread
                ON messages(thread_id, id);
        """)
        # 旧库升级：补 pinned 列
        cols = {row[1] for row in c.execute("PRAGMA table_info(conversations)").fetchall()}
        if "pinned" not in cols:
            c.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        c.commit()

    # ---- 会话操作 ----

    def upsert_conversation(
        self,
        thread_id: str,
        user_id: str,
        title: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        c = self._conn()
        c.execute(
            """INSERT INTO conversations (thread_id, user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(thread_id) DO UPDATE SET
                   title = CASE WHEN excluded.title != '' THEN excluded.title ELSE conversations.title END,
                   updated_at = excluded.updated_at""",
            (thread_id, user_id, title, now, now),
        )
        c.commit()

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = (
            self._conn()
            .execute(
                """SELECT thread_id, title, created_at, updated_at, pinned,
                      (SELECT COUNT(*) FROM messages m WHERE m.thread_id = c.thread_id) AS msg_count
               FROM conversations c
               WHERE user_id = ?
               ORDER BY pinned DESC, updated_at DESC
               LIMIT ?""",
                (user_id, limit),
            )
            .fetchall()
        )
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["pinned"] = bool(d.get("pinned"))
            out.append(d)
        return out

    def rename_conversation(self, thread_id: str, user_id: str, title: str) -> bool:
        title = (title or "").strip()
        if not title:
            return False
        if len(title) > 80:
            title = title[:80]
        c = self._conn()
        cur = c.execute(
            """UPDATE conversations SET title = ?, updated_at = ?
               WHERE thread_id = ? AND user_id = ?""",
            (title, datetime.now(UTC).isoformat(), thread_id, user_id),
        )
        c.commit()
        return cur.rowcount > 0

    def set_pinned(self, thread_id: str, user_id: str, pinned: bool) -> bool:
        c = self._conn()
        cur = c.execute(
            """UPDATE conversations SET pinned = ?, updated_at = ?
               WHERE thread_id = ? AND user_id = ?""",
            (1 if pinned else 0, datetime.now(UTC).isoformat(), thread_id, user_id),
        )
        c.commit()
        return cur.rowcount > 0

    def delete_conversation(self, thread_id: str, user_id: str) -> bool:
        c = self._conn()
        cur = c.execute(
            "DELETE FROM conversations WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id),
        )
        c.commit()
        return cur.rowcount > 0

    # ---- 消息操作 ----

    def add_message(
        self,
        thread_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
        title_hint: str = "",
    ) -> None:
        """追加一条消息，并确保父会话记录已存在（不存在则自动创建）。"""
        self.upsert_conversation(thread_id, user_id, title=title_hint)
        now = datetime.now(UTC).isoformat()
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        c = self._conn()
        c.execute(
            "INSERT INTO messages (thread_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, meta_json, now),
        )
        # 更新会话的 updated_at 时间戳
        c.execute(
            "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
            (now, thread_id),
        )
        c.commit()

    def get_messages(
        self, thread_id: str, user_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        if user_id is not None:
            owner = (
                self._conn()
                .execute(
                    "SELECT 1 FROM conversations WHERE thread_id = ? AND user_id = ?",
                    (thread_id, user_id),
                )
                .fetchone()
            )
            if owner is None:
                return []
        rows = (
            self._conn()
            .execute(
                "SELECT role, content, metadata, created_at FROM messages WHERE thread_id = ? ORDER BY id LIMIT ?",
                (thread_id, limit),
            )
            .fetchall()
        )
        result = []
        for r in rows:
            d = dict(r)
            if d.get("metadata"):
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None
