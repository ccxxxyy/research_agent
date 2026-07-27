"""ConversationStore 置顶 / 重命名。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_agent.memory.conversation_store import ConversationStore

if TYPE_CHECKING:
    from pathlib import Path


def test_rename_and_pin(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "c.db")
    store.upsert_conversation("t1", "u1", title="旧标题")
    store.upsert_conversation("t2", "u1", title="另一条")

    assert store.rename_conversation("t1", "u1", "新标题")
    assert store.set_pinned("t1", "u1", True)

    rows = store.list_conversations("u1")
    assert rows[0]["thread_id"] == "t1"
    assert rows[0]["title"] == "新标题"
    assert rows[0]["pinned"] is True
    assert rows[1]["thread_id"] == "t2"
    assert rows[1]["pinned"] is False

    assert store.set_pinned("t1", "u1", False)
    rows2 = store.list_conversations("u1")
    assert all(not r["pinned"] for r in rows2)


def test_migrate_adds_pinned_column(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE conversations (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO conversations VALUES ('t9','u9','legacy','2020-01-01','2020-01-02');
        """
    )
    conn.commit()
    conn.close()

    store = ConversationStore(db)
    rows = store.list_conversations("u9")
    assert len(rows) == 1
    assert rows[0]["pinned"] is False
    assert store.set_pinned("t9", "u9", True)
    assert store.list_conversations("u9")[0]["pinned"] is True
