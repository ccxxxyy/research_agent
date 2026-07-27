"""watchlist_store 单测。"""

from __future__ import annotations

import pytest

from research_agent.memory.watchlist_store import _MAX_PER_MARKET, WatchlistStore


@pytest.fixture()
def store(tmp_path):
    return WatchlistStore(db_path=tmp_path / "wl.db")


def test_add_list_remove(store: WatchlistStore):
    item = store.add_item(
        "u1",
        "CN_A",
        "600519",
        asset_class="equity",
        display_name="贵州茅台",
        exchange="SH",
    )
    assert item["symbol"] == "600519"
    items = store.list_items("u1", "CN_A")
    assert len(items) == 1
    assert items[0]["display_name"] == "贵州茅台"
    assert store.remove_item("u1", "CN_A", "600519") is True
    assert store.list_items("u1", "CN_A") == []


def test_markets_isolated(store: WatchlistStore):
    store.add_item("u1", "CN_A", "600519", display_name="茅台")
    store.add_item("u1", "US", "AAPL", display_name="Apple")
    assert len(store.list_items("u1", "CN_A")) == 1
    assert len(store.list_items("u1", "US")) == 1
    assert store.list_items("u2", "CN_A") == []


def test_limit(store: WatchlistStore):
    for i in range(_MAX_PER_MARKET):
        store.add_item("u1", "US", f"T{i:03d}", display_name=f"T{i}")
    with pytest.raises(ValueError, match="limit"):
        store.add_item("u1", "US", "OVERFLOW")
