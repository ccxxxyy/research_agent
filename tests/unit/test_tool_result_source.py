"""工具结果真实 source 提取（供 SSE tool_done / 前端标签）。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from research_agent.api.routes.supervisor import _emit_specialist_internal
from research_agent.market.tool_result_source import (
    coerce_tool_payload,
    extract_tool_result_source,
)


def test_extract_from_dict_payload() -> None:
    src, url = extract_tool_result_source(
        {
            "price": 1.0,
            "source": "yahoo_chart",
            "source_url": "https://finance.yahoo.com/quote/AAPL",
        }
    )
    assert src == "yahoo_chart"
    assert url == "https://finance.yahoo.com/quote/AAPL"


def test_extract_from_json_string() -> None:
    src, url = extract_tool_result_source(
        json.dumps(
            {"source": "eastmoney_us", "source_url": "https://quote.eastmoney.com/us/AAPL.html"}
        )
    )
    assert src == "eastmoney_us"
    assert "eastmoney" in (url or "")


def test_extract_mixed_index_sources() -> None:
    src, url = extract_tool_result_source(
        {
            "source": "eastmoney_us+yahoo_chart",
            "source_url": "https://quote.eastmoney.com/center/gridlist.html#us_stocks",
        }
    )
    assert src == "eastmoney_us+yahoo_chart"
    assert url is not None


def test_extract_from_mcp_text_blocks() -> None:
    src, url = extract_tool_result_source(
        [
            {
                "type": "text",
                "text": json.dumps(
                    {"source": "yahoo_search", "source_url": "https://finance.yahoo.com/"}
                ),
            }
        ]
    )
    assert src == "yahoo_search"
    assert url == "https://finance.yahoo.com/"


def test_extract_ignores_error_without_source() -> None:
    src, url = extract_tool_result_source({"error": "timeout", "context": "us_get_quote"})
    assert src is None
    assert url is None


def test_coerce_markdown_fence() -> None:
    payload = coerce_tool_payload('```json\n{"source": "yfinance"}\n```')
    assert payload == {"source": "yfinance"}


def test_emit_tool_done_includes_runtime_source() -> None:
    import asyncio

    frames: asyncio.Queue[str | None] = asyncio.Queue()
    _emit_specialist_internal(
        "us_data_expert",
        "us_data_expert",
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "symbol": "AAPL",
                            "source": "yahoo_chart",
                            "source_url": "https://finance.yahoo.com/quote/AAPL",
                        }
                    ),
                    name="us_get_quote",
                    tool_call_id="1",
                )
            ]
        },
        frames,
    )
    frame = frames.get_nowait()
    assert frame is not None
    assert "tool_done" in frame
    assert "yahoo_chart" in frame
    assert "finance.yahoo.com/quote/AAPL" in frame


def test_emit_tool_call_does_not_claim_source() -> None:
    import asyncio

    frames: asyncio.Queue[str | None] = asyncio.Queue()
    _emit_specialist_internal(
        "us_data_expert",
        "us_data_expert",
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "us_get_quote",
                            "args": {"symbol": "AAPL"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        frames,
    )
    frame = frames.get_nowait()
    assert frame is not None
    assert "tool_call" in frame
    assert "yahoo_chart" not in frame
    assert "eastmoney" not in frame
