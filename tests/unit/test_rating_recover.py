"""从工具消息回收研报评级。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from research_agent.graph.confidence_gate import apply_confidence_gate_to_messages
from research_agent.text.rating_recover import (
    extract_ratings_from_messages,
    recover_ratings_in_text,
)


def test_recover_table_cell_weiqude() -> None:
    summary = {
        "used": True,
        "ratings_sample": ["买入", "买入"],
        "institutions_sample": [
            {"institution": "东吴证券", "rating": "买入"},
            {"institution": "国金证券", "rating": "买入"},
        ],
    }
    text = "研报评级\t未取得\n其余观察照常。"
    out = recover_ratings_in_text(text, summary)
    assert "未取得" not in out
    assert "东吴证券买入" in out


def test_recover_from_specialist_prose_digest() -> None:
    """last_message 模式下只有专家正文：从「研报评级：东吴…」回收。"""
    msgs = [
        HumanMessage(content="分析宏发"),
        AIMessage(
            content=(
                "llm_digest 要点：研报评级：东吴证券买入、国金证券买入；新闻情绪均分0.2。\n"
                "其余财务略。"
            )
        ),
        AIMessage(content="研报评级\t未取得\n综合看中期可跟踪。"),
    ]
    hit = extract_ratings_from_messages(msgs)
    assert hit is not None
    out = recover_ratings_in_text(str(msgs[-1].content), hit)
    assert "未取得" not in out
    assert "东吴证券买入" in out


def test_extract_from_tool_json() -> None:
    payload = {
        "llm_digest": "研报评级：东吴证券买入",
        "analyst_summary": {
            "used": True,
            "ratings_sample": ["买入"],
            "institutions_sample": [{"institution": "东吴证券", "rating": "买入"}],
        },
        "ratings_sample": ["买入"],
    }
    msgs = [
        HumanMessage(content="分析宏发"),
        ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="t1",
            name="sentiment_get_stock_sentiment_report",
        ),
        AIMessage(content="研报评级\t未取得\n结论：中期可跟踪。"),
    ]
    hit = extract_ratings_from_messages(msgs)
    assert hit is not None
    assert hit["used"] is True

    gated = apply_confidence_gate_to_messages(msgs)
    last = gated[-1]
    assert isinstance(last, AIMessage)
    assert "未取得" not in str(last.content)
    assert "东吴证券买入" in str(last.content) or "买入" in str(last.content)
