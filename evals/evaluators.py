"""研究 supervisor 的评估器集合。

评估器为每次实验运行评分：

1. routing_accuracy — 预期与实际专家集合之间的确定性 Jaccard 相似度。
2. reply_quality — LLM 作为评判（LIGHT 层），对相关性、完整性和事实性进行 1-5 分评分。
3. memory_persistence — 确定性检查：长期记忆是否已写入（或对匿名用户正确跳过）。
4. keyword_coverage — 确定性检查：回复是否包含预期关键词。
5. tool_selection_precision — 确定性检查：是否路由了不必要的专家（惩罚过度路由）。
6. market_routing_accuracy — 预期市场 vs 实际 ``MarketResolution.market``。
7. market_isolation — 美股问句不得命中 A 股专家；A 股问句不得命中美股专家。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


# ---------------------------------------------------------------------------
# 1. 路由准确率（确定性）
# ---------------------------------------------------------------------------


def routing_accuracy(run: Run, example: Example) -> dict:
    """预期与实际专家集合之间的 Jaccard 相似度。"""
    expected = set(example.inputs.get("expected_specialists") or [])
    outputs = run.outputs or {}
    actual = set(outputs.get("specialists_reached") or [])

    if not expected and not actual:
        score = 1.0
    elif not expected or not actual:
        score = 0.0
    else:
        score = len(expected & actual) / len(expected | actual)

    return {
        "key": "routing_accuracy",
        "score": score,
        "comment": f"expected={sorted(expected)}, actual={sorted(actual)}",
    }


# ---------------------------------------------------------------------------
# 2. 回复质量（LLM 作为评判）
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an evaluation judge. Score the assistant's reply on THREE dimensions,
each 1-5 (1=terrible, 5=excellent):

1. **Relevance**: Does the reply address the user's question?
2. **Completeness**: Are all sub-questions answered?
3. **Factuality**: Are claims supported (no hallucinated numbers)?

User query: {query}
Expected keywords (should appear): {keywords}
Assistant reply:
---
{reply}
---

Respond ONLY with valid JSON (no markdown fences):
{{"relevance": <int>, "completeness": <int>, "factuality": <int>, "reasoning": "<one sentence>"}}
"""


def _build_reply_quality_evaluator(llm_caller):
    """工厂方法：返回一个使用 llm_caller 进行评判的评估器。

    ``llm_caller`` 必须是一个异步可调用对象 ``(prompt: str) -> str``，返回原始 LLM 文本响应。
    """

    async def reply_quality(run: Run, example: Example) -> dict:
        outputs = run.outputs or {}
        reply = outputs.get("reply", "")
        query = example.inputs.get("query", "")
        keywords = example.inputs.get("expected_reply_keywords") or []

        if not reply.strip():
            return {"key": "reply_quality", "score": 0.0, "comment": "empty reply"}

        prompt = _JUDGE_PROMPT.format(
            query=query,
            keywords=", ".join(keywords) if keywords else "(none)",
            reply=reply[:3000],
        )

        try:
            raw = await llm_caller(prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return {
                    "key": "reply_quality",
                    "score": 0.5,
                    "comment": f"judge returned unparseable response: {raw[:200]}",
                }
            parsed = json.loads(match.group())
            scores = [
                parsed.get("relevance", 3),
                parsed.get("completeness", 3),
                parsed.get("factuality", 3),
            ]
            avg = sum(scores) / len(scores)
            normalized = (avg - 1) / 4  # 1-5 → 0-1
            return {
                "key": "reply_quality",
                "score": round(normalized, 3),
                "comment": parsed.get("reasoning", ""),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "key": "reply_quality",
                "score": 0.5,
                "comment": f"judge error: {exc}",
            }

    return reply_quality


def create_reply_quality_evaluator(*, model: str = "", api_key: str = "", api_base: str = ""):
    """便捷方法：构建由 OpenAI 兼容模型支持的评估器。

    当参数为空时，回退到环境变量 ``OPENAI_API_KEY`` / ``OPENAI_API_BASE``。
    """
    from openai import AsyncOpenAI

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if api_base:
        client_kwargs["base_url"] = api_base

    client = AsyncOpenAI(**client_kwargs)
    chosen_model = model or "qwen-plus"

    async def _call(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        return resp.choices[0].message.content or ""

    return _build_reply_quality_evaluator(_call)


# ---------------------------------------------------------------------------
# 3. 记忆持久化（确定性）
# ---------------------------------------------------------------------------


def memory_persistence(run: Run, example: Example) -> dict:
    """检查长期记忆是否在应写入时已被写入。"""
    user_id = example.inputs.get("user_id", "anonymous")
    outputs = run.outputs or {}
    reply = outputs.get("reply", "")
    memory_saved = outputs.get("memory_saved", False)

    if user_id == "anonymous":
        score = 0.0 if memory_saved else 1.0
        comment = "匿名用户: 正确跳过" if score == 1.0 else "匿名用户: 意外保存"
    elif not reply.strip():
        score = 1.0 if not memory_saved else 0.0
        comment = "空回复: 正确跳过保存" if score == 1.0 else "空回复: 意外保存"
    else:
        score = 1.0 if memory_saved else 0.0
        comment = "正确保存" if score == 1.0 else "缺失: 应已保存"

    return {"key": "memory_persistence", "score": score, "comment": comment}


# ---------------------------------------------------------------------------
# 4. 关键词覆盖率（确定性）
# ---------------------------------------------------------------------------


def keyword_coverage(run: Run, example: Example) -> dict:
    """回复中预期关键词的命中率（大小写不敏感）。

    对于没有预期关键词的样本（如纯寒暄），返回 1.0。
    """
    outputs = run.outputs or {}
    reply = (outputs.get("reply") or "").lower()
    keywords: list[str] = example.inputs.get("expected_reply_keywords") or []

    if not keywords:
        return {"key": "keyword_coverage", "score": 1.0, "comment": "无预期关键词"}

    hits = [kw for kw in keywords if kw.lower() in reply]
    score = len(hits) / len(keywords)

    return {
        "key": "keyword_coverage",
        "score": round(score, 3),
        "comment": f"命中 {len(hits)}/{len(keywords)}: {hits}",
    }


# ---------------------------------------------------------------------------
# 5. 工具选择精确度（确定性）
# ---------------------------------------------------------------------------


def tool_selection_precision(run: Run, example: Example) -> dict:
    """惩罚路由了不必要专家的情况。

    precision = |expected ∩ actual| / |actual|（actual 为空时返回 1.0）。
    与 routing_accuracy（Jaccard）互补：Jaccard 惩罚遗漏，precision 惩罚过度路由。
    """
    expected = set(example.inputs.get("expected_specialists") or [])
    outputs = run.outputs or {}
    actual = set(outputs.get("specialists_reached") or [])

    if not actual:
        score = 1.0 if not expected else 0.0
        comment = "无路由" if not expected else "应路由但未路由"
    else:
        score = len(expected & actual) / len(actual)
        extra = sorted(actual - expected)
        comment = f"precision={score:.2f}" + (f", 多余: {extra}" if extra else "")

    return {
        "key": "tool_selection_precision",
        "score": round(score, 3),
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# 6. 市场判定准确率（确定性）
# ---------------------------------------------------------------------------


def _normalize_market_label(raw: str) -> str:
    text = raw.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "US": "US",
        "USA": "US",
        "US_STOCK": "US",
        "CN": "CN_A",
        "CN_A": "CN_A",
        "A": "CN_A",
        "ASHARE": "CN_A",
        "A_SHARE": "CN_A",
        "MIXED": "MIXED",
        "UNKNOWN": "UNKNOWN",
    }
    return aliases.get(text, text)


def market_routing_accuracy(run: Run, example: Example) -> dict:
    """预期 ``expected_market`` 与实际 ``outputs.market`` 是否一致。

    未标注 ``expected_market`` 的样本记 1.0（不影响历史 A 股集均值）。
    """
    expected_raw = (example.inputs.get("expected_market") or "").strip()
    if not expected_raw:
        return {
            "key": "market_routing_accuracy",
            "score": 1.0,
            "comment": "无 expected_market 标注",
        }

    expected = _normalize_market_label(expected_raw)
    outputs = run.outputs or {}
    actual = _normalize_market_label(str(outputs.get("market") or ""))
    score = 1.0 if actual == expected else 0.0
    return {
        "key": "market_routing_accuracy",
        "score": score,
        "comment": f"expected={expected}, actual={actual or '(missing)'}",
    }


# ---------------------------------------------------------------------------
# 7. 跨市场隔离（确定性）
# ---------------------------------------------------------------------------

_CN_ONLY_SPECIALISTS = frozenset(
    {
        "data_expert",
        "news_expert",
        "report_expert",
        "fund_expert",
        "sentiment_expert",
    }
)
_US_ONLY_SPECIALISTS = frozenset(
    {
        "us_data_expert",
        "us_filing_expert",
        "us_news_expert",
        "us_sentiment_expert",
    }
)


def market_isolation(run: Run, example: Example) -> dict:
    """惩罚跨市场误路由（ADR-0006 平行隔离）。

    * ``expected_market=US`` → 不得出现 A 股专用专家
    * ``expected_market=CN_A`` → 不得出现美股专用专家
    * 未标注 / MIXED / UNKNOWN → 记 1.0（不做隔离判定）
    """
    expected = (example.inputs.get("expected_market") or "").strip().upper().replace("-", "_")
    outputs = run.outputs or {}
    actual = set(outputs.get("specialists_reached") or [])

    if expected in {"US", "USA"}:
        bad = sorted(actual & _CN_ONLY_SPECIALISTS)
        score = 0.0 if bad else 1.0
        comment = "隔离通过" if not bad else f"美股问句误路由到 A 股专家: {bad}"
    elif expected in {"CN_A", "CN", "A"}:
        bad = sorted(actual & _US_ONLY_SPECIALISTS)
        score = 0.0 if bad else 1.0
        comment = "隔离通过" if not bad else f"A 股问句误路由到美股专家: {bad}"
    else:
        score = 1.0
        comment = "无单市场隔离约束"

    return {
        "key": "market_isolation",
        "score": score,
        "comment": comment,
    }
