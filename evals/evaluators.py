"""LangSmith evaluators for the research supervisor.

Three evaluators score each experiment run:

1. **routing_accuracy** — deterministic Jaccard similarity between
   expected and actual specialist sets.
2. **reply_quality** — LLM-as-judge (LIGHT tier) scoring relevance,
   completeness, and factuality on a 1-5 scale.
3. **memory_persistence** — deterministic check that long-term memory
   was written (or correctly skipped for anonymous users).
"""

from __future__ import annotations

import json
import re
from typing import Any

from langsmith.schemas import Example, Run


# ---------------------------------------------------------------------------
# 1. Routing accuracy (deterministic)
# ---------------------------------------------------------------------------


def routing_accuracy(run: Run, example: Example) -> dict:
    """Jaccard similarity between expected and actual specialists."""
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
# 2. Reply quality (LLM-as-judge)
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
    """Factory: returns an evaluator that uses *llm_caller* for judging.

    ``llm_caller`` must be an async callable ``(prompt: str) -> str``
    that returns the raw LLM text response.
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
    """Convenience: build an evaluator backed by an OpenAI-compatible model.

    Falls back to env vars ``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` when
    arguments are empty.
    """
    from openai import AsyncOpenAI

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if api_base:
        client_kwargs["base_url"] = api_base

    client = AsyncOpenAI(**client_kwargs)
    chosen_model = model or "gpt-4o-mini"

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
# 3. Memory persistence (deterministic)
# ---------------------------------------------------------------------------


def memory_persistence(run: Run, example: Example) -> dict:
    """Check that long-term memory was written when it should have been."""
    user_id = example.inputs.get("user_id", "anonymous")
    outputs = run.outputs or {}
    reply = outputs.get("reply", "")
    memory_saved = outputs.get("memory_saved", False)

    if user_id == "anonymous":
        # Anonymous users should NOT trigger a save
        score = 0.0 if memory_saved else 1.0
        comment = "anonymous: correctly skipped" if score == 1.0 else "anonymous: unexpected save"
    elif not reply.strip():
        score = 1.0 if not memory_saved else 0.0
        comment = "empty reply: save correctly skipped" if score == 1.0 else "empty reply: unexpected save"
    else:
        score = 1.0 if memory_saved else 0.0
        comment = "saved correctly" if score == 1.0 else "MISSING: should have saved"

    return {"key": "memory_persistence", "score": score, "comment": comment}
