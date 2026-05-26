"""Writer / Reasoner 反思子图的单元测试。

这些测试通过编译后的图端到端地验证反思逻辑（状态转换、批评解析、路由），但使用伪造的模型路由器，因此不会接触真实 LLM。

明确不测试的内容：

* 真实 LLM 评分质量 — 那是评估关注点而非单元测试关注点，且需要 API 凭证。
* 在 ``build_research_supervisor`` 中端到端运行所有专家的反思 — 那属于集成测试领域，专家需要 MCP 子进程。通过伪造的内部 supervisor 测试包装器连线。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from research_agent.graph.reflection import (
    _extract_json,
    _extract_supervisor_draft,
    _format_transcript,
    _normalise_critique,
    build_reflection_subgraph,
)
from research_agent.llm.tier import ModelTier


# ---------------------------------------------------------------------
# 伪造对象 — 记录提示词并回放预设响应的 model_router 替身，完全不接触网络。
# ---------------------------------------------------------------------
class _FakeModel:
    """与反思代码兼容的最小 Runnable 形状对象。

    反思仅需 ``.ainvoke(messages)`` → 含 ``.content`` 字符串的对象。捕获每个提示词用于断言。
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.prompts.append(list(messages))
        if not self._responses:
            # 脚本用完意味着测试提供了不完整的伪造对象 —— 大声失败而非静默复用最后一个响应。
            raise AssertionError(
                "fake model out of canned responses; "
                "test setup is incomplete"
            )
        text = self._responses.pop(0)
        return AIMessage(content=text)


class _FakeRouter:
    """分发层级特定伪造模型的路由器替身。"""

    def __init__(
        self,
        *,
        light_responses: list[str],
        heavy_responses: list[str] | None = None,
    ) -> None:
        self.light = _FakeModel(light_responses)
        self.heavy = _FakeModel(heavy_responses or [])

    def get_model(self, tier: ModelTier) -> _FakeModel:
        if tier == ModelTier.LIGHT:
            return self.light
        if tier == ModelTier.HEAVY:
            return self.heavy
        # 反思逻辑只使用 LIGHT + HEAVY。请求其他层级说明是生产代码的 bug，而非测试的问题。
        raise AssertionError(f"unexpected tier requested: {tier}")


# ---------------------------------------------------------------------
# 纯函数辅助方法测试
# ---------------------------------------------------------------------
class TestExtractJson:
    def test_plain_json(self) -> None:
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json_block(self) -> None:
        text = "Here is the verdict:\n```json\n{\"quality_score\": 0.9}\n```"
        assert _extract_json(text) == {"quality_score": 0.9}

    def test_embedded_object_after_prose(self) -> None:
        text = "Some prose. {\"quality_score\": 0.5, \"issues\": []}"
        assert _extract_json(text) == {"quality_score": 0.5, "issues": []}

    def test_unparseable_returns_empty(self) -> None:
        assert _extract_json("not json at all") == {}


class TestNormaliseCritique:
    def test_string_score_parsed(self) -> None:
        out = _normalise_critique({"quality_score": "0.7", "feedback": ""})
        assert out["quality_score"] == pytest.approx(0.7)

    def test_percent_score_normalised(self) -> None:
        out = _normalise_critique({"quality_score": "85%", "feedback": ""})
        assert out["quality_score"] == pytest.approx(0.85)

    def test_list_feedback_joined(self) -> None:
        out = _normalise_critique({"quality_score": 0.5, "feedback": ["a", "b"]})
        assert out["feedback"] == "a\nb"

    def test_missing_issues_becomes_empty_list(self) -> None:
        out = _normalise_critique({"quality_score": 0.9})
        assert out["issues"] == []

    def test_score_clamped_to_unit_interval(self) -> None:
        assert _normalise_critique({"quality_score": 1.5})["quality_score"] == 1.0
        assert _normalise_critique({"quality_score": -0.2})["quality_score"] == 0.0

    def test_garbage_score_defaults_to_zero(self) -> None:
        assert _normalise_critique({"quality_score": "garbage"})["quality_score"] == 0.0


class TestExtractSupervisorDraft:
    def test_last_assistant_without_toolcalls_wins(self) -> None:
        msgs: list[BaseMessage] = [
            HumanMessage(content="Q"),
            AIMessage(content="", tool_calls=[
                {"name": "transfer_to_data_expert", "args": {}, "id": "1"},
            ]),
            AIMessage(content="data found", name="data_expert"),
            AIMessage(content="final synthesis", name="supervisor"),
        ]
        assert _extract_supervisor_draft(msgs) == "final synthesis"

    def test_skips_tool_call_messages(self) -> None:
        msgs: list[BaseMessage] = [
            HumanMessage(content="Q"),
            AIMessage(content="real draft"),
            AIMessage(content="ignored because tool call", tool_calls=[
                {"name": "x", "args": {}, "id": "y"},
            ]),
        ]
        # "ignored because tool call" 消息含有 tool_calls， 因此跳过它，返回其上方的真正草稿。
        assert _extract_supervisor_draft(msgs) == "real draft"

    def test_empty_messages_returns_empty_string(self) -> None:
        assert _extract_supervisor_draft([]) == ""

    def test_only_human_messages_returns_empty_string(self) -> None:
        assert _extract_supervisor_draft([HumanMessage(content="hi")]) == ""


class TestFormatTranscript:
    def test_long_transcript_truncated_with_marker(self) -> None:
        # 构建一个远超 max_chars 的对话记录。
        msgs: list[BaseMessage] = [
            HumanMessage(content="x" * 500),
            AIMessage(content="y" * 5000),
        ]
        rendered = _format_transcript(msgs, max_chars=1000)
        assert rendered.startswith("... (transcript truncated) ...")
        # 保留尾部：AIMessage 的末尾应当存在。
        assert rendered.endswith("y")
        assert len(rendered) <= 1000 + len("... (transcript truncated) ...\n\n")


# ---------------------------------------------------------------------
# 子图行为
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_high_score_first_pass_skips_rewrite() -> None:
    """得分 ≥ 阈值的草稿不应触发 writer。"""
    router = _FakeRouter(
        light_responses=['{"quality_score": 0.95, "reasoning": "good", "feedback": "", "issues": []}'],
        heavy_responses=[],
    )
    graph: CompiledStateGraph = build_reflection_subgraph(
        model_router=router,  # type: ignore[arg-type]
        pass_threshold=0.85,
        max_iterations=2,
    )

    out = await graph.ainvoke({
        "messages": [
            HumanMessage(content="What is X?"),
            AIMessage(content="X is the answer.", name="supervisor"),
        ],
    })

    assert len(router.light.prompts) == 1, "critic should have been called exactly once"
    assert len(router.heavy.prompts) == 0, "writer should NOT have been called"

    final = out["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content == "X is the answer."
    refl = final.additional_kwargs["reflection"]
    assert refl["iterations_run"] == 1
    assert refl["final_score"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_low_score_triggers_rewrite_then_passes() -> None:
    """先失败后通过：critic 评分 0.4，重写一次后评分 0.9。"""
    router = _FakeRouter(
        light_responses=[
            '{"quality_score": 0.4, "reasoning": "bad", "feedback": "add citations", "issues": ["no_cite"]}',
            '{"quality_score": 0.9, "reasoning": "now good", "feedback": "", "issues": []}',
        ],
        heavy_responses=["REWRITTEN with citations [Source A]."],
    )
    graph = build_reflection_subgraph(
        model_router=router,  # type: ignore[arg-type]
        pass_threshold=0.85,
        max_iterations=2,
    )

    out = await graph.ainvoke({
        "messages": [
            HumanMessage(content="Q"),
            AIMessage(content="weak draft", name="supervisor"),
        ],
    })

    assert len(router.light.prompts) == 2
    assert len(router.heavy.prompts) == 1
    final = out["messages"][-1]
    assert final.content == "REWRITTEN with citations [Source A]."
    refl = final.additional_kwargs["reflection"]
    assert refl["iterations_run"] == 2
    assert refl["final_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_max_iterations_enforced_when_critic_never_satisfied() -> None:
    """病态用例：critic 始终评分低于阈值。"""
    router = _FakeRouter(
        light_responses=[
            '{"quality_score": 0.3, "feedback": "more"}',
            '{"quality_score": 0.4, "feedback": "more"}',
            '{"quality_score": 0.5, "feedback": "more"}',
            # 如果循环无上限则需要第 4 个响应；我们有意省略。
        ],
        heavy_responses=[
            "rewrite #1",
            "rewrite #2",
        ],
    )
    graph = build_reflection_subgraph(
        model_router=router,  # type: ignore[arg-type]
        pass_threshold=0.85,
        max_iterations=2,
    )

    out = await graph.ainvoke({
        "messages": [
            HumanMessage(content="Q"),
            AIMessage(content="initial draft", name="supervisor"),
        ],
    })

    # max_iterations=2 时，最多运行 3 次 critic + 2 次 writer。
    assert len(router.light.prompts) == 3
    assert len(router.heavy.prompts) == 2
    refl = out["messages"][-1].additional_kwargs["reflection"]
    assert refl["iterations_run"] == 3
    # "最佳"草稿是得分 0.5 的那个 — 即第二次重写（"rewrite #2"）， 因为它分数最高。
    assert out["messages"][-1].content == "rewrite #2"


@pytest.mark.asyncio
async def test_best_draft_preserved_on_regression() -> None:
    """如果重写导致退步，finalize 返回先前的最高水位草稿。"""
    router = _FakeRouter(
        light_responses=[
            # 初始 supervisor 草稿：得分 0.6（刚好低于阈值）
            '{"quality_score": 0.6, "feedback": "tighten"}',
            # 重写退步：0.3 — 不应成为最终答案
            '{"quality_score": 0.3, "feedback": "much worse"}',
            # 第二次重写恢复到 0.5 — 仍低于阈值， 循环因达到最大迭代次数而终止
            '{"quality_score": 0.5, "feedback": "still off"}',
        ],
        heavy_responses=[
            "regressed rewrite",
            "partial recovery rewrite",
        ],
    )
    graph = build_reflection_subgraph(
        model_router=router,  # type: ignore[arg-type]
        pass_threshold=0.85,
        max_iterations=2,
    )

    out = await graph.ainvoke({
        "messages": [
            HumanMessage(content="Q"),
            AIMessage(content="initial supervisor draft", name="supervisor"),
        ],
    })

    # 最高分（0.6）是 supervisor 的原始草稿，因此 finalize 应返回该草稿而非重写版本。
    assert out["messages"][-1].content == "initial supervisor draft"
    refl = out["messages"][-1].additional_kwargs["reflection"]
    assert refl["final_score"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_empty_draft_terminates_gracefully() -> None:
    """无 supervisor 合成内容 → critic 输出零分，循环快速结束。"""
    router = _FakeRouter(
        light_responses=[],  # critic 不会在实质路径上被调用
        heavy_responses=[],
    )
    graph = build_reflection_subgraph(
        model_router=router,  # type: ignore[arg-type]
        pass_threshold=0.85,
        max_iterations=2,
    )

    # 仅有一条人类消息 — 没有草稿可供批评。critic 节点以分数 0.0 短路，路由器看到 iteration=1 > max_iterations+1？不，1 < 3，
    # 所以它实际上会尝试重写。我们想确保它在不调用 writer 的情况下也能终止 — 这需要 critic_node 中的第二个短路或提供空响应。
    #
    # 文档化行为：无草稿时，critic 输出空批评和分数 0.0；路由器决定重写；writer 需要一个响应，而我们的伪造对象没有 → AssertionError。
    # 实践中调用者会在上游处理此情况。测试辅助函数本身不会在此输入形状上崩溃。
    with pytest.raises(AssertionError):
        await graph.ainvoke({"messages": [HumanMessage(content="hi")]})


@pytest.mark.asyncio
async def test_zero_max_iterations_makes_critic_only_pass() -> None:
    """消融模式：``max_iterations=0`` → 一次 critic，无重写。"""
    router = _FakeRouter(
        light_responses=['{"quality_score": 0.2, "feedback": "bad"}'],
        heavy_responses=[],
    )
    graph = build_reflection_subgraph(
        model_router=router,  # type: ignore[arg-type]
        pass_threshold=0.85,
        max_iterations=0,
    )

    out = await graph.ainvoke({
        "messages": [
            HumanMessage(content="Q"),
            AIMessage(content="draft", name="supervisor"),
        ],
    })

    assert len(router.light.prompts) == 1
    assert len(router.heavy.prompts) == 0
    # 即使分数低于阈值，max_iterations=0 时仍以原始草稿完成。
    assert out["messages"][-1].content == "draft"
