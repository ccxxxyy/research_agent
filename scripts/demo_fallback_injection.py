"""演示通过注入模型故障来展示 LangChain 的 ``with_fallbacks``。

运行::

    uv run python scripts/demo_fallback_injection.py

本脚本 **不是** 单元测试 — 它是一个带叙述的演示。它证明``ModelRouter.get_model(tier)`` 产生的 runnable 在主模型抛异常时能透明降级到备用模型。

运行四个场景：
    1. 主模型正常、备用正常           → 主模型回答
    2. 主模型始终失败、备用正常       → 备用回答（核心保障）
    3. 主模型正常、备用始终失败       → 主模型回答（备用不被触及）
    4. 主模型失败、备用也失败         → 错误浮出给调用方
    5. 流式调用在故障下的表现         → 流式路径相同语义

每个场景使用注入的 *桩* LLM，因此结果是确定性的，可在完全离线环境下运行，无需访问任何真实 API。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.runnables import RunnableWithFallbacks

CALL_LOG: list[str] = []
"""模块级审计跟踪：每次桩模型被调用时追加。

放在模块作用域是因为 BaseChatModel 是 pydantic 模型 — 在子类上声明可变类属性会被误解为 pydantic 字段。
"""


def _reset_log() -> None:
    CALL_LOG.clear()


# ---------------------------------------------------------------------------
# 桩聊天模型（确定性、离线）
# ---------------------------------------------------------------------------

class _StubChatModel(BaseChatModel):
    """用于故障注入演示的最小进程内聊天模型。

    - 若 ``should_fail`` 为 True，每次 ``_generate`` 调用都会抛异常。
    - 否则返回一条 AIMessage，其 content 包含实例 ``label``，使调用方能辨别是 **哪个** 模型回答的。
    """

    label: str
    should_fail: bool = False
    failure_message: str = "simulated upstream failure"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        CALL_LOG.append(self.label)
        if self.should_fail:
            raise RuntimeError(f"[{self.label}] {self.failure_message}")
        text = f"answer-from-{self.label}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # 在产出第一个 chunk 前抛异常以模拟流式故障。
        CALL_LOG.append(f"{self.label}:stream")
        if self.should_fail:
            raise RuntimeError(f"[{self.label}] {self.failure_message} (stream)")
        yield ChatGenerationChunk(message=AIMessageChunk(content=f"answer-from-{self.label}"))

    @property
    def _llm_type(self) -> str:
        return "stub"


# ---------------------------------------------------------------------------
# 场景运行器
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _describe(runnable: Any) -> str:
    return type(runnable).__name__


def _invoke(runnable: Any, question: str) -> str:
    _reset_log()
    try:
        resp = runnable.invoke([HumanMessage(content=question)])
        return f"成功 -> {resp.content}   (路径: {CALL_LOG})"
    except Exception as e:
        return f"失败 -> {type(e).__name__}: {e}   (路径: {CALL_LOG})"


def scenario_1_both_healthy() -> None:
    _banner("场景 1 — 主模型正常、备用正常")
    primary = _StubChatModel(label="primary")
    backup = _StubChatModel(label="backup")
    chain: RunnableWithFallbacks = primary.with_fallbacks([backup])
    print(f"  链类型        : {_describe(chain)}")
    print(f"  调用结果      : {_invoke(chain, 'hello')}")
    print("  预期          : 主模型回答，备用未被触及。")


def scenario_2_primary_fails() -> None:
    _banner("场景 2 — 主模型始终失败、备用正常   [核心保障]")
    primary = _StubChatModel(label="primary", should_fail=True)
    backup = _StubChatModel(label="backup")
    chain = primary.with_fallbacks([backup])
    print(f"  链类型        : {_describe(chain)}")
    print(f"  调用结果      : {_invoke(chain, 'hello')}")
    print("  预期          : 尝试主模型，抛异常，备用接管，")
    print("                  调用方透明地收到 'answer-from-backup'。")


def scenario_3_backup_never_needed() -> None:
    _banner("场景 3 — 主模型正常、备用会失败")
    primary = _StubChatModel(label="primary")
    backup = _StubChatModel(label="backup", should_fail=True)
    chain = primary.with_fallbacks([backup])
    print(f"  调用结果      : {_invoke(chain, 'hello')}")
    print("  预期          : 主模型成功；备用未被使用，")
    print("                  其潜在故障在被需要之前一直隐藏。")


def scenario_4_both_fail() -> None:
    _banner("场景 4 — 主模型失败、备用也失败   [错误浮出]")
    primary = _StubChatModel(label="primary", should_fail=True)
    backup = _StubChatModel(label="backup", should_fail=True)
    chain = primary.with_fallbacks([backup])
    print(f"  调用结果      : {_invoke(chain, 'hello')}")
    print("  预期          : 两者均被尝试，最终异常传播给调用方。")


def scenario_5_streaming() -> None:
    _banner("场景 5 — 流式路径：主模型在打开时失败，备用正常流式输出")
    primary = _StubChatModel(label="primary", should_fail=True)
    backup = _StubChatModel(label="backup")
    chain = primary.with_fallbacks([backup])

    _reset_log()
    try:
        chunks: list[str] = []
        for chunk in chain.stream([HumanMessage(content="hello")]):
            chunks.append(str(chunk.content))
        print(f"  流式内容      : {''.join(chunks)}")
        print(f"  路径          : {CALL_LOG}")
        print("  预期          : 主模型流抛异常，备用流被打开，")
        print("                  调用方仍收到连贯的流。")
    except Exception as e:
        print(f"  流式失败      : {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 附加 — 证明真实 ModelRouter 使用同一机制
# ---------------------------------------------------------------------------

def scenario_6_real_router_shape() -> None:
    _banner("场景 6 — 真实 ModelRouter.get_model(HEAVY) 使用同一链路")
    from research_agent.config import get_settings
    from research_agent.llm.provider import ModelRouter
    from research_agent.llm.tier import ModelTier

    settings = get_settings()
    router = ModelRouter(settings.llm)

    heavy = router.get_model(ModelTier.HEAVY)
    medium = router.get_model(ModelTier.MEDIUM)
    light = router.get_model(ModelTier.LIGHT)

    print(f"  HEAVY  -> {_describe(heavy)}   (预期 RunnableWithFallbacks)")
    print(f"  MEDIUM -> {_describe(medium)}  (预期 RunnableWithFallbacks)")
    print(f"  LIGHT  -> {_describe(light)}   (预期普通 ChatOpenAI — 无进一步降级)")

    print(
        "\n  观察：定义了 FALLBACK_CHAIN 条目的层级会被 .with_fallbacks(...) "
        "\n  包装，因此生产环境中 DashScope 上 deepseek-v3.2 的宕机会"
        "\n  透明降级到 qwen-turbo，无需修改调用方代码。"
    )


# ---------------------------------------------------------------------------
# 要点总结
# ---------------------------------------------------------------------------

def takeaway() -> None:
    _banner("要点总结")
    print(
        """
  * with_fallbacks([backup]) 返回一个与原始 LLM API 兼容的Runnable。调用方代码无需改动。
  * 顺序很重要：列表从左到右尝试；第一个成功的胜出。
  * 主模型的异常在链内部被捕获而非重新抛出，除非所有备用也失败。
  * 它与 ChatOpenAI(max_retries=2) 层叠良好：重试先吸收临时的5xx / 网络抖动；降级吸收持续性宕机或配额耗尽。Checkpointer 再吸收整个进程崩溃。三者合一：三层容错。
  * 层级映射中的设计选择：降级以可用性换质量（HEAVY -> MEDIUM -> LIGHT），而非在同一层级重试。这比"同层重试"（max_retries 已处理的）更符合真实宕机模式。""".rstrip()
    )


def main() -> None:
    scenario_1_both_healthy()
    scenario_2_primary_fails()
    scenario_3_backup_never_needed()
    scenario_4_both_fail()
    scenario_5_streaming()
    scenario_6_real_router_shape()
    takeaway()


if __name__ == "__main__":
    main()
