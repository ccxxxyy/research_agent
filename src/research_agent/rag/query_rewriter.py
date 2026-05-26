"""基于 LLM 的查询重写器，用于 corrective RAG 循环。

查询重写由 ``knowledge_expert`` Agent 的系统提示词驱动：
当 ``knowledge_search`` 返回 ``quality == "low"`` 时，Agent 重写查询并重试（最多 3 次）。
本类将同一模式封装为可独立调用的组件，以便进行单元测试、
"同一模式"指：knowledge_expert 的提示词里已经描述了的改写行为。knowledge_expert 的系统提示词告诉它："如果 quality == 'low'，用更具体的关键词改写查询，再调一次 knowledge_search"这个改写行为是 knowledge_expert（一个 LLM agent）自己做的——它看到 quality='low' 后，自己想一个新的搜索词。QueryRewriter 类把同样的逻辑（改写查询）封装成一个独立的、可直接调用的组件。

在 Agent 循环外使用，"Corrective RAG 代码"时可以直接指向。

典型用法::

    rewriter = QueryRewriter(model=model_router.get_model(ModelTier.LIGHT))
    better_query = await rewriter.rewrite(
        original_query="ESG 碳中和",
        context="Previous search returned low-quality results about ESG "
                "but nothing on carbon neutrality commitments.",
    )
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

_SYSTEM_PROMPT = """\
你是纠正式 RAG 流水线中的搜索查询重写器。

你的输入是：
  1. 产生了低质量检索结果的原始用户查询。
  2. （可选）描述出了什么问题的上下文。

你的任务：生成一个更具体、使用更精确术语的改进查询，使其更可能从中英文金融知识库中检索到相关分块。

规则：
  - 仅输出重写后的查询文本，不加任何解释。
  - 保持原始查询的语言（中文查询用中文重写，英文查询用英文重写）。
  - 如果原始查询含糊，添加可能的领域关键词（如 "ROE"、"毛利率"、"年报"、"ESG"、具体公司名称）。
  - 不得编造事实或改变用户的意图。
"""


class QueryRewriter:
    """重写搜索查询以提升检索质量。

    Parameters
    ----------
    model:
        任何 LangChain 兼容的聊天模型（``BaseChatModel``）。通常使用 LIGHT 级别模型，因为重写是一个简单的分类/生成任务。
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    async def rewrite(
        self,
        original_query: str,
        context: str = "",
    ) -> str:
        """返回重写后的查询字符串。

        若 LLM 调用失败或返回空值，回退到 ``original_query`` —纠正循环不应因重写器故障而崩溃。
        """
        user_content = f"Original query: {original_query}"
        if context:
            user_content += f"\nContext: {context}"

        try:
            response = await self._model.ainvoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            rewritten = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            ).strip()
            if not rewritten:
                return original_query
            logger.debug(
                "QueryRewriter: '{}' → '{}'", original_query, rewritten
            )
            return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryRewriter failed ({}); using original", exc)
            return original_query


__all__ = ["QueryRewriter"]
