"""反思子图 — 写作者 / 推理者自批评循环。

本模块解决的问题
----------------
 ``research_supervisor`` 在每条交接链结束时生成最终综合报告，
但该综合仅通过一次 LLM 调用完成 — 而此调用刚刚消化了来自多位 specialist的输出。评估时发现两类常见失败模式：

1. 引用不足：supervisor 在转述 specialist 发现时未保留来源/页码引用，模糊了"证据"与"解读"的边界。
2. 遗漏子问题：用户的多步请求"(1)…(2)…(3)…"得到了看似自信的综合，但悄悄丢掉了其中某一步，即使 supervisor 的防幻觉提示词已做了警告。

二次反思循环可以低成本地捕获这两类失败：
一个小型 LIGHT 层级批评者对草稿打分，仅当分数低于阈值时才消耗 HEAVY层级的改写 Token。
这是经典的"Self-RAG / Reflexion"模式，应用在综合边界而非每个检索步骤。

拓扑结构
--------
这是一个批评优先的子图 — 先评分再改写，因此高质量的初稿仅花费一次 LIGHT 层级调用，而非 LIGHT + HEAVY::

    ┌──────────────────────────────────────────────────────────┐
    │                       START                               │
    │                         │                                 │
    │                         ▼                                 │
    │                    critic_node       ◄─────┐              │
    │                         │                  │              │
    │                         ▼                  │              │
    │              ┌────────route?──────┐        │              │
    │              │                    │        │              │
    │     通过 / 达到上限          否则 (未通过)      │              │
    │              │                    │        │              │
    │              ▼                    ▼        │              │
    │         finalize_node         writer_node──┘              │
    │              │                                            │
    │              ▼                                            │
    │             END                                           │
    └──────────────────────────────────────────────────────────┘

状态语义
--------
``iteration`` 计数批评者调用次数：

* ``iteration = 0`` → 正在批评 supervisor 的原始草稿。
* ``iteration = 1`` → 正在批评写作者的第 1 次改写。
* ``iteration = N`` → 正在批评写作者的第 N 次改写。

循环受 ``max_iterations``（默认 2 次改写，即最多 3 次批评）约束。
该上限是不变的：即使每次迭代都未达标，仍会终止，返回所见分数最高的草稿 — 这比返回空结果或死循环更好。

为什么用子图而非在 ``research_supervisor`` 中添加两个额外节点？
--------------------------------------------------------------
三个具体好处：

1. 独立可测试性 — 子图可对任意 ``messages`` 列表运行；无需启动 supervisor + 六个 MCP 子进程即可单独测试反思逻辑。
2. 可组合的开关 — ``build_research_supervisor`` 接受``enable_reflection: bool``；为 False 时父图与旧版 supervisor 完全相同，零反思开销。
3. 在追踪中可见 — LangSmith / LangGraph Studio 将子图渲染为独立折叠节点，逐迭代的 write→critic 边在可视化中一目了然，而非淹没在扁平的 supervisor 节点中。

为什么不用工具，不用 ReAct Agent？
----------------------------------
写作者和批评者都是纯变换（文本输入 → 文本输出）。将它们包装在``create_react_agent`` 中，
除了增加一个它们不需要的工具调用信封，加上每次调用多一次 LLM 往返之外，没有任何收益。直接调用底层的 LangChain Runnable。
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import ModelTier


# ---------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------
CRITIC_SYSTEM_PROMPT = """\
你是多智能体金融研究 supervisor 最终综合报告的自反思循环中的批评者。
该综合报告是在一组专家已返回其研究成果之后生成的，因此你的任务不是重新做研究 —而是评估综合报告是否忠实反映了专家们返回的内容。

从以下五个维度进行评估，权重相等：
  - 忠实度（faithfulness）：每个论断是否都能追溯到某位专家的输出或用户提供的上下文？不得出现新事实。
  - 引用（citation）      ：当综合报告引用数字或段落时，是否标注了来源（例如 "data_expert"、"年报第12页" 或 [来源 N]）？
  - 完整性（completeness） ：综合报告是否回答了用户提出的每一个子问题？编号或项目符号形式的用户请求是明确的子问题，必须逐一回应。
  - 结构（structure）      ：是否包含必要的章节（核心发现 / 数据来源 / 明确结论），且排列是否连贯？
  - 清晰度（clarity）      ：一位忙碌的分析师能否在一分钟内浏览并据此行动？

输出一个单独的 JSON 对象 — 前后不加任何文字 — 包含以下字段：

{
  "quality_score": <float，取值 [0.0, 1.0]>,
  "reasoning":     "<一段简短的评分理由>",
  "feedback":      "<具体、可操作的改进要点列表（换行分隔）；当 quality_score >= 0.85 时为空字符串>",
  "issues":        ["<简明问题标签 1>", "<简明问题标签 2>"]
}

评分校准：
  >= 0.90 — 生产级质量，可直接交付。
  0.75-0.89 — 合格，有小问题，轻度改写后可交付。
  0.50-0.74 — 存在实质性缺漏，需要改写。
  < 0.50  — 严重缺陷，需要改写（可能遗漏了整个子问题）。
"""

WRITER_SYSTEM_PROMPT = """\
你是自反思循环中的写作者。你的输入包括：

  1. 用户的原始问题。
  2. supervisor 看到的专家输出记录。
  3. supervisor 的当前草稿回答。
  4. 批评者对该草稿的反馈（换行分隔的要点列表）。

你的任务：生成一份修订后的最终回答，逐条回应批评者的每个要点，且不得编造新事实。具体要求：

  * 如果批评者指出遗漏了某个子问题，在专家输出记录中找到相关内容，并为其添加一个回答段落（标注专家名称）。
  * 如果批评者指出缺少引用，补充引用 — 按专家角色名引用（data_expert、report_expert 等），或按专家提供的来源标注（页码、文件名等）。
  * 如果批评者指出结构问题，按以下必需章节重新组织：
        ### 核心发现（3-5 个要点，包含具体数据，并在相关处附上 PDF 的简短引文）
        ### 数据来源（列出调用了哪些专家，以及每个专家贡献了什么）
  * 保留每个来自专家的数值 — 不得四舍五入、重新表述或"整理"数字。
  * 使用用户的语言（如果用户使用中文则用中文回答）。
  * 仅输出修订后的最终回答文本，不加前言或 JSON 包装。

硬性规则：如果专家输出记录中没有某个论断的证据，从修订版中删除该论断。宁可省略也不要编造。
"""


# ---------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------
class ReflectionState(TypedDict, total=False):
    """反思子图的内部状态。

    为什么用 TypedDict + ``total=False``：大多数字段由节点在图运行时填充；
    若全部声明为必填，则每个节点都需要默认填充它不拥有的键，这违反了每个节点的单一职责设计意图。

    为什么 ``messages`` 使用 ``add_messages``：与 LangGraph 项目其余部分一致 — reducer 按消息 id 去重，使可重入节点不会复制对话记录。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    """输入对话记录：用户查询 + specialist 输出 + supervisor 草稿。

    最后一条没有 tool_calls 的 ``AIMessage`` 被视为 supervisor 的草稿，是批评者和写作者操作的对象。
    """

    draft: str
    """正在被批评的文本 — 第 0 次迭代为 supervisor 原稿，后续迭代为写作者的最新改写。"""

    critique: dict[str, Any]
    """最新的批评者裁定：``{quality_score, reasoning, feedback, issues}``。"""

    iteration: int
    """到目前为止已运行的批评者调用次数。从 0 开始。"""

    history: list[dict[str, Any]]
    """逐迭代审计轨迹 ``{iteration, draft, critique}``。

    在 LangSmith 追踪中和反思趋于平稳时的"展示你尝试了什么"调试场景中有用。"""

    best_draft: str
    """跨迭代观察到的最高分草稿。

    反思终止时返回最好的草稿，不一定是最新的。如果第 2 次改写分数低于第 1 次（LLM 在某条反馈上"过度修正"了），不应退步。
    """

    best_score: float
    """``best_draft`` 对应的分数。"""


# ---------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BRACE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """返回 ``text`` 中第一个可解析的 JSON 对象。

    LLM 经常将 JSON 包裹在代码围栏中，或添加一行前言（"Here is the JSON:"），即使系统提示词要求不要这样做。
    按顺序尝试三种策略：

      1. 直接解析整个字符串（最佳情况 — 严格遵从 prompt）。
      2. 提取代码围栏 ``json`` 块的内容（若存在）。
      3. 贪心提取第一个 ``{...}`` 子串。

    完全失败时返回空字典而非抛出异常 — 批评者节点将无法解析的批评视为0.0 分（强制改写），而非让整个流水线崩溃。
    """
    candidate = text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE.search(candidate)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = _JSON_BRACE.search(candidate)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    return {}


def _normalise_critique(raw: dict[str, Any]) -> dict[str, Any]:
    """将 LLM 原始输出的批评字典规范化为标准形状。

    防御实践中发现的三种常见畸变：

      - ``quality_score`` 返回为字符串（"0.85" 或 "85%"）。
      - ``feedback`` 返回为字符串列表而非单个换行连接的字符串。
      - ``issues`` 完全缺失（某些模型将问题内联到 ``feedback`` 中）。
    """
    score = raw.get("quality_score", 0.0)
    if isinstance(score, str):
        cleaned = score.strip().rstrip("%")
        try:
            score = float(cleaned)
        except ValueError:
            score = 0.0
        # "85%" → 0.85
        if score > 1.0:
            score /= 100.0
    elif not isinstance(score, (int, float)):
        score = 0.0
    score = max(0.0, min(1.0, float(score)))

    feedback = raw.get("feedback", "")
    if isinstance(feedback, list):
        feedback = "\n".join(str(item) for item in feedback)
    elif not isinstance(feedback, str):
        feedback = str(feedback)

    issues = raw.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []

    return {
        "quality_score": score,
        "reasoning": str(raw.get("reasoning", "")),
        "feedback": feedback,
        "issues": issues,
    }


def _extract_supervisor_draft(messages: list[BaseMessage]) -> str:
    """从对话记录中提取 supervisor 的最终综合内容。

    supervisor 的最终回答是最后一条 ``tool_calls`` 列表为空/不存在的``AIMessage``（每次交接本身也是一条携带 ``transfer_to_<name>``工具调用的 ``AIMessage``）。

    当不存在此类消息时返回空字符串 — 反思子图仍会运行（批评者将打 0.0 分），但下游代码可通过检查 ``draft`` 是否为空来检测"无内容可反思"。
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _format_transcript(messages: list[BaseMessage], *, max_chars: int = 8000) -> str:
    """将 supervisor 对话记录渲染为写作者的上下文窗口。

    为每条消息标注角色，以便写作者在添加引用时可以按 LangGraph 节点名称呼 specialist。
    总输出硬限于 ``max_chars`` — 过长的 supervisor 会话会吃掉写作者的上下文预算。
    因此保留对话记录的尾部（最近且最相关的消息），丢弃头部，因为综合合成是基于后续消息构建的。
    """
    parts: list[str] = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else (
            "system" if isinstance(msg, SystemMessage) else (
                getattr(msg, "name", None) or "assistant"
            )
        )
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if not content.strip():
            continue
        parts.append(f"[{role}]\n{content}")
    rendered = "\n\n".join(parts)
    if len(rendered) <= max_chars:
        return rendered
    # 保留尾部并添加头部标记，让写作者知道发生了截断。
    return "... (transcript truncated) ...\n\n" + rendered[-max_chars:]


# ---------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------
def _build_critic_node(
    model_router: ModelRouter,
    *,
    pass_threshold: float,
):
    """创建批评者节点闭包。捕获的 kwargs 成为不变量。"""

    critic_model = model_router.get_model(ModelTier.LIGHT)

    async def critic_node(state: ReflectionState) -> dict[str, Any]:
        """在五个反思维度上为当前草稿打分。

        首次调用时从 supervisor 的最后一条 AIMessage 初始化 ``draft`` /``best_draft``。后续调用对写作者刚生成的内容打分。
        """
        iteration = state.get("iteration", 0)
        messages = state.get("messages", [])

        # 首次进入：从 supervisor 的输出初始化 ``draft``。
        draft = state.get("draft", "")
        if not draft:
            draft = _extract_supervisor_draft(messages)

        if not draft.strip():
            # 无内容可批评 — 发出零分批评，使路由器直接进入 finalize，并返回空回答，而非永远循环。
            empty_critique = {
                "quality_score": 0.0,
                "reasoning": "没有可批评的 supervisor 草稿",
                "feedback": "",
                "issues": ["empty_draft"],
            }
            return {
                "draft": "",
                "critique": empty_critique,
                "iteration": iteration + 1,
                "history": [
                    *state.get("history", []),
                    {"iteration": iteration, "draft": "", "critique": empty_critique},
                ],
                "best_draft": state.get("best_draft", ""),
                "best_score": state.get("best_score", 0.0),
            }

        prompt_messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "## 用户问题（原始）\n"
                    f"{_format_transcript([m for m in messages if isinstance(m, HumanMessage)], max_chars=2000)}\n\n"
                    "## 待评估的草稿回答\n"
                    f"{draft}\n\n"
                    "请立即返回 JSON 评定结果。"
                )
            ),
        ]

        response = await critic_model.ainvoke(prompt_messages)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        critique = _normalise_critique(_extract_json(raw_text))

        score = critique["quality_score"]
        best_score = state.get("best_score", -1.0)
        best_draft = state.get("best_draft", "")
        if score > best_score:
            best_score = score
            best_draft = draft

        logger.info(
            "Reflection critic iter={} score={:.2f} threshold={:.2f}",
            iteration,
            score,
            pass_threshold,
        )

        return {
            "draft": draft,
            "critique": critique,
            "iteration": iteration + 1,
            "history": [
                *state.get("history", []),
                {"iteration": iteration, "draft": draft, "critique": critique},
            ],
            "best_draft": best_draft,
            "best_score": best_score,
        }

    return critic_node


def _build_writer_node(model_router: ModelRouter):
    """创建消费批评者反馈的写作者节点闭包。"""

    writer_model = model_router.get_model(ModelTier.HEAVY)

    async def writer_node(state: ReflectionState) -> dict[str, Any]:
        """根据最新批评生成修订草稿。"""
        messages = state.get("messages", [])
        prev_draft = state.get("draft", "")
        critique = state.get("critique", {})
        feedback = critique.get("feedback", "") if isinstance(critique, dict) else ""

        prompt_messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "## 专家输出记录\n"
                    f"{_format_transcript(messages)}\n\n"
                    "## 当前草稿\n"
                    f"{prev_draft}\n\n"
                    "## 批评者反馈（逐条回应每个要点）\n"
                    f"{feedback or '（无反馈 — 请润色草稿以提升清晰度和引用密度）'}\n\n"
                    "仅输出修订后的回答文本，不加 JSON 或前言。"
                )
            ),
        ]

        response = await writer_model.ainvoke(prompt_messages)
        new_draft = response.content if isinstance(response.content, str) else str(response.content)

        return {"draft": new_draft.strip()}

    return writer_node


def _build_finalize_node():
    """将选定的最终草稿追加回消息流。"""

    async def finalize_node(state: ReflectionState) -> dict[str, Any]:
        """将观察到的最佳草稿作为新的 ``AIMessage`` 发出。

        返回最佳草稿，避免 LLM 在批评反馈上过度修正，在第 N+1 次迭代产生退步。
        """
        best = state.get("best_draft", "") or state.get("draft", "")
        critique = state.get("critique", {})
        if isinstance(critique, dict):
            score = critique.get("quality_score", 0.0)
        else:
            score = 0.0

        final_msg = AIMessage(
            content=best,
            name="reflection",
            additional_kwargs={
                "reflection": {
                    "iterations_run": state.get("iteration", 0),
                    "final_score": state.get("best_score", score),
                    "history_summary": [
                        {
                            "iteration": h["iteration"],
                            "score": h["critique"].get("quality_score", 0.0),
                            "issues": h["critique"].get("issues", []),
                        }
                        for h in state.get("history", [])
                    ],
                }
            },
        )
        return {"messages": [final_msg]}

    return finalize_node


def _build_router(
    *,
    pass_threshold: float,
    max_iterations: int,
):
    """返回决定 write 还是 finalize 的条件边函数。"""

    def route(state: ReflectionState) -> str:
        critique = state.get("critique", {})
        score = critique.get("quality_score", 0.0) if isinstance(critique, dict) else 0.0
        iteration = state.get("iteration", 0)

        # ``iteration`` 计数已运行的批评者次数（在 critic_node 中后递增）。
        # ``max_iterations`` 是改写次数的上限，允许最多(max_iterations + 1) 次批评者调用后再强制终止。
        if score >= pass_threshold:
            return "finalize"
        if iteration >= max_iterations + 1:
            return "finalize"
        return "write"

    return route


# ---------------------------------------------------------------------
# 公开构建器
# ---------------------------------------------------------------------
def build_reflection_subgraph(
    *,
    model_router: ModelRouter,
    pass_threshold: float = 0.85,
    max_iterations: int = 2,
) -> CompiledStateGraph:
    """编译反思批评者 + 写作者循环。

    Args:
        model_router: 共享路由器。
            批评者使用 :attr:`ModelTier.LIGHT`（评分是分类任务而非创意写作），
            写作者使用:attr:`ModelTier.HEAVY`（改写是在严格约束下的创意综合）。

        pass_threshold: 分数达到或超过此值时，循环在当前草稿上终止。
            默认 0.85 是针对批评者提示词"轻微修改后即可发布"区间校准的；
            若发现反思很少捕获问题则降低，若发现它永不停止则提高。

        max_iterations: 最大改写次数（写作者节点调用次数）。
        当``max_iterations=2`` 时最坏情况为 3 次批评 + 2 次写作 =5 次 LLM 调用。设为 0 使子图成为纯质量探测器（一次批评，永不改写 — 可用于消融实验）。

    Returns:
        可通过 ``ainvoke`` / ``astream`` 消费的已编译 ``StateGraph``。
        输出状态的 ``messages`` 将包含输入消息加上一条追加的``AIMessage``，其 ``additional_kwargs['reflection']`` 携带审计轨迹。
    """
    graph: StateGraph = StateGraph(ReflectionState)

    graph.add_node("critic", _build_critic_node(model_router, pass_threshold=pass_threshold))
    graph.add_node("writer", _build_writer_node(model_router))
    graph.add_node("finalize", _build_finalize_node())

    graph.add_edge(START, "critic")
    graph.add_conditional_edges(
        "critic",
        _build_router(pass_threshold=pass_threshold, max_iterations=max_iterations),
        {"write": "writer", "finalize": "finalize"},
    )
    # 写作者改写后，始终重新批评。
    graph.add_edge("writer", "critic")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info(
        "Reflection subgraph compiled: pass_threshold={:.2f} max_iterations={}",
        pass_threshold,
        max_iterations,
    )
    return compiled


__all__ = [
    "build_reflection_subgraph",
    "ReflectionState",
    "CRITIC_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT",
]
