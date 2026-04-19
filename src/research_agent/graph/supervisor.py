"""Supervisor — the central orchestrator that builds and compiles the research graph.

Graph topology:

    planner → retrieve → grade_retrieval ─┬─→ rewrite_query → retrieve (loop)
                                           └─→ analyze → write → reason ─┬─→ reflect → write (loop)
                                                                          └─→ finalize
"""

from __future__ import annotations

from functools import partial

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from loguru import logger

from research_agent.graph.edges import should_reflect, should_retry_retrieval
from research_agent.graph.nodes.analyst import analyst_node
from research_agent.graph.nodes.reasoner import reasoner_node
from research_agent.graph.nodes.retriever import (
    grade_retrieval_node,
    retrieve_node,
    rewrite_query_node,
)
from research_agent.graph.nodes.writer import finalize_node, writer_node
from research_agent.graph.state import ResearchPhase, ResearchState
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName
from research_agent.rag.grader import RetrievalGrader
from research_agent.rag.query_rewriter import QueryRewriter
from research_agent.rag.retriever import HybridRetriever

PLANNER_SYSTEM_PROMPT = """\
You are a research planning expert. Given a research query, break it down into
2-4 specific sub-questions that, when answered, will comprehensively address
the original query.

Respond with only the sub-questions, one per line.
"""


async def planner_node(state: ResearchState, *, model_router: ModelRouter) -> dict:
    """Break down the research query into sub-questions for retrieval."""
    query = state["query"]
    logger.info("Planner: decomposing query='{}'", query)

    model = model_router.for_agent(AgentName.SUPERVISOR)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Research query: {query}"),
    ]

    response = await model.ainvoke(messages)
    content = str(response.content)
    sub_questions = [q.strip() for q in content.strip().split("\n") if q.strip()]

    return {
        "research_plan": sub_questions,
        "retrieval_queries": sub_questions,
        "phase": ResearchPhase.PLANNING,
        "active_agent": "supervisor",
    }


async def error_handler_node(state: ResearchState) -> dict:
    """Handle errors by logging and setting failed state."""
    logger.error("Error in research pipeline: {}", state["error"])
    return {"phase": ResearchPhase.FAILED}


def build_research_graph(
    *,
    model_router: ModelRouter,
    hybrid_retriever: HybridRetriever,
    checkpointer: BaseCheckpointSaver | None = None,
    memory_store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the full research StateGraph.

    The graph implements:
    - Supervisor-driven planning and task decomposition
    - Corrective RAG with retrieval grading and query rewriting loop
    - Reflection loop with quality scoring and iterative revision
    - Checkpoint-based fault recovery at every node
    - Human-in-the-loop interrupt before finalization
    """

    grader = RetrievalGrader(model_router)
    rewriter = QueryRewriter(model_router)

    # --- Bind dependencies to node functions via partial ---

    _planner = partial(planner_node, model_router=model_router)
    _retrieve = partial(retrieve_node, retriever=hybrid_retriever)
    _grade = partial(grade_retrieval_node, grader=grader)
    _rewrite = partial(rewrite_query_node, rewriter=rewriter)
    _analyze = partial(analyst_node, model_router=model_router)
    _write = partial(writer_node, model_router=model_router)
    _reason = partial(reasoner_node, model_router=model_router)

    # --- Build the graph ---

    graph = StateGraph(ResearchState)  # type: ignore[arg-type]

    # Add nodes (type: ignore needed — LangGraph stubs don't fully type async callables)
    graph.add_node("plan", _planner)  # type: ignore[arg-type]
    graph.add_node("retrieve", _retrieve)  # type: ignore[arg-type]
    graph.add_node("grade_retrieval", _grade)  # type: ignore[arg-type]
    graph.add_node("rewrite_query", _rewrite)  # type: ignore[arg-type]
    graph.add_node("analyze", _analyze)  # type: ignore[arg-type]
    graph.add_node("write", _write)  # type: ignore[arg-type]
    graph.add_node("reason", _reason)  # type: ignore[arg-type]
    graph.add_node("finalize", finalize_node)  # type: ignore[arg-type]
    graph.add_node("error_handler", error_handler_node)  # type: ignore[arg-type]

    # Define edges
    graph.set_entry_point("plan")

    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade_retrieval")

    # Corrective RAG loop: grade → (retry | proceed)
    graph.add_conditional_edges(
        "grade_retrieval",
        should_retry_retrieval,
        {
            "rewrite_query": "rewrite_query",
            "analyze": "analyze",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")

    graph.add_edge("analyze", "write")
    graph.add_edge("write", "reason")

    # Reflection loop: reason → (reflect & revise | finalize)
    graph.add_conditional_edges(
        "reason",
        should_reflect,
        {
            "reflect": "write",   # Send back to writer with feedback
            "finalize": "finalize",
        },
    )

    graph.add_edge("finalize", END)
    graph.add_edge("error_handler", END)

    # --- Compile with checkpoint persistence ---

    compiled = graph.compile(
        checkpointer=checkpointer,
        store=memory_store,
        interrupt_before=["finalize"],  # Human-in-the-loop: pause before final output
    )

    logger.info("Research graph compiled successfully")
    return compiled
