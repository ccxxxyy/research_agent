"""LangChain callbacks for streaming and usage tracking."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from research_agent.llm.usage_tracker import UsageTracker


class UsageTrackingCallback(AsyncCallbackHandler):
    """Tracks token usage for every LLM call and feeds it to UsageTracker."""

    def __init__(self, tracker: UsageTracker, agent_name: str = "unknown") -> None:
        self.tracker = tracker
        self.agent_name = agent_name

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                usage = getattr(gen, "generation_info", {}) or {}
                token_usage = usage.get("token_usage") or response.llm_output or {}
                if isinstance(token_usage, dict):
                    self.tracker.record(
                        agent_name=self.agent_name,
                        model_name=token_usage.get("model_name", "unknown"),
                        prompt_tokens=token_usage.get("prompt_tokens", 0),
                        completion_tokens=token_usage.get("completion_tokens", 0),
                    )


class StreamingLogCallback(AsyncCallbackHandler):
    """Emits agent execution events for SSE streaming."""

    def __init__(self, agent_name: str = "unknown") -> None:
        self.agent_name = agent_name

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        pass  # Will emit SSE event: {"agent": self.agent_name, "status": "thinking"}

    async def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        pass  # Will emit SSE event: {"agent": self.agent_name, "status": "tool_calling"}

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass  # Will stream token chunks via SSE
