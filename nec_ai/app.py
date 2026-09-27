"""Assembly: settings -> LLM + tools + memory -> Agent.

The CLI, the API server and the voice adapter all build their agent here, so
they share exactly the same tools and rules.
"""

from __future__ import annotations

import logging

from nec_ai.agent.core import Agent
from nec_ai.agent.planner import UpdatePlanTool
from nec_ai.config.settings import Settings, get_settings
from nec_ai.llm import LLMProvider, create_llm
from nec_ai.memory.short_term import ConversationMemory
from nec_ai.observability.trace import TraceWriter
from nec_ai.tools.fetch import FetchUrlTool
from nec_ai.tools.registry import ToolRegistry
from nec_ai.tools.web_search import WebSearchTool, create_provider

logger = logging.getLogger("nec.app")


def build_registry(settings: Settings) -> ToolRegistry:
    """Every tool enabled by the configuration."""
    registry = ToolRegistry(
        [
            UpdatePlanTool(),
            WebSearchTool(create_provider(settings)),
            FetchUrlTool(),
        ]
    )
    logger.info("tools: %s", ", ".join(registry.names))
    return registry


def build_agent(
    settings: Settings | None = None,
    *,
    llm: LLMProvider | None = None,
    memory: ConversationMemory | None = None,
) -> Agent:
    settings = settings or get_settings()
    tracer = TraceWriter(settings.traces_dir) if settings.trace_enabled else None
    return Agent(
        llm or create_llm(settings),
        build_registry(settings),
        settings,
        memory=memory,
        tracer=tracer,
    )
