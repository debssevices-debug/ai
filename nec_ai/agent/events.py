"""Events the agent emits while it works.

They are what a client streams to show progress ("🔎 Recherche web...") and
what the trace records, so every step of a run is visible and explainable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    STARTED = "agent.started"
    THINKING = "agent.thinking"
    PLAN = "agent.plan"
    WAITING = "agent.waiting"
    """Paused before an LLM call: quota pacing, or a retry after an outage."""

    MESSAGE = "agent.message"
    """Text the model said while still working (e.g. announcing a search)."""

    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    CONFIRMATION_REQUIRED = "confirmation.required"
    CONFIRMATION_RESOLVED = "confirmation.resolved"
    FINAL = "agent.final"
    ERROR = "agent.error"


#: Names used in traces, matching the observability vocabulary.
TRACE_NAMES: dict[EventType, str] = {
    EventType.STARTED: "USER_REQUEST",
    EventType.PLAN: "PLAN_CREATED",
    EventType.TOOL_STARTED: "TOOL_SELECTED",
    EventType.TOOL_COMPLETED: "TOOL_RESULT",
    EventType.TOOL_FAILED: "TOOL_RESULT",
    EventType.FINAL: "FINAL_RESPONSE",
    EventType.ERROR: "ERROR",
}


@dataclass
class AgentEvent:
    type: EventType
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }
