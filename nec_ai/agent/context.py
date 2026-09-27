"""Working memory for one task.

Lives for the duration of a single request: the plan, what each tool call
returned, and notes the agent wants to keep while it works. It is what lets the
agent say "I already searched for X" instead of searching again, and it is what
the trace records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolObservation:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


@dataclass
class TaskContext:
    request: str
    plan: list[str] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0

    def record(
        self, tool: str, arguments: dict[str, Any], ok: bool, content: str
    ) -> None:
        summary = " ".join(content.split())[:300]
        self.observations.append(ToolObservation(tool, arguments, ok, summary))

    def calls_to(self, tool: str) -> list[ToolObservation]:
        return [o for o in self.observations if o.tool == tool]
