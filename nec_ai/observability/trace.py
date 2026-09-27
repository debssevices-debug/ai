"""Per-run traces: one JSONL file per request.

    data/traces/2026-09-27/ab12cd34ef56.jsonl

Each line is one event (USER_REQUEST, PLAN_CREATED, TOOL_SELECTED,
TOOL_RESULT, FINAL_RESPONSE...), so ``why did the agent do that?`` can always be
answered after the fact. Writing is best effort: a full disk never breaks a run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from nec_ai.agent.events import TRACE_NAMES, AgentEvent

logger = logging.getLogger("nec.trace")


class TraceWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._failed = False

    def path_for(self, run_id: str) -> Path:
        day = time.strftime("%Y-%m-%d")
        return self.directory / day / f"{run_id}.jsonl"

    def record(self, event: AgentEvent) -> None:
        if self._failed:
            return
        entry = event.to_dict()
        entry["trace"] = TRACE_NAMES.get(event.type, str(event.type).upper())
        try:
            path = self.path_for(event.run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            self._failed = True
            logger.warning("tracing disabled: %s", exc)
