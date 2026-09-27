"""Short-term memory: the recent turns of each conversation.

Only user requests and final answers are kept, plus a one-line note of the
tools used. Raw tool output stays in the task's working memory and is not
replayed on every later turn, which keeps requests small.
"""

from __future__ import annotations

from collections import defaultdict

from nec_ai.llm.base import Message


class ConversationMemory:
    def __init__(self, max_messages: int = 40) -> None:
        self.max_messages = max_messages
        self._sessions: dict[str, list[Message]] = defaultdict(list)

    def history(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    def add_exchange(
        self,
        session_id: str,
        request: str,
        answer: str,
        tools_used: list[str] | None = None,
    ) -> None:
        turns = self._sessions[session_id]
        turns.append(Message.user(request))
        note = ""
        if tools_used:
            note = f"\n\n(outils utilisés : {', '.join(dict.fromkeys(tools_used))})"
        turns.append(Message.assistant(answer + note))
        overflow = len(turns) - self.max_messages
        if overflow > 0:
            # Drop whole exchanges from the front so a turn is never half-kept.
            del turns[: overflow + (overflow % 2)]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def sessions(self) -> list[str]:
        return list(self._sessions)
