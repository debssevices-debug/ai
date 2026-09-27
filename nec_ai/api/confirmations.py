"""Remote confirmations: a risky tool call waits for the client's yes or no.

During a streamed run, the agent emits ``confirmation.required`` with an id.
The client answers ``POST /v1/confirmations/{id}``. Without an answer within
``CONFIRMATION_TIMEOUT`` seconds, or if the client disconnects, the action is
refused. A client can only answer its own confirmations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nec_ai.tools.registry import ConfirmationHandler, ConfirmationRequest


@dataclass
class _Pending:
    client: str
    future: asyncio.Future[bool]


class ConfirmationBroker:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._pending: dict[str, _Pending] = {}

    def handler_for(self, client: str) -> ConfirmationHandler:
        async def wait_for_answer(request: ConfirmationRequest) -> bool:
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._pending[request.id] = _Pending(client, future)
            try:
                return await asyncio.wait_for(future, timeout=self.timeout)
            except TimeoutError:
                return False
            finally:
                self._pending.pop(request.id, None)

        return wait_for_answer

    def resolve(self, confirmation_id: str, client: str, approved: bool) -> bool:
        """True when a pending confirmation of this client was answered."""
        pending = self._pending.get(confirmation_id)
        if pending is None or pending.client != client or pending.future.done():
            return False
        pending.future.set_result(approved)
        return True

    def __len__(self) -> int:
        return len(self._pending)
