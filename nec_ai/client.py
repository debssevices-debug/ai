"""Client for a remote NEC server (the base of `nec remote` and a future Windows app).

client = RemoteClient("https://nec.example.com", api_key)
async for event in client.stream("Cherche les nouvelles de Microsoft"):
    ...
    if event.type is EventType.CONFIRMATION_REQUIRED:
        await client.confirm(event.data["id"], approve=True)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from nec_ai.agent.events import AgentEvent, EventType


class RemoteError(Exception):
    """The server refused or failed the request. The message is user-facing."""


class RemoteClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            # Reading a stream may legitimately wait minutes between events.
            timeout=httpx.Timeout(timeout, read=None),
            transport=transport,
        )

    async def __aenter__(self) -> RemoteClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        return (await self._request("GET", "/health")).json()

    async def tools(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/v1/tools")).json()

    async def ask(self, message: str, session_id: str = "default") -> dict[str, Any]:
        response = await self._request(
            "POST", "/v1/chat", json={"message": message, "session_id": session_id}
        )
        return response.json()

    async def confirm(self, confirmation_id: str, *, approve: bool) -> None:
        await self._request(
            "POST", f"/v1/confirmations/{confirmation_id}", json={"approve": approve}
        )

    async def forget(self, session_id: str) -> None:
        await self._request("DELETE", f"/v1/sessions/{session_id}")

    async def stream(
        self, message: str, session_id: str = "default"
    ) -> AsyncIterator[AgentEvent]:
        payload = {"message": message, "session_id": session_id}
        try:
            async with self._client.stream(
                "POST", "/v1/chat/stream", json=payload
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise RemoteError(_describe(response))
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif line == "" and data_lines:
                        event = _to_event("\n".join(data_lines))
                        data_lines = []
                        if event is not None:
                            yield event
        except httpx.ConnectError as exc:
            raise RemoteError(
                f"Serveur injoignable ({self._client.base_url})."
            ) from exc
        except httpx.HTTPError as exc:
            raise RemoteError(f"Connexion interrompue : {type(exc).__name__}") from exc

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise RemoteError(
                f"Serveur injoignable ({self._client.base_url})."
            ) from exc
        except httpx.HTTPError as exc:
            raise RemoteError(f"Erreur réseau : {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise RemoteError(_describe(response))
        return response


def _describe(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = response.text[:200]
    if response.status_code == 401:
        return f"Clé API refusée par le serveur ({detail}). Vérifie NEC_API_KEY."
    if response.status_code == 429:
        return f"Trop de requêtes : {detail}"
    return f"Erreur {response.status_code} : {detail}"


def _to_event(raw: str) -> AgentEvent | None:
    try:
        payload = json.loads(raw)
        return AgentEvent(
            EventType(payload["type"]),
            payload.get("run_id", ""),
            payload.get("data") or {},
            payload.get("timestamp", 0.0),
        )
    except (ValueError, KeyError, TypeError):
        return None
