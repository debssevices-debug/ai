"""API authentication and rate limiting.

* Every ``/v1`` route needs ``Authorization: Bearer <key>`` with a key from
  ``API_KEYS``. Keys are compared in constant time and never logged; a client is
  identified by a short hash of its key.
* Without any key configured, the API refuses to serve, except on localhost
  with ``API_ALLOW_NO_AUTH=true`` (local development).
* Each key gets ``RATE_LIMIT_PER_MINUTE`` requests per minute; beyond it the
  API answers 429 with ``Retry-After``.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from nec_ai.config.settings import Settings

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
ANONYMOUS = "local"


def client_id(key: str) -> str:
    """Stable, non-reversible identifier for a key (logs, sessions, limits)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def auth_problem(settings: Settings) -> str | None:
    """Why the server must not start with this configuration, or None."""
    if settings.api_keys:
        weak = [k for k in settings.api_keys if len(k) < 24]
        if weak:
            return "API_KEYS contains a key shorter than 24 characters; generate one with `nec new-key`."
        return None
    if settings.api_allow_no_auth and settings.api_host in LOCAL_HOSTS:
        return None
    if settings.api_allow_no_auth:
        return (
            "API_ALLOW_NO_AUTH only works when API_HOST is 127.0.0.1: an API "
            "reachable from the network must require a key."
        )
    return (
        "No API key configured. Generate one with `nec new-key` and put it in API_KEYS."
    )


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        self._keys = [k.encode("utf-8") for k in settings.api_keys]
        self._open = (
            not self._keys
            and settings.api_allow_no_auth
            and (settings.api_host in LOCAL_HOSTS)
        )

    def __call__(self, request: Request) -> str:
        """FastAPI dependency: returns the caller's client id or raises 401."""
        if self._open:
            return ANONYMOUS
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise _unauthorized("Missing bearer token.")
        candidate = token.strip().encode("utf-8")
        # Compare against every key so timing does not reveal which one matched.
        matched = False
        for key in self._keys:
            matched |= secrets.compare_digest(candidate, key)
        if not matched:
            raise _unauthorized("Invalid API key.")
        return client_id(token.strip())


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


class RateLimiter:
    """Sliding one-minute window per client."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        hits = self._hits[client]
        while hits and now - hits[0] >= 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            retry = max(1, int(60 - (now - hits[0])) + 1)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit reached ({self.per_minute}/min). Retry in {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        hits.append(now)
