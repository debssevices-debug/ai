"""Network policy for tools that fetch URLs.

Same rules as the browser's URL policy (http/https only, no cloud metadata, no
private or loopback address unless explicitly allowed, optional allow/block
lists), plus a DNS check: a public-looking hostname that resolves to a private
address is refused too. Every redirect hop is checked again by the caller.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})
METADATA_HOSTS = frozenset(
    {"169.254.169.254", "metadata.google.internal", "metadata.goog", "100.100.100.200"}
)
LOOPBACK_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})


def _is_private(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    )


def _matches(host: str, pattern: str) -> bool:
    return host == pattern or host.endswith(f".{pattern}")


def check_url(
    url: str,
    *,
    allowed_hosts: tuple[str, ...] = (),
    blocked_hosts: tuple[str, ...] = (),
    allow_private: bool = False,
) -> str | None:
    """Return why ``url`` may not be fetched, or None when it may."""
    raw = (url or "").strip()
    if not raw:
        return "No URL was given."
    try:
        parts = urlsplit(raw)
    except ValueError:
        return f"{raw!r} is not a valid URL."
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return (
            f"Only http:// and https:// URLs are allowed (got {scheme or 'none'}://)."
        )
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return f"{raw!r} has no hostname."
    if host in METADATA_HOSTS:
        return "That address is a cloud metadata endpoint and is always blocked."
    if allowed_hosts and not any(_matches(host, p) for p in allowed_hosts):
        return f"{host} is not on the allowlist."
    if any(_matches(host, p) for p in blocked_hosts):
        return f"{host} is blocked by policy."
    if not allow_private and (host in LOOPBACK_HOSTS or _is_private(host)):
        return f"{host} is a private or loopback address and is blocked."
    return None


async def check_resolved(host: str, *, allow_private: bool = False) -> str | None:
    """Refuse a hostname that resolves to a private address (DNS rebinding, SSRF)."""
    if allow_private:
        return None
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror:
        return f"{host} could not be resolved."
    for info in infos:
        address = str(info[4][0])
        if address in METADATA_HOSTS or _is_private(address):
            return f"{host} resolves to a private address and is blocked."
    return None
