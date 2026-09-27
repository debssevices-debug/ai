"""fetch_url and the network policy, against a local HTTP server."""

from __future__ import annotations

import http.server
import socketserver
import threading
from collections.abc import Iterator

import pytest

from nec_ai.config.settings import Settings
from nec_ai.security.net import check_resolved, check_url
from nec_ai.tools.base import ToolContext
from nec_ai.tools.fetch import FetchUrlTool, html_to_text
from nec_ai.tools.registry import ToolRegistry

PAGES = {
    "/page": (
        "text/html; charset=utf-8",
        "<html><head><title>Tarifs Odigo</title><style>.x{}</style></head><body>"
        "<h1>Nos offres</h1><script>alert(1)</script><p>Essentiel : 89 € par agent</p>"
        "<ul><li>Voix</li><li>Chat</li></ul></body></html>",
    ),
    "/long": ("text/plain", "abcdefghij" * 100),
    "/pdf": ("application/pdf", "%PDF-1.4"),
}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/redirect-meta":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
            return
        if self.path == "/redirect-page":
            self.send_response(301)
            self.send_header("Location", "/page")
            self.end_headers()
            return
        if self.path not in PAGES:
            self.send_response(404)
            self.end_headers()
            return
        content_type, body = PAGES[self.path]
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _ctx(**overrides) -> ToolContext:
    base = {"fetch_allow_private": True, "fetch_max_chars": 300}
    return ToolContext(settings=Settings(_env_file=None, **{**base, **overrides}))


async def fetch(url: str, ctx: ToolContext | None = None, **extra):
    registry = ToolRegistry([FetchUrlTool()])
    run = await registry.execute("fetch_url", {"url": url, **extra}, ctx or _ctx())
    return run.result


async def test_html_is_read_as_clean_text(server: str) -> None:
    result = await fetch(f"{server}/page")
    assert result.ok
    assert "Titre : Tarifs Odigo" in result.content
    assert "89 € par agent" in result.content
    assert "- Voix" in result.content
    assert "alert(1)" not in result.content
    assert ".x{}" not in result.content
    assert result.content.startswith("<page_data>")


async def test_long_pages_are_paginated(server: str) -> None:
    first = await fetch(f"{server}/long")
    assert "start=300" in first.content
    second = await fetch(f"{server}/long", start=300)
    assert "Caractères 300-600 sur 1000" in second.content


async def test_redirects_are_followed(server: str) -> None:
    result = await fetch(f"{server}/redirect-page")
    assert result.ok
    assert result.data["url"].endswith("/page")


async def test_a_redirect_to_cloud_metadata_is_refused(server: str) -> None:
    result = await fetch(f"{server}/redirect-meta")
    assert not result.ok
    assert "metadata" in result.content


async def test_binary_files_are_refused(server: str) -> None:
    result = await fetch(f"{server}/pdf")
    assert not result.ok
    assert "not a text page" in result.content


async def test_http_errors_are_reported(server: str) -> None:
    result = await fetch(f"{server}/missing")
    assert not result.ok
    assert "404" in result.content


async def test_loopback_is_refused_by_default(server: str) -> None:
    result = await fetch(f"{server}/page", _ctx(fetch_allow_private=False))
    assert not result.ok
    assert "private or loopback" in result.content


async def test_an_unreachable_site_is_a_readable_error() -> None:
    result = await fetch("http://127.0.0.1:1/")
    assert not result.ok
    assert "unreachable" in result.content


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://x.fr",
        "http://169.254.169.254/",
        "http://metadata.google.internal/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://localhost:8000/",
        "http://[::1]/",
        "",
    ],
)
def test_dangerous_urls_are_refused(url: str) -> None:
    assert check_url(url) is not None


def test_public_urls_are_allowed() -> None:
    assert check_url("https://www.odigo.com/fr/") is None


def test_allow_and_block_lists() -> None:
    assert check_url("https://docs.python.org", allowed_hosts=("python.org",)) is None
    assert check_url("https://evil.com", allowed_hosts=("python.org",)) is not None
    assert check_url("https://ads.evil.com", blocked_hosts=("evil.com",)) is not None


async def test_a_hostname_resolving_to_a_private_address_is_refused() -> None:
    assert await check_resolved("localhost") is not None
    assert await check_resolved("localhost", allow_private=True) is None


def test_html_to_text_handles_broken_markup() -> None:
    title, text = html_to_text("<title>T</title><p>un<p>deux<div>trois")
    assert title == "T"
    assert text.split() == ["un", "deux", "trois"]
