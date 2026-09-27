"""URL policy and action classification. No browser required."""

from __future__ import annotations

import pytest

from browser import (
    ActionRisk,
    BrowserConfig,
    check_request_url,
    check_url,
    classify_action,
    resolve_url,
    wrap_untrusted,
)

BASE = BrowserConfig()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://C:/Windows/System32/config/SAM",
        "chrome://settings",
        "about:blank",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
    ],
)
def test_blocks_dangerous_schemes(url: str) -> None:
    assert check_url(url, BASE) is not None


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/iam/",
        "https://metadata.google.internal/computeMetadata/v1/",
        "http://100.100.100.200/latest/meta-data/",
    ],
)
def test_blocks_cloud_metadata_even_with_loopback_allowed(url: str) -> None:
    relaxed = BrowserConfig(allow_loopback=True, allowed_hosts=())
    assert check_url(url, relaxed) is not None


def test_blocks_loopback_by_default() -> None:
    """SSRF path to internal services. Production default must stay closed."""
    assert check_url("http://127.0.0.1:8080/admin", BASE) is not None
    assert check_url("http://localhost:3000/", BASE) is not None
    assert check_url("http://10.0.0.5/internal", BASE) is not None
    assert check_url("http://192.168.1.1/router", BASE) is not None


def test_loopback_allowed_when_configured() -> None:
    relaxed = BrowserConfig(allow_loopback=True)
    assert check_url("http://127.0.0.1:8080/admin", relaxed) is None


def test_allowlist_permits_subdomains() -> None:
    scoped = BrowserConfig(allowed_hosts=("example.com",))
    assert check_url("https://example.com/a", scoped) is None
    assert check_url("https://www.example.com/a", scoped) is None
    assert check_url("https://evil.com/a", scoped) is not None
    # A prefix must not pass as a subdomain.
    assert check_url("https://notexample.com/a", scoped) is not None


def test_blocklist_wins_over_allowlist() -> None:
    scoped = BrowserConfig(
        allowed_hosts=("example.com",), blocked_hosts=("bad.example.com",)
    )
    assert check_url("https://bad.example.com/x", scoped) is not None


@pytest.mark.parametrize(
    "url,expected",
    [
        ("", "No URL"),
        ("   ", "No URL"),
        ("example.com", "no scheme"),
        ("https://", "no hostname"),
    ],
)
def test_rejects_malformed_urls(url: str, expected: str) -> None:
    reason = check_url(url, BASE)
    assert reason is not None
    assert expected in reason


def test_accepts_ordinary_urls() -> None:
    assert check_url("https://www.iana.org/domains/example", BASE) is None
    assert check_url("http://example.com:8080/path?q=1", BASE) is None


def test_subdomain_bypass_attempt_on_allowlist() -> None:
    scoped = BrowserConfig(allowed_hosts=("example.com",))
    assert check_url("https://example.com.evil.net/", scoped) is not None


def test_request_check_ignores_non_http_schemes() -> None:
    """data: and blob: subresources are normal and must not be blocked."""
    assert check_request_url("data:image/png;base64,AAAA", BASE) is None
    assert check_request_url("blob:http://example.com/1234", BASE) is None


def test_request_check_enforces_host_rules() -> None:
    assert check_request_url("http://169.254.169.254/x", BASE) is not None
    assert check_request_url("http://127.0.0.1:9000/x", BASE) is not None
    assert check_request_url("https://example.com/x", BASE) is None


def test_resolve_relative_href() -> None:
    assert (
        resolve_url("https://example.com/a/b/page.html", "../c.html")
        == "https://example.com/a/c.html"
    )


@pytest.mark.parametrize(
    "tool,args,expected",
    [
        ("open_page", {}, ActionRisk.READ),
        ("read_page", {}, ActionRisk.READ),
        ("click", {}, ActionRisk.READ),
        ("type_text", {}, ActionRisk.FILL),
        ("fill_form", {"submit": False}, ActionRisk.FILL),
        ("fill_form", {"submit": True}, ActionRisk.COMMIT),
        ("type_text", {"submit": True}, ActionRisk.COMMIT),
        ("download_file", {}, ActionRisk.COMMIT),
        ("run_steps", {}, ActionRisk.COMMIT),
    ],
)
def test_action_risk_is_structural(tool: str, args: dict, expected: ActionRisk) -> None:
    assert classify_action(tool, args) is expected


def test_action_risk_ignores_argument_wording() -> None:
    """Meaning is the model's job. Only the shape of the call is read here."""
    assert classify_action("click", {"text": "supprimer mon compte"}) is ActionRisk.READ
    assert classify_action("type_text", {"text": "x", "into": "y"}) is ActionRisk.FILL


def test_untrusted_wrapper_marks_page_data() -> None:
    wrapped = wrap_untrusted("hello")
    assert wrapped.startswith("<page_data>")
    assert wrapped.endswith("</page_data>")
