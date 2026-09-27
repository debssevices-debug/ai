"""Configuration: defaults, parsing, and secrets never leaking."""

from __future__ import annotations

from pathlib import Path

import pytest

from nec_ai.config.settings import Settings


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_defaults_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None)
    assert s.max_agent_iterations == 20
    assert s.terminal_enabled is False
    assert s.filesystem_write_enabled is False
    assert s.api_host == "127.0.0.1"
    assert s.api_allow_no_auth is False


def test_max_iterations_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings(monkeypatch, MAX_AGENT_ITERATIONS="5").max_agent_iterations == 5


def test_max_iterations_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _settings(monkeypatch, MAX_AGENT_ITERATIONS="0")


def test_csv_lists_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, API_KEYS=" a , b,,c ", FILESYSTEM_ROOTS="/tmp,/var")
    assert s.api_keys == ["a", "b", "c"]
    assert s.filesystem_roots == ["/tmp", "/var"]


def test_secrets_are_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, GOOGLE_API_KEY="super-secret-value")
    assert "super-secret-value" not in repr(s)
    assert "super-secret-value" not in str(s.model_dump())
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "super-secret-value"


def test_filesystem_roots_default_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None)
    assert s.resolved_filesystem_roots() == [Path.home().resolve()]


def test_log_level_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings(monkeypatch, LOG_LEVEL="debug").log_level == "DEBUG"
