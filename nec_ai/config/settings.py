"""Application settings, read once from the environment and ``.env``.

Every value has a safe default so the agent starts with no configuration at
all; only the LLM key is needed to actually answer. Secrets are typed as
``SecretStr`` so they never show up in a log line or a ``repr``.

Resolution order (highest first): real environment variables, ``.env.local``,
``.env``. ``.env.local`` is what the LiveKit tooling writes, so both work.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

#: Repository root: ``nec_ai/config/settings.py`` -> three levels up. Anchored to
#: this file so the ``.env`` is found whatever the working directory is.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: Literal["gemini", "openai"] = "gemini"
    llm_model: str = ""
    """Empty means the provider's default model."""

    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    """Any OpenAI-compatible endpoint (OpenRouter, a local server, ...)."""

    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    llm_max_retry_wait: float = 60.0
    """Longest wait accepted before a retry (e.g. a quota reset). Beyond it, fail."""

    llm_requests_per_minute: int = 0
    """Client-side pacing. 0 = no limit. Gemini free tier: set 5 (or 10 for Lite)."""
    llm_temperature: float = 0.3

    # ── Agent ────────────────────────────────────────────────────────────
    max_agent_iterations: int = Field(default=20, ge=1, le=100)
    agent_checkpoint_step: int = 7
    """At this step the agent is reminded to answer if it is not progressing.
    0 disables it."""
    tool_timeout: float = 60.0
    max_tool_output_chars: int = 12_000
    """Tool output longer than this is truncated before it reaches the LLM."""

    agent_language: str = "fr"

    # ── Web search ───────────────────────────────────────────────────────
    search_provider: Literal["duckduckgo", "brave", "serper"] = "duckduckgo"
    search_api_key: SecretStr | None = None
    search_max_results: int = Field(default=6, ge=1, le=20)
    search_region: str = "fr-fr"
    search_timeout: float = 15.0

    fetch_timeout: float = 15.0
    fetch_max_bytes: int = 3 * 1024 * 1024
    fetch_max_chars: int = 8_000
    fetch_allow_private: bool = False
    """Allow fetching private/loopback addresses. Local testing only (SSRF risk)."""

    # ── Filesystem / terminal ───────────────────────────────────────────
    filesystem_roots: CsvList = Field(default_factory=list)
    """Directories the agent may read. Empty: the user's home directory."""

    filesystem_write_enabled: bool = False
    terminal_enabled: bool = False
    terminal_timeout: float = 60.0
    applications_enabled: bool = False

    # ── Memory ───────────────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"
    history_max_messages: int = 40

    # ── API server ───────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_keys: CsvList = Field(default_factory=list)
    """Accepted bearer tokens. Empty disables the API unless API_ALLOW_NO_AUTH."""

    api_allow_no_auth: bool = False
    rate_limit_per_minute: int = 30
    server_url: str = "http://127.0.0.1:8000"
    """Where a client (Windows app, CLI --remote) reaches the API."""

    confirmation_timeout: float = 120.0

    # ── Logging / tracing ────────────────────────────────────────────────
    log_level: str = "INFO"
    trace_enabled: bool = True
    """Write one JSONL trace per request under ``data_dir/traces``."""

    @field_validator("filesystem_roots", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper() or "INFO"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    def resolved_filesystem_roots(self) -> list[Path]:
        roots = self.filesystem_roots or [str(Path.home())]
        return [Path(r).expanduser().resolve() for r in roots]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings. Tests build their own ``Settings(...)``."""
    return Settings()
