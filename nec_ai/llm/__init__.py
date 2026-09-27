"""LLM providers behind one interface. Pick one with ``LLM_PROVIDER``."""

from __future__ import annotations

from nec_ai.config.settings import Settings
from nec_ai.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
    Message,
    ToolCall,
    complete_with_retry,
)


def create_llm(settings: Settings) -> LLMProvider:
    """Build the configured provider. Raises LLMError when its key is missing."""
    if settings.llm_provider == "gemini":
        if settings.google_api_key is None:
            raise LLMError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        from nec_ai.llm.gemini import GeminiProvider

        return GeminiProvider(
            settings.google_api_key.get_secret_value(), settings.llm_model
        )

    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")
        from nec_ai.llm.openai import OpenAIProvider

        return OpenAIProvider(
            settings.openai_api_key.get_secret_value(),
            settings.llm_model,
            base_url=settings.openai_base_url,
        )

    raise LLMError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}")


__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "LLMUnavailableError",
    "Message",
    "ToolCall",
    "complete_with_retry",
    "create_llm",
]
