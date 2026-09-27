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
    """Build the configured provider. Raises LLMError when its key is missing.

    ``ollama`` runs a model on this machine and needs no key at all.
    """
    if settings.llm_provider == "gemini":
        if settings.google_api_key is None:
            raise LLMError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        from nec_ai.llm.gemini import GeminiProvider

        return GeminiProvider(
            settings.google_api_key.get_secret_value(), settings.llm_model
        )

    if settings.llm_provider == "claude":
        if settings.anthropic_api_key is None:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        from nec_ai.llm.claude import ClaudeProvider

        return ClaudeProvider(
            settings.anthropic_api_key.get_secret_value(),
            settings.llm_model,
            fallbacks=settings.claude_fallbacks,
        )

    if settings.llm_provider == "claude_code":
        from nec_ai.llm.claude_code import ClaudeCodeProvider

        return ClaudeCodeProvider(settings.llm_model, settings.claude_code_path)

    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")
        from nec_ai.llm.openai import OpenAIProvider

        return OpenAIProvider(
            settings.openai_api_key.get_secret_value(),
            settings.llm_model,
            base_url=settings.openai_base_url,
        )

    if settings.llm_provider == "ollama":
        from nec_ai.llm.ollama import OllamaProvider

        return OllamaProvider(
            settings.llm_model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
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
