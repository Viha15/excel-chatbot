"""Runtime settings, read from environment variables (and an optional .env file).

LLM_PROVIDER selects the backend:
    ollama  -> local model via Ollama's OpenAI-compatible endpoint (default, for local testing)
    azure   -> Azure OpenAI deployment
    openai  -> OpenAI API
All three are driven through the same OpenAI SDK, so nothing else in the app changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# Longest question accepted by /chat, after cleaning. Enforced by guardrails.check_question;
# the request model derives its payload bound from it.
MAX_QUESTION_CHARS = 500


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    azure_endpoint: str
    azure_api_key: str
    azure_deployment: str
    azure_api_version: str
    openai_api_key: str
    openai_model: str
    llm_timeout_seconds: float
    max_upload_mb: int
    max_uncompressed_mb: int  # zip-bomb guard: size of the workbook once unpacked
    rate_limit_per_minute: int  # per client, for /upload and /chat (0 = off)
    max_result_rows: int  # rows returned to the client
    max_llm_rows: int  # rows of a *result* shown to the LLM when phrasing the answer
    max_distinct_values: int  # distinct values listed per text column in the schema profile
    profile_sample_rows: int  # example rows per sheet in the schema profile
    answer_with_llm: bool  # False: phrase the answer from the result with a template (one LLM call)
    structured_output: bool  # constrain the plan to PLAN_JSON_SCHEMA (opt-in, see README)
    ollama_keep_alive: str  # how long Ollama keeps the model in memory after warm-up


@lru_cache
def get_settings() -> Settings:
    """Build the Settings object from environment variables, with a default for each one.

    Cached with lru_cache: the environment is read on the first call only and every later
    call returns the same object, so a change to .env needs a restart of the API.
    Usage: `settings = get_settings()`, or `Depends(get_settings)` in a FastAPI endpoint.
    """
    env = os.getenv
    provider = env("LLM_PROVIDER", "ollama").lower()
    return Settings(
        llm_provider=provider,
        ollama_base_url=env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=env("OLLAMA_MODEL", "llama3.2"),
        azure_endpoint=env("AZURE_OPENAI_ENDPOINT", ""),
        azure_api_key=env("AZURE_OPENAI_API_KEY", ""),
        azure_deployment=env("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_api_version=env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        openai_api_key=env("OPENAI_API_KEY", ""),
        openai_model=env("OPENAI_MODEL", "gpt-4o-mini"),
        llm_timeout_seconds=float(env("LLM_TIMEOUT_SECONDS", "180")),
        max_upload_mb=int(env("MAX_UPLOAD_MB", "50")),
        max_uncompressed_mb=int(env("MAX_UNCOMPRESSED_MB", "300")),
        rate_limit_per_minute=int(env("RATE_LIMIT_PER_MINUTE", "30")),
        max_result_rows=int(env("MAX_RESULT_ROWS", "200")),
        max_llm_rows=int(env("MAX_LLM_ROWS", "20")),
        max_distinct_values=int(env("MAX_DISTINCT_VALUES", "25")),
        profile_sample_rows=int(env("PROFILE_SAMPLE_ROWS", "0")),
        # Default: skip the second LLM call on local Ollama, where every call costs seconds.
        answer_with_llm=_flag(env("ANSWER_WITH_LLM"), default=provider != "ollama"),
        structured_output=_flag(env("STRUCTURED_OUTPUT"), default=False),
        ollama_keep_alive=env("OLLAMA_KEEP_ALIVE", "30m"),
    )


def _flag(value: str | None, default: bool) -> bool:
    """Turn the text of an environment variable into True/False.

    Unset or blank returns `default`; "1", "true", "yes" and "on" (any case) are True;
    anything else is False. Used by get_settings() for the on/off settings.
    """
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
