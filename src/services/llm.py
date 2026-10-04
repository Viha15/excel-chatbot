"""LLM access. One class covers Ollama, Azure OpenAI and OpenAI, because all three
speak the OpenAI chat-completions API; only the client construction differs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import APITimeoutError, AzureOpenAI, BadRequestError, OpenAI, UnprocessableEntityError

from src.config import Settings


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: Usage) -> None:
        """Add another call's token counts to this one. Used by pipeline.answer_question() to
        total the tokens over all LLM calls for a question."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


class LLM(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = 500,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, Usage]:
        """Send a conversation to the model and return (reply text, token usage).

        The interface the rest of the app depends on: ChatLLM implements it for the real
        providers, and the tests' FakeLLM implements it with scripted replies.
        """
        ...


# (provider, model) pairs whose server rejected json_schema output; they get plain JSON mode.
_NO_SCHEMA_SUPPORT: set[tuple[str, str]] = set()


class ChatLLM:
    def __init__(self, client: Any, model: str, provider: str):
        """Wrap an OpenAI-SDK client. `model` is the model name (the deployment name on
        Azure) and `provider` is "ollama", "azure" or "openai". Created by build_llm()."""
        self.client, self.model, self.provider = client, model, provider

    def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = 500,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, Usage]:
        """Send `messages` to the model and return (reply text, token usage).

        json_mode: ask for a JSON object (used for the plan); leave off for plain prose.
        max_tokens: cap on the reply length, which also bounds generation time.
        schema: constrain the output to this JSON schema (structured outputs) where the
        server supports it; otherwise fall back to plain JSON mode, once per model.

        Uses temperature 0, so the same question gives the same plan. A timeout or any other
        failure is raised as LLMError. Called by pipeline.answer_question().
        """
        # max_tokens caps generation time: a plan is ~100 tokens, small models can otherwise ramble.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        key = (self.provider, self.model)
        if schema is not None and key not in _NO_SCHEMA_SUPPORT:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "query_plan", "schema": schema},
            }
        elif json_mode or schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            try:
                response = self.client.chat.completions.create(**kwargs)
            except (BadRequestError, UnprocessableEntityError):
                if kwargs.get("response_format", {}).get("type") != "json_schema":
                    raise
                _NO_SCHEMA_SUPPORT.add(key)
                kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
        except APITimeoutError as exc:
            raise LLMError(
                f"{self.provider} did not answer within the timeout. The model may still be loading "
                "or is too slow on this machine: try a smaller model or raise LLM_TIMEOUT_SECONDS."
            ) from exc
        except Exception as exc:
            raise LLMError(f"{self.provider} call failed: {exc}") from exc
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            Usage(getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0),
        )


def build_llm(settings: Settings) -> ChatLLM:
    """Create the model client for settings.llm_provider ("ollama", "azure" or "openai").

    All three use the OpenAI SDK; only the client construction differs. Raises LLMError for
    missing credentials or an unknown provider. Called per /chat request through
    dependencies.get_llm(), and by the evaluation script.
    """
    timeout = settings.llm_timeout_seconds
    if settings.llm_provider == "ollama":
        # max_retries=0: the SDK would otherwise silently retry a timed-out call twice,
        # tripling the wait on a slow local model.
        client = OpenAI(
            base_url=f"{settings.ollama_base_url}/v1", api_key="ollama", timeout=timeout, max_retries=0
        )
        return ChatLLM(client, settings.ollama_model, "ollama")
    if settings.llm_provider == "azure":
        if not (settings.azure_endpoint and settings.azure_api_key):
            raise LLMError("LLM_PROVIDER=azure needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
        client = AzureOpenAI(
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            timeout=timeout,
            max_retries=1,  # one retry for transient cloud errors such as 429
        )
        return ChatLLM(client, settings.azure_deployment, "azure")
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("LLM_PROVIDER=openai needs OPENAI_API_KEY.")
        return ChatLLM(
            OpenAI(api_key=settings.openai_api_key, timeout=timeout, max_retries=1),
            settings.openai_model,
            "openai",
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use ollama, azure or openai.")


def parse_json(text: str) -> dict[str, Any]:
    """Parse the model's JSON, tolerating code fences or stray text around the object.

    Tries the whole text first, then the first {...} block inside it; raises LLMError if
    neither is valid JSON. Called by pipeline.answer_question() on the planning reply.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Model did not return valid JSON: {text[:200]!r}")


def warm_up_ollama(settings: Settings) -> None:
    """Load the model into memory before the first question and keep it there.

    Uses Ollama's native endpoint, which accepts keep_alive (the OpenAI-compatible one does
    not). Best effort: if Ollama is not running yet, the first question just loads it.

    Started in a background thread by main.lifespan() when the provider is Ollama.
    """
    try:
        httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            timeout=settings.llm_timeout_seconds,
            json={"model": settings.ollama_model, "keep_alive": settings.ollama_keep_alive},
        )
    except httpx.HTTPError:
        pass
