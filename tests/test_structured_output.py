"""Structured output: the plan request carries the JSON schema; servers that reject it
fall back to plain JSON mode, once per model."""

from types import SimpleNamespace

import httpx
from openai import BadRequestError

from src.models.plan import PLAN_JSON_SCHEMA
from src.services.llm import _NO_SCHEMA_SUPPORT, ChatLLM


class FakeCompletions:
    """Stands in for the SDK's `client.chat.completions`, without any network call."""

    def __init__(self, reject_schema: bool):
        """`reject_schema=True` imitates a server that does not support json_schema output.
        `formats` records the response format of every request received."""
        self.reject_schema, self.formats = reject_schema, []

    def create(self, **kwargs):
        """Record the requested response format, then either reject a json_schema request
        with HTTP 400 or return a canned reply shaped like the SDK's response."""
        fmt = kwargs.get("response_format", {}).get("type")
        self.formats.append(fmt)
        if self.reject_schema and fmt == "json_schema":
            request = httpx.Request("POST", "http://test/v1/chat/completions")
            raise BadRequestError("no json_schema", response=httpx.Response(400, request=request), body=None)
        message = SimpleNamespace(content='{"sheet": "S"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def make_llm(reject_schema: bool, model: str) -> tuple[ChatLLM, FakeCompletions]:
    """Create a ChatLLM wired to a FakeCompletions, and return both. Clears any remembered
    "no schema support" entry for this model so each test starts fresh."""
    completions = FakeCompletions(reject_schema)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    _NO_SCHEMA_SUPPORT.discard(("ollama", model))
    return ChatLLM(client, model, "ollama"), completions


def test_plan_request_is_constrained_by_the_schema():
    llm, calls = make_llm(reject_schema=False, model="supports")
    llm.complete([{"role": "user", "content": "q"}], json_mode=True, schema=PLAN_JSON_SCHEMA)
    assert calls.formats == ["json_schema"]


def test_falls_back_to_json_mode_once_when_schema_is_rejected():
    llm, calls = make_llm(reject_schema=True, model="no-schema")
    for _ in range(2):
        text, _ = llm.complete([{"role": "user", "content": "q"}], json_mode=True, schema=PLAN_JSON_SCHEMA)
        assert text == '{"sheet": "S"}'
    assert calls.formats == ["json_schema", "json_object", "json_object"]
