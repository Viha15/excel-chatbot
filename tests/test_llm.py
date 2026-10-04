from dataclasses import replace

import pytest

from src.config import get_settings
from src.services.llm import LLMError, build_llm, parse_json


def test_ollama_client_does_not_silently_retry():
    """A timed-out call must not be retried by the SDK: that tripled the wait and made the UI time out."""
    llm = build_llm(replace(get_settings(), llm_provider="ollama"))
    assert llm.client.max_retries == 0
    assert str(llm.client.base_url).rstrip("/").endswith("/v1")


def test_missing_azure_credentials_fail_clearly():
    settings = replace(get_settings(), llm_provider="azure", azure_endpoint="", azure_api_key="")
    with pytest.raises(LLMError, match="AZURE_OPENAI_ENDPOINT"):
        build_llm(settings)


def test_parse_json_tolerates_code_fences():
    assert parse_json('```json\n{"sheet": "Sales"}\n```') == {"sheet": "Sales"}
