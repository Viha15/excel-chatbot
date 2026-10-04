from __future__ import annotations

import os

# Tests never call a real model. Pin the provider so the defaults (e.g. ANSWER_WITH_LLM)
# don't depend on the developer's .env, and the Ollama warm-up never runs.
os.environ["LLM_PROVIDER"] = "openai"
os.environ["ANSWER_WITH_LLM"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"  # tested on its own in test_guardrails.py

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dependencies import WORKBOOKS, get_llm
from src.main import app
from src.services.llm import Usage
from src.services.workbook import Workbook, load_workbook

SAMPLE = Path(__file__).parent / "data" / "sample.xlsx"


@pytest.fixture(scope="session")
def sample_bytes() -> bytes:
    """The raw bytes of tests/data/sample.xlsx, read once per test session."""
    return SAMPLE.read_bytes()


@pytest.fixture(scope="session")
def workbook(sample_bytes: bytes) -> Workbook:
    """The sample workbook loaded through the app's own loader. Tests that call the
    executor directly take this fixture."""
    return load_workbook(sample_bytes, "sample.xlsx")


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    """Ground truth read straight with pandas, independent of the app's loader."""
    return pd.read_excel(SAMPLE, sheet_name="Sales_Data")


@pytest.fixture(scope="session")
def targets() -> pd.DataFrame:
    """The Targets sheet read straight with pandas, as ground truth like `sales`."""
    return pd.read_excel(SAMPLE, sheet_name="Targets")


class FakeLLM:
    """Returns scripted replies in order and records every prompt it was sent."""

    def __init__(self, *replies: str | dict[str, Any]):
        """Script the replies, in order. A dict (a plan) is stored as JSON text, a str as
        it is. Usage: `FakeLLM(plan_dict, "answer sentence")`."""
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, json_mode=False, max_tokens=500, schema=None):
        """Same signature as the real model's complete(): record the prompt in `calls` and
        return the next scripted reply with a fixed usage of 100 + 10 tokens. Fails the
        test if the app makes more calls than were scripted."""
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self.replies.pop(0), Usage(prompt_tokens=100, completion_tokens=10)


@pytest.fixture
def client():
    """A test client that calls the API in-process (no server needed). Starts each test
    with an empty workbook store and removes dependency overrides afterwards."""
    WORKBOOKS.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    WORKBOOKS.clear()


@pytest.fixture
def use_llm(client):
    """Returns a function that makes the API use a given FakeLLM instead of a real model.
    Usage: `fake = use_llm(FakeLLM(plan, "answer"))`, then inspect `fake.calls`."""

    def _install(fake: FakeLLM) -> FakeLLM:
        """Override the get_llm dependency with `fake` and return it."""
        app.dependency_overrides[get_llm] = lambda: fake
        return fake

    return _install


@pytest.fixture
def uploaded(client, sample_bytes) -> str:
    """Upload the sample workbook through POST /upload and return its workbook_id, ready
    to be used in a /chat request."""
    response = client.post("/upload", files={"file": ("sample.xlsx", sample_bytes)})
    assert response.status_code == 200, response.text
    return response.json()["workbook_id"]
