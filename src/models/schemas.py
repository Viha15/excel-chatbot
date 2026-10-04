"""Request and response bodies of the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.config import MAX_QUESTION_CHARS


class ChatRequest(BaseModel):
    workbook_id: str
    # The question limit is MAX_QUESTION_CHARS, enforced by check_question after cleaning (400
    # with a readable message). This is only a payload bound, with room for stray whitespace.
    question: str = Field(min_length=3, max_length=2 * MAX_QUESTION_CHARS)


class ChatResponse(BaseModel):
    answer: str
    plan: dict[str, Any] | None
    result: dict[str, Any] | None
    sources: list[dict[str, Any]]
    usage: dict[str, int]
    failed_attempts: list[str] = []  # why earlier LLM plans were rejected and retried
    guardrails: list[str] = []  # guardrails that changed the answer, e.g. an ungrounded number
