"""Chat endpoint: ask a question about an uploaded workbook."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.config import Settings, get_settings
from src.dependencies import get_llm, get_workbook, rate_limit
from src.models.schemas import ChatRequest, ChatResponse
from src.services.guardrails import GuardrailError, check_question
from src.services.llm import LLM, LLMError
from src.services.pipeline import answer_question

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(
    request: ChatRequest, llm: LLM = Depends(get_llm), settings: Settings = Depends(get_settings)
) -> ChatResponse:
    """POST /chat: answer one question about an uploaded workbook.

    Looks up the workbook (404), cleans and checks the question before any LLM call (400),
    runs the plan -> execute -> answer pipeline (502 if the model fails), and returns the
    answer with the plan, the computed table, the sources and the token usage.
    """
    workbook = get_workbook(request.workbook_id)
    try:
        question = check_question(request.question)
    except GuardrailError as exc:
        logger.warning("chat refused workbook=%s reason=%s", request.workbook_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    started = time.perf_counter()
    try:
        out = answer_question(workbook, question, llm, settings)
    except LLMError as exc:
        logger.error(
            "chat llm_error workbook=%s seconds=%.1f error=%s", request.workbook_id, _since(started), exc
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    # One line per question: what it cost (tokens, time, retries) and what came back.
    logger.info(
        "chat workbook=%s seconds=%.1f prompt_tokens=%d completion_tokens=%d retries=%d rows=%s "
        "guardrails=%d question=%r",
        request.workbook_id,
        _since(started),
        out.usage.prompt_tokens,
        out.usage.completion_tokens,
        len(out.failed_attempts),
        out.result.total_rows if out.result else None,
        len(out.guardrails),
        question,
    )

    result = None
    sources: list[dict[str, Any]] = []
    if out.result:
        result = {
            "columns": out.result.columns,
            "rows": out.result.rows,
            "total_rows": out.result.total_rows,
            "steps": out.result.steps,
        }
        sources = [asdict(s) for s in out.result.sources]
    return ChatResponse(
        answer=out.answer,
        plan=out.plan.model_dump(exclude_defaults=True) if out.plan else None,
        result=result,
        sources=sources,
        usage=asdict(out.usage),
        failed_attempts=out.failed_attempts,
        guardrails=out.guardrails,
    )


def _since(started: float) -> float:
    """Seconds elapsed since `started` (a time.perf_counter() value). Used for the log lines."""
    return time.perf_counter() - started
