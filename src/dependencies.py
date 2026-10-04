"""What the routers share: the in-memory workbook store and the FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from src.config import Settings, get_settings
from src.services.guardrails import GuardrailError, RateLimiter
from src.services.llm import LLM, LLMError, build_llm
from src.services.workbook import Workbook

# In-memory store: fine for a single-process demo. See README "Scaling" for production.
WORKBOOKS: dict[str, Workbook] = {}


_limiter: RateLimiter | None = None


def rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Guardrail: requests per client per minute on the endpoints that cost compute or tokens."""
    global _limiter
    if _limiter is None or _limiter.per_minute != settings.rate_limit_per_minute:
        _limiter = RateLimiter(settings.rate_limit_per_minute)
    try:
        _limiter.check(request.client.host if request.client else "unknown")
    except GuardrailError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


def get_llm(settings: Settings = Depends(get_settings)) -> LLM:
    """Dependency that provides the model client for the configured provider.

    A configuration problem (e.g. a missing API key) becomes HTTP 500. Used by /chat through
    `Depends(get_llm)`; tests replace it with a fake model via `app.dependency_overrides`.
    """
    try:
        return build_llm(settings)
    except LLMError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def get_workbook(workbook_id: str) -> Workbook:
    """Return the stored workbook for this id, or raise HTTP 404 if it was never uploaded.

    Used by /chat and /workbooks/{workbook_id}.
    """
    if workbook_id not in WORKBOOKS:
        raise HTTPException(status_code=404, detail="Unknown workbook_id. Upload the file first.")
    return WORKBOOKS[workbook_id]
