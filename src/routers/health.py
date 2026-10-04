"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src.config import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """GET /health: report that the API is running and which LLM provider is configured.

    The Streamlit UI calls it to show "API connected".
    """
    return {"status": "ok", "llm_provider": get_settings().llm_provider}
