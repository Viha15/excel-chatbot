"""FastAPI service: upload a workbook, then ask questions about it.

This module only creates the application and attaches the routers; the endpoints live in
src/routers, the logic in src/services and the data shapes in src/models.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import get_settings
from src.routers import chat, health, workbooks
from src.services.llm import warm_up_ollama

# One log line per upload and per question (tokens, seconds, retries), written to the console.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # no extra line for every call to the LLM


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup/shutdown handler, passed to FastAPI(lifespan=...).

    The code before `yield` runs once when the API starts: on Ollama it preloads the model in
    a background thread so the first question does not wait for it. There is no shutdown work.
    """
    settings = get_settings()
    if settings.llm_provider == "ollama":  # load the model in the background, don't block startup
        threading.Thread(target=warm_up_ollama, args=(settings,), daemon=True).start()
    yield


app = FastAPI(title="Excel Intelligence Chatbot", version="1.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(workbooks.router)
app.include_router(chat.router)
