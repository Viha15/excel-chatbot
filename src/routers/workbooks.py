"""Workbook endpoints: upload an .xlsx file, and describe an uploaded one."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.config import Settings, get_settings
from src.dependencies import WORKBOOKS, get_workbook, rate_limit
from src.services.guardrails import GuardrailError, check_upload
from src.services.workbook import Workbook, WorkbookError, load_workbook

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", dependencies=[Depends(rate_limit)])
async def upload(file: UploadFile = File(...), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """POST /upload: accept an .xlsx file, check it, load it and store it in memory.

    Checks the extension, emptiness and size, then runs the upload guardrail (real workbook,
    no macros, no zip bomb) and parses the sheets. Returns a new `workbook_id` plus the sheet
    description; that id is what /chat needs. Bad files get 400, oversized ones 413.
    """
    if not (file.filename or "").lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")
    # Check the size before loading the file: the declared size first, then read at most one
    # byte past the limit, so an oversized upload is never held in memory in full.
    limit = settings.max_upload_mb * 1024 * 1024
    too_large = HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    if file.size is not None and file.size > limit:
        raise too_large
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise too_large
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        check_upload(content, settings.max_uncompressed_mb)
        workbook = load_workbook(
            content,
            file.filename or "upload.xlsx",
            settings.max_distinct_values,
            settings.profile_sample_rows,
        )
    except (GuardrailError, WorkbookError) as exc:
        logger.warning("upload refused file=%r bytes=%d reason=%s", file.filename, len(content), exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workbook_id = uuid.uuid4().hex
    WORKBOOKS[workbook_id] = workbook
    logger.info(
        "upload workbook=%s file=%r bytes=%d sheets=%d rows=%d",
        workbook_id,
        workbook.filename,
        len(content),
        len(workbook.profiles),
        sum(p.rows for p in workbook.profiles.values()),
    )
    return {"workbook_id": workbook_id, **_describe(workbook)}


@router.get("/workbooks/{workbook_id}")
def describe(workbook_id: str) -> dict[str, Any]:
    """GET /workbooks/{workbook_id}: return the sheet description of an uploaded workbook.

    Answers 404 if the id is unknown.
    """
    return {"workbook_id": workbook_id, **_describe(get_workbook(workbook_id))}


def _describe(workbook: Workbook) -> dict[str, Any]:
    """Summarise a workbook for the client: filename and, per sheet, its name, row count,
    header row and column names with types.

    Built from the profiles created at upload time; the data itself is not read. Used by
    /upload and /workbooks/{workbook_id}, and shown in the UI's sidebar.
    """
    return {
        "filename": workbook.filename,
        "sheets": [
            {
                "name": p.name,
                "rows": p.rows,
                "header_row": p.header_row,
                "columns": [{"name": c.name, "type": c.kind} for c in p.columns],
            }
            for p in workbook.profiles.values()
        ],
    }
