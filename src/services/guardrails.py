"""Guardrails around the LLM and the uploaded data.

Input
  check_question      length, control characters, prompt-injection phrases (rejected before
                      any LLM call, so they cost nothing)
  check_upload        the .xlsx is a real zip, has no macros, and does not decompress into
                      something huge (zip bomb)
  safe_profile_value  cell text shown to the LLM in the schema profile is length-capped, and
                      cells that read like instructions to a model are left out

Output
  ungrounded_numbers  numbers in an LLM-written answer that do not appear in the computed
                      result; if there are any, the answer is replaced by the template answer

Usage
  RateLimiter         requests per client per minute (protects cost and the local model)

The query plan itself is already constrained elsewhere: it can only express whitelisted
operations, every sheet and column is validated, and nothing the model writes is executed
as code.
"""

from __future__ import annotations

import io
import re
import threading
import time
import unicodedata
import zipfile
from collections import defaultdict, deque
from typing import Any

from src.config import MAX_QUESTION_CHARS


class GuardrailError(ValueError):
    """A request was refused by a guardrail. The message is safe to show to the user."""


# --- input --------------------------------------------------------------------------------


# Phrases that try to steer the model instead of asking about the data. Deliberately narrow:
# a question that merely mentions "instructions" or "system" is not refused.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b"
    r".{0,40}\b(instructions?|prompts?|rules?|messages?)\b",
    r"\b(reveal|show|print|repeat|leak)\b.{0,40}\b(system|hidden|initial)\s+(prompt|instructions?|message)\b",
    r"\byou\s+are\s+now\b",
    r"\b(act|behave)\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?(unrestricted|jailbroken|different)\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"</?\s*(system|assistant|instructions?)\s*>",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)


def looks_like_injection(text: str) -> bool:
    """Return True if the text contains a phrase that tries to steer the model
    ("ignore previous instructions", "reveal the system prompt", role tags, ...).

    Used by check_question() for the user's question and by safe_profile_value() for cell
    text from the spreadsheet.
    """
    return bool(_INJECTION.search(text or ""))


def check_question(question: str) -> str:
    """Return the cleaned question, or raise GuardrailError.

    Removes control characters and collapses whitespace, then refuses a question that is
    too short, longer than MAX_QUESTION_CHARS, or that looks like a prompt injection.
    Called by the /chat endpoint before any LLM call, so a refused question costs no tokens.
    """
    # drop control characters (keeps letters, digits, punctuation, spaces in any script)
    cleaned = "".join(ch for ch in question if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) < 3:
        raise GuardrailError("Please ask a question about the workbook.")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise GuardrailError(f"Questions are limited to {MAX_QUESTION_CHARS} characters.")
    if looks_like_injection(cleaned):
        raise GuardrailError(
            "This looks like an instruction to the assistant rather than a question "
            "about the workbook, so it was not processed."
        )
    return cleaned


def check_upload(content: bytes, max_uncompressed_mb: int, max_entries: int = 5_000) -> None:
    """An .xlsx is a zip archive. Refuse files that are not, that carry macros, or that
    would decompress to far more than their size suggests (zip bomb).

    Only the zip index is read; nothing is decompressed. Returns nothing on success and
    raises GuardrailError otherwise. Called by the /upload endpoint before the file is parsed.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise GuardrailError("The file is not a valid .xlsx workbook.") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > max_entries:
            raise GuardrailError("The workbook contains an unusually large number of parts.")
        names = {e.filename.lower() for e in entries}
        if "[content_types].xml" not in names:
            raise GuardrailError("The file is not a valid .xlsx workbook.")
        if any(n.endswith("vbaproject.bin") for n in names):
            raise GuardrailError("Workbooks with macros are not accepted.")
        total = sum(e.file_size for e in entries)
        if total > max_uncompressed_mb * 1024 * 1024:
            raise GuardrailError(f"The workbook expands to more than {max_uncompressed_mb} MB of data.")


MAX_PROFILE_VALUE_CHARS = 60


def safe_profile_value(value: Any) -> Any | None:
    """Prepare one distinct cell value for the schema profile sent to the LLM. Returns None
    for values that should not be shown at all (text that reads like model instructions).

    Non-text values pass through unchanged; long text is cut to MAX_PROFILE_VALUE_CHARS.
    Called by workbook._profile() for every distinct value it would list.
    """
    if not isinstance(value, str):
        return value
    if looks_like_injection(value):
        return None
    text = " ".join(value.split())
    return text if len(text) <= MAX_PROFILE_VALUE_CHARS else text[: MAX_PROFILE_VALUE_CHARS - 1] + "…"


# --- output -------------------------------------------------------------------------------

_ANSWER_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")


def _numbers(text: str) -> list[float]:
    """Extract every number written in a text as a float ("110,207.50" -> 110207.5).

    Helper for ungrounded_numbers(), used on the LLM's answer and on the question.
    """
    out = []
    for raw in _ANSWER_NUMBER.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def ungrounded_numbers(
    answer: str, rows: list[dict[str, Any]], total_rows: int, question: str = ""
) -> list[float]:
    """Numbers the LLM wrote that are not in the computed result (allowing for rounding).
    Small counts (<= the number of result rows) and numbers from the question are allowed.

    An empty list means the answer is grounded. Called by pipeline.answer_question() after the
    LLM has phrased the answer; if anything is returned, the sentence is replaced by the
    template answer.
    """
    known = [
        float(v) for r in rows for v in r.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    known += [float(total_rows), float(len(rows))] + _numbers(question)
    bad = []
    for n in _numbers(answer):
        if n.is_integer() and 0 <= n <= max(total_rows, 10):  # "2 countries", list numbering
            continue
        if not any(abs(n - k) <= max(0.01, 0.005 * abs(k)) for k in known):
            bad.append(n)
    return bad


# --- usage --------------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window limit per client key. In memory: fine for one process."""

    def __init__(self, per_minute: int, window_seconds: float = 60.0):
        """Create a limiter allowing `per_minute` requests per client within the window.

        Keeps, per client key, the times of its recent requests, and a lock so simultaneous
        requests are counted correctly. `per_minute <= 0` switches the limiter off.
        """
        self.per_minute, self.window = per_minute, window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Record one request for this client, or raise GuardrailError if it is over the limit.

        Forgets requests older than the window, then counts what is left. `key` identifies
        the client; dependencies.rate_limit() passes the client's IP address.
        """
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.per_minute:
                raise GuardrailError(f"Too many requests: the limit is {self.per_minute} per minute.")
            hits.append(now)
