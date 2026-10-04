"""The question -> answer pipeline.

    1. plan    (LLM)    question + schema profile -> QueryPlan JSON
    2. execute (pandas) QueryPlan -> exact result          [retry step 1 once on PlanError]
    3. answer  (LLM)    question + small result -> sentence
                        (or a code template when ANSWER_WITH_LLM is off: one LLM call per question,
                        and Ollama can reuse its cached prompt prefix between questions)

The raw sheets never go to the model. The response carries the computed table and the
source rows, so the numbers can be checked independently of the model's wording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.config import Settings
from src.models.plan import PLAN_JSON_SCHEMA, QueryPlan, fill_values_from_question
from src.services import prompts
from src.services.executor import PlanError, QueryResult, execute
from src.services.guardrails import ungrounded_numbers
from src.services.llm import LLM, LLMError, Usage, parse_json
from src.services.workbook import Workbook, profile_as_text

PLAN_ATTEMPTS = 2
PLAN_MAX_TOKENS = 400  # a plan is ~50-150 tokens
ANSWER_MAX_TOKENS = 300


@dataclass
class ChatResult:
    answer: str
    plan: QueryPlan | None
    result: QueryResult | None
    usage: Usage
    failed_attempts: list[str] = field(default_factory=list)  # why earlier plans were rejected
    guardrails: list[str] = field(default_factory=list)  # guardrails that changed the answer


def answer_question(workbook: Workbook, question: str, llm: LLM, settings: Settings) -> ChatResult:
    """Answer one question about a workbook: plan with the LLM, execute with pandas, phrase.

    1. Sends the schema profile and the question to the model and gets a JSON plan.
    2. Validates and executes the plan; on an invalid plan the error goes back to the model
       for one retry (PLAN_ATTEMPTS in total), after which it gives up with a message.
    3. Phrases the answer with a code template, or with a second LLM call whose numbers are
       checked against the result (settings.answer_with_llm).

    `question` must already be cleaned by guardrails.check_question(). Returns a ChatResult
    with the answer, plan, result, token usage, failed attempts and guardrail notes.
    Called by the /chat endpoint and by the evaluation script.
    """
    usage = Usage()
    messages = [
        {"role": "system", "content": prompts.PLAN_SYSTEM},
        {
            "role": "user",
            "content": prompts.PLAN_USER.format(schema=profile_as_text(workbook.profiles), question=question),
        },
    ]

    plan: QueryPlan | None = None
    error = ""
    failed: list[str] = []
    for _ in range(PLAN_ATTEMPTS):
        text, u = llm.complete(
            messages,
            json_mode=True,
            max_tokens=PLAN_MAX_TOKENS,
            schema=PLAN_JSON_SCHEMA if settings.structured_output else None,
        )
        usage.add(u)
        try:
            raw, notes = fill_values_from_question(parse_json(text), question)
            plan = QueryPlan.model_validate(raw)
            if plan.intent == "unanswerable":
                return ChatResult(_unanswerable(plan.reason), plan, None, usage, failed)
            result = execute(workbook, plan, settings.max_result_rows, question=question)
            result.steps[1:1] = notes  # after the 'sheet' step
            break
        except (PlanError, ValidationError, LLMError) as exc:
            error = _short_error(exc)
            failed.append(f"{error} | model wrote: {' '.join(text.split())[:800]}")
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": prompts.PLAN_RETRY.format(error=error)},
            ]
    else:
        return ChatResult(
            f"I could not turn that question into a valid query: {error}", plan, None, usage, failed
        )

    # Report the plan as executed, after the executor's repairs, so it matches the result.
    plan = result.plan or plan

    if not settings.answer_with_llm:
        return ChatResult(template_answer(result, plan), plan, result, usage, failed)

    answer, u = llm.complete(
        [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": _answer_prompt(question, result, settings.max_llm_rows)},
        ],
        max_tokens=ANSWER_MAX_TOKENS,
    )
    usage.add(u)
    # Output guardrail: every number in the model's sentence must come from the result.
    invented = ungrounded_numbers(answer, result.rows, result.total_rows, question)
    if invented:
        note = (
            f"The model's answer contained numbers not in the computed result {invented[:5]}; "
            "it was replaced by the answer built directly from the result."
        )
        return ChatResult(template_answer(result, plan), plan, result, usage, failed, [note])
    return ChatResult(answer.strip(), plan, result, usage, failed)


def template_answer(result: QueryResult, plan: QueryPlan | None = None, max_lines: int = 10) -> str:
    """Phrase a result without an LLM. Every number comes straight from the result, and the
    answer says what was computed (e.g. "max of Revenue_EUR"), so a misread question is visible.

    Used by answer_question() when the LLM answer is switched off, and as the replacement
    when the LLM's sentence contains numbers that are not in the result.
    """
    text = _template_body(result, max_lines)
    if plan and plan.metrics:
        what = ", ".join(f"{m.name()} = {m.agg} of {m.column or 'rows'}" for m in plan.metrics)
        if plan.group_by:
            what += f", per {', '.join(plan.group_by)}"
        text += f"\n\nComputed as: {what}."
    return text


def _template_body(result: QueryResult, max_lines: int) -> str:
    """Write the main sentence of the template answer, choosing the wording by the shape of
    the result: nothing matched, a column search, a single row, a short list (up to
    `max_lines` rows), or a pointer to the table for anything larger.

    Helper for template_answer().
    """
    sheets = ", ".join(s.sheet for s in result.sources)
    if result.total_rows == 0:
        return f"No rows matched in {sheets}." if sheets else "Nothing matched."
    if result.columns == ["sheet", "column"]:  # find_columns
        found = "; ".join(f"{r['sheet']} ({r['column']})" for r in result.rows)
        return f"Matching columns: {found}."
    if result.total_rows == 1:
        parts = ", ".join(f"{k}: {_fmt(v)}" for k, v in result.rows[0].items())
        return f"{parts} (from {sheets}, {result.sources[0].rows_matched} rows used)."
    if result.total_rows <= max_lines and len(result.columns) <= 8:
        lines = [" · ".join(f"{k}: {_fmt(v)}" for k, v in r.items()) for r in result.rows]
        return f"{result.total_rows} results from {sheets}:\n" + "\n".join(f"- {line}" for line in lines)
    return f"{result.total_rows} rows from {sheets}; see the table below."


def _fmt(value: Any) -> str:
    """Format one result value for display: floats as "110,207.50", integers as "1,200",
    everything else as plain text. Helper for _template_body()."""
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _answer_prompt(question: str, result: QueryResult, max_rows: int) -> str:
    """Build the user message for the answer-phrasing LLM call: the question, the executed
    steps, and at most `max_rows` rows of the computed result as JSON (with a note when
    rows were cut). Used by answer_question() when settings.answer_with_llm is on."""
    shown: list[dict[str, Any]] = result.rows[:max_rows]
    truncated = f", first {len(shown)} shown" if result.total_rows > len(shown) else ""
    return prompts.ANSWER_USER.format(
        question=question,
        steps="; ".join(result.steps),
        total_rows=result.total_rows,
        truncated=truncated,
        rows=json.dumps(shown, default=str, ensure_ascii=False),
    )


def _short_error(exc: Exception) -> str:
    """Readable error for the retry prompt (and the user): no pydantic URLs or input dumps.

    A pydantic ValidationError is summarised as up to five "field.path: message" items; any
    other error is returned as its text. Used by answer_question() when a plan fails.
    """
    if isinstance(exc, ValidationError):
        problems = [f"{'.'.join(map(str, e['loc'])) or 'plan'}: {e['msg']}" for e in exc.errors()[:5]]
        return "Invalid plan structure. " + "; ".join(problems)
    return str(exc)


def _unanswerable(reason: str) -> str:
    """Build the reply for a question the workbook cannot answer, appending the model's
    reason if it gave one. Used by answer_question() for intent "unanswerable"."""
    return "This workbook does not contain the information needed to answer that." + (
        f" {reason}" if reason else ""
    )
