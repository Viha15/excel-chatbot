"""Run the assignment's ten example questions against the configured LLM and check every
answer against values computed independently with pandas from the sample workbook.

    uv run python -m eval.run_examples                      # model from .env
    uv run python -m eval.run_examples --model qwen2.5:3b   # try another Ollama model
    uv run python -m eval.run_examples --provider azure     # or another provider

Use it to choose a model and to catch regressions after prompt or code changes.
The checks are deliberately tolerant about naming (any column alias is fine); they look
for the right numbers and labels in the computed result.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings, get_settings
from src.services.llm import LLM, LLMError, build_llm
from src.services.pipeline import ChatResult, answer_question
from src.services.workbook import load_workbook

SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "data" / "sample.xlsx"


def _values(out: ChatResult) -> list[Any]:
    """Every cell value of the result rows as one flat list (empty if there is no result).
    Helper for _has_number() and _has_text()."""
    rows = out.result.rows if out.result else []
    return [v for r in rows for v in r.values()]


def _has_number(out: ChatResult, expected: float) -> bool:
    """True if any value in the result is a number within 0.01 of `expected`. The column's
    name does not matter, so any alias the model chose is accepted."""
    return any(isinstance(v, (int, float)) and abs(v - expected) < 0.01 for v in _values(out))


def _has_text(out: ChatResult, expected: str) -> bool:
    """True if any value in the result is text equal to `expected`, ignoring case."""
    return any(isinstance(v, str) and v.casefold() == expected.casefold() for v in _values(out))


def _rows(out: ChatResult) -> int:
    """Number of rows the query matched, or -1 if there is no result (so it can never
    equal a real count)."""
    return out.result.total_rows if out.result else -1


@dataclass
class Case:
    question: str
    expected: str  # human-readable, shown when the check fails
    check: Callable[[ChatResult], bool]


def build_cases(path: Path = SAMPLE) -> list[Case]:
    """Build the ten example questions, each with a check for its correct answer.

    The correct values are computed here with plain pandas from the workbook at `path`,
    independently of the app, so the app is compared with a separate source of truth.
    Used by evaluate() when no cases are passed in.
    """
    sales = pd.read_excel(path, sheet_name="Sales_Data")
    targets = pd.read_excel(path, sheet_name="Targets").set_index("Country")
    rev = sales.groupby("Country").Revenue_EUR.sum()
    profit = sales.groupby("Country").Profit_EUR.sum()
    avg = sales.groupby("Country").Revenue_EUR.mean().round(2)
    cat = sales.groupby("Product_Category").Revenue_EUR.sum()
    germany = sales[sales.Country == "Germany"]
    below = {c for c in rev.index if rev[c] < targets.loc[c, "Revenue_Target_EUR"]}
    over_10k = int((sales.Revenue_EUR > 10000).sum())

    return [
        Case(
            "What is the total revenue for Germany?",
            f"{rev['Germany']:,.2f}",
            lambda o: _rows(o) == 1 and _has_number(o, rev["Germany"]),
        ),
        Case("Show all records for Germany.", f"{len(germany)} rows", lambda o: _rows(o) == len(germany)),
        Case(
            "What is the average revenue by country?",
            "5 rows with each country's mean",
            lambda o: _rows(o) == len(avg) and all(_has_number(o, v) for v in avg),
        ),
        Case(
            "Which country has the highest total revenue?",
            f"{rev.idxmax()} ({rev.max():,.2f})",
            lambda o: (
                bool(o.result and o.result.rows)
                and rev.idxmax() in map(str, o.result.rows[0].values())
                and _has_number(o, rev.max())
            ),
        ),
        Case(
            "How many records have revenue greater than 10,000?",
            str(over_10k),
            lambda o: _has_number(o, over_10k) and _rows(o) == 1,
        ),
        Case(
            "Compare Germany and France.",
            "2 rows: Germany and France, revenue each",
            lambda o: (
                _rows(o) == 2
                and _has_text(o, "Germany")
                and _has_text(o, "France")
                and _has_number(o, rev["Germany"])
                and _has_number(o, rev["France"])
            ),
        ),
        Case(
            "Which product category generated the highest revenue?",
            f"{cat.idxmax()} ({cat.max():,.2f})",
            lambda o: _has_text(o, cat.idxmax()) and _has_number(o, cat.max()),
        ),
        Case(
            "Which worksheet contains information about revenue?",
            "Sales_Data and Targets",
            lambda o: _has_text(o, "Sales_Data") and _has_text(o, "Targets"),
        ),
        Case(
            "What are the total revenue and profit by country?",
            "5 rows with revenue and profit",
            lambda o: _rows(o) == len(rev) and all(_has_number(o, v) for v in [*rev, *profit]),
        ),
        Case(
            "Which countries are below their revenue target?",
            ", ".join(sorted(below)),
            lambda o: bool(o.result) and {str(r.get("Country")) for r in o.result.rows} == below,
        ),
    ]


@dataclass
class Outcome:
    case: Case
    passed: bool
    seconds: float
    retries: int
    tokens: int
    note: str


def evaluate(llm: LLM, settings: Settings, cases: list[Case] | None = None) -> list[Outcome]:
    """Run every case through the real pipeline (answer_question) and check the result.

    Returns one Outcome per question with pass/fail, seconds, retries, tokens and a note
    explaining a failure. A model that cannot be reached counts as a failed case and the
    run continues. Takes the LLM as an argument, so main() passes a real model and the
    tests pass a scripted fake.
    """
    workbook = load_workbook(
        SAMPLE.read_bytes(), SAMPLE.name, settings.max_distinct_values, settings.profile_sample_rows
    )
    outcomes = []
    for case in cases or build_cases():
        start = time.perf_counter()
        try:
            out = answer_question(workbook, case.question, llm, settings)
            passed = out.result is not None and case.check(out)
            note = "" if passed else (out.answer.splitlines()[0][:90] if out.answer else "no answer")
            outcomes.append(
                Outcome(
                    case,
                    passed,
                    time.perf_counter() - start,
                    len(out.failed_attempts),
                    out.usage.prompt_tokens + out.usage.completion_tokens,
                    note,
                )
            )
        except LLMError as exc:
            outcomes.append(Outcome(case, False, time.perf_counter() - start, 0, 0, str(exc)[:90]))
    return outcomes


def main() -> None:
    """Command-line entry point: `uv run python -m eval.run_examples [--provider P] [--model M]`.

    Applies the optional provider and model overrides to the settings, runs the evaluation
    against that model, and prints one line per question plus a summary.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", choices=["ollama", "azure", "openai"])
    parser.add_argument("--model", help="Ollama model, Azure deployment or OpenAI model")
    args = parser.parse_args()

    settings = get_settings()
    if args.provider:
        settings = replace(settings, llm_provider=args.provider)
    if args.model:
        field = {"ollama": "ollama_model", "azure": "azure_deployment", "openai": "openai_model"}
        settings = replace(settings, **{field[settings.llm_provider]: args.model})
    model = {
        "ollama": settings.ollama_model,
        "azure": settings.azure_deployment,
        "openai": settings.openai_model,
    }[settings.llm_provider]
    print(f"Evaluating {settings.llm_provider} / {model} on {SAMPLE.name}\n")

    outcomes = evaluate(build_llm(settings), settings)
    for i, o in enumerate(outcomes, 1):
        mark = "PASS" if o.passed else "FAIL"
        extra = f"  expected {o.case.expected}; got: {o.note}" if not o.passed else ""
        print(
            f"{i:>2}. {mark}  {o.seconds:5.1f}s  {o.tokens:>5} tok  retries {o.retries}  "
            f"{o.case.question}{extra}"
        )
    passed = sum(o.passed for o in outcomes)
    print(
        f"\n{passed}/{len(outcomes)} passed · {sum(o.tokens for o in outcomes):,} tokens · "
        f"{sum(o.seconds for o in outcomes):.0f}s total"
    )


if __name__ == "__main__":
    main()
