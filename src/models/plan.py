"""The query plan: a small, typed description of WHAT to compute.

The LLM produces a plan as JSON; the executor turns it into pandas operations.
The plan can only express whitelisted operations (no code, no eval), which keeps
execution safe and every number reproducible.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Op = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains"]
Agg = Literal["sum", "mean", "count", "min", "max", "median", "nunique"]
AGG_SYNONYMS = {
    "avg": "mean",
    "average": "mean",
    "total": "sum",
    "add": "sum",
    "maximum": "max",
    "highest": "max",
    "minimum": "min",
    "lowest": "min",
    "number": "count",
    "count_rows": "count",
    "distinct": "nunique",
    "count_distinct": "nunique",
    "unique": "nunique",
}


NULL_TEXT = {"none", "null", "nil", "n/a", ""}


def _is_null_text(value: Any) -> bool:
    """Models trained on Python sometimes write the string "None" for null.

    Returns True for "none", "null", "nil", "n/a" or an empty string (any case).
    Used by Lenient._drop_nulls().
    """
    return isinstance(value, str) and value.strip().lower() in NULL_TEXT


class Lenient(BaseModel):
    """Accept the JSON small models actually write: optional fields spelled out as null."""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        """Runs before validation on every plan piece: turns null-like strings into None
        and removes None entries for optional fields, so they take their defaults.
        A required field keeps its None (Filter.value may really be null)."""
        if not isinstance(data, dict):
            return data
        # keep null only where it is meaningful and the field is required (Filter.value)
        required = {n for n, f in cls.model_fields.items() if f.is_required()}
        data = {k: (None if _is_null_text(v) else v) for k, v in data.items()}
        return {k: v for k, v in data.items() if v is not None or k in required}


class Filter(Lenient):
    """Row filter applied BEFORE aggregation: column <op> value."""

    column: str
    op: Op
    value: Any


class Metric(Lenient):
    """An aggregate. column=None with agg='count' counts rows."""

    agg: Agg
    column: str | None = None
    alias: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _agg_synonyms(cls, data: Any) -> Any:
        """Map the names models use for aggregations (average, total, ...) to ours."""
        if isinstance(data, dict) and isinstance(data.get("agg"), str):
            agg = data["agg"].strip().lower()
            data = {**data, "agg": AGG_SYNONYMS.get(agg, agg)}
        return data

    def name(self) -> str:
        """Name of this metric's column in the result: the alias if given, otherwise
        "<agg>_<column>" (e.g. "sum_Revenue_EUR"), or "row_count" when rows are counted.
        Used by the executor and in the "Computed as" line of the answer."""
        return self.alias or (f"{self.agg}_{self.column}" if self.column else "row_count")


class Join(Lenient):
    """Bring in columns from another sheet (left join), e.g. targets per country."""

    sheet: str
    left_on: str
    right_on: str


class Condition(Lenient):
    """Filter applied AFTER aggregation/join. Compare to a value or to another column."""

    column: str
    op: Op
    value: Any = None
    other_column: str | None = None


class Sort(Lenient):
    by: str
    descending: bool = True


class QueryPlan(Lenient):
    intent: Literal["query", "find_columns", "unanswerable"] = "query"
    sheet: str | None = None
    filters: list[Filter] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    join: Join | None = None
    having: list[Condition] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)  # columns to show for row listings
    sort: Sort | None = None
    limit: int | None = Field(default=None, ge=1)
    keywords: list[str] = Field(default_factory=list)  # for intent=find_columns
    reason: str = ""  # why unanswerable, or a short note

    @model_validator(mode="before")
    @classmethod
    def _tidy(cls, data: Any) -> Any:
        """Normalise other shapes small models produce, before field validation.

        Wraps a single item into a list, drops empty or half-filled join/sort blocks, turns
        a limit written as text into a number, drops a limit below 1, and drops an unknown
        intent so it defaults to "query".
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for key in ("filters", "metrics", "having", "group_by", "columns", "keywords"):
            value = data.get(key)
            if isinstance(value, (dict, str)):  # single item instead of a list
                data[key] = [value]
        for key, required in (("join", ("sheet", "left_on", "right_on")), ("sort", ("by",))):
            value = data.get(key)
            if isinstance(value, dict) and not all(value.get(r) for r in required):
                data.pop(key)  # empty or half-filled block: treat as absent
        limit = data.get("limit")
        if isinstance(limit, str) and limit.strip().isdigit():
            data["limit"] = limit = int(limit)
        if isinstance(limit, int) and limit < 1:
            data.pop("limit")
        if data.get("intent") not in ("query", "find_columns", "unanswerable"):
            data.pop("intent", None)  # e.g. "compare" or "aggregate": it is a normal query
        return data


_VALUE = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "array", "items": {"type": ["string", "number"]}},
    ]
}
_OPS = ["==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains"]

# JSON schema handed to the model for constrained ("structured") output: the decoder can only
# produce JSON of this shape, so invalid operators, aggregations or missing values cannot occur.
PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["query", "find_columns", "unanswerable"]},
        "sheet": {"type": "string"},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["column", "op", "value"],
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string", "enum": _OPS},
                    "value": _VALUE,
                },
            },
        },
        "group_by": {"type": "array", "items": {"type": "string"}},
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["agg"],
                "properties": {
                    "agg": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "min", "max", "median", "nunique"],
                    },
                    "column": {"type": ["string", "null"]},
                    "alias": {"type": "string"},
                },
            },
        },
        "join": {
            "type": "object",
            "required": ["sheet", "left_on", "right_on"],
            "properties": {
                "sheet": {"type": "string"},
                "left_on": {"type": "string"},
                "right_on": {"type": "string"},
            },
        },
        "having": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["column", "op"],
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string", "enum": _OPS},
                    "value": _VALUE,
                    "other_column": {"type": "string"},
                },
            },
        },
        "columns": {"type": "array", "items": {"type": "string"}},
        "sort": {
            "type": "object",
            "required": ["by"],
            "properties": {"by": {"type": "string"}, "descending": {"type": "boolean"}},
        },
        "limit": {"type": "integer", "minimum": 1},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?\s*(?:k|m)?(?![\w])", re.IGNORECASE)


def numbers_in(question: str) -> list[float]:
    """Numbers written in the question: '10,000' -> 10000.0, '2.5k' -> 2500.0.

    Digits inside words such as "Q3" are not counted. Used by fill_values_from_question().
    """
    found = []
    for raw in _NUMBER.findall(question or ""):
        text = raw.replace(",", "").strip().lower()
        factor = {"k": 1_000, "m": 1_000_000}.get(text[-1], 1)
        try:
            found.append(float(text.rstrip("km")) * factor)
        except ValueError:
            continue
    return found


def fill_values_from_question(data: Any, question: str) -> tuple[Any, list[str]]:
    """Small models sometimes write {"column": "Revenue_EUR", "op": ">"} and leave the number
    out, although the question states it ("greater than 10,000"). Fill comparison conditions
    that lack a value from the numbers in the question: one number fills all of them (they
    are then duplicates and are merged), otherwise numbers are used in order if the counts
    match. Returns the patched plan dict and notes for the steps.

    Works on the raw dict, so it must run before QueryPlan.model_validate() (a filter with
    no value would fail validation). Called by pipeline.answer_question().
    """
    if not isinstance(data, dict):
        return data, []
    notes: list[str] = []
    numbers = numbers_in(question)
    for key in ("filters", "having"):
        items = data.get(key)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue
        missing = [
            i
            for i in items
            if isinstance(i, dict)
            and "value" not in i
            and not i.get("other_column")
            and i.get("op") in (">", ">=", "<", "<=", "==", "!=")
        ]
        if missing and (len(numbers) == 1 or len(numbers) == len(missing)):
            for n, item in enumerate(missing):
                value = numbers[0] if len(numbers) == 1 else numbers[n]
                item["value"] = int(value) if value.is_integer() else value
                notes.append(
                    f"took value {item['value']:,} for {item.get('column')} {item['op']} from the question"
                )
        unique: list[Any] = []
        for item in items:  # drop exact duplicates (models repeat a filter on retry)
            if item not in unique:
                unique.append(item)
        data = {**data, key: unique}
    return data, notes
