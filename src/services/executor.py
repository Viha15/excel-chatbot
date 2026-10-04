"""Deterministic execution of a QueryPlan with pandas. No LLM is involved here.

Every column and sheet name in the plan is validated against the workbook; a bad plan
raises PlanError with a message precise enough to send back to the LLM for a retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.models.plan import Condition, Filter, Metric, QueryPlan, Sort
from src.services.workbook import EXCEL_ROW, Workbook, _plain, kind_of

MAX_CITED_ROWS = 20
NUMERIC_AGGS = {"sum", "mean", "median"}


class PlanError(ValueError):
    """The plan refers to something that does not exist or cannot be computed."""


@dataclass
class Source:
    sheet: str
    rows_matched: int
    excel_rows: list[int] = field(default_factory=list)  # first MAX_CITED_ROWS row numbers


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    sources: list[Source]
    steps: list[str]
    plan: QueryPlan | None = None  # the plan as executed, after repairs


def execute(workbook: Workbook, plan: QueryPlan, max_rows: int = 200, question: str = "") -> QueryResult:
    """Run a validated plan against the workbook with pandas and return the result.

    Repairs predictable plan mistakes first, then applies the steps in a fixed order:
    filters -> aggregate (or list rows) -> join -> having -> sort -> limit. Every sheet and
    column name is checked; a problem raises PlanError with a message the LLM can act on.

    max_rows: how many result rows are returned (total_rows still gives the full count).
    question: the user's question, used by the repairs that depend on its wording.
    Returns a QueryResult with the rows, the sources (sheet and Excel rows) and a log of
    the steps. Called by pipeline.answer_question().
    """
    notes: list[str] = []
    if plan.intent == "find_columns":
        plan = _list_values_not_columns(plan, workbook, question, notes)
    if plan.intent == "find_columns":
        return _find_columns(workbook, plan.keywords, question)
    if plan.intent == "unanswerable":
        raise PlanError(plan.reason or "The question cannot be answered from this workbook.")

    sheet = _resolve_sheet(workbook, plan.sheet)
    df = workbook.sheets[sheet]
    steps = [f'sheet "{sheet}" ({len(df)} rows)', *notes]
    plan = _merge_equalities(plan, steps)
    plan = _group_compared_values(plan, question, steps)
    plan = _which_has_most_means_total(plan, question, steps)
    plan = _compare_metrics_not_raw_columns(plan, steps)
    plan = _drop_match_all_conditions(plan, df, steps)
    plan = _drop_unused_join(plan, workbook, steps)
    plan = _keep_named_values(plan, steps)

    for f in plan.filters:
        df = df[_mask(df, f.column, f.op, f.value)]
        steps.append(f"filter {f.column} {f.op} {f.value!r}")
    matched = df
    sources = [Source(sheet, len(matched), matched[EXCEL_ROW].head(MAX_CITED_ROWS).tolist())]

    if plan.metrics or plan.group_by:
        result = _aggregate(df, plan.group_by, plan.metrics)
        steps.append(
            f"aggregate {[m.name() for m in plan.metrics] or ['row_count']}"
            + (f" by {plan.group_by}" if plan.group_by else "")
        )
    else:
        cols = [_col(df, c) for c in plan.columns] or [c for c in df.columns if c != EXCEL_ROW]
        result = df[cols + [EXCEL_ROW]]

    if plan.join:
        other = _resolve_sheet(workbook, plan.join.sheet)
        right = workbook.sheets[other].drop(columns=[EXCEL_ROW])
        left_on, right_on = _col(result, plan.join.left_on), _col(right, plan.join.right_on)
        result = result.merge(
            right, how="left", left_on=left_on, right_on=right_on, suffixes=("", f"_{other}")
        )
        if right_on != left_on and right_on in result.columns:
            result = result.drop(columns=[right_on])
        sources.append(Source(other, len(right)))
        steps.append(f'join "{other}" on {left_on} = {right_on}')

    for cond in plan.having:
        if (
            cond.other_column
            and _has_col(result, cond.column)
            and _has_col(result, cond.other_column)
            and _col(result, cond.column) == _col(result, cond.other_column)
        ):
            guess = _guess_comparison_column(result, cond, plan, workbook, question)
            if guess is None:
                raise PlanError(
                    f"The condition compares {cond.column} with itself. other_column must be a different "
                    f"column, e.g. one of: {_numeric_columns_elsewhere(workbook, sheet)} "
                    "(add a join to that sheet)."
                )
            steps.append(
                f"compared {cond.column} with {guess} instead of itself (best match for the question)"
            )
            cond = cond.model_copy(update={"other_column": guess})
        try:
            result = result[_condition_mask(result, cond)]
        except PlanError as exc:
            raise PlanError(
                f"{exc}. 'having' runs after aggregation, so it can only use the "
                "group_by columns, metric aliases and joined columns. Conditions on "
                "raw columns belong in 'filters'."
            ) from exc
        target = cond.other_column if cond.other_column else repr(cond.value)
        steps.append(f"keep rows where {cond.column} {cond.op} {target}")

    if plan.sort:
        result = result.sort_values(_col(result, plan.sort.by), ascending=not plan.sort.descending)
        steps.append(f"sort by {plan.sort.by} {'desc' if plan.sort.descending else 'asc'}")
    if plan.limit:
        result = result.head(plan.limit)
        steps.append(f"limit {plan.limit}")

    if EXCEL_ROW in result.columns:  # row listing: cite exactly the rows returned
        sources[0].excel_rows = result[EXCEL_ROW].head(MAX_CITED_ROWS).tolist()
        sources[0].rows_matched = len(result)
        result = result.drop(columns=[EXCEL_ROW])

    total = len(result)
    records = [{k: _clean(v) for k, v in r.items()} for r in result.head(max_rows).to_dict("records")]
    return QueryResult(list(result.columns), records, total, sources, steps, plan)


# --- plan pieces -------------------------------------------------------------------------


def _merge_equalities(plan: QueryPlan, steps: list[str]) -> QueryPlan:
    """Country == Germany AND Country == France can never match a row; the question
    ("compare Germany and France") means either value, i.e. Country in [Germany, France]."""
    by_column: dict[str, list[Filter]] = {}
    for f in plan.filters:
        if f.op == "==":
            by_column.setdefault(_norm(f.column), []).append(f)
    merged = {k: v for k, v in by_column.items() if len(v) > 1}
    if not merged:
        return plan
    filters: list[Filter] = []
    for f in plan.filters:
        group = merged.get(_norm(f.column)) if f.op == "==" else None
        if group is None:
            filters.append(f)
        elif f is group[0]:
            values = [g.value for g in group]
            filters.append(Filter(column=f.column, op="in", value=values))
            steps.append(f"combined {len(values)} '==' filters on {f.column} into 'in' {values}")
    return plan.model_copy(update={"filters": filters})


def _keep_named_values(plan: QueryPlan, steps: list[str]) -> QueryPlan:
    """If the question names N values of a grouped column (filter 'in' [A, B]), a limit
    below N would silently drop some of the values the user asked about."""
    if not plan.limit or not plan.group_by:
        return plan
    groups = {_norm(g) for g in plan.group_by}
    for f in plan.filters:
        if (
            f.op == "in"
            and _norm(f.column) in groups
            and isinstance(f.value, list)
            and plan.limit < len(f.value)
        ):
            steps.append(f"removed limit {plan.limit}: the question names {len(f.value)} {f.column} values")
            return plan.model_copy(update={"limit": None})
    return plan


def _compare_metrics_not_raw_columns(plan: QueryPlan, steps: list[str]) -> QueryPlan:
    """Column-to-column condition on a raw column that is aggregated, e.g. Revenue_EUR <
    Revenue_Target_EUR next to sum(Revenue_EUR): after aggregation only the sum exists, and a
    target per group is compared with the group's total."""
    if not plan.metrics or not any(h.other_column for h in plan.having):
        return plan
    available = {_norm(g) for g in plan.group_by} | {_norm(m.name()) for m in plan.metrics}
    having: list[Condition] = []
    for h in plan.having:
        using = [m for m in plan.metrics if m.column and _norm(m.column) == _norm(h.column)]
        if h.other_column and _norm(h.column) not in available and len(using) == 1:
            steps.append(f"compare {using[0].name()} (the aggregated {h.column}) with {h.other_column}")
            h = h.model_copy(update={"column": using[0].name()})
        having.append(h)
    return plan.model_copy(update={"having": having})


def _words(text: str) -> set[str]:
    """Split text into its lowercase words longer than two characters
    ("Revenue_Target_EUR" -> {"revenue", "target", "eur"}). Used by
    _guess_comparison_column() to compare column names with the question."""
    return {w for w in re.split(r"[^a-z0-9]+", text.casefold()) if len(w) > 2}


def _guess_comparison_column(
    result: pd.DataFrame, cond: Condition, plan: QueryPlan, workbook: Workbook, question: str
) -> str | None:
    """Repair: the model compared a column with itself (e.g. Total Revenue < Total Revenue)
    while a joined sheet is present. Pick the joined numeric column whose name shares the most
    words with the question ("revenue target" -> Revenue_Target_EUR). Only a unique best match
    counts; otherwise the plan goes back to the model."""
    if not plan.join or not question:
        return None
    try:
        other = workbook.sheets[_resolve_sheet(workbook, plan.join.sheet)]
    except PlanError:
        return None
    asked = _words(question)
    scored = sorted(
        (
            (len(_words(c) & asked), c)
            for c in other.columns
            if c != EXCEL_ROW
            and c in result.columns
            and kind_of(other[c]) == "number"
            and _norm(c) != _norm(cond.column)
        ),
        reverse=True,
    )
    if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return scored[0][1]


def _numeric_columns_elsewhere(workbook: Workbook, sheet: str) -> list[str]:
    """List the numeric columns of every sheet except `sheet`. Used in the error message
    for a column compared with itself, as candidates the model can pick from."""
    return [
        c
        for name, d in workbook.sheets.items()
        if name != sheet
        for c in d.columns
        if c != EXCEL_ROW and kind_of(d[c]) == "number"
    ]


WHICH_MOST = re.compile(r"\bwhich\b.*\b(highest|most|largest|biggest|top|best)\b", re.IGNORECASE)
SINGLE_ROW_WORDS = re.compile(
    r"\b(single|individual|one|order|transaction|record|row|deal|line)\b", re.IGNORECASE
)


def _which_has_most_means_total(plan: QueryPlan, question: str, steps: list[str]) -> QueryPlan:
    """'Which category generated the highest revenue?' asks for the category with the largest
    total. Small models plan it as max per category (the largest single row), which answers a
    different question. Applies only when the question does not mention a single order/row."""
    if not (
        plan.group_by
        and len(plan.metrics) == 1
        and plan.metrics[0].agg == "max"
        and WHICH_MOST.search(question or "")
        and not SINGLE_ROW_WORDS.search(question or "")
    ):
        return plan
    metric = plan.metrics[0].model_copy(update={"agg": "sum"})
    steps.append(
        f"used sum instead of max for {metric.column}: 'which ... highest' asks for the "
        "largest total, not the largest single row"
    )
    return plan.model_copy(
        update={
            "metrics": [metric],
            "limit": plan.limit or 1,
            "sort": plan.sort or Sort(by=metric.name(), descending=True),
        }
    )


COMPARE_WORDS = re.compile(r"\b(compare|comparison|versus|vs\.?)\b", re.IGNORECASE)


def _group_compared_values(plan: QueryPlan, question: str, steps: list[str]) -> QueryPlan:
    """'Compare Germany and France' asks for one result per value. If the plan filters on
    several values but does not group, group by that column."""
    if plan.group_by or not plan.metrics or not COMPARE_WORDS.search(question or ""):
        return plan
    for f in plan.filters:
        if f.op == "in" and isinstance(f.value, list) and len(f.value) > 1:
            steps.append(f"grouped by {f.column}: the question compares {f.value}")
            return plan.model_copy(update={"group_by": [f.column]})
    return plan


MATCH_ALL = {"all", "any", "*", "every", "everything", "everyone", "all values"}


def _is_match_all(value: Any) -> bool:
    """True if a value (or any item of a list) is a word meaning "everything", such as
    "all", "any", "*" or "all countries". Used by _drop_match_all_conditions()."""
    if isinstance(value, list):
        return any(_is_match_all(v) for v in value)
    return isinstance(value, str) and (
        value.strip().casefold() in MATCH_ALL or value.strip().casefold().startswith("all ")
    )


def _drop_match_all_conditions(plan: QueryPlan, df: pd.DataFrame, steps: list[str]) -> QueryPlan:
    """Small models express "every country" as Country == 'all'. Such a condition means
    no condition at all; applied literally it matches nothing. A value that really occurs
    in the column (a place called "All Saints") is of course kept."""

    def occurs(column: str, value: Any) -> bool:
        """True if the value (or one of a list) really exists in that column's data."""
        if not _has_col(df, column):
            return False
        values = value if isinstance(value, list) else [value]
        present = set(df[_col(df, column)].dropna().astype(str).str.strip().str.casefold())
        return any(str(v).strip().casefold() in present for v in values)

    def keep(cond: Filter | Condition) -> bool:
        """False (and a note in the steps) for a "match everything" condition whose value
        is not real data; True for every other condition."""
        if (
            getattr(cond, "other_column", None) is None
            and cond.op in ("==", "in", "contains")
            and _is_match_all(cond.value)
            and not occurs(cond.column, cond.value)
        ):
            steps.append(f"ignored {cond.column} {cond.op} {cond.value!r} (means: all values)")
            return False
        return True

    filters = [f for f in plan.filters if keep(f)]
    having = [h for h in plan.having if keep(h)]
    if len(filters) == len(plan.filters) and len(having) == len(plan.having):
        return plan
    return plan.model_copy(update={"filters": filters, "having": having})


def _drop_unused_join(plan: QueryPlan, workbook: Workbook, steps: list[str]) -> QueryPlan:
    """A join only matters if something uses the other sheet's columns (a condition, the
    sort, or the shown columns). Otherwise it just adds noise columns, so skip it."""
    if not plan.join:
        return plan
    try:
        other = workbook.sheets[_resolve_sheet(workbook, plan.join.sheet)]
    except PlanError:
        return plan  # reported properly when the join runs
    joined = {_norm(c) for c in other.columns if c != EXCEL_ROW} - {_norm(plan.join.right_on)}
    used = {_norm(h.column) for h in plan.having} | {
        _norm(h.other_column) for h in plan.having if h.other_column
    }
    used |= {_norm(c) for c in plan.columns}
    if plan.sort:
        used.add(_norm(plan.sort.by))
    self_compared = any(h.other_column and _norm(h.column) == _norm(h.other_column) for h in plan.having)
    if used & joined or self_compared:  # a self-comparison is repaired from the joined columns later
        return plan
    steps.append(f'skipped join with "{plan.join.sheet}" (none of its columns are used)')
    return plan.model_copy(update={"join": None})


def _has_col(df: pd.DataFrame, name: str) -> bool:
    """True if the table has a column matching `name` (tolerantly, as _col() does),
    without raising. Used by the repairs to test for a column before acting."""
    try:
        _col(df, name)
        return True
    except PlanError:
        return False


def _aggregate(df: pd.DataFrame, group_by: list[str], metrics: list[Metric]) -> pd.DataFrame:
    """Compute the metrics, per group if `group_by` is given, otherwise over the whole table.

    With no metrics it counts rows. sum, mean and median are refused on a non-numeric
    column (PlanError). Returns one row per group, or a single row. Called by execute().
    """
    keys = [_col(df, g) for g in group_by]
    metrics = metrics or [Metric(agg="count")]
    named: dict[str, pd.NamedAgg] = {}
    for m in metrics:
        if m.column is None:
            if m.agg != "count":
                raise PlanError(f"Aggregation '{m.agg}' needs a column.")
            named[m.name()] = pd.NamedAgg(column=EXCEL_ROW, aggfunc="count")
            continue
        column = _col(df, m.column)
        if m.agg in NUMERIC_AGGS and kind_of(df[column]) != "number":
            raise PlanError(f"Cannot compute {m.agg} of '{column}': it is not a numeric column.")
        named[m.name()] = pd.NamedAgg(column=column, aggfunc=m.agg)

    if keys:
        return df.groupby(keys, dropna=False).agg(**named).reset_index()
    return pd.DataFrame([{name: df[a.column].agg(a.aggfunc) for name, a in named.items()}])


def _mask(df: pd.DataFrame, column: str, op: str, value: Any) -> pd.Series:
    """Build the True/False column that says which rows satisfy `column <op> value`.

    Text is compared ignoring case and surrounding spaces; numbers and dates are converted
    first; empty cells never match. Usage: `df[_mask(df, "Country", "==", "Germany")]`.
    Called by execute() for each filter and by _condition_mask().
    """
    col = _col(df, column)
    series = df[col]
    kind = kind_of(series)

    if op in ("in", "not in"):
        values = value if isinstance(value, list) else [value]
        if kind == "text":
            wanted = {str(v).strip().casefold() for v in values}
            hit = series.astype("string").str.strip().str.casefold().isin(wanted).fillna(False)
        else:
            hit = series.isin([_coerce_value(v, kind, col) for v in values])
        return hit if op == "in" else ~hit

    if op == "contains":
        return series.astype("string").str.contains(str(value), case=False, regex=False).fillna(False)

    if kind == "text":
        left = series.astype("string").str.strip().str.casefold()
        right: Any = str(value).strip().casefold()
    else:
        left, right = series, _coerce_value(value, kind, col)
    return _compare(left, op, right).fillna(False).astype(bool)


def _condition_mask(df: pd.DataFrame, cond: Condition) -> pd.Series:
    """True/False column for a 'having' condition: compares a column with a value, or,
    when `other_column` is set, two columns row by row (only the six comparison
    operators are allowed there). Called by execute()."""
    if cond.other_column is None:
        return _mask(df, cond.column, cond.op, cond.value)
    left, right = df[_col(df, cond.column)], df[_col(df, cond.other_column)]
    if cond.op not in ("==", "!=", ">", ">=", "<", "<="):
        raise PlanError(f"Operator '{cond.op}' cannot compare two columns.")
    return _compare(left, cond.op, right).fillna(False).astype(bool)


def _compare(left: Any, op: str, right: Any) -> pd.Series:
    """Apply a comparison operator given as text ("==", "<", ...) to two operands.

    A dictionary maps each allowed operator to a function, so no model output is ever
    evaluated as code. Used by _mask() and _condition_mask().
    """
    return {
        "==": lambda: left == right,
        "!=": lambda: left != right,
        ">": lambda: left > right,
        ">=": lambda: left >= right,
        "<": lambda: left < right,
        "<=": lambda: left <= right,
    }[op]()


def _coerce_value(value: Any, kind: str, column: str) -> Any:
    """Convert a plan value to the type of the column it is compared with: a float for a
    number column ("€1,200" -> 1200.0), a Timestamp for a date column, unchanged otherwise.
    Raises PlanError if the value cannot be converted. Used by _mask()."""
    try:
        if kind == "number":
            return float(str(value).replace(",", "").replace("€", "").strip())
        if kind == "date":
            return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PlanError(f"Value {value!r} is not a valid {kind} for column '{column}'.") from exc
    return value


LOCATION_WORDS = re.compile(
    r"\b(sheets?|worksheets?|tabs?|columns?|fields?|where|contains?|containing)\b", re.IGNORECASE
)
LIST_WORDS = re.compile(
    r"\b(different|distinct|unique|list|all|show|name|what\s+are|which\s+are)\b", re.IGNORECASE
)


def _list_values_not_columns(
    plan: QueryPlan, workbook: Workbook, question: str, notes: list[str]
) -> QueryPlan:
    """Repair: 'What are the different products?' planned as a column-name search.

    The question asks for the values of a column, not for where the column is. If it does
    not mention sheets or columns, and every word of a column's name occurs in it ("product
    categories" -> Product_Category, "products" -> Product), list that column's values
    instead. The column with the most words wins, so "product categories" is not read as
    Product. With the column in several sheets, the sheet where it is unique per row (a
    lookup table such as a catalog) is used, otherwise the one with the most distinct
    values. Anything unclear stays a column search.
    """
    if not question or LOCATION_WORDS.search(question) or not LIST_WORDS.search(question):
        return plan

    # "What are the prices for products?": the model named the columns it wants (Product,
    # List_Price_EUR) but called it a column search. List them from the sheet that has them all.
    named = [c for c in plan.columns if c.strip()]
    having_all = [name for name, d in workbook.sheets.items() if all(_has_col(d, c) for c in named)]
    if len(named) > 1 and having_all:
        chosen = next((s for s in having_all if _norm(s) == _norm(plan.sheet or "")), having_all[0])
        columns = [_col(workbook.sheets[chosen], c) for c in named]
        notes.append(f"listed {columns}: the question asks for their values, not for where they are")
        return QueryPlan(sheet=chosen, columns=columns)

    asked = set(re.findall(r"[a-z0-9]+", question.casefold()))
    asked |= {w[:-1] for w in asked if w.endswith("s")}  # products -> product
    asked |= {w[:-3] + "y" for w in asked if w.endswith("ies")}  # categories -> category

    def name_words(name: str) -> set[str]:
        """The words of a column name: "Product_Category" -> {"product", "category"}."""
        return set(re.findall(r"[a-z0-9]+", name.casefold()))

    found = [
        (p, c)
        for p in workbook.profiles.values()
        for c in p.columns
        if c.kind == "text" and name_words(c.name) and name_words(c.name) <= asked
    ]
    most = max((len(name_words(c.name)) for _, c in found), default=0)
    found = [(p, c) for p, c in found if len(name_words(c.name)) == most]
    if not found or len({_norm(c.name) for _, c in found}) != 1:
        return plan
    lookups = [(p, c) for p, c in found if c.distinct == p.rows]
    profile, column = lookups[0] if len(lookups) == 1 else max(found, key=lambda pc: pc[1].distinct)
    one_row_per_value = column.distinct == profile.rows

    # Other words of the question that name a column of the same sheet are shown next to it
    # ("prices for products" -> Product, List_Price_EUR). A word that only matches a column of
    # another sheet makes the question unclear, so it stays a column search.
    extras: list[str] = []
    for word in sorted(asked - name_words(column.name) - STOP_WORDS):
        if len(word) < 3 or LIST_WORDS.fullmatch(word):  # "list" is not the List_Price column
            continue
        here = [c.name for c in profile.columns if c is not column and word in name_words(c.name)]
        others = [p for p in workbook.profiles.values() if p is not profile]
        if here:
            extras += [name for name in here if name not in extras]
        elif any(word in name_words(c.name) for p in others for c in p.columns):
            return plan
    if extras and not one_row_per_value:
        return plan

    shown = [column.name, *extras]
    notes.append(
        f"listed the values of {', '.join(shown)}: the question asks for them, not for where they are"
    )
    if one_row_per_value:  # list the rows, which cites them
        return QueryPlan(sheet=profile.name, columns=shown)
    return QueryPlan(sheet=profile.name, group_by=[column.name])  # distinct values with row counts


STOP_WORDS = {
    "which",
    "what",
    "where",
    "sheet",
    "sheets",
    "worksheet",
    "worksheets",
    "column",
    "columns",
    "contains",
    "contain",
    "information",
    "about",
    "data",
    "does",
    "the",
    "is",
    "are",
    "has",
    "have",
    "with",
    "for",
    "and",
    "any",
    "there",
    "a",
    "an",
    "of",
}


def _find_columns(workbook: Workbook, keywords: list[str], question: str = "") -> QueryResult:
    """Answer "which sheet contains ...?": list every (sheet, column) whose column name
    contains one of the keywords.

    If the model gave no keywords, the question's own words (minus STOP_WORDS) are used.
    Searches column names only, not the data; no match raises PlanError. Called by
    execute() for intent "find_columns".
    """
    words = [k.casefold() for k in keywords if k.strip()]
    if not words:  # the model chose find_columns but gave no keywords: use the question's words
        words = [w for w in re.findall(r"[a-z0-9]{3,}", question.casefold()) if w not in STOP_WORDS]
        keywords = words
    if not words:
        raise PlanError("find_columns needs at least one keyword.")
    rows = [
        {"sheet": p.name, "column": c.name}
        for p in workbook.profiles.values()
        for c in p.columns
        if any(w in c.name.casefold() for w in words)
    ]
    if not rows:
        raise PlanError(
            f"No column name contains {keywords}. find_columns searches column names only; "
            "to look for values in the data (e.g. a country), use intent 'query' with filters."
        )
    sheets = sorted({r["sheet"] for r in rows})
    sources = [Source(s, workbook.profiles[s].rows) for s in sheets]
    return QueryResult(["sheet", "column"], rows, len(rows), sources, [f"search column names for {keywords}"])


# --- name resolution ---------------------------------------------------------------------


def _norm(name: str) -> str:
    """Reduce a name to lowercase letters and digits, so "Revenue_EUR", "revenue eur" and
    "RevenueEUR" all compare equal. Used wherever sheet and column names are matched."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _col(df: pd.DataFrame, name: str) -> str:
    """Resolve a column name tolerantly (case, spaces, underscores); raise PlanError if absent.

    Returns the column's real name in `df`. The error lists the available columns, which is
    what lets the model correct itself on the retry. Usage: `df[_col(df, "revenue eur")]`.
    """
    visible = [c for c in df.columns if c != EXCEL_ROW]
    if name in visible:
        return name
    for candidate in visible:
        if _norm(candidate) == _norm(name):
            return candidate
    raise PlanError(f"Column '{name}' does not exist here. Available columns: {visible}")


def _resolve_sheet(workbook: Workbook, name: str | None) -> str:
    """Return the real name of the sheet the plan refers to, matched tolerantly.

    With no name given, a workbook that has exactly one sheet uses that sheet. Otherwise
    raises PlanError listing the available sheets. Used by execute() and the join repairs.
    """
    names = list(workbook.sheets)
    if not name:
        if len(names) == 1:
            return names[0]
        raise PlanError(f"The plan must name a sheet. Available sheets: {names}")
    for candidate in names:
        if candidate == name or _norm(candidate) == _norm(name):
            return candidate
    raise PlanError(f"Sheet '{name}' does not exist. Available sheets: {names}")


def _clean(value: Any) -> Any:
    """Make one result value JSON-friendly: empty values become None, pandas/numpy types
    become plain Python, and floats are rounded to 2 decimals. Used by execute() on every
    cell of the returned rows."""
    if value is None or (not isinstance(value, (list, dict, str)) and pd.isna(value)):
        return None
    value = _plain(value)
    return round(value, 2) if isinstance(value, float) else value
