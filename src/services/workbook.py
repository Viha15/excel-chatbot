"""Loading an .xlsx workbook into DataFrames and building a compact schema profile.

The profile is the ONLY view of the workbook the LLM gets when planning a query:
sheet names, row counts, column names and types, value ranges, the distinct values of
low-cardinality text columns, and optional sample rows (none by default). Its size depends on the number of
columns, not the number of rows, so a 1-million-row sheet costs the same tokens as a
100-row one.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.services.guardrails import safe_profile_value

EXCEL_ROW = "_excel_row"  # hidden column: original row number in the sheet, used for citations
HEADER_SCAN_ROWS = 10


class WorkbookError(ValueError):
    """The file could not be read as a usable workbook."""


@dataclass
class ColumnProfile:
    name: str
    kind: str  # number | date | text | bool
    non_null: int
    distinct: int
    values: list[Any] | None = None  # distinct values, only for low-cardinality text columns
    min: Any = None
    max: Any = None


@dataclass
class SheetProfile:
    name: str
    rows: int
    header_row: int  # 1-based Excel row number of the detected header
    columns: list[ColumnProfile]
    sample: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Workbook:
    filename: str
    sheets: dict[str, pd.DataFrame]
    profiles: dict[str, SheetProfile]


def load_workbook(
    content: bytes, filename: str, max_distinct_values: int = 25, sample_rows: int = 0
) -> Workbook:
    """Read every sheet of an .xlsx file into a clean table and build its schema profile.

    `content` is the raw file bytes. Empty sheets are skipped; a file that cannot be read,
    or that has no non-empty sheet, raises WorkbookError. Called by the /upload endpoint
    (and by the evaluation script and the tests).
    """
    try:
        raw_sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    except Exception as exc:  # openpyxl raises several unrelated exception types
        raise WorkbookError(f"Could not read '{filename}' as an .xlsx workbook: {exc}") from exc

    sheets: dict[str, pd.DataFrame] = {}
    profiles: dict[str, SheetProfile] = {}
    for name, raw in raw_sheets.items():
        df, header_row = _to_table(raw)
        if df is None:
            continue  # empty sheet
        sheets[name] = df
        profiles[name] = _profile(name, df, header_row, max_distinct_values, sample_rows)

    if not sheets:
        raise WorkbookError(f"'{filename}' contains no non-empty worksheets.")
    return Workbook(filename=filename, sheets=sheets, profiles=profiles)


def _to_table(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, int]:
    """Find the header row, name the columns, drop empty rows/columns, fix dtypes.

    Also adds the hidden EXCEL_ROW column with each row's original Excel row number.
    Returns (table, Excel row number of the header), or (None, 0) for an empty sheet.
    Called by load_workbook() for each sheet.
    """
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return None, 0

    header_idx = _detect_header(raw)
    header = raw.loc[header_idx]
    body = raw.loc[raw.index > header_idx].copy()
    body.columns = _column_names(header)
    body = body.dropna(axis=1, how="all")
    # Excel row number = pandas index (0-based, from the raw sheet) + 1
    body[EXCEL_ROW] = body.index + 1
    body = body.reset_index(drop=True)

    for col in body.columns:
        if col != EXCEL_ROW:
            body[col] = _coerce(body[col])
    return body, int(header_idx) + 1


def _detect_header(raw: pd.DataFrame) -> Any:
    """First all-text row (within the first few) that is about as wide as the table itself.

    Comparing against the widest nearby row, rather than the sheet width, stops a one-cell
    title above a narrow table from being taken as the header.
    """
    scan = raw.loc[raw.index[: HEADER_SCAN_ROWS * 2]]
    widest = int(scan.notna().sum(axis=1).max())
    needed = max(min(2, widest), math.ceil(0.8 * widest))
    for idx in raw.index[:HEADER_SCAN_ROWS]:
        values = raw.loc[idx].dropna()
        if len(values) >= needed and all(isinstance(v, str) for v in values):
            return idx
    return raw.index[0]


def _column_names(header: pd.Series) -> list[str]:
    """Turn the header row into a list of unique column names.

    An empty header cell becomes "Column_<position>"; a duplicate gets a "_2", "_3" suffix.
    Called by _to_table().
    """
    names: list[str] = []
    for i, value in enumerate(header):
        name = str(value).strip() if pd.notna(value) and str(value).strip() else f"Column_{i + 1}"
        base, n = name, 2
        while name in names:
            name, n = f"{base}_{n}", n + 1
        names.append(name)
    return names


def _coerce(series: pd.Series) -> pd.Series:
    """Read with header=None, every column starts as object. Recover numbers and dates.

    Tries bool, then number, then date, and falls back to text. The rule is all-or-nothing:
    one non-numeric cell keeps the whole column as text. Called by _to_table() per column.
    """
    non_null = series.dropna()
    if non_null.empty:
        return series
    if all(isinstance(v, bool) for v in non_null):
        return series.astype("boolean")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(non_null):
        return numeric
    if all(isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat") for v in non_null):
        dates = pd.to_datetime(series, errors="coerce")
        if dates.notna().sum() == len(non_null):
            return dates
    return series.astype("string")


def kind_of(series: pd.Series) -> str:
    """Return the simple type of a column: "bool", "number", "date" or "text".

    Used for the schema profile and by the executor to decide how to compare and aggregate.
    """
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "text"


def _profile(
    name: str, df: pd.DataFrame, header_row: int, max_distinct: int, sample_rows: int
) -> SheetProfile:
    """Build the description of one sheet that the LLM sees instead of the data.

    Per column: type, non-empty and distinct counts, plus either the distinct values (text
    columns with at most `max_distinct` of them) or the min/max (number and date columns).
    `sample_rows` example rows are added if asked for. Called by load_workbook().
    """
    columns: list[ColumnProfile] = []
    for col in df.columns:
        if col == EXCEL_ROW:
            continue
        s = df[col]
        kind = kind_of(s)
        non_null = s.dropna()
        profile = ColumnProfile(
            name=col, kind=kind, non_null=int(non_null.size), distinct=int(non_null.nunique())
        )
        if kind == "text" and 0 < profile.distinct <= max_distinct:
            shown = (safe_profile_value(v) for v in sorted(non_null.unique().tolist()))
            profile.values = [v for v in shown if v is not None]
        elif kind in ("number", "date") and not non_null.empty:
            profile.min, profile.max = _plain(non_null.min()), _plain(non_null.max())
        columns.append(profile)

    visible = df.drop(columns=[EXCEL_ROW])
    sample = [
        {k: _plain(v) for k, v in row.items() if pd.notna(v)}
        for row in visible.head(sample_rows).to_dict("records")
    ]
    return SheetProfile(name=name, rows=len(df), header_row=header_row, columns=columns, sample=sample)


def _plain(value: Any) -> Any:
    """numpy / pandas scalars -> JSON-friendly Python values.

    Timestamps become ISO text (date only when there is no time part) and floats are rounded
    to 4 decimals. Used for the profile and by the executor when cleaning result values.
    """
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def profile_as_text(profiles: dict[str, SheetProfile]) -> str:
    """Render the schema profile for the planning prompt, as compactly as possible.

    One line per column: name, type, then a range or the distinct values joined with "|".
    A value list already shown for a same-named column in an earlier sheet is referenced
    instead of repeated (e.g. Country in both Sales and Targets).

    Called by pipeline.answer_question() for every question; the returned text is the only
    description of the workbook the model receives.
    """
    lines: list[str] = []
    shown: dict[str, tuple[str, list[Any]]] = {}  # column name -> (sheet, values)
    for p in profiles.values():
        lines.append(f"{p.name} ({p.rows} rows)")
        for c in p.columns:
            detail = f" {c.name} {c.kind}"
            if c.values is not None:
                earlier = shown.get(c.name)
                if earlier and earlier[1] == c.values:
                    detail += f" (values as {earlier[0]})"
                else:
                    detail += ": " + "|".join(map(str, c.values))
                    shown[c.name] = (p.name, c.values)
            elif c.min is not None:
                detail += f" {c.min}..{c.max}"
            elif c.kind == "text":
                detail += f", {c.distinct} distinct"
            if c.non_null < p.rows:
                detail += f", {p.rows - c.non_null} empty"
            lines.append(detail)
        for row in p.sample:
            lines.append(f" e.g. {row}")
    return "\n".join(lines)
