from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.models.plan import QueryPlan
from src.services.executor import execute
from src.services.pipeline import _answer_prompt
from src.services.workbook import WorkbookError, load_workbook, profile_as_text


def to_xlsx(sheets: dict[str, pd.DataFrame], startrow: int = 0) -> bytes:
    """Write DataFrames to an in-memory .xlsx file and return its bytes, one sheet per
    entry. `startrow` leaves that many blank rows above each table."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False, startrow=startrow)
    return buf.getvalue()


def test_loads_all_sheets_with_types(workbook):
    assert list(workbook.sheets) == ["Sales_Data", "Targets", "Product_Catalog"]
    kinds = {c.name: c.kind for c in workbook.profiles["Sales_Data"].columns}
    assert kinds["Revenue_EUR"] == "number"
    assert kinds["Order_Date"] == "date"
    assert kinds["Country"] == "text"


def test_detects_header_below_a_title_block():
    """Real workbooks often have a title and a blank line above the table."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        pd.DataFrame([["Quarterly report"], [None]]).to_excel(
            writer, sheet_name="R", index=False, header=False
        )
        pd.DataFrame({"Region": ["North", "South"], "Sales": [10, 20]}).to_excel(
            writer, sheet_name="R", index=False, startrow=2
        )
    wb = load_workbook(buf.getvalue(), "titled.xlsx")
    assert wb.profiles["R"].header_row == 3
    assert [c.name for c in wb.profiles["R"].columns] == ["Region", "Sales"]
    assert (
        execute(wb, QueryPlan(sheet="R", metrics=[{"agg": "sum", "column": "Sales"}])).rows[0]["sum_Sales"]
        == 30
    )


def test_empty_values_and_blank_rows_are_handled():
    df = pd.DataFrame({"Name": ["a", None, None, "d"], "Amount": [1.0, 2.0, np.nan, 4.0]})
    wb = load_workbook(to_xlsx({"S": df}, startrow=2), "gaps.xlsx")
    assert wb.profiles["S"].rows == 3  # the fully blank row is dropped
    assert wb.profiles["S"].header_row == 3
    result = execute(wb, QueryPlan(sheet="S", metrics=[{"agg": "sum", "column": "Amount"}]))
    assert result.rows[0]["sum_Amount"] == 7.0


def test_rejects_file_that_is_not_a_workbook():
    with pytest.raises(WorkbookError):
        load_workbook(b"definitely not excel", "fake.xlsx")


def test_planning_prompt_size_does_not_grow_with_row_count():
    """The core cost constraint: the LLM sees a profile, not the data."""
    rng = np.random.default_rng(0)

    def schema_for(n: int) -> str:
        df = pd.DataFrame(
            {
                "Country": rng.choice(["DE", "FR", "NL"], n),
                "Revenue": rng.uniform(0, 1000, n).round(2),
                "OrderID": [f"O-{i}" for i in range(n)],
            }
        )
        return profile_as_text(load_workbook(to_xlsx({"Sales": df}), "big.xlsx").profiles)

    small, large = schema_for(100), schema_for(20_000)
    assert len(large) < 1_000
    assert abs(len(large) - len(small)) < 50  # only the row count / ranges change
    assert "O-19999" not in large  # high-cardinality values are summarised, not listed


def test_answer_prompt_caps_rows_sent_to_llm(workbook):
    result = execute(workbook, QueryPlan(sheet="Sales_Data"))  # all 120 rows
    prompt = _answer_prompt("list everything", result, max_rows=20)
    assert "120 rows, first 20 shown" in prompt
    assert prompt.count('"Order_ID"') == 20
