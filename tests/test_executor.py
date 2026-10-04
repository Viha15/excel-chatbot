"""The ten example questions from the brief, expressed as the plans the LLM should produce.

Expected values are computed directly with pandas from the raw file, so these tests check
the executor's arithmetic, not a copy of it."""

from __future__ import annotations

import pytest

from src.models.plan import QueryPlan
from src.services.executor import PlanError, execute


def run(workbook, **plan):
    """Build a QueryPlan from keyword arguments and execute it against the workbook.
    Usage: `run(workbook, sheet="Sales_Data", metrics=[{"agg": "count"}])`."""
    return execute(workbook, QueryPlan(**plan))


def test_total_revenue_for_germany(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        filters=[{"column": "Country", "op": "==", "value": "Germany"}],
        metrics=[{"agg": "sum", "column": "Revenue_EUR"}],
    )
    expected = sales.loc[sales.Country == "Germany", "Revenue_EUR"].sum()
    assert result.rows[0]["sum_Revenue_EUR"] == pytest.approx(expected)
    assert result.sources[0].rows_matched == (sales.Country == "Germany").sum()


def test_show_all_records_for_germany_cites_excel_rows(workbook, sales):
    result = run(
        workbook, sheet="Sales_Data", filters=[{"column": "Country", "op": "==", "value": "germany"}]
    )  # case-insensitive
    german = sales[sales.Country == "Germany"]
    assert result.total_rows == len(german)
    assert {r["Order_ID"] for r in result.rows} == set(german.Order_ID)
    # header is Excel row 1, so pandas index i is Excel row i + 2
    assert result.sources[0].excel_rows[:3] == [i + 2 for i in german.index[:3]]


def test_average_revenue_by_country(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Country"],
        metrics=[{"agg": "mean", "column": "Revenue_EUR", "alias": "avg_revenue"}],
    )
    got = {r["Country"]: r["avg_revenue"] for r in result.rows}
    expected = sales.groupby("Country").Revenue_EUR.mean().round(2).to_dict()
    assert got == pytest.approx(expected)


def test_country_with_highest_total_revenue(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Country"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}],
        sort={"by": "revenue", "descending": True},
        limit=1,
    )
    assert result.rows[0]["Country"] == sales.groupby("Country").Revenue_EUR.sum().idxmax()


def test_count_records_with_revenue_over_10000(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        filters=[{"column": "Revenue_EUR", "op": ">", "value": "10,000"}],
        metrics=[{"agg": "count"}],
    )
    assert result.rows[0]["row_count"] == (sales.Revenue_EUR > 10000).sum()


def test_compare_germany_and_france(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        filters=[{"column": "Country", "op": "in", "value": ["Germany", "France"]}],
        group_by=["Country"],
        metrics=[
            {"agg": "sum", "column": "Revenue_EUR"},
            {"agg": "sum", "column": "Profit_EUR"},
            {"agg": "count"},
        ],
    )
    assert sorted(r["Country"] for r in result.rows) == ["France", "Germany"]
    france = next(r for r in result.rows if r["Country"] == "France")
    assert france["row_count"] == (sales.Country == "France").sum()


def test_product_category_with_highest_revenue(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Product_Category"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR"}],
        sort={"by": "sum_Revenue_EUR"},
        limit=1,
    )
    assert result.rows[0]["Product_Category"] == sales.groupby("Product_Category").Revenue_EUR.sum().idxmax()


def test_which_worksheet_contains_revenue(workbook):
    result = run(workbook, intent="find_columns", keywords=["revenue"])
    assert {r["sheet"] for r in result.rows} == {"Sales_Data", "Targets"}


def test_total_revenue_and_profit_by_country(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Country"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR"}, {"agg": "sum", "column": "Profit_EUR"}],
    )
    assert result.total_rows == sales.Country.nunique()
    assert sum(r["sum_Profit_EUR"] for r in result.rows) == pytest.approx(sales.Profit_EUR.sum(), abs=0.05)


def test_countries_below_revenue_target(workbook, sales, targets):
    """Aggregate one sheet, join another, compare two columns."""
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Country"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}],
        join={"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
        having=[{"column": "revenue", "op": "<", "other_column": "Revenue_Target_EUR"}],
    )
    actual = sales.groupby("Country").Revenue_EUR.sum()
    target = targets.set_index("Country").Revenue_Target_EUR
    expected = {c for c in actual.index if actual[c] < target[c]}
    assert {r["Country"] for r in result.rows} == expected
    assert [s.sheet for s in result.sources] == ["Sales_Data", "Targets"]


# --- validation --------------------------------------------------------------------------


def test_unknown_column_error_lists_available_columns(workbook):
    with pytest.raises(PlanError, match="Available columns"):
        run(workbook, sheet="Sales_Data", metrics=[{"agg": "sum", "column": "Turnover"}])


def test_unknown_sheet_is_rejected(workbook):
    with pytest.raises(PlanError, match="does not exist"):
        run(workbook, sheet="Invoices", metrics=[{"agg": "count"}])


def test_sum_of_text_column_is_rejected(workbook):
    with pytest.raises(PlanError, match="not a numeric column"):
        run(workbook, sheet="Sales_Data", metrics=[{"agg": "sum", "column": "Country"}])


def test_column_names_are_resolved_tolerantly(workbook):
    result = run(workbook, sheet="sales data", metrics=[{"agg": "max", "column": "revenue eur"}])
    assert result.rows[0]["max_revenue eur"] == 18000.0


def test_filter_on_nonexistent_value_returns_empty_result(workbook):
    result = run(workbook, sheet="Sales_Data", filters=[{"column": "Country", "op": "==", "value": "Italy"}])
    assert result.total_rows == 0


def test_having_on_aggregated_values_is_left_after_aggregation(workbook, sales):
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Country"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}],
        having=[{"column": "revenue", "op": ">", "value": 120000}],
    )
    totals = sales.groupby("Country").Revenue_EUR.sum()
    assert {r["Country"] for r in result.rows} == set(totals[totals > 120000].index)
    assert not any("moved" in step for step in result.steps)


def test_unknown_having_column_error_explains_filters_vs_having(workbook):
    with pytest.raises(PlanError, match="belong in 'filters'"):
        run(
            workbook,
            sheet="Sales_Data",
            group_by=["Country"],
            metrics=[{"agg": "sum", "column": "Revenue_EUR"}],
            having=[{"column": "Margin", "op": ">", "value": 1}],
        )


def test_match_all_placeholder_and_unused_join_are_ignored(workbook, sales):
    """Seen with llama3.2 for 'total revenue and profit by country'."""
    result = run(
        workbook,
        sheet="Sales_Data",
        filters=[{"column": "Country", "op": "==", "value": "all"}],
        group_by=["Country"],
        metrics=[
            {"agg": "sum", "column": "Revenue_EUR", "alias": "Total Revenue"},
            {"agg": "sum", "column": "Profit_EUR", "alias": "Total Profit"},
        ],
        join={"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
        having=[{"column": "Country", "op": "==", "value": "all"}],
        sort={"by": "Total Revenue", "descending": True},
    )
    assert result.columns == ["Country", "Total Revenue", "Total Profit"]  # no target columns
    assert result.rows[0]["Country"] == sales.groupby("Country").Revenue_EUR.sum().idxmax()
    assert result.total_rows == sales.Country.nunique()
    assert any("skipped join" in s for s in result.steps)


def test_join_is_kept_when_its_columns_are_used(workbook):
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Country"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}],
        join={"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
        having=[{"column": "revenue", "op": "<", "other_column": "Revenue_Target_EUR"}],
    )
    assert "Revenue_Target_EUR" in result.columns


def test_value_that_really_exists_is_not_treated_as_match_all():
    import io

    import pandas as pd

    from src.services.workbook import load_workbook

    buf = io.BytesIO()
    pd.DataFrame({"Parish": ["All Saints", "St Mary"], "Visitors": [5, 7]}).to_excel(buf, index=False)
    wb = load_workbook(buf.getvalue(), "p.xlsx")
    result = execute(
        wb,
        QueryPlan(
            sheet="Sheet1",
            filters=[{"column": "Parish", "op": "==", "value": "All Saints"}],
            metrics=[{"agg": "sum", "column": "Visitors"}],
        ),
    )
    assert result.rows[0]["sum_Visitors"] == 5


def test_two_equalities_on_one_column_mean_either_value_and_keep_both(workbook, sales):
    """Seen with llama3.2 for 'compare Germany and France': == Germany AND == France, limit 1."""
    result = run(
        workbook,
        sheet="Sales_Data",
        filters=[
            {"column": "Country", "op": "==", "value": "Germany"},
            {"column": "Country", "op": "==", "value": "France"},
        ],
        group_by=["Country"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR"}],
        sort={"by": "sum_Revenue_EUR"},
        limit=1,
    )
    assert sorted(r["Country"] for r in result.rows) == ["France", "Germany"]


def test_raw_column_compared_to_target_uses_its_aggregate(workbook, sales, targets):
    """Seen with llama3.2: having Revenue_EUR < other_column Revenue_Target_EUR."""
    result = run(
        workbook,
        sheet="Sales_Data",
        group_by=["Country"],
        metrics=[{"agg": "sum", "column": "Revenue_EUR", "alias": "Total Revenue"}],
        join={"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
        having=[{"column": "Revenue_EUR", "op": "<", "other_column": "Revenue_Target_EUR"}],
    )
    assert {r["Country"] for r in result.rows} == {"France", "Germany"}


def test_compare_question_groups_by_the_compared_values(workbook):
    """Seen with llama3.2: 'Compare Germany and France' planned as one combined total."""
    result = execute(
        workbook,
        QueryPlan(
            sheet="Sales_Data",
            filters=[{"column": "Country", "op": "in", "value": ["Germany", "France"]}],
            metrics=[{"agg": "sum", "column": "Revenue_EUR"}],
        ),
        question="Compare Germany and France.",
    )
    assert sorted(r["Country"] for r in result.rows) == ["France", "Germany"]
    assert result.plan.group_by == ["Country"]  # 'Computed as' describes the executed plan


def test_comparing_a_column_with_itself_is_rejected_with_candidates(workbook):
    """Seen with llama3.2: having Revenue_EUR < other_column 'Total Revenue' (its own sum)."""
    with pytest.raises(PlanError, match="with itself.*Revenue_Target_EUR"):
        run(
            workbook,
            sheet="Sales_Data",
            group_by=["Country"],
            metrics=[{"agg": "sum", "column": "Revenue_EUR", "alias": "Total Revenue"}],
            join={"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
            having=[{"column": "Revenue_EUR", "op": "<", "other_column": "Total Revenue"}],
        )


def test_find_columns_without_keywords_uses_the_question(workbook):
    result = execute(
        workbook,
        QueryPlan(intent="find_columns"),
        question="Which worksheet contains information about revenue?",
    )
    assert {r["sheet"] for r in result.rows} == {"Sales_Data", "Targets"}


def test_find_columns_for_data_values_asks_for_a_query_instead(workbook):
    """Seen with llama3.2: 'Compare Germany and France' planned as a column-name search."""
    with pytest.raises(PlanError, match="use intent 'query'"):
        run(workbook, intent="find_columns", keywords=["Germany", "France"])


def test_question_about_values_planned_as_column_search_lists_the_values(workbook):
    """Seen with llama3.2: 'what are the different products?' planned as find_columns [Product]."""
    import pandas as pd

    from tests.conftest import SAMPLE

    catalog = pd.read_excel(SAMPLE, sheet_name="Product_Catalog")
    result = execute(
        workbook,
        QueryPlan(intent="find_columns", keywords=["Product"]),
        question="what are the different products?",
    )
    assert {r["Product"] for r in result.rows} == set(catalog.Product)
    assert result.sources[0].sheet == "Product_Catalog"  # the lookup table, one row per product
    assert any("listed the values of Product" in s for s in result.steps)


def test_two_word_column_is_preferred_over_the_one_word_column_it_contains(workbook, sales):
    """'product categories' means Product_Category, not Product (the model's keyword)."""
    result = execute(
        workbook,
        QueryPlan(intent="find_columns", keywords=["Product"]),
        question="what are the different product categories",
    )
    assert "Product" not in result.columns
    assert sorted(r["Product_Category"] for r in result.rows) == sorted(sales.Product_Category.unique())


def test_columns_named_by_the_model_are_listed_from_the_sheet_that_has_them(workbook):
    """Exact plan of llama3.2 for 'what are the prices for products': the right columns, but
    as a column search on the wrong sheet with a stray metric."""
    import pandas as pd

    from tests.conftest import SAMPLE

    catalog = pd.read_excel(SAMPLE, sheet_name="Product_Catalog")
    plan = QueryPlan.model_validate(
        {
            "intent": "find_columns",
            "sheet": "Sales_Data",
            "metrics": [{"agg": "sum", "column": "List_Price_EUR", "alias": "Total List Price"}],
            "columns": ["Product_ID", "Product", "List_Price_EUR"],
        }
    )
    result = execute(workbook, plan, question="what are the prices for products")
    assert result.columns == ["Product_ID", "Product", "List_Price_EUR"]
    assert {r["Product"]: r["List_Price_EUR"] for r in result.rows} == dict(
        zip(catalog.Product, catalog.List_Price_EUR, strict=True)
    )
    assert result.sources[0].sheet == "Product_Catalog"


def test_other_columns_named_in_the_question_are_shown_next_to_the_values(workbook):
    """Exact plan of llama3.1 for 'list products with their prices': keywords only."""
    result = execute(
        workbook,
        QueryPlan(intent="find_columns", keywords=["product", "price"]),
        question="list products with their prices",
    )
    assert result.columns == ["Product", "List_Price_EUR"]
    assert result.total_rows == 8 and result.sources[0].sheet == "Product_Catalog"


def test_plural_keyword_matches_the_column_and_each_value_is_listed_once(workbook, sales):
    result = execute(
        workbook,
        QueryPlan(intent="find_columns", keywords=["countries"]),
        question="List the different countries",
    )
    assert sorted(r["Country"] for r in result.rows) == sorted(sales.Country.unique())


def test_values_of_a_repeated_column_are_grouped_with_their_row_counts():
    import io

    import pandas as pd

    from src.services.workbook import load_workbook

    buf = io.BytesIO()
    pd.DataFrame({"Region": ["North", "South", "North"], "Sales": [1, 2, 3]}).to_excel(buf, index=False)
    wb = load_workbook(buf.getvalue(), "r.xlsx")
    result = execute(wb, QueryPlan(intent="find_columns"), question="What are the different regions?")
    assert {r["Region"]: r["row_count"] for r in result.rows} == {"North": 2, "South": 1}


def test_question_about_where_data_is_stays_a_column_search(workbook):
    result = execute(
        workbook,
        QueryPlan(intent="find_columns", keywords=["Product"]),
        question="Which sheets contain the different products?",
    )
    assert result.columns == ["sheet", "column"]


def test_which_has_highest_planned_with_max_uses_the_total(workbook, sales):
    """Seen with llama3.2 for 'which product category generated the highest revenue'."""
    cats = sales.groupby("Product_Category").Revenue_EUR.sum()
    result = execute(
        workbook,
        QueryPlan(
            sheet="Sales_Data",
            group_by=["Product_Category"],
            metrics=[{"agg": "max", "column": "Revenue_EUR", "alias": "Highest Revenue"}],
        ),
        question="Which product category generated the highest revenue?",
    )
    assert result.total_rows == 1
    assert result.rows[0]["Product_Category"] == cats.idxmax()
    assert result.rows[0]["Highest Revenue"] == pytest.approx(cats.max())


def test_max_is_kept_when_the_question_is_about_a_single_order(workbook, sales):
    result = execute(
        workbook,
        QueryPlan(
            sheet="Sales_Data",
            group_by=["Product_Category"],
            metrics=[{"agg": "max", "column": "Revenue_EUR"}],
        ),
        question="Which product category has the highest single order?",
    )
    assert result.plan.metrics[0].agg == "max"


def test_self_comparison_is_repaired_from_the_question(workbook):
    """Exact plan llama3.2 wrote first for 'which countries are below their revenue target'."""
    plan = QueryPlan.model_validate(
        {
            "intent": "query",
            "sheet": "Sales_Data",
            "filters": [
                {
                    "column": "Country",
                    "op": "in",
                    "value": ["Belgium", "France", "Germany", "Netherlands", "Spain"],
                }
            ],
            "group_by": ["Country"],
            "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "Total Revenue"}],
            "join": {"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
            "having": [{"column": "Revenue_EUR", "op": "<", "other_column": "Total Revenue", "value": 95000}],
            "columns": ["Country"],
            "sort": {"by": "Total Revenue", "descending": True},
            "limit": 10,
        }
    )
    result = execute(workbook, plan, question="Which countries are below their revenue target?")
    assert {r["Country"] for r in result.rows} == {"France", "Germany"}
    assert any("instead of itself" in s and "Revenue_Target_EUR" in s for s in result.steps)


def test_self_comparison_without_a_clear_match_still_goes_back_to_the_model(workbook):
    with pytest.raises(PlanError, match="with itself"):
        execute(
            workbook,
            QueryPlan(
                sheet="Sales_Data",
                group_by=["Country"],
                metrics=[{"agg": "sum", "column": "Revenue_EUR", "alias": "rev"}],
                join={"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
                having=[{"column": "rev", "op": "<", "other_column": "rev"}],
            ),
            question="Which countries are doing badly?",
        )  # no word matches a target column
