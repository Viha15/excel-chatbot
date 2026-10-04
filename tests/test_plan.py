"""The plan model accepts the JSON shapes small local models actually produce."""

import pytest
from pydantic import ValidationError

from src.models.plan import QueryPlan
from src.services.executor import execute
from src.services.pipeline import _short_error

# Seen with llama3.2: every field written out, unused ones as null.
ALL_NULLS = {
    "intent": "query",
    "sheet": "Sales_Data",
    "filters": [{"column": "Country", "op": "==", "value": "Germany"}],
    "group_by": None,
    "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": None}],
    "join": None,
    "having": None,
    "columns": None,
    "sort": None,
    "limit": None,
    "keywords": None,
    "reason": None,
}


def test_null_fields_fall_back_to_defaults(workbook, sales):
    plan = QueryPlan.model_validate(ALL_NULLS)
    assert plan.having == [] and plan.keywords == [] and plan.reason == ""
    result = execute(workbook, plan)
    assert result.rows[0]["sum_Revenue_EUR"] == pytest.approx(
        sales.loc[sales.Country == "Germany", "Revenue_EUR"].sum()
    )


def test_empty_blocks_zero_limit_and_single_items_are_normalised():
    plan = QueryPlan.model_validate(
        {
            "sheet": "Sales_Data",
            "group_by": "Country",
            "metrics": {"agg": "sum", "column": "Revenue_EUR"},
            "join": {"sheet": None, "left_on": None, "right_on": None},
            "sort": {"by": None, "descending": True},
            "limit": 0,
            "intent": "",
        }
    )
    assert plan.group_by == ["Country"] and len(plan.metrics) == 1
    assert plan.join is None and plan.sort is None and plan.limit is None
    assert plan.intent == "query"


def test_filter_value_null_is_kept_rather_than_dropped():
    plan = QueryPlan.model_validate({"sheet": "S", "filters": [{"column": "A", "op": "==", "value": None}]})
    assert plan.filters[0].value is None


def test_validation_errors_are_summarised_for_the_retry_prompt():
    with pytest.raises(ValidationError) as info:
        QueryPlan.model_validate({"sheet": "S", "metrics": [{"agg": "percentile", "column": "x"}]})
    message = _short_error(info.value)
    assert message.startswith("Invalid plan structure.")
    assert "metrics.0.agg" in message and "pydantic.dev" not in message


@pytest.mark.parametrize(
    "word,agg",
    [("average", "mean"), ("AVG", "mean"), ("total", "sum"), ("highest", "max"), ("distinct", "nunique")],
)
def test_aggregation_synonyms_are_accepted(word, agg):
    plan = QueryPlan.model_validate({"sheet": "S", "metrics": [{"agg": word, "column": "x"}]})
    assert plan.metrics[0].agg == agg


def test_template_answer_states_what_was_computed(workbook):
    from src.services.pipeline import template_answer

    plan = QueryPlan(
        sheet="Sales_Data",
        group_by=["Product_Category"],
        metrics=[{"agg": "max", "column": "Revenue_EUR", "alias": "Highest Revenue"}],
    )
    answer = template_answer(execute(workbook, plan), plan)
    assert "Highest Revenue = max of Revenue_EUR, per Product_Category" in answer


def test_plan_schema_rejects_the_shapes_small_models_produced():
    jsonschema = pytest.importorskip("jsonschema")
    from src.models.plan import PLAN_JSON_SCHEMA

    validator = jsonschema.Draft202012Validator(PLAN_JSON_SCHEMA)
    assert validator.is_valid({"sheet": "S", "filters": [{"column": "a", "op": ">", "value": 1}]})
    assert not validator.is_valid({"filters": [{"column": "a", "op": ">"}]})
    assert not validator.is_valid({"filters": [{"column": "a", "op": "greater than", "value": 1}]})
    assert not validator.is_valid({"intent": "compare"})


def test_python_none_strings_are_treated_as_null():
    """Seen with llama3.2: "sheet": "None", "column": "None"."""
    plan = QueryPlan.model_validate(
        {"sheet": "None", "limit": "null", "metrics": [{"agg": "count", "column": "None"}]}
    )
    assert plan.sheet is None and plan.limit is None and plan.metrics[0].column is None


def test_numbers_in_question():
    from src.models.plan import numbers_in

    assert numbers_in("revenue greater than 10,000?") == [10000.0]
    assert numbers_in("between 2.5k and 1m") == [2500.0, 1_000_000.0]
    assert numbers_in("Show Q3 records for Germany") == []  # 'Q3' is not a number


def test_value_is_not_guessed_when_the_question_is_ambiguous():
    from src.models.plan import fill_values_from_question

    raw = {"filters": [{"column": "A", "op": ">"}]}
    patched, notes = fill_values_from_question(raw, "between 10 and 20")  # 2 numbers, 1 filter
    assert "value" not in patched["filters"][0] and notes == []
