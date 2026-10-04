"""The evaluation's checks are themselves correct: with the right plans, all ten pass."""

from dataclasses import replace

from eval.run_examples import evaluate
from src.config import get_settings
from tests.conftest import FakeLLM

S, R = "Sales_Data", "Revenue_EUR"
CORRECT_PLANS = [
    {
        "sheet": S,
        "filters": [{"column": "Country", "op": "==", "value": "Germany"}],
        "metrics": [{"agg": "sum", "column": R}],
    },
    {"sheet": S, "filters": [{"column": "Country", "op": "==", "value": "Germany"}]},
    {"sheet": S, "group_by": ["Country"], "metrics": [{"agg": "mean", "column": R}]},
    {
        "sheet": S,
        "group_by": ["Country"],
        "metrics": [{"agg": "sum", "column": R, "alias": "rev"}],
        "sort": {"by": "rev"},
        "limit": 1,
    },
    {"sheet": S, "filters": [{"column": R, "op": ">", "value": 10000}], "metrics": [{"agg": "count"}]},
    {
        "sheet": S,
        "filters": [{"column": "Country", "op": "in", "value": ["Germany", "France"]}],
        "group_by": ["Country"],
        "metrics": [{"agg": "sum", "column": R}],
    },
    {
        "sheet": S,
        "group_by": ["Product_Category"],
        "metrics": [{"agg": "sum", "column": R, "alias": "rev"}],
        "sort": {"by": "rev"},
        "limit": 1,
    },
    {"intent": "find_columns", "keywords": ["revenue"]},
    {
        "sheet": S,
        "group_by": ["Country"],
        "metrics": [{"agg": "sum", "column": R}, {"agg": "sum", "column": "Profit_EUR"}],
    },
    {
        "sheet": S,
        "group_by": ["Country"],
        "metrics": [{"agg": "sum", "column": R, "alias": "rev"}],
        "join": {"sheet": "Targets", "left_on": "Country", "right_on": "Country"},
        "having": [{"column": "rev", "op": "<", "other_column": "Revenue_Target_EUR"}],
    },
]


def test_correct_plans_pass_every_check():
    settings = replace(get_settings(), answer_with_llm=False)
    outcomes = evaluate(FakeLLM(*CORRECT_PLANS), settings)
    assert [o.case.question for o in outcomes if not o.passed] == []


def test_a_wrong_plan_is_caught():
    """The product-category question answered with the average instead of the total must fail."""
    plans = list(CORRECT_PLANS)
    plans[6] = {"sheet": S, "group_by": ["Product_Category"], "metrics": [{"agg": "mean", "column": R}]}
    outcomes = evaluate(FakeLLM(*plans), replace(get_settings(), answer_with_llm=False))
    assert [i for i, o in enumerate(outcomes) if not o.passed] == [6]
