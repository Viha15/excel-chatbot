"""End-to-end through FastAPI, with a scripted FakeLLM in place of a real model."""

from __future__ import annotations

from dataclasses import replace

from src.config import get_settings
from src.main import app
from tests.conftest import FakeLLM

GERMANY_PLAN = {
    "sheet": "Sales_Data",
    "filters": [{"column": "Country", "op": "==", "value": "Germany"}],
    "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "total_revenue"}],
}


def test_upload_describes_sheets(client, sample_bytes):
    body = client.post("/upload", files={"file": ("sample.xlsx", sample_bytes)}).json()
    assert [s["name"] for s in body["sheets"]] == ["Sales_Data", "Targets", "Product_Catalog"]
    assert client.get(f"/workbooks/{body['workbook_id']}").status_code == 200


def test_upload_rejects_wrong_extension(client):
    response = client.post("/upload", files={"file": ("data.csv", b"a,b\n1,2")})
    assert response.status_code == 400


def test_upload_rejects_a_file_over_the_size_limit(client):
    app.dependency_overrides[get_settings] = lambda: replace(get_settings(), max_upload_mb=1)
    too_big = b"0" * (1024 * 1024 + 1)  # one byte over 1 MB
    response = client.post("/upload", files={"file": ("big.xlsx", too_big)})
    assert response.status_code == 413 and "exceeds 1 MB" in response.json()["detail"]


def test_upload_rejects_an_empty_file(client):
    response = client.post("/upload", files={"file": ("empty.xlsx", b"")})
    assert response.status_code == 400 and "empty" in response.json()["detail"]


def test_upload_rejects_corrupt_xlsx(client):
    response = client.post("/upload", files={"file": ("broken.xlsx", b"not a zip")})
    assert response.status_code == 400


def test_chat_returns_answer_exact_result_and_sources(client, uploaded, use_llm, sales):
    fake = use_llm(FakeLLM(GERMANY_PLAN, "Total revenue for Germany is €X."))
    body = client.post(
        "/chat", json={"workbook_id": uploaded, "question": "What is the total revenue for Germany?"}
    ).json()

    expected = round(sales.loc[sales.Country == "Germany", "Revenue_EUR"].sum(), 2)
    assert body["result"]["rows"] == [{"total_revenue": expected}]
    assert body["sources"][0]["sheet"] == "Sales_Data"
    assert body["usage"] == {"prompt_tokens": 200, "completion_tokens": 20}
    # the answer prompt got the computed number, not the raw sheet
    answer_prompt = fake.calls[1][1]["content"]
    assert str(expected) in answer_prompt
    assert "ORD-1001" not in answer_prompt


def test_response_shows_the_plan_as_executed_after_repairs(client, uploaded, use_llm):
    """The model plans a column search; the executor lists the values instead. The response
    must show the plan that produced the result, not the one that was replaced."""
    use_llm(FakeLLM({"intent": "find_columns", "sheet": "Sales_Data", "keywords": ["Product"]}, "ok"))
    body = client.post(
        "/chat", json={"workbook_id": uploaded, "question": "what are the different products?"}
    ).json()
    assert body["plan"] == {"sheet": "Product_Catalog", "columns": ["Product"]}
    assert body["sources"][0]["sheet"] == "Product_Catalog"


def test_each_question_is_logged_with_its_token_usage(client, uploaded, use_llm, caplog):
    use_llm(FakeLLM(GERMANY_PLAN, "ok"))
    with caplog.at_level("INFO", logger="src.routers.chat"):
        client.post("/chat", json={"workbook_id": uploaded, "question": "Total revenue for Germany?"})
    line = next(r.getMessage() for r in caplog.records if r.name == "src.routers.chat")
    assert "prompt_tokens=200" in line and "completion_tokens=20" in line and "retries=0" in line


def test_raw_rows_are_never_sent_when_planning(client, uploaded, use_llm, sales):
    fake = use_llm(FakeLLM(GERMANY_PLAN, "ok"))
    client.post("/chat", json={"workbook_id": uploaded, "question": "Total revenue for Germany?"})
    planning_prompt = fake.calls[0][1]["content"]
    order_ids_in_prompt = [oid for oid in sales.Order_ID if oid in planning_prompt]
    assert order_ids_in_prompt == []  # no data rows at all (PROFILE_SAMPLE_ROWS defaults to 0)


def test_planning_prompt_stays_within_budget(client, uploaded, use_llm):
    """~850 tokens for the sample workbook. Fails if a change bloats the prompt."""
    fake = use_llm(FakeLLM(GERMANY_PLAN, "ok"))
    client.post("/chat", json={"workbook_id": uploaded, "question": "Total revenue for Germany?"})
    prompt_chars = sum(len(m["content"]) for m in fake.calls[0])
    assert prompt_chars < 3_000  # ~3.3 characters per token -> under ~920 tokens


def test_invalid_plan_is_retried_with_the_error(client, uploaded, use_llm):
    bad = {"sheet": "Sales_Data", "metrics": [{"agg": "sum", "column": "Turnover"}]}
    fake = use_llm(FakeLLM(bad, GERMANY_PLAN, "ok"))
    response = client.post("/chat", json={"workbook_id": uploaded, "question": "German revenue?"})
    assert response.status_code == 200
    retry_message = fake.calls[1][-1]["content"]
    assert "Turnover" in retry_message and "Revenue_EUR" in retry_message


def test_gives_up_after_two_bad_plans(client, uploaded, use_llm):
    use_llm(FakeLLM("not json", {"sheet": "Nope"}))
    body = client.post("/chat", json={"workbook_id": uploaded, "question": "anything?"}).json()
    assert body["answer"].startswith("I could not turn that question")
    assert body["result"] is None


def test_unanswerable_question_skips_execution(client, uploaded, use_llm):
    use_llm(FakeLLM({"intent": "unanswerable", "reason": "There is no employee data."}))
    body = client.post("/chat", json={"workbook_id": uploaded, "question": "Who is the CEO?"}).json()
    assert "no employee data" in body["answer"]


def test_unknown_workbook_is_404(client, use_llm):
    use_llm(FakeLLM())
    response = client.post("/chat", json={"workbook_id": "missing", "question": "hello?"})
    assert response.status_code == 404


def test_template_mode_uses_a_single_llm_call(client, uploaded, use_llm, sales):
    """ANSWER_WITH_LLM=false (the Ollama default): plan with the LLM, phrase with code."""
    app.dependency_overrides[get_settings] = lambda: replace(get_settings(), answer_with_llm=False)
    fake = use_llm(FakeLLM(GERMANY_PLAN))  # only one scripted reply: a second call would fail
    body = client.post("/chat", json={"workbook_id": uploaded, "question": "Revenue for Germany?"}).json()

    expected = sales.loc[sales.Country == "Germany", "Revenue_EUR"].sum()
    assert len(fake.calls) == 1
    assert f"{expected:,.2f}" in body["answer"]
    assert "Sales_Data" in body["answer"]


def test_failed_attempts_are_reported(client, uploaded, use_llm):
    bad = {"sheet": "Sales_Data", "metrics": [{"agg": "sum", "column": "Turnover"}]}
    use_llm(FakeLLM(bad, GERMANY_PLAN, "ok"))
    body = client.post("/chat", json={"workbook_id": uploaded, "question": "German revenue?"}).json()
    assert len(body["failed_attempts"]) == 1 and "Turnover" in body["failed_attempts"][0]


def test_missing_filter_value_is_taken_from_the_question(client, uploaded, use_llm, sales):
    """Exact output of llama3.2 for 'How many records have revenue greater than 10,000?'."""
    no_value = {
        "intent": "query",
        "sheet": "Sales_Data",
        "filters": [{"column": "Revenue_EUR", "op": ">"}, {"column": "Revenue_EUR", "op": ">"}],
        "metrics": [{"agg": "count", "column": None}],
        "limit": 120,
    }
    fake = use_llm(FakeLLM(no_value, "ok"))
    body = client.post(
        "/chat",
        json={"workbook_id": uploaded, "question": "How many records have revenue greater than 10,000?"},
    ).json()
    assert body["result"]["rows"] == [{"row_count": int((sales.Revenue_EUR > 10000).sum())}]
    assert "took value 10,000 for Revenue_EUR > from the question" in body["result"]["steps"]
    assert body["failed_attempts"] == [] and len(fake.calls) == 2  # no retry needed
