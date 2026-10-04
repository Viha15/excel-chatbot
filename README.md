# Excel Intelligence Chatbot

A FastAPI service that lets you upload an `.xlsx` workbook and ask questions about it in plain language.

The LLM never sees the spreadsheet data. It sees a compact **schema profile** and returns a **query plan** (JSON). The application checks that plan against the workbook and executes it with **pandas**, so every number is computed, not generated. The answer sentence is then written either by a second, small LLM call or, by default on Ollama, by a code template, which keeps local answers fast.

```
question ──► [LLM] plan ──► [pandas] execute ──► [LLM] phrase answer ──► answer + result table + sources
               ▲                 │
               └── schema profile (no data rows)      invalid plan? error goes back to the LLM once
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`, `brew install uv`, or see its install page). uv manages the virtual environment and, if needed, Python itself (3.11+).

```bash
uv sync                           # creates .venv and installs the locked dependencies
cp .env.example .env              # then choose a provider (below)
```

Every command below is run through `uv run`, which uses that environment; there is nothing to activate. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; change them with `uv add <package>` / `uv remove <package>`.

### Choosing the LLM

Set `LLM_PROVIDER` in `.env`. The same code path serves all three, because each speaks the OpenAI chat-completions API.

**Ollama (local, default)**

```bash
ollama pull llama3.2              # 3B, fast on a laptop; any instruction model that can return JSON works
ollama serve                      # if it is not already running
```
```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

**Azure OpenAI**
```
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=<deployment name, e.g. gpt-4o>
AZURE_OPENAI_API_VERSION=2024-10-21
```

**OpenAI**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=<key>
OPENAI_MODEL=gpt-4o-mini
```

Small local models plan less reliably than GPT-4o-class models; the one-retry loop helps, but expect more "could not turn that question into a valid query" answers from a 7–8B model.

## Run

```bash
uv run uvicorn src.main:app --reload
```

Interactive docs: http://localhost:8000/docs

### Web UI (Streamlit)

With the API running, open a second terminal and run:

```bash
uv run streamlit run ui.py
```

It opens http://localhost:8501. Upload a workbook in the sidebar, then type a question. Each answer shows the computed table, the source sheets and Excel row numbers, and under *How this was computed* the steps, the query plan and the tokens used.

The UI is a thin client of the API (it calls `/upload` and `/chat`), so both always give the same answers. Set `API_URL` if the API runs somewhere other than `http://localhost:8000`.

```bash
# upload
curl -F "file=@tests/data/sample.xlsx" http://localhost:8000/upload
# -> {"workbook_id": "3f2c...", "sheets": [...]}

# ask
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"workbook_id": "3f2c...", "question": "Which countries are below their revenue target?"}'
```

`/chat` returns:

| field | contents |
|---|---|
| `answer` | the model's sentence, written only from the computed result |
| `plan` | the query plan as executed: the model's plan after any repairs, so it matches the result |
| `result` | the computed table (`columns`, `rows`, `total_rows`, `steps`) |
| `sources` | sheets used, rows matched, and the Excel row numbers of the matched records |
| `usage` | prompt and completion tokens spent on this question |
| `failed_attempts` | why earlier plans were rejected and what the model wrote, if it had to retry |

Other endpoints: `GET /health`, `GET /workbooks/{id}` (detected sheets, header rows, column types).

### Logs

The API writes one line per upload and per question to the console, which is what you would feed into monitoring to track LLM cost:

```
INFO src.routers.chat: chat workbook=3f2c... seconds=6.4 prompt_tokens=880 completion_tokens=30 retries=0 rows=1 guardrails=0 question='What is the total revenue for Germany?'
```

Refused questions and uploads are logged as warnings with the reason, and LLM failures as errors. Spreadsheet data is not logged, only the question, counts and timings.

## Speed with Ollama

What the code does to keep local answers quick:

- **One LLM call per question.** On Ollama the answer sentence is built from the computed result by a template (`ANSWER_WITH_LLM`, default off for Ollama, on for Azure/OpenAI). The numbers are identical either way; only the wording differs.
- **Prompt reuse.** The planning prompt starts with the same instructions and workbook schema for every question, so Ollama reuses its cached work and only processes the new question. This works because no second, different prompt runs in between.
- **Model loaded at startup.** The API loads the model when it starts and asks Ollama to keep it in memory for `OLLAMA_KEEP_ALIVE` (default 30 minutes), so the first question doesn't wait for loading.
- **Short outputs.** Plans are capped at 400 tokens, so a small model can't ramble.

What you can do on your machine:

- Use a small model: `llama3.2` (3B) is the default. `qwen2.5:3b` is a good alternative for JSON output. Larger models (`llama3.1` 8B) plan more reliably but are several times slower on a CPU.
- Check where it runs: `ollama ps` while a question is processing shows `100% GPU` or `100% CPU`. CPU-only is the main reason for slow answers.
- Ask the second question after the first: the first one fills Ollama's cache, so later ones on the same workbook are faster.
- If a call still runs past `LLM_TIMEOUT_SECONDS` (default 180), the API returns a clear timeout message; the UI waits as long as the API needs.

## Guardrails

The model only proposes a query; code decides what runs. On top of that design (whitelisted operations, every sheet and column validated, nothing the model writes executed as code), `src/services/guardrails.py` adds:

| Where | Guardrail | What happens |
|---|---|---|
| Question | Length limit (500 characters), control characters removed | refused with 400 / cleaned |
| Question | Prompt-injection phrases ("ignore previous instructions", "reveal the system prompt", "you are now…", role tags) | refused with 400 **before any LLM call**, so they cost no tokens. Deliberately narrow: "what are the instructions in the Notes column?" still works |
| Upload | Must be a real .xlsx zip; macro workbooks (`vbaProject.bin`) refused; zip-bomb check (`MAX_UNCOMPRESSED_MB`, default 300) | refused with 400 before parsing |
| Prompt | Cell text that reads like instructions to a model is left out of the schema profile; long cell text is cut to 60 characters; the planning prompt tells the model the schema and question are data, not instructions | a spreadsheet cannot steer the model |
| Answer | When the LLM writes the answer sentence (`ANSWER_WITH_LLM=true`), every number in it must appear in the computed result (allowing for rounding) | otherwise the sentence is replaced by the answer built directly from the result, and `guardrails` in the response says so (the UI shows it as a warning) |
| Usage | Rate limit per client (`RATE_LIMIT_PER_MINUTE`, default 30) on `/upload` and `/chat` | 429 |

Limits: the injection check is pattern-based and catches the common phrasings, not every possible one; the rate limit is in memory (per process).

## Pre-commit hooks

```bash
uv sync                           # includes pre-commit and ruff
git init                          # if the folder is not a repository yet
uv run pre-commit install         # runs the hooks on every commit
uv run pre-commit run --all-files # or run them now
```

`.pre-commit-config.yaml` runs, on every commit:

- **File hygiene**: trailing whitespace, final newline, valid YAML, every Python file parses, no merge-conflict markers, no leftover `breakpoint()`, no files over 1 MB.
- **Secrets**: no private keys, no `.env` file (it holds API keys), no API-key-looking strings in code.
- **Ruff**: lint (errors, import order, common bugs, modern syntax) and formatting, line length 110. Configured in the hook itself, so there is no separate config file.
- **Tests**: the full pytest suite (about 5 seconds; no LLM needed).

## Evaluating a model

```bash
uv run python -m eval.run_examples                      # the model configured in .env
uv run python -m eval.run_examples --model qwen2.5:3b   # another Ollama model
uv run python -m eval.run_examples --provider azure     # another provider
```

Runs the ten example questions through the real LLM and checks each answer against values computed independently with pandas from `tests/data/sample.xlsx`. It prints pass/fail, time, tokens and retries per question, and a total. Use it to choose a model and to catch regressions after changing a prompt. The checks themselves are tested: correct plans pass all ten, and a plan that answers "highest revenue" with `max` instead of `sum` fails.

## Tests

```bash
uv run pytest
```

106 tests, no LLM or network needed (a scripted fake model is injected through FastAPI's dependency override):

- `test_executor.py` — the ten example questions below as plans, checked against values computed independently with pandas from the raw file; plus validation errors (unknown column or sheet, sum of a text column).
- `test_workbook.py` — multi-sheet loading, type detection, a header on row 3, blank rows and empty cells, a corrupt file, and the cost constraint itself: the planning prompt for a 20,000-row sheet is under 1,000 characters and the same size as for 100 rows.
- `test_plan.py` — plans written the way small local models write them (null fields, empty blocks, single items instead of lists) are accepted.
- `test_guardrails.py` — injection attempts refused before any LLM call, ordinary questions pass, macro files and zip bombs refused, instruction-like cell text kept out of the prompt, invented numbers replaced, rate limit enforced.
- `test_eval.py` — the evaluation's checks pass for correct plans and catch a wrong one.
- `test_structured_output.py` — the plan request carries the JSON schema; a server that rejects it falls back to JSON mode once.
- `test_llm.py` — provider wiring: no silent SDK retries on Ollama, clear error for missing Azure credentials, tolerant JSON parsing.
- `test_ui.py` — the Streamlit page renders and reports clearly when the API is unreachable.
- `test_api.py` — upload validation; the full chat flow; that the planning prompt contains no data rows; that an invalid plan is retried with the error message; that the service gives up cleanly after two bad plans.

## How the LLM is kept cheap

| call | what the model receives | grows with |
|---|---|---|
| plan | question + schema profile: per column its name, type, range, and — for low-cardinality text columns only — the distinct values (capped at 25), joined with `|`; a value list repeated in another sheet is referenced, not repeated. No data rows by default (`PROFILE_SAMPLE_ROWS`) | number of **columns** |
| answer (optional, `ANSWER_WITH_LLM`) | question + the computed result, capped at 20 rows | size of the **answer** |

Nothing grows with the number of rows. For the sample workbook the planning prompt is about 850 tokens (instructions ~430, schema ~400); sending the workbook itself as CSV would be about 4,400 tokens for just 120 rows, and grows with every row. Listing distinct values of columns like `Country` is what lets the model map "Germany" to the right column and spelling without seeing any rows. High-cardinality columns such as `Order_ID` are summarised as a count.

## Key technical decisions

- **Plan, don't generate code.** The model outputs a constrained JSON plan (filters, group-by, aggregates, one join, post-aggregation conditions, sort, limit) validated by Pydantic. An alternative is to have the model write pandas or SQL and run it; that is more expressive but needs sandboxing and is harder to validate. The plan format covers all ten example questions, including the cross-sheet target comparison.
- **Structured output is opt-in.** `STRUCTURED_OUTPUT=true` sends a JSON schema (`PLAN_JSON_SCHEMA` in `src/models/plan.py`) so the model's decoder can only produce valid operators, aggregations and complete filters; a server that rejects it falls back to plain JSON mode once. It is off by default because with a 3B local model it made the model fill every optional field with placeholders (`"None"`, `limit 1000`), which hurt accuracy. Whether it helps depends on the model, which is what `python -m eval.run_examples` measures (see "Evaluating a model").
- **Never return a silently empty answer for an impossible query.** A condition that compares a column with itself is rejected, and the retry message lists the numeric columns of the other sheets as candidates.
- **Accept how small models write JSON.** Local models often spell out unused fields as `null`, send an empty `join` block, `"limit": 0`, or a single string where a list is expected. The plan model normalises these before validation instead of rejecting the plan; validation errors that remain are summarised into a short message for the retry.
- **Repair predictable mistakes in code.** Small models make the same few mistakes repeatedly. The executor fixes these before running the plan; every repair is recorded in the steps, and none costs another LLM call:

  | The model writes | The executor does |
  |---|---|
  | `Country == 'all'` to mean every country | ignores the condition, unless `all` is a real value in that column |
  | a join whose columns nothing uses | skips the join |
  | `Country == Germany` and `Country == France` (no row can satisfy both) | combines them into `Country in [Germany, France]` |
  | a limit smaller than the number of values the question names | drops the limit |
  | `average`, `total`, `highest` as the aggregation | maps them to `mean`, `sum`, `max` |
  | one combined total for "compare A and B" | groups by the compared column |
  | `max` for "which X has the highest Y" | uses the total, unless the question mentions a single order or row |
  | a comparison with no value (`Revenue_EUR >`) | takes the number from the question ("greater than 10,000"), only when that is unambiguous |
  | a raw column compared with a target, next to its sum | compares the group's total |
  | a column compared with itself | picks the joined column whose name best matches the question, or rejects the plan |
  | a column-name search for "what are the different products?" | lists that column's values, unless the question asks about sheets or columns |

  Repairs are kept only where the evaluation shows the model needs them, or where the mistake would give a silently wrong answer. Other malformed plans (an unknown operator, a raw-column condition in `having`) are rejected and go back to the model for the retry.
- **Validate, then let the model fix it.** Every sheet and column in the plan is resolved against the real workbook (tolerating case, spaces and underscores). If something is wrong, the precise error — including the list of valid names — goes back to the model for one retry.
- **Misreadings stay visible.** Some mistakes can't be repaired safely: "highest revenue per category" planned as `max` (largest single order) instead of `sum` is a valid query, just not the one asked. The answer therefore always states what was computed ("Computed as: Highest Revenue = max of Revenue_EUR, per Product_Category"), so a misread question is obvious rather than silently wrong.
- **Numbers never pass through the model.** The API returns the computed table alongside the prose answer, so the figures can be checked even if the model phrases them badly.
- **Citations.** Each loaded row keeps its original Excel row number; listings cite exactly the rows returned, aggregates report how many rows were matched.
- **Generic loading.** The header row is detected (first mostly-text row within the first ten), fully blank rows and columns are dropped, duplicate or missing headers get unique names, and numeric/date types are recovered per column. Nothing refers to the sample workbook's column names.
- **One interface, three providers.** Ollama, Azure OpenAI and OpenAI all go through the OpenAI SDK; only client construction differs (`src/services/llm.py`). Temperature 0 and JSON mode for planning.

## Project layout

```
pyproject.toml  dependencies (uv); uv.lock pins the exact versions
ui.py           Streamlit front end (client of the API)
eval/
  run_examples.py   accuracy check of the ten example questions against a real LLM
src/
  main.py           creates the FastAPI app and attaches the routers
  config.py         settings from environment / .env
  dependencies.py   in-memory workbook store, rate limit, LLM client, workbook lookup
  routers/          the HTTP endpoints
    health.py         GET /health
    workbooks.py      POST /upload, GET /workbooks/{id}
    chat.py           POST /chat
  models/           the data shapes
    schemas.py        ChatRequest / ChatResponse (API bodies)
    plan.py           QueryPlan model (what the LLM may ask for)
  services/         the logic
    pipeline.py       plan -> execute -> answer pipeline, retry on invalid plans
    workbook.py       loading, header detection, type recovery, schema profile
    executor.py       plan validation and pandas execution, citations
    llm.py            provider selection (ollama / azure / openai)
    prompts.py        the two prompts
    guardrails.py     input, upload, prompt and answer guardrails, rate limit
tests/
  data/sample.xlsx   the provided workbook
```

## Limitations

- **Workbooks live in memory** in a single process and are lost on restart.
- **One table per sheet.** A sheet with several tables side by side or stacked is read as a single table. Numbers stored as formatted text (e.g. `"€1,200"`) stay text.
- **Plan expressiveness.** One join per question, no date bucketing ("revenue per month"), no calculated columns (e.g. margin = profit / revenue), no multi-step questions.
- **Business meaning is up to the question.** "Total revenue" sums all orders including `Cancelled` ones; the model adds a `Status` filter only if the question asks for it.
- **Whole workbook is loaded into pandas.** Fine to tens or hundreds of thousands of rows.

## Scaling to much larger files

1. Convert each sheet to Parquet once at upload, and execute plans with **DuckDB** over the Parquet files instead of in-memory pandas — the plan maps directly onto SQL (`WHERE`, `GROUP BY`, `JOIN`, `HAVING`, `ORDER BY`, `LIMIT`), and DuckDB streams from disk.
2. Build the profile from a streamed pass (openpyxl `read_only=True`) rather than loading the sheet.
3. Store files in blob storage and profiles/metadata in a database, so any API instance can serve any workbook.
4. Cache plans by (workbook, normalised question) and results by plan.
5. Move large uploads to a background job and poll for readiness.

## Example questions and responses

Each example shows the plan the model is expected to produce and the result the executor computes from `tests/data/sample.xlsx`. The same plans are asserted in `tests/test_executor.py`. The `answer` sentence depends on the model and is not reproduced here.

### 1. What is the total revenue for Germany?

```json
{"sheet": "Sales_Data", "filters": [{"column": "Country", "op": "==", "value": "Germany"}], "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "total_revenue"}]}
```

| total_revenue |
|---|
| 110,207.50 |

Sources: `Sales_Data` (23 rows)

### 2. Show all records for Germany.

```json
{"sheet": "Sales_Data", "filters": [{"column": "Country", "op": "==", "value": "Germany"}], "columns": ["Order_ID", "Order_Date", "Product", "Revenue_EUR"]}
```

| Order_ID | Order_Date | Product | Revenue_EUR |
|---|---|---|---|
| ORD-1001 | 2026-06-27 | Secure Laptop | 4,940.00 |
| ORD-1002 | 2026-05-14 | Implementation | 1,500.00 |
| ORD-1003 | 2026-06-04 | Reporting Cloud | 6,077.50 |
| ORD-1004 | 2026-07-19 | Secure Laptop | 7,020.00 |
| ORD-1005 | 2026-07-10 | Implementation | 8,100.00 |

Sources: `Sales_Data` (23 rows) · 23 rows, first 5 shown

### 3. What is the average revenue by country?

```json
{"sheet": "Sales_Data", "group_by": ["Country"], "metrics": [{"agg": "mean", "column": "Revenue_EUR", "alias": "avg_revenue"}]}
```

| Country | avg_revenue |
|---|---|
| Belgium | 5,744.48 |
| France | 4,704.76 |
| Germany | 4,791.63 |
| Netherlands | 5,065.00 |
| Spain | 6,867.14 |

Sources: `Sales_Data` (120 rows)

### 4. Which country has the highest total revenue?

```json
{"sheet": "Sales_Data", "group_by": ["Country"], "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}], "sort": {"by": "revenue"}, "limit": 1}
```

| Country | revenue |
|---|---|
| Belgium | 166,590.00 |

Sources: `Sales_Data` (120 rows)

### 5. How many records have revenue greater than 10,000?

```json
{"sheet": "Sales_Data", "filters": [{"column": "Revenue_EUR", "op": ">", "value": 10000}], "metrics": [{"agg": "count"}]}
```

| row_count |
|---|
| 20 |

Sources: `Sales_Data` (20 rows)

### 6. Compare Germany and France.

```json
{"sheet": "Sales_Data", "filters": [{"column": "Country", "op": "in", "value": ["Germany", "France"]}], "group_by": ["Country"], "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}, {"agg": "sum", "column": "Profit_EUR", "alias": "profit"}, {"agg": "count", "alias": "orders"}]}
```

| Country | revenue | profit | orders |
|---|---|---|---|
| France | 98,800.00 | 35,410.63 | 21 |
| Germany | 110,207.50 | 44,170.75 | 23 |

Sources: `Sales_Data` (44 rows)

### 7. Which product category generated the highest revenue?

```json
{"sheet": "Sales_Data", "group_by": ["Product_Category"], "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}], "sort": {"by": "revenue"}, "limit": 1}
```

| Product_Category | revenue |
|---|---|
| Services | 250,720.00 |

Sources: `Sales_Data` (120 rows)

### 8. Which worksheet contains information about revenue?

```json
{"intent": "find_columns", "keywords": ["revenue"]}
```

| sheet | column |
|---|---|
| Sales_Data | Revenue_EUR |
| Targets | Revenue_Target_EUR |

Sources: `Sales_Data` (120 rows); `Targets` (5 rows)

### 9. What are the total revenue and profit by country?

```json
{"sheet": "Sales_Data", "group_by": ["Country"], "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}, {"agg": "sum", "column": "Profit_EUR", "alias": "profit"}], "sort": {"by": "revenue"}}
```

| Country | revenue | profit |
|---|---|---|
| Belgium | 166,590.00 | 67,405.93 |
| Spain | 144,210.00 | 59,881.00 |
| Netherlands | 131,690.00 | 54,169.75 |
| Germany | 110,207.50 | 44,170.75 |
| France | 98,800.00 | 35,410.63 |

Sources: `Sales_Data` (120 rows)

### 10. Which countries are below their revenue target?

```json
{"sheet": "Sales_Data", "group_by": ["Country"], "metrics": [{"agg": "sum", "column": "Revenue_EUR", "alias": "revenue"}], "join": {"sheet": "Targets", "left_on": "Country", "right_on": "Country"}, "having": [{"column": "revenue", "op": "<", "other_column": "Revenue_Target_EUR"}]}
```

| Country | revenue | Revenue_Target_EUR |
|---|---|---|
| France | 98,800.00 | 165,000 |
| Germany | 110,207.50 | 210,000 |

Sources: `Sales_Data` (120 rows); `Targets` (5 rows)
