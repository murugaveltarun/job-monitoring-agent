# Enterprise Job-Monitoring Agent (Databricks)

A LangGraph ReAct agent built on the **Mosaic AI Agent Framework** that
answers natural-language questions about Databricks job runs by querying a
`job_logs` Delta table. Designed to be served behind a Databricks Model
Serving endpoint and eventually called from MS Teams.

## Layout

```
.
├── config.yaml                # single source of truth (table, LLM, warehouse, UC)
├── requirements.txt           # cluster + serving deps
├── pyproject.toml             # local dev deps + pytest + ruff
├── chain.py                   # MLflow chain entrypoint (loaded by Model Serving)
├── 01_setup_table.py          # notebook: create job_logs + seed fake data
├── 02_agent.py                # notebook: interactive agent + eval harness
├── 03_log_and_register.py     # notebook: log → UC register → deploy serving endpoint
├── job_monitor/               # importable package (the actual logic)
│   ├── config.py              # pydantic config model + load_config()
│   ├── sql_guard.py           # sqlglot-based read-only SQL guardrails
│   ├── seed.py                # pure, deterministic fake-data generator
│   ├── tools.py               # SQL executors (spark / warehouse) + tool factory
│   └── agent.py               # build_agent + system prompt
└── tests/                     # pytest suite — runs locally, no Databricks needed
    ├── test_config.py
    ├── test_sql_guard.py
    └── test_seed.py
```

## How it fits together

- **Notebooks** are thin orchestration. All business logic lives in
  `job_monitor/` so it can be unit-tested locally (no Spark, no Databricks)
  and re-used by both the interactive notebook and the deployed chain.
- **`02_agent.py`** runs the agent against the in-cluster `SparkSession` —
  fast iteration loop for prompt + tool changes.
- **`chain.py`** is what Model Serving actually loads. It builds the same
  agent but with a `databricks-sql-connector`-backed executor so it works
  without a Spark session. `03_log_and_register.py` ships it.
- **`sql_guard.py`** parses every LLM-generated SQL with sqlglot and rejects
  anything that isn't a single SELECT/WITH against the configured table.
  This is the hard safety boundary — see `tests/test_sql_guard.py`.

## Run order

1. Upload the folder to a **Databricks Repo** (or Git folder) so all four
   files (`config.yaml`, `chain.py`, the notebooks, and the `job_monitor/`
   package) sit at the same level.
2. Edit `config.yaml`:
   - `table.*` — where to materialize `job_logs` (defaults to `main.default`).
   - `llm.endpoint` — the foundation-model endpoint name in your workspace.
   - `warehouse.http_path` — **required for step 4**; leave null while
     iterating in `02_agent.py`.
   - `registry.uc_model_name` — UC name to register the deployed agent under.
3. Run **`01_setup_table.py`** once. Recreates `job_logs` and seeds it with
   ~500 deterministic fake runs.
4. Run **`02_agent.py`**. Loads the agent, runs a demo question, then a
   ~8-case eval harness that fails loudly if the agent regresses.
5. Run **`03_log_and_register.py`** once you're happy. Logs `chain.py` to
   MLflow, registers a new version in UC, and creates/updates the Model
   Serving endpoint via `databricks-agents`.

## Local development

The package layer (everything under `job_monitor/`) is plain Python — no
Spark, no Databricks runtime required:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

`pyspark`, `mlflow`, `databricks-langchain`, etc. are runtime dependencies
of the Databricks cluster and the Model Serving image — they will show as
"not installed" in your local IDE. That's expected; don't add them to
`pyproject.toml` dependencies.

## Adding new tables (cluster usage, billing, etc.)

When more system tables come online:

1. Add a new tool factory in `job_monitor/tools.py` (or extend
   `build_query_tool` to take multiple allowed tables).
2. Extend `ALLOWED_TABLE_REFS` and the system prompt's schema description.
3. Add test cases to `tests/test_sql_guard.py` for the new allowlist.
4. Re-run `03_log_and_register.py` to ship.

## Security model

- The tool only accepts queries that sqlglot parses as a single `SELECT` /
  `WITH` / set-operation against an explicitly allowlisted table.
- All DDL/DML node types are rejected, anywhere in the parse tree (covers
  `WITH evil AS (DELETE FROM x) SELECT ...` and similar).
- Auth is workspace-identity-based: in serving, the `DatabricksSQLWarehouse`
  resource declared in `03_log_and_register.py` causes Model Serving to
  inject short-lived credentials scoped to that warehouse only.
- `instructions.txt` (the original PoC brief) is gitignored.
