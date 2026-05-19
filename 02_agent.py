# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Enterprise Job Monitoring Agent (interactive)
# MAGIC
# MAGIC LangGraph ReAct agent on top of Mosaic AI. Loads config from `./config.yaml`
# MAGIC and queries `job_logs` via the in-cluster Spark session (good for
# MAGIC interactive development — for the deployed version see `03_log_and_register`).
# MAGIC
# MAGIC Run `01_setup_table` first.

# COMMAND ----------

# MAGIC %pip install -q -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging
from typing import TYPE_CHECKING

import mlflow
from pyspark.sql import SparkSession

from job_monitor.agent import ask, build_agent
from job_monitor.config import load_config
from job_monitor.tools import build_query_tool, make_spark_executor

if TYPE_CHECKING:  # pragma: no cover — declared for the IDE only
    def display(_: object) -> None: ...
    class _DBUtils: ...
    dbutils: _DBUtils

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agent_notebook")

mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md ## Build the agent

# COMMAND ----------

cfg = load_config("./config.yaml")

executor = make_spark_executor(spark, cfg.agent.max_rows_returned)
tool = build_query_tool(
    executor,
    table_fqn=cfg.table.fqn,
    table_name=cfg.table.name,
    schema_name=cfg.table.schema_name,
    max_rows=cfg.agent.max_rows_returned,
)
agent = build_agent(cfg, [tool])

# COMMAND ----------

# MAGIC %md ## Demo

# COMMAND ----------

print(ask(agent, "Which jobs failed in the last 7 days and how many times each?"))

# COMMAND ----------

# MAGIC %md ## Eval harness
# MAGIC
# MAGIC Smoke tests covering the question shapes the agent will get from Teams:
# MAGIC failures, long-running, currently-running, status mix, per-job stats.
# MAGIC Each row enforces a light *behavioral* assertion against the rendered
# MAGIC answer (presence of key terms / a digit) so the cell fails loudly if
# MAGIC the agent regresses, instead of just printing pretty output.

# COMMAND ----------

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    question: str
    must_contain: tuple[str, ...] = ()       # case-insensitive substrings
    must_match: tuple[str, ...] = ()         # regex patterns (case-insensitive)


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        "How many job runs are in the table overall, and what's the breakdown by status?",
        must_contain=("SUCCESS", "FAILED"),
        must_match=(r"\d{2,}",),
    ),
    EvalCase(
        "Which jobs failed yesterday?",
        must_match=(r"job[-_ ]\d+|no .*failed", ),
    ),
    EvalCase(
        "What are the top 5 longest-running successful runs in the last 30 days?",
        must_contain=("success",),
        must_match=(r"\d+",),
    ),
    EvalCase(
        "Are any jobs currently running right now? If so, for how long?",
        must_match=(r"running|no .*running", ),
    ),
    EvalCase(
        "Which job has the highest failure rate in the last 30 days?",
        must_match=(r"job[-_ ]\d+|\d+%|\d+/\d+", ),
    ),
    EvalCase(
        "Show me the most recent run of every job and its status.",
        must_match=(r"job[-_ ]\d+", ),
    ),
    EvalCase(
        "Did any runs hit a TIMEOUT in the last week? List the job names.",
        must_match=(r"timeout|no .*timeout", ),
    ),
    EvalCase(
        "What's the average run duration per job, sorted by slowest first?",
        must_match=(r"\d+", ),
    ),
]


def _check(answer: str, case: EvalCase) -> list[str]:
    failures: list[str] = []
    low = answer.lower()
    for term in case.must_contain:
        if term.lower() not in low:
            failures.append(f"missing required substring: {term!r}")
    for pattern in case.must_match:
        if not re.search(pattern, answer, flags=re.IGNORECASE):
            failures.append(f"no match for pattern: {pattern!r}")
    return failures


results: list[tuple[int, EvalCase, str, list[str]]] = []
for i, case in enumerate(EVAL_CASES, 1):
    print(f"\n{'=' * 80}\nQ{i}: {case.question}\n{'-' * 80}")
    try:
        answer = ask(agent, case.question)
    except Exception as e:
        answer = ""
        failures = [f"exception: {type(e).__name__}: {e}"]
    else:
        failures = _check(answer, case)
    print(answer or "<no answer>")
    if failures:
        print(f"\n!! ASSERTION FAILED: {failures}")
    results.append((i, case, answer, failures))

passed = sum(1 for _, _, _, f in results if not f)
print(f"\n\n{'=' * 80}\nEval: {passed}/{len(results)} cases passed")
failed_idxs = [i for i, _, _, f in results if f]
if failed_idxs:
    raise AssertionError(f"Eval cases failed: {failed_idxs}")

# COMMAND ----------

# MAGIC %md ## Next
# MAGIC
# MAGIC Run `03_log_and_register` to log this agent to MLflow / Unity Catalog
# MAGIC and deploy it behind a Model Serving endpoint.
