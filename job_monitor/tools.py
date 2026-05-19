"""LangChain tool for read-only access to the job_logs table.

Provides two SQL executors so the same tool definition works in both contexts:
  * `make_spark_executor` — for interactive notebook use (in-cluster spark).
  * `make_warehouse_executor` — for the served model (Databricks SQL Connector
    + warehouse). This is what the MLflow-logged chain uses, because the
    serving runtime does not have a SparkSession.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool

from .sql_guard import UnsafeSQLError, validate_select

logger = logging.getLogger(__name__)

Executor = Callable[[str], pd.DataFrame]


def make_spark_executor(spark: Any, max_rows: int) -> Executor:
    """Build an executor that runs queries against an in-cluster SparkSession."""

    def execute(sql: str) -> pd.DataFrame:
        return spark.sql(sql).limit(max_rows).toPandas()

    return execute


def make_warehouse_executor(http_path: str, max_rows: int) -> Executor:
    """Build an executor that runs queries against a Databricks SQL warehouse.

    Auth is delegated to databricks-sdk's `Config`, which auto-detects:
      - Model Serving: DATABRICKS_HOST + DATABRICKS_TOKEN (PAT)
      - Databricks Apps: DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET (OAuth M2M)
      - Local dev: ~/.databrickscfg profile
    Resolution happens at query time so the executor can be built before any
    of those env vars are present (e.g. inside mlflow.langchain.log_model).
    """
    from databricks import sql as dbsql  # imported lazily; not needed in notebooks
    from databricks.sdk.core import Config

    def execute(sql: str) -> pd.DataFrame:
        cfg = Config()
        host = cfg.host.replace("https://", "").rstrip("/")
        wrapped = f"SELECT * FROM ({sql}) AS _q LIMIT {max_rows}"
        with dbsql.connect(
            server_hostname=host,
            http_path=http_path,
            credentials_provider=lambda: cfg.authenticate,
        ) as conn, conn.cursor() as cur:
            cur.execute(wrapped)
            return cur.fetchall_arrow().to_pandas()

    return execute


def _build_description(table_fqn: str, max_rows: int) -> str:
    return f"""Run a read-only Spark SQL SELECT against the job_logs table and return
the results as a Markdown table.

Table: {table_fqn}

Schema:
    job_id          STRING       -- stable job identifier, e.g. 'job-0007'
    job_name        STRING       -- human-readable name
    run_id          STRING       -- unique per run
    run_start_time  TIMESTAMP    -- UTC start
    run_end_time    TIMESTAMP    -- UTC end; NULL while status='RUNNING'
    status          STRING       -- SUCCESS | FAILED | RUNNING | TIMEOUT | CANCELLED

Rules:
  - Only SELECT / WITH queries; no DDL or DML.
  - Only this one table may be referenced.
  - Treat data as UTC; use current_timestamp() for "now".
  - Duration seconds = unix_timestamp(run_end_time) - unix_timestamp(run_start_time).
  - Always include a sensible LIMIT (tool also caps at {max_rows} rows).
"""


def build_query_tool(
    executor: Executor,
    *,
    table_fqn: str,
    table_name: str,
    schema_name: str,
    max_rows: int,
) -> StructuredTool:
    """Build the `query_job_logs` tool bound to the given executor."""
    allowed_tables: set[str] = {
        table_name.lower(),
        f"{schema_name}.{table_name}".lower(),
        table_fqn.lower(),
    }
    description = _build_description(table_fqn, max_rows)

    def query_job_logs(sql: str) -> str:
        try:
            validate_select(sql, allowed_tables)
        except UnsafeSQLError as e:
            logger.warning("Query rejected by guard: %s | sql=%r", e, sql)
            return f"ERROR: query rejected — {e}"

        try:
            df = executor(sql)
        except Exception as e:  # noqa: BLE001 — surface back to LLM so it can retry
            logger.exception("SQL execution failed")
            return f"ERROR: SQL execution failed — {type(e).__name__}: {e}"

        if df.empty:
            return "Query returned 0 rows."
        return df.to_markdown(index=False)

    return StructuredTool.from_function(
        func=query_job_logs,
        name="query_job_logs",
        description=description,
    )
