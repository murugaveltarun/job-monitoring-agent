"""LangChain tool for read-only access to the job_logs table.

Provides two SQL executors so the same tool definition works in both contexts:
  * `make_spark_executor` — for interactive notebook use (in-cluster spark).
  * `make_warehouse_executor` — for the served model (Databricks SQL Connector
    + warehouse). This is what the MLflow-logged chain uses, because the
    serving runtime does not have a SparkSession.
"""

from __future__ import annotations

import logging
import os
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

    Auth comes from env vars `DATABRICKS_HOST` / `DATABRICKS_TOKEN`. In Model
    Serving these are injected automatically when the model is logged with a
    `DatabricksSQLWarehouse` resource.
    """
    from databricks import sql as dbsql  # imported lazily; not needed in notebooks

    def execute(sql: str) -> pd.DataFrame:
        # Env vars are read at query time, not at executor construction. At
        # log_model time (notebook) the chain is built without DATABRICKS_HOST/
        # _TOKEN set; Model Serving injects them at request time via the
        # DatabricksSQLWarehouse resource.
        host = os.environ["DATABRICKS_HOST"].replace("https://", "").rstrip("/")
        token = os.environ["DATABRICKS_TOKEN"]
        # The LIMIT is enforced by the tool wrapper too, but doubling up here
        # protects against an enormous result set being pulled into memory.
        wrapped = f"SELECT * FROM ({sql}) AS _q LIMIT {max_rows}"
        with dbsql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token,
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
