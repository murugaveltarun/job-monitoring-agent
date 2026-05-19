"""MLflow chain entrypoint.

This file is what `mlflow.langchain.log_model(lc_model="chain.py", ...)` loads.
At import time it builds the agent and registers it with `mlflow.models.set_model`.

The chain uses the Databricks SQL warehouse executor (not in-cluster spark)
because Model Serving runs the chain in a stateless Python process without a
SparkSession. `cfg.warehouse.http_path` must be set in `config.yaml` and the
serving endpoint must be given a `DatabricksSQLWarehouse` resource so the
runtime injects `DATABRICKS_HOST` / `DATABRICKS_TOKEN`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from langchain_core.runnables import RunnableLambda

from job_monitor.agent import build_agent
from job_monitor.config import load_config
from job_monitor.tools import build_query_tool, make_warehouse_executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve config.yaml relative to this file so it works regardless of CWD.
HERE = Path(__file__).resolve().parent
cfg = load_config(HERE / "config.yaml")

if not cfg.warehouse.http_path:
    raise RuntimeError(
        "warehouse.http_path is not set in config.yaml. "
        "The deployed chain requires a SQL warehouse to query job_logs."
    )

executor = make_warehouse_executor(cfg.warehouse.http_path, cfg.agent.max_rows_returned)
tool = build_query_tool(
    executor,
    table_fqn=cfg.table.fqn,
    table_name=cfg.table.name,
    schema_name=cfg.table.schema_name,
    max_rows=cfg.agent.max_rows_returned,
)
agent = build_agent(cfg, [tool])


def _final_message(state: dict) -> str:
    # Agent Framework's agents.deploy() only accepts ChatCompletionResponse or
    # StringResponse signatures. LangGraph returns the full state dict, which
    # MLflow infers as Any. Returning the last message's content makes the
    # output schema a string → StringResponse-compatible.
    return state["messages"][-1].content


chain = agent | RunnableLambda(_final_message)

mlflow.models.set_model(chain)
