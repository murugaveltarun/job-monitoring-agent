"""LangGraph ReAct agent factory."""

from __future__ import annotations

import logging
from typing import Any

from databricks_langchain import ChatDatabricks
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from .config import Config

logger = logging.getLogger(__name__)


def system_prompt(table_fqn: str) -> str:
    return f"""You are an enterprise job-monitoring assistant for a Databricks workspace.

You answer questions about Databricks job runs by querying exactly one table:

    {table_fqn}

Columns:
    job_id          STRING
    job_name        STRING
    run_id          STRING
    run_start_time  TIMESTAMP  (UTC)
    run_end_time    TIMESTAMP  (UTC; NULL while status='RUNNING')
    status          STRING     one of SUCCESS, FAILED, RUNNING, TIMEOUT, CANCELLED

Use the `query_job_logs` tool to run read-only SELECT queries. Guidelines:
  - Always write Spark SQL.
  - Treat the data as UTC. Use `current_timestamp()` for "now".
  - Duration in seconds = unix_timestamp(run_end_time) - unix_timestamp(run_start_time).
  - For "long running": prefer the duration expression above; flag anything over
    1 hour as long-running unless the user gives a threshold.
  - For "currently running" / "in flight" use `status = 'RUNNING'`.
  - Convert relative times ("yesterday", "last week") to explicit ranges in SQL.
  - If a query errors, read the error and fix the SQL — don't give up after one try.
  - Answer concisely. Cite the numbers you found. Don't paste the raw SQL into
    the final answer unless the user asks for it.
"""


def build_agent(cfg: Config, tools: list[BaseTool]) -> Any:
    """Build a LangGraph ReAct agent from config + tools."""
    llm = ChatDatabricks(
        endpoint=cfg.llm.endpoint,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )
    logger.info(
        "Built ChatDatabricks endpoint=%s, temperature=%s, max_tokens=%s",
        cfg.llm.endpoint, cfg.llm.temperature, cfg.llm.max_tokens,
    )
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt(cfg.table.fqn),
    )


def ask(agent: Any, question: str) -> str:
    """Single-turn convenience helper; returns the final assistant message."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
