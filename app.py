"""Databricks App entrypoint — FastAPI wrapper around the job-monitor agent.

Same agent as chain.py, different front door. The App runs on dedicated
Apps compute (not Model Serving), so it works on trial workspaces where
serverless serving is blocked.

POST /chat accepts a list of messages and returns the agent's final reply,
gated by a shared secret in the X-App-Secret header so Azure Bot Service
(or any HTTP client) can call it without workspace SSO.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from job_monitor.agent import build_agent
from job_monitor.config import load_config
from job_monitor.tools import build_query_tool, make_warehouse_executor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

HERE = Path(__file__).resolve().parent
cfg = load_config(HERE / "config.yaml")

if not cfg.warehouse.http_path:
    raise RuntimeError(
        "warehouse.http_path is not set in config.yaml. "
        "The App requires a SQL warehouse to query job_logs."
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

# When unset, /chat is open — fine for a private workspace App, but set this
# in app.yaml before exposing the App to Azure Bot Service.
SHARED_SECRET = os.environ.get("APP_SHARED_SECRET")

app = FastAPI(title="Job Monitor Agent", version="0.1.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    x_app_secret: str | None = Header(default=None, alias="X-App-Secret"),
) -> ChatResponse:
    if SHARED_SECRET and x_app_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="invalid or missing X-App-Secret")

    result = agent.invoke({"messages": [m.model_dump() for m in req.messages]})
    return ChatResponse(reply=result["messages"][-1].content)
