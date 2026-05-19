# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Log + register the agent (Unity Catalog + Model Serving)
# MAGIC
# MAGIC Logs `chain.py` as an MLflow LangChain model, registers the version in
# MAGIC Unity Catalog under `cfg.registry.uc_model_name`, then deploys it
# MAGIC behind a Model Serving endpoint via `databricks-agents`.
# MAGIC
# MAGIC Prerequisites:
# MAGIC  - `01_setup_table` has been run.
# MAGIC  - `cfg.warehouse.http_path` in `config.yaml` points at a SQL warehouse
# MAGIC    that has SELECT on the job_logs table.
# MAGIC  - The cluster identity has permission to register UC models in the
# MAGIC    target catalog/schema and to create serving endpoints.

# COMMAND ----------

# MAGIC %pip install -q -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging
from typing import TYPE_CHECKING

import mlflow
from databricks import agents
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
)

from job_monitor.config import load_config

if TYPE_CHECKING:  # pragma: no cover
    def display(_: object) -> None: ...
    class _DBUtils: ...
    dbutils: _DBUtils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("log_and_register")

# COMMAND ----------

# MAGIC %md ## Load config + sanity-check warehouse

# COMMAND ----------

cfg = load_config("./config.yaml")

if not cfg.warehouse.http_path:
    raise ValueError(
        "config.yaml: warehouse.http_path is required to deploy the chain. "
        "Set it to the HTTP path of a Databricks SQL warehouse, e.g. "
        "/sql/1.0/warehouses/abc123."
    )

warehouse_id = cfg.warehouse.http_path.rstrip("/").split("/")[-1]
log.info("Will log chain.py and grant access to warehouse_id=%s", warehouse_id)

# COMMAND ----------

# MAGIC %md ## Log + register

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="job_monitor_agent"):
    info = mlflow.langchain.log_model(
        lc_model="chain.py",
        artifact_path="agent",
        # Bundle the package + config so the chain loads inside Model Serving.
        code_paths=["job_monitor", "config.yaml"],
        # Resources tell serving which secrets/credentials to inject at runtime.
        resources=[
            DatabricksServingEndpoint(endpoint_name=cfg.llm.endpoint),
            DatabricksSQLWarehouse(warehouse_id=warehouse_id),
        ],
        pip_requirements="requirements.txt",
        input_example={"messages": [{"role": "user", "content": "How many runs failed yesterday?"}]},
    )

log.info("Logged model: %s", info.model_uri)

uc_version = mlflow.register_model(
    model_uri=info.model_uri,
    name=cfg.registry.uc_model_name,
)
log.info("Registered %s version %s", cfg.registry.uc_model_name, uc_version.version)

# COMMAND ----------

# MAGIC %md ## Deploy to Model Serving
# MAGIC
# MAGIC `agents.deploy` creates (or updates) a serving endpoint named after the
# MAGIC UC model. After it returns, the endpoint is the URL the Databricks App
# MAGIC (or the eventual Teams connector) will call.

# COMMAND ----------

deployment = agents.deploy(
    model_name=cfg.registry.uc_model_name,
    model_version=uc_version.version,
    scale_to_zero=True,
)
log.info("Deployed: endpoint=%s, query_url=%s", deployment.endpoint_name, deployment.query_endpoint)
