# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Setup `job_logs` table
# MAGIC
# MAGIC Drops and recreates the job-logs table defined in `./config.yaml`, then
# MAGIC seeds it with deterministic fake data from `job_monitor.seed`.
# MAGIC Run once before `02_agent`.

# COMMAND ----------

# MAGIC %pip install -q -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging
from typing import TYPE_CHECKING

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from job_monitor.config import load_config
from job_monitor.seed import generate_runs

# Notebook globals are injected by the Databricks runtime; declare them for
# static type-checkers so the IDE doesn't flag them.
if TYPE_CHECKING:  # pragma: no cover
    def display(_: object) -> None: ...
    class _DBUtils: ...
    dbutils: _DBUtils

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("setup_table")

# COMMAND ----------

# MAGIC %md ## Load config

# COMMAND ----------

cfg = load_config("./config.yaml")
fqn = cfg.table.fqn
log.info("Target table: %s", fqn)

# COMMAND ----------

# MAGIC %md ## Recreate the table (Delta, Unity Catalog)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.table.catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.table.catalog}.{cfg.table.schema_name}")
spark.sql(f"DROP TABLE IF EXISTS {fqn}")
spark.sql(f"""
    CREATE TABLE {fqn} (
        job_id          STRING    COMMENT 'Stable identifier of the job definition',
        job_name        STRING    COMMENT 'Human-readable job name',
        run_id          STRING    COMMENT 'Unique identifier for this run',
        run_start_time  TIMESTAMP COMMENT 'When the run started (UTC)',
        run_end_time    TIMESTAMP COMMENT 'When the run finished (UTC); NULL if still RUNNING',
        status          STRING    COMMENT 'SUCCESS | FAILED | RUNNING | TIMEOUT | CANCELLED'
    )
    USING DELTA
    COMMENT 'Synthetic Databricks job run logs for the monitoring agent'
""")
log.info("Created %s", fqn)

# COMMAND ----------

# MAGIC %md ## Generate + insert fake runs

# COMMAND ----------

runs = generate_runs(cfg.seed)

schema = StructType([
    StructField("job_id",         StringType(),    False),
    StructField("job_name",       StringType(),    False),
    StructField("run_id",         StringType(),    False),
    StructField("run_start_time", TimestampType(), False),
    StructField("run_end_time",   TimestampType(), True),
    StructField("status",         StringType(),    False),
])

rows = [run.as_dict() for run in runs]
df = spark.createDataFrame(rows, schema=schema)
df.write.mode("append").saveAsTable(fqn)
log.info("Inserted %d rows into %s", df.count(), fqn)

# COMMAND ----------

# MAGIC %md ## Sanity check

# COMMAND ----------

display(spark.sql(f"""
    SELECT status, COUNT(*) AS run_count
    FROM {fqn}
    GROUP BY status
    ORDER BY run_count DESC
"""))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {fqn} ORDER BY run_start_time DESC LIMIT 10"))
