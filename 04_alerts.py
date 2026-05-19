# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Alerts: failed + long-running jobs → Teams
# MAGIC
# MAGIC Scheduled Databricks job that scans `job_logs` for new failures and
# MAGIC long-running runs since the last poll, formats a Teams MessageCard,
# MAGIC and POSTs it to a channel's Incoming Webhook URL.
# MAGIC
# MAGIC Prereqs:
# MAGIC  - `01_setup_table` has been run.
# MAGIC  - A Teams channel Incoming Webhook URL stored as a Databricks secret:
# MAGIC      `databricks secrets put-secret job-monitor-app teams_webhook_url`
# MAGIC  - This notebook scheduled as a Databricks Job. The schedule interval
# MAGIC    must match `LOOKBACK_MINUTES` below (e.g. every 15 min → 15).

# COMMAND ----------

import logging
from typing import TYPE_CHECKING

import requests

from job_monitor.config import load_config

if TYPE_CHECKING:  # pragma: no cover
    class _DBUtils: ...
    dbutils: _DBUtils
    spark: object

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("alerts")

cfg = load_config("./config.yaml")

# Window must match the job schedule. With a 15-min cron we look back 15 min
# for newly-completed failures; long-runners are independent of the window.
LOOKBACK_MINUTES = 15
LONG_RUNNING_THRESHOLD_MINUTES = 60

WEBHOOK_URL = dbutils.secrets.get(scope="job-monitor-app", key="teams_webhook_url")

# COMMAND ----------

# MAGIC %md ## Query for failures + long-runners

# COMMAND ----------

failures_df = spark.sql(f"""
    SELECT job_id, job_name, run_id, run_end_time, status
    FROM {cfg.table.fqn}
    WHERE status IN ('FAILED', 'TIMEOUT')
      AND run_end_time > current_timestamp() - INTERVAL {LOOKBACK_MINUTES} MINUTES
    ORDER BY run_end_time DESC
""").toPandas()

long_running_df = spark.sql(f"""
    SELECT job_id, job_name, run_id, run_start_time,
           CAST((unix_timestamp(current_timestamp()) - unix_timestamp(run_start_time)) / 60 AS INT) AS minutes_running
    FROM {cfg.table.fqn}
    WHERE status = 'RUNNING'
      AND run_start_time < current_timestamp() - INTERVAL {LONG_RUNNING_THRESHOLD_MINUTES} MINUTES
    ORDER BY run_start_time ASC
""").toPandas()

log.info("failures=%d long_running=%d", len(failures_df), len(long_running_df))

# COMMAND ----------

# MAGIC %md ## Bail early if nothing to alert on

# COMMAND ----------

if failures_df.empty and long_running_df.empty:
    log.info("No alerts to send")
    dbutils.notebook.exit("ok: no alerts")

# COMMAND ----------

# MAGIC %md ## Build Teams MessageCard + POST

# COMMAND ----------

sections: list[str] = []

if not failures_df.empty:
    rows = "\n".join(
        f"- **{r.job_name}** (run `{r.run_id}`) — {r.status} at {r.run_end_time}"
        for r in failures_df.itertuples()
    )
    sections.append(f"### Failed runs in last {LOOKBACK_MINUTES}m\n{rows}")

if not long_running_df.empty:
    rows = "\n".join(
        f"- **{r.job_name}** (run `{r.run_id}`) — running {r.minutes_running}m"
        for r in long_running_df.itertuples()
    )
    sections.append(f"### Long-running (> {LONG_RUNNING_THRESHOLD_MINUTES}m)\n{rows}")

payload = {
    "@type": "MessageCard",
    "@context": "https://schema.org/extensions",
    "summary": "Job monitor alert",
    "themeColor": "EE0000",
    "title": "Job Monitor Alert",
    "text": "\n\n".join(sections),
}

resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
resp.raise_for_status()
log.info("Posted to Teams: status=%s body=%s", resp.status_code, resp.text[:200])
