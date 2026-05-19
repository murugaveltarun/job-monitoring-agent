"""SQL guard accept/reject cases.

These are the security-critical tests for the agent — any regression here
means the LLM could potentially run unintended SQL. Keep this list growing as
new attack patterns are spotted.
"""

from __future__ import annotations

import pytest

from job_monitor.sql_guard import UnsafeSQLError, validate_select


ALLOWED = {"job_logs", "default.job_logs", "main.default.job_logs"}


# ---- happy path ----------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * FROM job_logs",
    "SELECT * FROM default.job_logs LIMIT 10",
    "SELECT * FROM main.default.job_logs LIMIT 10",
    "select count(*) from job_logs where status = 'FAILED'",
    "WITH recent AS (SELECT * FROM job_logs WHERE run_start_time > current_timestamp() - INTERVAL 7 DAYS) SELECT job_id, COUNT(*) FROM recent GROUP BY job_id",
    "SELECT a.job_id FROM job_logs a JOIN job_logs b ON a.run_id = b.run_id",
    "SELECT * FROM job_logs UNION ALL SELECT * FROM job_logs",
])
def test_accepts_safe_select(sql: str) -> None:
    validate_select(sql, ALLOWED)


# ---- top-level statement type -------------------------------------------

@pytest.mark.parametrize("sql", [
    "INSERT INTO job_logs VALUES ('x', 'y', 'z', current_timestamp(), current_timestamp(), 'SUCCESS')",
    "UPDATE job_logs SET status = 'SUCCESS'",
    "DELETE FROM job_logs",
    "DROP TABLE job_logs",
    "CREATE TABLE evil (x INT)",
    "TRUNCATE TABLE job_logs",
    "MERGE INTO job_logs USING job_logs ON 1=1 WHEN MATCHED THEN DELETE",
])
def test_rejects_dml_ddl(sql: str) -> None:
    with pytest.raises(UnsafeSQLError):
        validate_select(sql, ALLOWED)


# ---- multi-statement ----------------------------------------------------

def test_rejects_multi_statement() -> None:
    with pytest.raises(UnsafeSQLError, match="one statement"):
        validate_select("SELECT * FROM job_logs; DROP TABLE job_logs", ALLOWED)


# ---- table allowlist ----------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * FROM system.information_schema.tables",
    "SELECT * FROM users",
    "SELECT * FROM job_logs JOIN secrets ON job_logs.job_id = secrets.id",
    "SELECT * FROM other_catalog.default.job_logs",
])
def test_rejects_disallowed_table(sql: str) -> None:
    with pytest.raises(UnsafeSQLError, match="not allowed"):
        validate_select(sql, ALLOWED)


def test_rejects_subquery_against_disallowed_table() -> None:
    sql = "SELECT * FROM job_logs WHERE job_id IN (SELECT id FROM secrets)"
    with pytest.raises(UnsafeSQLError, match="not allowed"):
        validate_select(sql, ALLOWED)


def test_rejects_cte_with_disallowed_table() -> None:
    sql = "WITH evil AS (SELECT * FROM users) SELECT * FROM evil"
    with pytest.raises(UnsafeSQLError, match="not allowed"):
        validate_select(sql, ALLOWED)


# ---- parse errors -------------------------------------------------------

def test_rejects_garbage() -> None:
    with pytest.raises(UnsafeSQLError):
        validate_select("not sql at all !!", ALLOWED)
