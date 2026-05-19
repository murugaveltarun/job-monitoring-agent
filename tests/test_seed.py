"""Fake-data generator: shape, determinism, status invariants."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from job_monitor.config import SeedConfig
from job_monitor.seed import JobRun, generate_runs


FIXED_NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def small_cfg() -> SeedConfig:
    return SeedConfig(num_jobs=5, num_runs=200, num_days=14, random_seed=123)


def test_row_count_matches_config(small_cfg: SeedConfig) -> None:
    runs = generate_runs(small_cfg, now=FIXED_NOW)
    assert len(runs) == small_cfg.num_runs


def test_is_deterministic_given_now_and_seed(small_cfg: SeedConfig) -> None:
    a = generate_runs(small_cfg, now=FIXED_NOW)
    b = generate_runs(small_cfg, now=FIXED_NOW)
    assert a == b


def test_different_seed_changes_output(small_cfg: SeedConfig) -> None:
    other = SeedConfig(**{**small_cfg.model_dump(), "random_seed": 999})
    a = generate_runs(small_cfg, now=FIXED_NOW)
    b = generate_runs(other, now=FIXED_NOW)
    assert a != b


def test_running_rows_have_null_end_time(small_cfg: SeedConfig) -> None:
    runs = generate_runs(small_cfg, now=FIXED_NOW)
    for r in runs:
        if r.status == "RUNNING":
            assert r.run_end_time is None, f"RUNNING row has end_time: {r}"
        else:
            assert r.run_end_time is not None, f"{r.status} row missing end_time: {r}"


def test_end_after_start_for_finished_rows(small_cfg: SeedConfig) -> None:
    runs = generate_runs(small_cfg, now=FIXED_NOW)
    for r in runs:
        if r.run_end_time is not None:
            assert r.run_end_time > r.run_start_time, f"end <= start: {r}"


def test_jobs_pool_capped_to_num_jobs(small_cfg: SeedConfig) -> None:
    runs = generate_runs(small_cfg, now=FIXED_NOW)
    distinct_jobs = {r.job_id for r in runs}
    assert len(distinct_jobs) <= small_cfg.num_jobs


def test_runs_fall_within_window(small_cfg: SeedConfig) -> None:
    window_start = FIXED_NOW - timedelta(days=small_cfg.num_days)
    slack = timedelta(hours=1)  # RUNNING rows are pinned to "now - <1h"
    runs = generate_runs(small_cfg, now=FIXED_NOW)
    for r in runs:
        assert window_start - slack <= r.run_start_time <= FIXED_NOW + slack


def test_all_statuses_present_with_enough_rows() -> None:
    # 2000 rows with the configured weights should cover every status.
    cfg = SeedConfig(num_jobs=10, num_runs=2000, num_days=30, random_seed=42)
    runs = generate_runs(cfg, now=FIXED_NOW)
    seen = {r.status for r in runs}
    assert seen == {"SUCCESS", "FAILED", "TIMEOUT", "CANCELLED", "RUNNING"}


def test_run_id_is_unique(small_cfg: SeedConfig) -> None:
    runs = generate_runs(small_cfg, now=FIXED_NOW)
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids))


def test_as_dict_round_trips(small_cfg: SeedConfig) -> None:
    runs = generate_runs(small_cfg, now=FIXED_NOW)
    d = runs[0].as_dict()
    assert set(d) == {"job_id", "job_name", "run_id", "run_start_time", "run_end_time", "status"}
    assert isinstance(d["run_start_time"], datetime)
