"""Pure, deterministic generator for fake job-run rows.

Kept independent of pyspark so the logic is unit-testable. The notebook
converts the returned list into a Spark DataFrame.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from .config import SeedConfig

logger = logging.getLogger(__name__)


_JOB_NAME_TEMPLATES: tuple[str, ...] = (
    "ingest_{src}_raw",
    "transform_{src}_silver",
    "publish_{src}_gold",
    "ml_{src}_training",
    "report_{src}_daily",
)
_SOURCES: tuple[str, ...] = (
    "sales", "marketing", "finance", "ops", "product", "support", "web", "mobile",
)
_STATUS_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("SUCCESS",   72),
    ("FAILED",    15),
    ("TIMEOUT",    5),
    ("CANCELLED",  4),
    ("RUNNING",    4),
)
_LONG_RUNNING_FRACTION = 0.05  # fraction of runs that are deliberate outliers


@dataclass(frozen=True)
class JobRun:
    job_id: str
    job_name: str
    run_id: str
    run_start_time: datetime
    run_end_time: datetime | None
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


def _build_jobs(rng: random.Random, num_jobs: int) -> list[dict]:
    jobs: list[dict] = []
    for i in range(num_jobs):
        template = rng.choice(_JOB_NAME_TEMPLATES)
        source = rng.choice(_SOURCES)
        jobs.append({
            "job_id": f"job-{i + 1:04d}",
            "job_name": f"{template.format(src=source)}_{i + 1:02d}",
        })
    return jobs


def _random_status(rng: random.Random) -> str:
    statuses, weights = zip(*_STATUS_WEIGHTS, strict=True)
    return rng.choices(statuses, weights=weights, k=1)[0]


def _random_duration_s(rng: random.Random) -> int:
    if rng.random() < _LONG_RUNNING_FRACTION:
        return rng.randint(90 * 60, 6 * 3600)
    return rng.randint(30, 25 * 60)


def generate_runs(cfg: SeedConfig, now: datetime | None = None) -> list[JobRun]:
    """Generate `cfg.num_runs` fake runs over the past `cfg.num_days`.

    Deterministic for a given `cfg.random_seed` and `now`. If `now` is None,
    uses the current UTC time (non-deterministic across calls — pass an
    explicit `now` in tests).
    """
    rng = random.Random(cfg.random_seed)
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    window_start = now - timedelta(days=cfg.num_days)

    jobs = _build_jobs(rng, cfg.num_jobs)
    runs: list[JobRun] = []

    for _ in range(cfg.num_runs):
        job = rng.choice(jobs)
        start = window_start + timedelta(
            seconds=rng.randint(0, cfg.num_days * 24 * 3600)
        )
        status = _random_status(rng)

        if status == "RUNNING":
            start = now - timedelta(seconds=rng.randint(60, 3600))
            end: datetime | None = None
        elif status == "TIMEOUT":
            end = start + timedelta(seconds=rng.randint(2 * 3600, 8 * 3600))
        else:
            end = start + timedelta(seconds=_random_duration_s(rng))

        runs.append(JobRun(
            job_id=job["job_id"],
            job_name=job["job_name"],
            run_id=str(uuid.UUID(int=rng.getrandbits(128))),
            run_start_time=start,
            run_end_time=end,
            status=status,
        ))

    logger.info("Generated %d fake runs across %d jobs", len(runs), cfg.num_jobs)
    return runs
