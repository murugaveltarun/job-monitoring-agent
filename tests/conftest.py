"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_monitor.config import Config, load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg(repo_root: Path) -> Config:
    return load_config(repo_root / "config.yaml")


@pytest.fixture(scope="session")
def allowed_tables(cfg: Config) -> set[str]:
    return {
        cfg.table.name.lower(),
        f"{cfg.table.schema_name}.{cfg.table.name}".lower(),
        cfg.table.fqn.lower(),
    }
