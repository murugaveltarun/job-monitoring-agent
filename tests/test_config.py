"""Config loading + validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from job_monitor.config import Config, load_config


def test_loads_real_config_file(cfg: Config) -> None:
    assert cfg.table.fqn == f"{cfg.table.catalog}.{cfg.table.schema_name}.{cfg.table.name}"
    assert cfg.llm.endpoint
    assert cfg.seed.num_runs > 0
    assert cfg.agent.max_rows_returned > 0


def test_schema_alias_works_via_yaml(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("""
        table:
          catalog: c
          schema: s
          name: t
        llm:
          endpoint: ep
    """).strip())
    loaded = load_config(p)
    assert loaded.table.schema_name == "s"
    assert loaded.table.fqn == "c.s.t"


def test_missing_required_field_raises(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("llm:\n  endpoint: ep\n")  # missing table block
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        load_config(p)


def test_negative_seed_count_rejected(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("""
        table: {catalog: c, schema: s, name: t}
        llm: {endpoint: ep}
        seed: {num_jobs: -1}
    """).strip())
    with pytest.raises(Exception):  # noqa: B017 — PositiveInt rejects -1
        load_config(p)
