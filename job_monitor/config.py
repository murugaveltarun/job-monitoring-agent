"""Typed configuration loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: Final[Path] = Path("./config.yaml")


class TableConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    catalog: str
    schema_name: str = Field(alias="schema")
    name: str

    @property
    def fqn(self) -> str:
        return f"{self.catalog}.{self.schema_name}.{self.name}"


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint: str
    temperature: float = 0.0
    max_tokens: PositiveInt = 1500


class SeedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_jobs: PositiveInt = 20
    num_runs: PositiveInt = 500
    num_days: PositiveInt = 30
    random_seed: int = 42


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_rows_returned: PositiveInt = 100


class WarehouseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    http_path: str | None = None


class RegistryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    uc_model_name: str = "main.default.job_monitor_agent"


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    table: TableConfig
    llm: LLMConfig
    seed: SeedConfig = SeedConfig()
    agent: AgentConfig = AgentConfig()
    warehouse: WarehouseConfig = WarehouseConfig()
    registry: RegistryConfig = RegistryConfig()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate config from a YAML file.

    Defaults to ./config.yaml resolved against the caller's current working
    directory. In a Databricks notebook, that is the notebook's folder.
    """
    path = Path(path)
    logger.info("Loading config from %s", path.resolve())
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
