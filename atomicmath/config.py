"""Configuration for the lineage synthesis pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


class InputCfg(BaseModel):
    """Hugging Face seed dataset configuration.

    Only the question and answer fields are required. Their column names are
    configurable so the same runner works on MathNet, MATH-500, or a custom
    dataset with different schema names.
    """

    dataset: str
    config_name: str | None = None
    split: str = "train"
    question_field: str = "question"
    answer_field: str = "answer"
    iteration_field: str = "iteration"
    memory_field: str = "memory"
    max_seeds: int = Field(default=100, ge=1)
    seed: int = 0

    @field_validator("config_name", mode="before")
    @classmethod
    def _blank_config_is_none(cls, value: str | None) -> str | None:
        return _none_if_blank(value)


class OutputCfg(BaseModel):
    """Local and optional Hugging Face output configuration."""

    dataset: str | None = None
    split: str = "train"
    private: bool = False
    push_to_hub: bool = False
    append_if_same_dataset: bool = True
    local_path: str = "./out/lineage/lineage.jsonl"
    summary_path: str = "./out/lineage/lineage_summary.json"

    @field_validator("dataset", mode="before")
    @classmethod
    def _blank_dataset_is_none(cls, value: str | None) -> str | None:
        return _none_if_blank(value)


class ModelsCfg(BaseModel):
    """LLM choices.

    The old config used `generators: [...]`; this schema accepts it as a
    fallback so existing config files fail less abruptly during migration.
    """

    model_config = ConfigDict(extra="ignore")

    generator: str | None = None
    refiner: str | None = None
    judge: str | None = None
    generators: list[str] | None = None
    verifiers: list[str] | None = None

    @model_validator(mode="after")
    def _fill_defaults(self) -> "ModelsCfg":
        if self.generator is None:
            self.generator = self.generators[0] if self.generators else "openai/gpt-5-mini"
        if self.refiner is None:
            self.refiner = self.generator
        if self.judge is None:
            self.judge = self.verifiers[0] if self.verifiers else "openai/gpt-5-mini"
        return self


class LineageCfg(BaseModel):
    candidates_per_iteration: int = Field(default=3, ge=1, le=8)
    generator_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    refiner_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_memory_items: int = Field(default=8, ge=0, le=40)
    max_question_chars: int = Field(default=6000, ge=500)
    max_answer_chars: int = Field(default=4000, ge=200)
    min_correctness: float = Field(default=1.0, ge=0.0, le=1.0)
    min_novelty: float = Field(default=0.45, ge=0.0, le=1.0)
    min_depth: float = Field(default=0.45, ge=0.0, le=1.0)
    min_non_stitched: float = Field(default=0.70, ge=0.0, le=1.0)
    min_solution_economy: float = Field(default=0.50, ge=0.0, le=1.0)
    continue_on_rejected: bool = False


class StorageCfg(BaseModel):
    cache_dir: str = "./cache"


class RuntimeCfg(BaseModel):
    dry_run: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: InputCfg
    output: OutputCfg = Field(default_factory=OutputCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    lineage: LineageCfg = Field(default_factory=LineageCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)

    @model_validator(mode="after")
    def _output_defaults(self) -> "Config":
        if self.output.dataset is None:
            self.output.dataset = self.input.dataset
        return self


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
