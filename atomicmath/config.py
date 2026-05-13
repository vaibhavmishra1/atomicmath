"""Config schema. Everything the pipeline needs comes from one YAML file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputCfg(BaseModel):
    dataset: str
    config_name: str | None = None  # e.g. ShadenA/MathNet "United_States"
    split: str = "train"
    question_field: str
    solution_field: str
    answer_field: str | None = None
    topic_field: str | None = None
    default_topic: str = "train"
    max_seeds: int | None = None  # cap rows ingested (scan dataset until this many kept)

    language_field: str | None = None
    language_filter: str | None = "English"
    problem_type_field: str | None = None
    problem_type_filter: list[str] | None = None
    country_field: str | None = None
    country_filter: str | None = None


class OutputCfg(BaseModel):
    dataset: str
    private: bool = True
    push_audit_sidecar: bool = True


class ModelsCfg(BaseModel):
    """Optional extra YAML keys (e.g. legacy `cohort:`) are ignored via pydantic extra='ignore'."""

    model_config = ConfigDict(extra="ignore")

    embedder: str = "openai/text-embedding-3-small"
    extractor: str  # LLM for structured JSON (exemplar bootstrap); T≈0 where used
    generators: list[str]  # realizer rotates; composer uses composer.model or generators[0]
    verifiers: list[str]  # one or more; how many run is gate.correctness_verifier_count
    judge: str  # topic cluster naming during ingest / auxiliary calls

    @model_validator(mode="after")
    def _check_models(self) -> "ModelsCfg":
        if len(self.verifiers) < 1:
            raise ValueError("need at least 1 verifier model in models.verifiers")
        if len(self.generators) < 1:
            raise ValueError("need at least 1 generator")
        return self


class FilterCfg(BaseModel):
    language: str | None = "en"  # set to null to disable (ASCII heuristic in ingest)
    max_question_tokens: int = 4000
    max_answer_tokens: int = 1000
    min_topic_size: int = 8  # topic normalization: mark seeds in smaller topics ineligible


class GateCfg(BaseModel):
    novelty_minhash_max: float = 0.60
    novelty_embed_max: float = 0.92
    correctness_verifier_count: int = Field(default=3, ge=1, le=8)
    correctness_consensus: int = 2
    affinity_log_threshold: float = 0.7
    answer_equivalence_model: str | None = "openai/gpt-5-mini"
    answer_equivalence_fallback: bool = True


class QualityCfg(BaseModel):
    enabled: bool = True
    model: str = "openai/gpt-5-mini"
    min_depth_score: float = Field(default=0.45, ge=0.0, le=1.0)
    min_contest_score: float = Field(default=0.50, ge=0.0, le=1.0)
    max_routine_score: float = Field(default=0.72, ge=0.0, le=1.0)
    # If false, only hard-fail on routine_score. If true, also require judge pass=true.
    require_judge_pass: bool = False


class MutationCfg(BaseModel):
    enabled: bool = True
    extraction_model: str | None = None  # default: models.extractor
    generation_model: str | None = None  # default: first models.generators entry; does plan + generate
    judge_model: str | None = None  # default: models.judge
    prompt_version: str = "single_question_plan_generate_v3"
    max_hinges_per_seed: int = Field(default=3, ge=1, le=5)
    min_solution_chars: int = 20
    generator_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    success_story_limit: int = Field(default=3, ge=0, le=10)
    failure_story_limit: int = Field(default=3, ge=0, le=10)
    global_memory_enabled: bool = True
    global_success_memory_limit: int = Field(default=5, ge=0, le=30)
    global_failure_memory_limit: int = Field(default=5, ge=0, le=30)
    global_memory_max_active: int = Field(default=300, ge=20, le=5000)
    global_memory_max_lesson_chars: int = Field(default=1800, ge=200, le=5000)
    global_memory_prioritize_topic: bool = True
    strict_correctness: bool = True
    min_hinge_preservation: float = Field(default=0.70, ge=0.0, le=1.0)
    min_mutation_quality: float = Field(default=0.60, ge=0.0, le=1.0)
    min_sharpness: float = Field(default=0.60, ge=0.0, le=1.0)
    min_non_stitched: float = Field(default=0.75, ge=0.0, le=1.0)
    min_solution_economy: float = Field(default=0.55, ge=0.0, le=1.0)
    min_novelty: float = Field(default=0.55, ge=0.0, le=1.0)
    max_seed_minhash_overlap: float = Field(default=0.75, ge=0.0, le=1.0)


class GeneratorCfg(BaseModel):
    rounds: int = 3
    candidates_per_round: int = 4
    fidelity_weight: float = 0.4
    correctness_weight: float = 0.4
    novelty_weight: float = 0.2
    fidelity_unrealizable_threshold: float = 0.30


class CurriculumCfg(BaseModel):
    """Bias brief sampling toward under-covered (primary_concept, secondary_concept) pairs."""

    coverage_smoothing: float = 1.0
    coverage_power: float = 1.0
    epsilon_uniform_pair: float = 0.05  # probability of ignoring coverage for exploration


class ComposerCfg(BaseModel):
    model: str | None = None  # default: first entry in models.generators
    temperature: float = 0.3


class RealizerCfg(BaseModel):
    temperature: float = 0.7


class ExemplarBootstrapCfg(BaseModel):
    model: str | None = None  # default: models.extractor
    max_bootstrap: int | None = None  # default: min(indexed seeds, input.max_seeds)


class CohortCfg(BaseModel):
    """Reserved; ignore legacy YAML keys under `cohort:`."""

    model_config = ConfigDict(extra="ignore")


class TaxonomyCfg(BaseModel):
    path: str = "default"  # "default" → packaged taxonomy; otherwise path to JSON


class StorageCfg(BaseModel):
    db_path: str = "./atomicmath_fresh.db"
    cache_dir: str = "./cache"


class RuntimeCfg(BaseModel):
    target_count: int = 2000
    max_attempts: int = 50000
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    dry_run: bool = False


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: InputCfg
    output: OutputCfg
    models: ModelsCfg
    taxonomy: TaxonomyCfg = Field(default_factory=TaxonomyCfg)
    filters: FilterCfg = Field(default_factory=FilterCfg)
    cohort: CohortCfg = Field(default_factory=CohortCfg)
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    gate: GateCfg = Field(default_factory=GateCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    curriculum: CurriculumCfg = Field(default_factory=CurriculumCfg)
    composer: ComposerCfg = Field(default_factory=ComposerCfg)
    realizer: RealizerCfg = Field(default_factory=RealizerCfg)
    exemplar_bootstrap: ExemplarBootstrapCfg = Field(default_factory=ExemplarBootstrapCfg)
    quality: QualityCfg = Field(default_factory=QualityCfg)
    mutation: MutationCfg = Field(default_factory=MutationCfg)

    @model_validator(mode="after")
    def _verifier_gate_consistent(self) -> "Config":
        n_ver = len(self.models.verifiers)
        n_run = self.gate.correctness_verifier_count
        if n_ver < n_run:
            raise ValueError(
                f"gate.correctness_verifier_count is {n_run} but models.verifiers has only "
                f"{n_ver} entr{'y' if n_ver == 1 else 'ies'}; add more models or lower correctness_verifier_count"
            )
        cons = self.gate.correctness_consensus
        if cons < 1 or cons > n_run:
            raise ValueError(
                "gate.correctness_consensus must satisfy 1 <= correctness_consensus <= "
                f"correctness_verifier_count (got consensus={cons}, verifier_count={n_run})"
            )
        return self


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
