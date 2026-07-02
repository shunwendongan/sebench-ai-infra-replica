from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from sebench_infra.benchmark.schemas import EvaluationReport


class ModelRole(StrEnum):
    TEACHER = "teacher"
    BASE = "base"
    STUDENT = "student"


class ProviderKind(StrEnum):
    MOCK = "mock"
    OPENAI_COMPATIBLE = "openai_compatible"


class TrainingSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    REJECTED = "rejected"


class TrainingTaskKind(StrEnum):
    TASK_SPEC_AUTHORING = "task_spec_authoring"
    ISSUE_TO_PATCH = "issue_to_patch"
    TRAJECTORY_ACTION = "trajectory_action"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: Literal["warning", "error"] = "error"


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    source_task_id: str
    instruction: str
    input: str = ""
    output: str
    training_task: TrainingTaskKind = TrainingTaskKind.TASK_SPEC_AUTHORING
    split: TrainingSplit
    teacher_model_id: str
    prompt_version: str
    prompt_hash: str
    dataset_version_id: str
    validation_status: Literal["accepted", "rejected"]
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    license_note: str = "synthetic_public_reproduction_only"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_trainable(self) -> bool:
        return self.validation_status == "accepted" and self.split != TrainingSplit.REJECTED


class DatasetVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    source_dataset_id: str | None = None
    prompt_version: str
    prompt_hash: str
    teacher_provider: ProviderKind
    teacher_model_id: str
    created_at: str
    seed: int = 13
    split_counts: dict[str, int] = Field(default_factory=dict)
    references: list[str] = Field(default_factory=list)
    examples: list[TrainingExample] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ModelRole
    provider: ProviderKind = ProviderKind.MOCK
    model: str = "mock"
    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    api_key_env: str | None = None
    prompt_version: str | None = None
    cost_per_1k_input_tokens: float = Field(default=0.0, ge=0.0)
    cost_per_1k_output_tokens: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_count: int = Field(ge=0)
    valid_task_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    aggregate_score: float = Field(ge=0.0, le=1.0)
    judge_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_sec_avg: float = Field(default=0.0, ge=0.0)
    cost_estimate_usd: float = Field(default=0.0, ge=0.0)
    schema_repair_count: int = Field(default=0, ge=0)


class ModelRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ModelRunConfig
    metrics: ModelMetrics
    evaluation_report: EvaluationReport | None = None
    report_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_id: str
    dataset_id: str
    created_at: str
    runs: list[ModelRunResult]
    best_role_by_pass_rate: ModelRole | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlamaFactoryExportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    dataset_name: str
    output_dir: str
    train_file: str
    validation_file: str
    test_file: str
    dataset_info_file: str
    train_config_file: str
    windows_script_file: str
    train_examples: int = Field(ge=0)
    validation_examples: int = Field(ge=0)
    test_examples: int = Field(ge=0)
    base_model: str
    finetuning_type: str = "lora"
    metadata: dict[str, Any] = Field(default_factory=dict)
