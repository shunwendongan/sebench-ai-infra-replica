from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskCategory(StrEnum):
    CODE_REPAIR = "code_repair"
    DATA_PIPELINE = "data_pipeline"
    NUMERIC_REASONING = "numeric_reasoning"
    SPATIAL_REASONING = "spatial_reasoning"
    BENCHMARK_AUTHORING = "benchmark_authoring"


class ScoringRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = Field(default=1.0, gt=0.0)
    kind: Literal["exact_match", "contains", "file_exists", "numeric_close", "metadata_flag"]
    expected: str | float | int | bool | None = None
    path: str | None = None
    tolerance: float = Field(default=1e-6, ge=0.0)


class TaskSpec(BaseModel):
    """Typed benchmark task contract.

    The schema is intentionally small: it captures the minimum information needed for
    a reproducible benchmark item without leaking private task formats.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    title: str
    category: TaskCategory
    prompt: str
    fixtures: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=lambda: ["submission/"])
    scoring: list[ScoringRule] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_artifacts", "allowed_paths")
    @classmethod
    def paths_must_be_relative(cls, value: list[str]) -> list[str]:
        for item in value:
            path = PurePosixPath(item)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"path must be relative and must not escape sandbox: {item}")
        return value

    @model_validator(mode="after")
    def ensure_score_rules(self) -> "TaskSpec":
        if not self.scoring:
            self.scoring = [
                ScoringRule(
                    name="has_summary",
                    kind="contains",
                    path="submission/answer.txt",
                    expected="",
                )
            ]
        return self


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    version: str = "0.1.0"
    source: str = "synthetic_public_reproduction"
    tasks: list[TaskSpec]
    references: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSpec
    runner_mode: Literal["local", "docker"] = "local"
    max_tasks: int | None = Field(default=None, gt=0)
    include_spatial_diagnostics: bool = True


class TaskRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["passed", "failed", "error"]
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    patch_hash: str | None = None
    failure_type: str | None = None


class CacheHitFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: bool = False
    repo_snapshot: bool = False
    repo_map: bool = False
    llm_prompt: bool = False
    judge_verdict: bool = False


class TaskMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_time_sec: float = Field(ge=0.0)
    judge_time_sec: float = Field(ge=0.0)
    agent_step_latency_sec: float = Field(ge=0.0)
    patch_apply_time_sec: float = Field(default=0.0, ge=0.0)
    hidden_test_time_sec: float = Field(default=0.0, ge=0.0)
    hidden_test_file_write_time_sec: float = Field(default=0.0, ge=0.0)
    python_subprocess_startup_time_sec: float = Field(default=0.0, ge=0.0)
    pytest_process_startup_time_sec: float = Field(default=0.0, ge=0.0)
    pytest_collection_time_sec: float = Field(default=0.0, ge=0.0)
    pytest_execution_time_sec: float = Field(default=0.0, ge=0.0)
    pytest_total_time_sec: float = Field(default=0.0, ge=0.0)
    judge_cache_lookup_time_sec: float = Field(default=0.0, ge=0.0)
    repo_checkout_time_sec: float = Field(default=0.0, ge=0.0)
    git_clone_time_sec: float = Field(default=0.0, ge=0.0)
    git_checkout_time_sec: float = Field(default=0.0, ge=0.0)
    snapshot_materialize_time_sec: float = Field(default=0.0, ge=0.0)
    cache_lock_wait_time_sec: float = Field(default=0.0, ge=0.0)
    tempdir_create_time_sec: float = Field(default=0.0, ge=0.0)
    judge_workspace_cleanup_time_sec: float = Field(default=0.0, ge=0.0)
    pytest_subprocess_count: int = Field(default=0, ge=0)
    memory_peak_mb: float | None = Field(default=None, ge=0.0)
    cache_hit_flags: CacheHitFlags = Field(default_factory=CacheHitFlags)


class BenchmarkEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_backend: str = "local_toy"
    system: str | None = None
    machine: str | None = None
    processor: str | None = None
    python_version: str | None = None
    mlx_version: str | None = None
    torch_version: str | None = None
    pytorch_mps_used: bool = False


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    patch_hash: str
    pass_fail: Literal["pass", "fail", "error"]
    metrics: TaskMetrics
    failure_type: str | None = None
    benchmark_environment: BenchmarkEnvironment = Field(default_factory=BenchmarkEnvironment)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def failures_need_type(self) -> "RunRecord":
        if self.pass_fail != "pass" and not self.failure_type:
            raise ValueError("failed or errored run records must include failure_type")
        return self


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset_id: str
    task_results: list[TaskRunResult]
    aggregate_score: float = Field(ge=0.0, le=1.0)
    reward_signal: float = Field(ge=0.0, le=1.0)
    regression_passed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_records: list[RunRecord] = Field(default_factory=list)
