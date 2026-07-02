from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sebench_infra.benchmark.schemas import ScoringRule, TaskCategory, TaskSpec
from sebench_infra.training_loop.models import (
    DatasetVersion,
    ProviderKind,
    TrainingExample,
    TrainingSplit,
    TrainingTaskKind,
)
from sebench_infra.training_loop.validation import prompt_hash, stable_hash


class ExternalBenchmarkSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: str
    split: str | None = None
    url: str | None = None
    license: str = "unknown_public_dataset_license_check_required"
    revision: str | None = None
    provenance: str = "public_swe_ecosystem"


class SWEIssueInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    patch: str | None = None
    hints_text: str | None = None
    created_at: str | None = None
    version: str | None = None
    source: ExternalBenchmarkSource
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_hf_row(
        cls,
        row: dict[str, Any],
        *,
        source: ExternalBenchmarkSource,
    ) -> SWEIssueInstance:
        return cls(
            instance_id=str(row.get("instance_id") or row.get("id") or row.get("task_id")),
            repo=str(row.get("repo") or row.get("repository") or ""),
            base_commit=str(row.get("base_commit") or row.get("commit") or ""),
            problem_statement=str(row.get("problem_statement") or row.get("issue") or ""),
            test_patch=_optional_str(row.get("test_patch")),
            fail_to_pass=_coerce_test_list(row.get("FAIL_TO_PASS") or row.get("fail_to_pass")),
            pass_to_pass=_coerce_test_list(row.get("PASS_TO_PASS") or row.get("pass_to_pass")),
            patch=_optional_str(row.get("patch") or row.get("gold_patch")),
            hints_text=_optional_str(row.get("hints_text") or row.get("hints")),
            created_at=_optional_str(row.get("created_at")),
            version=_optional_str(row.get("version")),
            source=source,
            metadata={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "instance_id",
                    "id",
                    "task_id",
                    "repo",
                    "repository",
                    "base_commit",
                    "commit",
                    "problem_statement",
                    "issue",
                    "test_patch",
                    "FAIL_TO_PASS",
                    "fail_to_pass",
                    "PASS_TO_PASS",
                    "pass_to_pass",
                    "patch",
                    "gold_patch",
                    "hints_text",
                    "hints",
                    "created_at",
                    "version",
                }
            },
        )

    def to_task_spec(self) -> TaskSpec:
        prompt = "\n".join(
            [
                "You are given a public SWE-bench-style software issue.",
                f"Instance: {self.instance_id}",
                f"Repository: {self.repo}",
                f"Base commit: {self.base_commit}",
                "",
                "Problem statement:",
                self.problem_statement.strip(),
                "",
                "Return only a unified diff patch that fixes the issue.",
                "Do not include explanations or test oracle contents.",
            ]
        )
        fixtures = {
            "public_swe": {
                "instance_id": self.instance_id,
                "repo": self.repo,
                "base_commit": self.base_commit,
                "source": self.source.model_dump(mode="json"),
                "fail_to_pass": self.fail_to_pass,
                "pass_to_pass": self.pass_to_pass,
                "has_gold_patch": bool(self.patch),
                "has_test_patch": bool(self.test_patch),
            }
        }
        return TaskSpec(
            task_id=_safe_task_id(self.instance_id),
            title=f"SWE issue fix: {self.instance_id}",
            category=TaskCategory.CODE_REPAIR,
            prompt=prompt,
            fixtures=fixtures,
            expected_artifacts=["submission/patch.diff"],
            allowed_paths=["submission/"],
            scoring=[
                ScoringRule(
                    name="patch_created",
                    kind="file_exists",
                    path="submission/patch.diff",
                    weight=0.2,
                ),
                ScoringRule(
                    name="swe_harness_resolved",
                    kind="metadata_flag",
                    expected="swe_resolved",
                    weight=0.8,
                ),
            ],
            tags=["public-swe", self.source.name, self.source.dataset],
            metadata={
                "external_instance_id": self.instance_id,
                "external_source": self.source.model_dump(mode="json"),
                "gold_patch_available_offline": bool(self.patch),
            },
        )

    def to_issue_to_patch_example(
        self,
        *,
        dataset_version_id: str,
        split: TrainingSplit,
        teacher_model_id: str = "public_gold_patch",
        prompt_version: str = "public-swe-issue-to-patch-v1",
    ) -> TrainingExample | None:
        if not self.patch:
            return None
        instruction = (
            "Generate a unified diff patch for the public SWE-bench-style issue. "
            "Use only the problem statement and repository metadata."
        )
        input_payload = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
        }
        return TrainingExample(
            example_id=stable_hash(f"{dataset_version_id}:{self.instance_id}:issue_to_patch"),
            source_task_id=_safe_task_id(self.instance_id),
            instruction=instruction,
            input=json.dumps(input_payload, ensure_ascii=False, indent=2),
            output=self.patch,
            training_task=TrainingTaskKind.ISSUE_TO_PATCH,
            split=split,
            teacher_model_id=teacher_model_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash(json.dumps(input_payload, ensure_ascii=False, sort_keys=True)),
            dataset_version_id=dataset_version_id,
            validation_status="accepted",
            license_note=self.source.license,
            metadata={
                "instance_id": self.instance_id,
                "source": self.source.model_dump(mode="json"),
            },
        )


def load_swe_instances_from_jsonl(
    path: Path,
    *,
    source: ExternalBenchmarkSource,
    limit: int | None = None,
) -> list[SWEIssueInstance]:
    instances: list[SWEIssueInstance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        instances.append(SWEIssueInstance.from_hf_row(json.loads(line), source=source))
        if limit is not None and len(instances) >= limit:
            break
    return instances


def load_swe_instances_from_json(
    path: Path,
    *,
    source: ExternalBenchmarkSource,
    limit: int | None = None,
) -> list[SWEIssueInstance]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        payload
        if isinstance(payload, list)
        else payload.get("instances", payload.get("rows", []))
    )
    instances = [
        SWEIssueInstance.from_hf_row(row, source=source)
        for row in rows[:limit]
    ]
    return instances


def load_swe_instances_from_hf(
    dataset_name: str,
    *,
    split: str = "test",
    source: ExternalBenchmarkSource | None = None,
    limit: int | None = None,
) -> list[SWEIssueInstance]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the training extra to load Hugging Face datasets") from exc

    dataset = load_dataset(dataset_name, split=split)
    rows = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        rows.append(row)
    resolved_source = source or ExternalBenchmarkSource(
        name="huggingface",
        dataset=dataset_name,
        split=split,
        url=f"https://huggingface.co/datasets/{dataset_name}",
    )
    return [SWEIssueInstance.from_hf_row(row, source=resolved_source) for row in rows]


def swe_instances_to_dataset_version(
    instances: list[SWEIssueInstance],
    *,
    dataset_version_id: str | None = None,
    seed: int = 13,
    prompt_version: str = "public-swe-issue-to-patch-v1",
) -> DatasetVersion:
    payload = "|".join(instance.instance_id for instance in instances)
    version_id = dataset_version_id or f"public-swe-data-{stable_hash(payload, length=12)}"
    examples: list[TrainingExample] = []
    for index, instance in enumerate(instances):
        example = instance.to_issue_to_patch_example(
            dataset_version_id=version_id,
            split=_split_for_index(index, seed),
            prompt_version=prompt_version,
        )
        if example is not None:
            examples.append(example)
    return DatasetVersion(
        dataset_version_id=version_id,
        source_dataset_id=instances[0].source.dataset if instances else None,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash(payload),
        teacher_provider=ProviderKind.MOCK,
        teacher_model_id="public_gold_patch",
        created_at=datetime.now(UTC).isoformat(),
        seed=seed,
        split_counts=_split_counts(examples),
        references=[instances[0].source.url] if instances and instances[0].source.url else [],
        examples=examples,
        metadata={
            "source": instances[0].source.model_dump(mode="json") if instances else None,
            "instance_count": len(instances),
            "trainable_count": len(examples),
            "training_task": TrainingTaskKind.ISSUE_TO_PATCH.value,
        },
    )


def _split_for_index(index: int, seed: int) -> TrainingSplit:
    bucket = (index + seed) % 10
    if bucket < 8:
        return TrainingSplit.TRAIN
    if bucket == 8:
        return TrainingSplit.VALIDATION
    return TrainingSplit.TEST


def _split_counts(examples: list[TrainingExample]) -> dict[str, int]:
    counts = {split.value: 0 for split in TrainingSplit}
    for example in examples:
        counts[example.split.value] += 1
    return counts


def _coerce_test_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
        return [str(decoded)]
    return [str(value)]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _safe_task_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "." for char in value)
