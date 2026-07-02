from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from sebench_infra.authoring.llm import LLMClient
from sebench_infra.authoring.prompts import AUTHORING_SYSTEM_PROMPT, REPAIR_SYSTEM_PROMPT
from sebench_infra.benchmark.schemas import TaskSpec
from sebench_infra.training_loop.models import (
    DatasetVersion,
    ProviderKind,
    TrainingExample,
    TrainingSplit,
    TrainingTaskKind,
    ValidationIssue,
)
from sebench_infra.training_loop.validation import (
    json_dumps_stable,
    prompt_hash,
    stable_hash,
    task_fingerprint,
    validate_task_for_training,
)


class TeacherDataGenerator:
    """Generate auditable SFT examples from teacher-authored benchmark tasks."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        teacher_provider: ProviderKind | str = ProviderKind.MOCK,
        teacher_model_id: str = "mock",
        prompt_version: str = "teacher-data-v1",
        seed: int = 13,
        max_repairs: int = 1,
    ) -> None:
        self.llm = llm
        self.teacher_provider = ProviderKind(teacher_provider)
        self.teacher_model_id = teacher_model_id
        self.prompt_version = prompt_version
        self.seed = seed
        self.max_repairs = max_repairs

    def generate(
        self,
        requirement: str,
        *,
        references: list[str] | None = None,
        count: int = 1,
        source_dataset_id: str | None = None,
    ) -> DatasetVersion:
        references = references or []
        version_id = self._dataset_version_id(requirement, references, count)
        prompt_digest = prompt_hash(f"{self.prompt_version}\n{requirement}")
        created_at = datetime.now(UTC).isoformat()
        examples: list[TrainingExample] = []
        seen: dict[str, str] = {}
        accepted_index = 0

        for index in range(max(1, count)):
            user_prompt = self._candidate_prompt(requirement, references, index)
            tasks, repair_count, raw_payload, error = self._author_tasks(user_prompt)
            if error:
                examples.append(
                    self._rejected_example(
                        version_id=version_id,
                        requirement=user_prompt,
                        raw_payload=raw_payload,
                        index=index,
                        issues=[
                            ValidationIssue(
                                code="schema_validation_failed",
                                message=error,
                            )
                        ],
                        repair_count=repair_count,
                    )
                )
                continue

            for task in tasks:
                issues = validate_task_for_training(task)
                fingerprint = task_fingerprint(task)
                if fingerprint in seen:
                    issues.append(
                        ValidationIssue(
                            code="duplicate_task",
                            message=f"duplicate of source task {seen[fingerprint]}",
                        )
                    )
                if issues:
                    examples.append(
                        self._task_example(
                            task=task,
                            version_id=version_id,
                            requirement=user_prompt,
                            split=TrainingSplit.REJECTED,
                            status="rejected",
                            issues=issues,
                            accepted_index=accepted_index,
                            repair_count=repair_count,
                        )
                    )
                    continue

                seen[fingerprint] = task.task_id
                split = self._split_for_index(accepted_index)
                examples.append(
                    self._task_example(
                        task=task,
                        version_id=version_id,
                        requirement=user_prompt,
                        split=split,
                        status="accepted",
                        issues=[],
                        accepted_index=accepted_index,
                        repair_count=repair_count,
                    )
                )
                accepted_index += 1

        return DatasetVersion(
            dataset_version_id=version_id,
            source_dataset_id=source_dataset_id,
            prompt_version=self.prompt_version,
            prompt_hash=prompt_digest,
            teacher_provider=self.teacher_provider,
            teacher_model_id=self.teacher_model_id,
            created_at=created_at,
            seed=self.seed,
            split_counts=self._split_counts(examples),
            references=references,
            examples=examples,
            metadata={
                "requirement_hash": stable_hash(requirement, length=12),
                "candidate_count": max(1, count),
                "accepted_count": sum(1 for example in examples if example.is_trainable),
                "rejected_count": sum(
                    1 for example in examples if example.validation_status == "rejected"
                ),
            },
        )

    def _author_tasks(
        self,
        requirement: str,
    ) -> tuple[list[TaskSpec], int, dict[str, Any], str | None]:
        repair_count = 0
        current_payload = self.llm.complete_json(AUTHORING_SYSTEM_PROMPT, requirement)
        last_error: ValidationError | None = None

        for attempt in range(self.max_repairs + 1):
            try:
                tasks = [TaskSpec.model_validate(item) for item in current_payload.get("tasks", [])]
                if not tasks:
                    return [], repair_count, current_payload, "teacher returned no tasks"
                return tasks, repair_count, current_payload, None
            except ValidationError as exc:
                last_error = exc
                if attempt >= self.max_repairs:
                    break
                repair_count += 1
                current_payload = self.llm.complete_json(
                    REPAIR_SYSTEM_PROMPT,
                    "\n\n".join(
                        [
                            f"Requirement:\n{requirement}",
                            f"Validation error:\n{exc}",
                            f"Payload:\n{current_payload}",
                        ]
                    ),
                )

        return [], repair_count, current_payload, str(last_error)

    def _task_example(
        self,
        *,
        task: TaskSpec,
        version_id: str,
        requirement: str,
        split: TrainingSplit,
        status: str,
        issues: list[ValidationIssue],
        accepted_index: int,
        repair_count: int,
    ) -> TrainingExample:
        output = task.model_dump_json()
        return TrainingExample(
            example_id=stable_hash(f"{version_id}:{task.task_id}:{status}:{accepted_index}"),
            source_task_id=task.task_id,
            instruction=_TRAINING_INSTRUCTION,
            input=requirement,
            output=output,
            training_task=TrainingTaskKind.TASK_SPEC_AUTHORING,
            split=split,
            teacher_model_id=self.teacher_model_id,
            prompt_version=self.prompt_version,
            prompt_hash=prompt_hash(requirement),
            dataset_version_id=version_id,
            validation_status="accepted" if status == "accepted" else "rejected",
            validation_issues=issues,
            metadata={
                "task_fingerprint": task_fingerprint(task),
                "schema_repair_count": repair_count,
            },
        )

    def _rejected_example(
        self,
        *,
        version_id: str,
        requirement: str,
        raw_payload: dict[str, Any],
        index: int,
        issues: list[ValidationIssue],
        repair_count: int,
    ) -> TrainingExample:
        return TrainingExample(
            example_id=stable_hash(f"{version_id}:invalid:{index}"),
            source_task_id=f"invalid-{index}",
            instruction=_TRAINING_INSTRUCTION,
            input=requirement,
            output=json_dumps_stable(raw_payload),
            training_task=TrainingTaskKind.TASK_SPEC_AUTHORING,
            split=TrainingSplit.REJECTED,
            teacher_model_id=self.teacher_model_id,
            prompt_version=self.prompt_version,
            prompt_hash=prompt_hash(requirement),
            dataset_version_id=version_id,
            validation_status="rejected",
            validation_issues=issues,
            metadata={"schema_repair_count": repair_count},
        )

    def _candidate_prompt(self, requirement: str, references: list[str], index: int) -> str:
        payload = {
            "requirement": requirement,
            "candidate_index": index,
            "references": references,
            "constraints": [
                "Use only public or synthetic data.",
                "Return TaskSpec-compatible JSON.",
                "Do not expose hidden judge answers in the prompt.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _dataset_version_id(self, requirement: str, references: list[str], count: int) -> str:
        payload = json.dumps(
            {
                "requirement": requirement,
                "references": references,
                "count": count,
                "prompt_version": self.prompt_version,
                "teacher_model_id": self.teacher_model_id,
                "seed": self.seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"sebench-student-data-{stable_hash(payload, length=12)}"

    def _split_for_index(self, index: int) -> TrainingSplit:
        bucket = (index + self.seed) % 10
        if bucket < 8:
            return TrainingSplit.TRAIN
        if bucket == 8:
            return TrainingSplit.VALIDATION
        return TrainingSplit.TEST

    def _split_counts(self, examples: list[TrainingExample]) -> dict[str, int]:
        counts = {split.value: 0 for split in TrainingSplit}
        for example in examples:
            counts[example.split.value] += 1
        return counts


_TRAINING_INSTRUCTION = (
    "Author one SE-Bench-style TaskSpec JSON object for the provided requirement. "
    "Use public/synthetic fixtures, relative allowed paths, deterministic scoring, "
    "and no private or employer-internal claims."
)
