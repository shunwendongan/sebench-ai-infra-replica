from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sebench_infra.benchmark.schemas import TaskSpec
from sebench_infra.training_loop.models import ValidationIssue

PRIVATE_CLAIM_PATTERNS = (
    r"\bbytedance\b",
    r"\bseed\b",
    r"\binternal\b",
    r"\bconfidential\b",
    r"\bprivate\b",
    "字节",
    "内部",
    "保密",
    "私有",
)

LEAKAGE_PATTERNS = (
    "hidden_judge",
    "hidden_tests",
    "secret",
    "gold answer",
    "ground truth",
    "参考答案",
    "隐藏测试",
)


def stable_hash(payload: str, *, length: int = 16) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def prompt_hash(prompt: str) -> str:
    return stable_hash(_normalize(prompt), length=12)


def task_fingerprint(task: TaskSpec) -> str:
    payload = {
        "title": _normalize(task.title),
        "category": task.category.value,
        "prompt": _normalize(task.prompt),
        "expected_artifacts": sorted(task.expected_artifacts),
        "allowed_paths": sorted(task.allowed_paths),
        "scoring": [rule.model_dump(mode="json") for rule in task.scoring],
    }
    return stable_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False), length=16)


def validate_task_for_training(task: TaskSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    searchable = "\n".join(
        [
            task.task_id,
            task.title,
            task.prompt,
            " ".join(task.tags),
            json.dumps(task.metadata, ensure_ascii=False, sort_keys=True),
        ]
    )

    for pattern in PRIVATE_CLAIM_PATTERNS:
        if re.search(pattern, searchable, flags=re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    code="private_claim",
                    message=f"task text contains private or employer-specific claim: {pattern}",
                )
            )

    prompt_lower = task.prompt.lower()
    for pattern in LEAKAGE_PATTERNS:
        if pattern.lower() in prompt_lower:
            issues.append(
                ValidationIssue(
                    code="judge_leakage",
                    message=f"prompt appears to expose hidden evaluation detail: {pattern}",
                )
            )

    if not task.expected_artifacts:
        issues.append(
            ValidationIssue(
                code="missing_expected_artifacts",
                message="task must declare at least one expected artifact for training data",
            )
        )
    if not task.allowed_paths:
        issues.append(
            ValidationIssue(
                code="missing_allowed_paths",
                message="task must constrain writable paths",
            )
        )
    if not task.scoring:
        issues.append(
            ValidationIssue(
                code="missing_scoring",
                message="task must include deterministic scoring rules",
            )
        )

    for rule in task.scoring:
        if rule.kind in {"contains", "exact_match", "numeric_close"} and rule.path is None:
            issues.append(
                ValidationIssue(
                    code="unanchored_scoring_rule",
                    message=f"scoring rule {rule.name!r} must bind to an artifact path",
                )
            )

    return issues


def deduplicate_tasks(tasks: list[TaskSpec]) -> tuple[list[TaskSpec], dict[str, str]]:
    seen: dict[str, str] = {}
    accepted: list[TaskSpec] = []
    duplicate_reasons: dict[str, str] = {}
    for task in tasks:
        fingerprint = task_fingerprint(task)
        if fingerprint in seen:
            duplicate_reasons[task.task_id] = seen[fingerprint]
            continue
        seen[fingerprint] = task.task_id
        accepted.append(task)
    return accepted, duplicate_reasons


def json_dumps_stable(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize(value: str) -> str:
    return " ".join(value.strip().split())
