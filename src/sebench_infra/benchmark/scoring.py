from collections.abc import Mapping
from math import isclose
from typing import Any

from sebench_infra.benchmark.schemas import ScoringRule, TaskRunResult, TaskSpec


class ScoreEngine:
    """Weighted deterministic scorer for local and judge-container results."""

    def score_task(self, task: TaskSpec, submission: Mapping[str, Any]) -> TaskRunResult:
        weighted_score = 0.0
        total_weight = sum(rule.weight for rule in task.scoring)
        rule_details: dict[str, Any] = {}
        metadata = submission.get("metadata", {})

        for rule in task.scoring:
            passed = self._score_rule(rule, submission)
            weighted_score += rule.weight * float(passed)
            rule_details[rule.name] = {"passed": passed, "kind": rule.kind}

        score = weighted_score / total_weight if total_weight else 0.0
        status = "passed" if score >= 0.999 else "failed"
        return TaskRunResult(
            task_id=task.task_id,
            status=status,
            score=round(score, 6),
            details={"rules": rule_details},
            artifacts={k: str(v) for k, v in submission.get("artifacts", {}).items()},
            patch_hash=metadata.get("patch_hash"),
            failure_type=metadata.get("failure_type") if status != "passed" else None,
        )

    def _score_rule(self, rule: ScoringRule, submission: Mapping[str, Any]) -> bool:
        artifacts = submission.get("artifacts", {})
        metadata = submission.get("metadata", {})
        value = artifacts.get(rule.path or "") if rule.path else submission.get("answer")

        if rule.kind == "file_exists":
            return bool(value)
        if rule.kind == "contains":
            expected = "" if rule.expected is None else str(rule.expected)
            return value is not None and expected in str(value)
        if rule.kind == "exact_match":
            return str(value).strip() == str(rule.expected).strip()
        if rule.kind == "numeric_close":
            try:
                return isclose(float(value), float(rule.expected), abs_tol=rule.tolerance)
            except (TypeError, ValueError):
                return False
        if rule.kind == "metadata_flag":
            return metadata.get(str(rule.expected)) is True
        raise ValueError(f"unsupported scoring rule: {rule.kind}")
