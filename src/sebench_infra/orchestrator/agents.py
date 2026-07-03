from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sebench_infra.benchmark.schemas import TaskSpec
from sebench_infra.orchestrator.workspace import _string_map, _write_files


@dataclass(frozen=True)
class AgentRunResult:
    artifacts: dict[str, str]
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PatchAgentRunner(Protocol):
    backend: str

    def run(
        self,
        *,
        task: TaskSpec,
        work_dir: Path,
        fixture: dict[str, Any],
        patch_required: bool,
    ) -> AgentRunResult:
        """Run an agent in the work directory and return submitted artifacts."""


class FixturePatchAgentRunner:
    """Deterministic runner backed by fixture-provided files or patches."""

    backend = "fixture"

    def run(
        self,
        *,
        task: TaskSpec,
        work_dir: Path,
        fixture: dict[str, Any],
        patch_required: bool,
    ) -> AgentRunResult:
        start = time.perf_counter()
        agent_patch = fixture.get("agent_patch")
        if patch_required or isinstance(agent_patch, str):
            _write_files(work_dir, {"submission/model.patch": str(agent_patch or "")})
            candidate_paths = ["submission/model.patch"]
        else:
            agent_files = _string_map(fixture.get("agent_files", {}))
            _write_files(work_dir, agent_files)
            candidate_paths = sorted(agent_files)

        artifacts = {}
        for path in candidate_paths:
            target = work_dir / path
            if target.is_file():
                artifacts[path] = target.read_text(encoding="utf-8")
        return AgentRunResult(
            artifacts=artifacts,
            latency_sec=time.perf_counter() - start,
            metadata={"agent_backend": self.backend},
        )


class CodexCliPatchAgentRunner:
    """Experimental local Codex CLI adapter for small smoke tests."""

    backend = "codex_cli"

    def __init__(
        self,
        *,
        codex_binary: str = "codex",
        model: str | None = None,
        timeout_sec: float = 300.0,
        run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.codex_binary = codex_binary
        self.model = model
        self.timeout_sec = timeout_sec
        self._run_command = run_command

    def run(
        self,
        *,
        task: TaskSpec,
        work_dir: Path,
        fixture: dict[str, Any],
        patch_required: bool,
    ) -> AgentRunResult:
        start = time.perf_counter()
        raw_output_path = work_dir / "artifacts/work/codex_last_message.txt"
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path = work_dir / "submission/model.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = _codex_patch_prompt(task)
        command = [
            self.codex_binary,
            "exec",
            "--cd",
            str(work_dir),
            "--sandbox",
            "workspace-write",
            "--ephemeral",
            "--output-last-message",
            str(raw_output_path),
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append(prompt)

        failure_reason: str | None = None
        try:
            completed = self._run_command(
                command,
                cwd=work_dir,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            completed = None
            failure_reason = "codex_cli_timeout"

        raw_output = ""
        if raw_output_path.exists():
            raw_output = raw_output_path.read_text(encoding="utf-8", errors="replace")
        elif completed is not None:
            raw_output = "\n".join([completed.stdout or "", completed.stderr or ""]).strip()

        if not patch_path.exists():
            extracted = _extract_unified_diff(raw_output)
            if extracted:
                patch_path.write_text(extracted, encoding="utf-8")

        artifacts = {}
        patch_validation_status = "missing_patch"
        if patch_path.is_file():
            patch_text = patch_path.read_text(encoding="utf-8")
            if patch_text.strip():
                artifacts["submission/model.patch"] = patch_text
                patch_validation_status = "collected"

        if completed is not None and completed.returncode != 0 and failure_reason is None:
            failure_reason = "codex_cli_error"

        return AgentRunResult(
            artifacts=artifacts,
            latency_sec=time.perf_counter() - start,
            metadata={
                "agent_backend": self.backend,
                "codex_binary": self.codex_binary,
                "codex_model": self.model,
                "codex_returncode": completed.returncode if completed is not None else None,
                "codex_raw_output_path": str(raw_output_path.relative_to(work_dir)),
                "agent_failure_reason": failure_reason,
                "patch_validation_status": patch_validation_status,
            },
        )


def build_patch_agent_runner(
    *,
    agent_backend: str = "fixture",
    codex_binary: str = "codex",
    codex_model: str | None = None,
    codex_timeout_sec: float = 300.0,
) -> PatchAgentRunner:
    if agent_backend == "fixture":
        return FixturePatchAgentRunner()
    if agent_backend == "codex_cli":
        return CodexCliPatchAgentRunner(
            codex_binary=codex_binary,
            model=codex_model,
            timeout_sec=codex_timeout_sec,
        )
    raise ValueError("agent_backend must be one of: fixture, codex_cli")


def _codex_patch_prompt(task: TaskSpec) -> str:
    return "\n".join(
        [
            "You are running inside a benchmark work directory.",
            "Do not look for or infer hidden tests or oracle answers.",
            "Use only files already present in this work directory and the public task prompt.",
            "Write exactly one unified diff patch to submission/model.patch.",
            "Do not modify files outside this work directory.",
            "",
            f"Task id: {task.task_id}",
            f"Title: {task.title}",
            "Prompt:",
            task.prompt,
            "",
            "Expected modified paths:",
            "\n".join(f"- {path}" for path in task.expected_artifacts) or "- n/a",
        ]
    )


def _extract_unified_diff(raw_output: str) -> str | None:
    fenced = re.search(r"```(?:diff)?\s*(.*?)```", raw_output, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        if _looks_like_unified_diff(candidate):
            return _ensure_trailing_newline(candidate)

    markers = [match.start() for match in re.finditer(r"(?m)^(diff --git |--- )", raw_output)]
    for start in markers:
        candidate = raw_output[start:].strip()
        if _looks_like_unified_diff(candidate):
            return _ensure_trailing_newline(candidate)
    return None


def _looks_like_unified_diff(value: str) -> bool:
    return (
        ("diff --git " in value and "\n@@ " in value)
        or (value.startswith("--- ") and "\n+++ " in value and "\n@@ " in value)
        or ("\n--- " in value and "\n+++ " in value and "\n@@ " in value)
    )


def _ensure_trailing_newline(value: str) -> str:
    return value if value.endswith("\n") else f"{value}\n"
