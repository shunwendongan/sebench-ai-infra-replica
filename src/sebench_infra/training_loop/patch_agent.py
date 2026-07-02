from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from sebench_infra.authoring.llm import LLMClient
from sebench_infra.benchmark.schemas import TaskSpec


class PatchClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict:
        """Return a JSON object with a patch field."""


class StaticPatchClient:
    def __init__(self, patch: str) -> None:
        self.patch = patch

    def complete_json(self, system: str, user: str) -> dict:
        return {"patch": self.patch}


class ModelPatchAgent:
    """Single-turn issue-to-patch agent for public SWE-style tasks."""

    def __init__(self, client: PatchClient | LLMClient) -> None:
        self.client = client

    def generate_patch(self, task: TaskSpec) -> str:
        payload = self.client.complete_json(_PATCH_SYSTEM_PROMPT, _task_to_patch_prompt(task))
        patch = payload.get("patch") or payload.get("diff") or payload.get("unified_diff")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("model response did not contain a non-empty patch")
        patch = _strip_fences(patch).strip()
        if not patch.endswith("\n"):
            patch += "\n"
        if not looks_like_unified_diff(patch):
            raise ValueError("model response is not a unified diff patch")
        return patch


def looks_like_unified_diff(patch: str) -> bool:
    return (
        ("diff --git " in patch and "\n@@ " in patch)
        or ("\n--- " in patch and "\n+++ " in patch and "\n@@ " in patch)
        or (patch.startswith("--- ") and "\n+++ " in patch and "\n@@ " in patch)
    )


def apply_unified_diff_to_files(
    files: dict[str, str],
    patch: str,
) -> tuple[bool, dict[str, str], str | None]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel_path, content in files.items():
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        check = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if check.returncode != 0:
            return False, files, check.stderr.strip() or check.stdout.strip()
        applied = subprocess.run(
            ["git", "apply", "-"],
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if applied.returncode != 0:
            return False, files, applied.stderr.strip() or applied.stdout.strip()

        updated: dict[str, str] = {}
        for path in root.rglob("*"):
            if path.is_file() and ".git" not in path.parts:
                updated[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        return True, updated, None


def _task_to_patch_prompt(task: TaskSpec) -> str:
    public_swe = task.fixtures.get("public_swe", {})
    lines = [
        task.prompt,
        "",
        "Public metadata:",
        f"- instance_id: {public_swe.get('instance_id', task.task_id)}",
        f"- repo: {public_swe.get('repo', 'unknown')}",
        f"- base_commit: {public_swe.get('base_commit', 'unknown')}",
        "",
        "Output JSON with exactly one key: patch.",
    ]
    return "\n".join(lines)


def _strip_fences(value: str) -> str:
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return value


_PATCH_SYSTEM_PROMPT = (
    "You are a software engineering patch agent. Return a JSON object containing "
    "only a unified diff patch. Do not include explanations."
)
