#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Template:
    name: str
    module_path: str
    broken: str
    fixed: str
    hidden_test: str


TEMPLATES = [
    Template(
        name="calc",
        module_path="pkg/calc.py",
        broken="def add(a, b):\n    return a - b\n",
        fixed="def add(a, b):\n    return a + b\n",
        hidden_test=(
            "from pkg.calc import add\n\n"
            "def test_add_positive_and_negative():\n"
            "    assert add(2, 3) == 5\n"
            "    assert add(-4, 6) == 2\n"
        ),
    ),
    Template(
        name="text",
        module_path="pkg/text.py",
        broken="def slugify(text):\n    return text.replace(' ', '_')\n",
        fixed="def slugify(text):\n    return text.strip().lower().replace(' ', '-')\n",
        hidden_test=(
            "from pkg.text import slugify\n\n"
            "def test_slugify():\n"
            "    assert slugify(' Hello World ') == 'hello-world'\n"
        ),
    ),
    Template(
        name="stats",
        module_path="pkg/stats.py",
        broken=(
            "def median(values):\n"
            "    ordered = sorted(values)\n"
            "    return ordered[len(ordered) // 2]\n"
        ),
        fixed=(
            "def median(values):\n"
            "    ordered = sorted(values)\n"
            "    n = len(ordered)\n"
            "    mid = n // 2\n"
            "    if n % 2:\n"
            "        return ordered[mid]\n"
            "    return (ordered[mid - 1] + ordered[mid]) / 2\n"
        ),
        hidden_test=(
            "from pkg.stats import median\n\n"
            "def test_median_even_and_odd():\n"
            "    assert median([3, 1, 2]) == 2\n"
            "    assert median([4, 1, 2, 3]) == 2.5\n"
        ),
    ),
    Template(
        name="config",
        module_path="pkg/config.py",
        broken="def env_bool(value):\n    return bool(value)\n",
        fixed=(
            "def env_bool(value):\n"
            "    normalized = str(value).strip().lower()\n"
            "    if normalized in {'1', 'true', 'yes', 'on'}:\n"
            "        return True\n"
            "    if normalized in {'0', 'false', 'no', 'off'}:\n"
            "        return False\n"
            "    raise ValueError(f'not a boolean: {value}')\n"
        ),
        hidden_test=(
            "from pkg.config import env_bool\n\n"
            "def test_env_bool():\n"
            "    assert env_bool('YES') is True\n"
            "    assert env_bool('off') is False\n"
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local real-git pytest benchmark fixtures."
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=128,
        help="Number of task instances to generate from the template repos.",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=ROOT / "examples/git_fixtures",
        help="Directory for generated local git repositories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "examples/git_pytest_benchmark.json",
        help="Manifest output path.",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def build_repo(root: Path, template: Template) -> tuple[Path, str]:
    repo = root / f"{template.name}_repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    write_text(repo / "pkg/__init__.py", "")
    write_text(repo / template.module_path, template.broken)
    write_text(
        repo / "README.md",
        f"# {template.name} fixture\n\nGenerated local git fixture for SE-Bench tests.\n",
    )
    run(["git", "init", "--quiet"], repo)
    run(["git", "config", "user.email", "sebench@example.local"], repo)
    run(["git", "config", "user.name", "SE Bench Fixture"], repo)
    run(["git", "add", "."], repo)
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_DATE": "2026-06-30T00:00:00+0000",
            "GIT_COMMITTER_DATE": "2026-06-30T00:00:00+0000",
        }
    )
    run(["git", "commit", "--quiet", "-m", f"base {template.name} fixture"], repo, env)
    return repo, run(["git", "rev-parse", "HEAD"], repo)


def unified_diff(path: str, before: str, after: str) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="\n",
        )
    )


def hidden_score() -> list[dict[str, Any]]:
    return [
        {
            "name": "hidden_judge_passed",
            "kind": "metadata_flag",
            "expected": "judge_passed",
            "weight": 1.0,
        }
    ]


def make_task(index: int, template: Template, repo: Path, base_commit: str) -> dict[str, Any]:
    patch = unified_diff(template.module_path, template.broken, template.fixed)
    try:
        repo_path = str(repo.resolve().relative_to(ROOT))
    except ValueError:
        repo_path = str(repo.resolve())
    return {
        "task_id": f"gitpytest.{template.name}.{index:04d}",
        "title": f"Fix {template.name} local git fixture #{index:04d}",
        "category": "code_repair",
        "prompt": (
            "Submit a unified diff patch that fixes the described bug. "
            "Hidden pytest tests will validate behavior."
        ),
        "fixtures": {
            "git_repo": {
                "repo_id": f"local-{template.name}",
                "path": repo_path,
                "base_commit": base_commit,
                "agent_patch": patch,
                "patch_allowed_paths": ["pkg/"],
                "hidden_judge": [
                    {
                        "kind": "pytest",
                        "files": {
                            f"tests/test_hidden_{template.name}_{index:04d}.py": (
                                template.hidden_test
                            )
                        },
                        "args": ["-q"],
                        "timeout_sec": 10,
                    }
                ],
            }
        },
        "expected_artifacts": [template.module_path],
        "allowed_paths": ["submission/model.patch"],
        "scoring": hidden_score(),
        "tags": ["local_git", "pytest", "patch_based", "round6"],
        "metadata": {
            "difficulty": "easy-medium",
            "submission_mode": "unified_diff_patch",
            "judge": "pytest",
        },
    }


def main() -> None:
    args = parse_args()
    args.fixture_root.mkdir(parents=True, exist_ok=True)
    repos = {
        template.name: build_repo(args.fixture_root, template)
        for template in TEMPLATES
    }
    tasks = []
    for index in range(args.tasks):
        template = TEMPLATES[index % len(TEMPLATES)]
        repo, base_commit = repos[template.name]
        tasks.append(make_task(index, template, repo, base_commit))

    manifest = {
        "dataset_id": "sebench-git-pytest-v0.1",
        "version": "0.1.0",
        "source": "local_real_git_pytest_true_loop",
        "references": [
            "https://arxiv.org/abs/2310.06770",
            "https://github.com/SWE-bench/SWE-bench",
        ],
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(tasks)} git pytest tasks to {args.out}")


if __name__ == "__main__":
    main()
