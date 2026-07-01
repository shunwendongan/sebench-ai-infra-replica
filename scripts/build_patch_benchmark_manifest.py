#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a file-artifact benchmark manifest into patch-submission tasks."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("examples/realistic_benchmark.json"),
        help="Source manifest with toy_repo.agent_files fixtures.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/patch_benchmark.json"),
        help="Patch-submission manifest output path.",
    )
    return parser.parse_args()


def unified_diff(path: str, before: str | None, after: str) -> str:
    before_lines = [] if before is None else before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    if after_lines and not after_lines[-1].endswith("\n"):
        after_lines[-1] += "\n"
    fromfile = "/dev/null" if before is None else f"a/{path}"
    tofile = f"b/{path}"
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=fromfile,
            tofile=tofile,
            lineterm="\n",
        )
    )


def patch_from_files(files: dict[str, str], agent_files: dict[str, str]) -> str:
    parts = []
    for path in sorted(agent_files):
        parts.append(unified_diff(path, files.get(path), agent_files[path]))
    return "".join(parts)


def convert_task(task: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(task))
    converted["task_id"] = f"patch.{task['task_id']}"
    converted["title"] = f"Patch submission: {task['title']}"
    converted["tags"] = sorted(set(converted.get("tags", [])) | {"patch_based", "round4"})
    toy_repo = converted["fixtures"]["toy_repo"]
    files = {str(path): str(content) for path, content in toy_repo.get("files", {}).items()}
    agent_files = {
        str(path): str(content) for path, content in toy_repo.get("agent_files", {}).items()
    }
    toy_repo["agent_patch"] = patch_from_files(files, agent_files)
    toy_repo["patch_allowed_paths"] = converted.get("allowed_paths", [])
    del toy_repo["agent_files"]
    converted["expected_artifacts"] = sorted(agent_files)
    converted["allowed_paths"] = ["submission/model.patch"]
    converted["metadata"] = {
        **converted.get("metadata", {}),
        "submission_mode": "unified_diff_patch",
    }
    return converted


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    manifest = {
        "dataset_id": "sebench-patch-v0.1",
        "version": source.get("version", "0.1.0"),
        "source": "patch_submission_true_loop",
        "references": source.get("references", []),
        "tasks": [convert_task(task) for task in source["tasks"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest['tasks'])} patch tasks to {args.out}")


if __name__ == "__main__":
    main()
