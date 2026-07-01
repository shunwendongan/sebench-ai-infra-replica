#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simulated-real-repo SE-Bench manifest for local validation."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/realistic_benchmark.json"),
        help="Manifest output path.",
    )
    return parser.parse_args()


def hidden_score() -> list[dict[str, Any]]:
    return [
        {
            "name": "hidden_judge_passed",
            "kind": "metadata_flag",
            "expected": "judge_passed",
            "weight": 1.0,
        }
    ]


def code_task(
    task_id: str,
    title: str,
    category: str,
    repo_id: str,
    path: str,
    broken: str,
    fixed: str,
    hidden_code: str,
    prompt: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "title": title,
        "category": category,
        "prompt": prompt or title,
        "fixtures": {
            "toy_repo": {
                "repo_id": repo_id,
                "files": {
                    "app/__init__.py": "",
                    path: broken,
                },
                "agent_files": {
                    path: fixed,
                },
                "hidden_judge": [
                    {
                        "kind": "python_inline_tests",
                        "code": hidden_code,
                    }
                ],
            }
        },
        "expected_artifacts": [path],
        "allowed_paths": ["app/"],
        "scoring": hidden_score(),
        "tags": ["simulated_real_repo", "true_loop", "round3"],
        "metadata": {
            "difficulty": "easy-medium",
            "failure_type_if_wrong": "wrong_edit",
            "repo_style": "python_package",
        },
    }


def build_tasks() -> list[dict[str, Any]]:
    tasks = [
        code_task(
            "realistic.metrics.percent_change",
            "Fix percent change calculation",
            "numeric_reasoning",
            "analytics-core",
            "app/metrics.py",
            "def percent_change(old, new):\n    return new - old\n",
            (
                "def percent_change(old, new):\n"
                "    if old == 0:\n"
                "        return None\n"
                "    return round((new - old) / old, 4)\n"
            ),
            (
                "from app.metrics import percent_change\n"
                "assert percent_change(100, 125) == 0.25\n"
                "assert percent_change(80, 60) == -0.25\n"
                "assert percent_change(0, 10) is None\n"
            ),
        ),
        code_task(
            "realistic.validation.email",
            "Tighten email validation",
            "code_repair",
            "user-service",
            "app/validators.py",
            "def is_valid_email(value):\n    return '@' in value\n",
            (
                "def is_valid_email(value):\n"
                "    if not isinstance(value, str) or value.count('@') != 1:\n"
                "        return False\n"
                "    local, domain = value.split('@')\n"
                "    return bool(local) and '.' in domain and ' ' not in value\n"
            ),
            (
                "from app.validators import is_valid_email\n"
                "assert is_valid_email('ada@example.com')\n"
                "assert not is_valid_email('ada example.com')\n"
                "assert not is_valid_email('ada@@example.com')\n"
            ),
        ),
        code_task(
            "realistic.csv.sum_amounts",
            "Parse CSV amounts",
            "data_pipeline",
            "analytics-core",
            "app/csv_tools.py",
            "def sum_amounts(text):\n    return len(text.splitlines())\n",
            (
                "import csv\n"
                "import io\n\n"
                "def sum_amounts(text):\n"
                "    rows = csv.DictReader(io.StringIO(text))\n"
                "    return sum(float(row['amount']) for row in rows)\n"
            ),
            (
                "from app.csv_tools import sum_amounts\n"
                "data = 'id,amount\\n1,10.5\\n2,2.25\\n'\n"
                "assert sum_amounts(data) == 12.75\n"
            ),
        ),
        code_task(
            "realistic.mapping.default",
            "Return mapping default safely",
            "code_repair",
            "platform-utils",
            "app/mapping.py",
            "def get_or_default(mapping, key, default=None):\n    return mapping[key]\n",
            (
                "def get_or_default(mapping, key, default=None):\n"
                "    return mapping.get(key, default)\n"
            ),
            (
                "from app.mapping import get_or_default\n"
                "assert get_or_default({'a': 1}, 'a') == 1\n"
                "assert get_or_default({'a': 1}, 'b', 5) == 5\n"
            ),
        ),
        code_task(
            "realistic.pricing.discount",
            "Apply percentage discount",
            "numeric_reasoning",
            "shop-core",
            "app/pricing.py",
            (
                "def discounted_total(items, discount):\n"
                "    subtotal = sum(items)\n"
                "    return subtotal + discount\n"
            ),
            (
                "def discounted_total(items, discount):\n"
                "    subtotal = sum(items)\n"
                "    return round(subtotal * (1 - discount), 2)\n"
            ),
            (
                "from app.pricing import discounted_total\n"
                "assert discounted_total([10, 20], 0.1) == 27.0\n"
                "assert discounted_total([9.99, 5.01], 0.2) == 12.0\n"
            ),
        ),
        code_task(
            "realistic.dates.weekday",
            "Parse ISO date weekday",
            "code_repair",
            "platform-utils",
            "app/dates.py",
            "def iso_weekday(value):\n    return int(value.split('-')[-1])\n",
            (
                "from datetime import date\n\n"
                "def iso_weekday(value):\n"
                "    return date.fromisoformat(value).isoweekday()\n"
            ),
            (
                "from app.dates import iso_weekday\n"
                "assert iso_weekday('2026-06-30') == 2\n"
                "assert iso_weekday('2026-07-05') == 7\n"
            ),
        ),
        code_task(
            "realistic.stats.median",
            "Fix median for even inputs",
            "numeric_reasoning",
            "analytics-core",
            "app/stats.py",
            (
                "def median(values):\n"
                "    ordered = sorted(values)\n"
                "    return ordered[len(ordered)//2]\n"
            ),
            (
                "def median(values):\n"
                "    ordered = sorted(values)\n"
                "    n = len(ordered)\n"
                "    mid = n // 2\n"
                "    if n % 2:\n"
                "        return ordered[mid]\n"
                "    return (ordered[mid - 1] + ordered[mid]) / 2\n"
            ),
            (
                "from app.stats import median\n"
                "assert median([3, 1, 2]) == 2\n"
                "assert median([4, 1, 2, 3]) == 2.5\n"
            ),
        ),
        code_task(
            "realistic.text.normalize_space",
            "Normalize whitespace",
            "code_repair",
            "platform-utils",
            "app/text.py",
            "def normalize_space(text):\n    return text.replace(' ', '')\n",
            "def normalize_space(text):\n    return ' '.join(text.split())\n",
            (
                "from app.text import normalize_space\n"
                "assert normalize_space('  hello   world\\n') == 'hello world'\n"
                "assert normalize_space('a\\tb') == 'a b'\n"
            ),
        ),
        code_task(
            "realistic.security.mask_token",
            "Mask API token display",
            "code_repair",
            "security-utils",
            "app/security.py",
            "def mask_token(token):\n    return token\n",
            (
                "def mask_token(token):\n"
                "    if len(token) <= 8:\n"
                "        return '***'\n"
                "    return token[:4] + '...' + token[-4:]\n"
            ),
            (
                "from app.security import mask_token\n"
                "assert mask_token('abcd1234wxyz') == 'abcd...wxyz'\n"
                "assert mask_token('short') == '***'\n"
            ),
        ),
        code_task(
            "realistic.inventory.merge",
            "Merge inventory counts",
            "data_pipeline",
            "shop-core",
            "app/inventory.py",
            "def merge_inventory(left, right):\n    return right\n",
            (
                "def merge_inventory(left, right):\n"
                "    merged = dict(left)\n"
                "    for sku, count in right.items():\n"
                "        merged[sku] = merged.get(sku, 0) + count\n"
                "    return merged\n"
            ),
            (
                "from app.inventory import merge_inventory\n"
                "assert merge_inventory({'a': 2}, {'a': 3, 'b': 1}) == {'a': 5, 'b': 1}\n"
            ),
        ),
        code_task(
            "realistic.records.filter_active",
            "Filter active records",
            "data_pipeline",
            "user-service",
            "app/records.py",
            "def active_records(records):\n    return records\n",
            (
                "def active_records(records):\n"
                "    return [row for row in records if row.get('active') is True]\n"
            ),
            (
                "from app.records import active_records\n"
                "rows = [{'id': 1, 'active': True}, {'id': 2, 'active': False}, {'id': 3}]\n"
                "assert active_records(rows) == [{'id': 1, 'active': True}]\n"
            ),
        ),
        code_task(
            "realistic.json.canonical",
            "Canonicalize JSON output",
            "data_pipeline",
            "platform-utils",
            "app/json_tools.py",
            "def canonical_json(payload):\n    return str(payload)\n",
            (
                "import json\n\n"
                "def canonical_json(payload):\n"
                "    return json.dumps(payload, sort_keys=True, separators=(',', ':'))\n"
            ),
            (
                "from app.json_tools import canonical_json\n"
                "assert canonical_json({'b': 2, 'a': 1}) == '{\"a\":1,\"b\":2}'\n"
            ),
        ),
        code_task(
            "realistic.geometry.distance",
            "Compute 2D distance",
            "numeric_reasoning",
            "analytics-core",
            "app/geometry.py",
            "def distance(a, b):\n    return abs(a[0] - b[0]) + abs(a[1] - b[1])\n",
            (
                "import math\n\n"
                "def distance(a, b):\n"
                "    return math.hypot(a[0] - b[0], a[1] - b[1])\n"
            ),
            (
                "from app.geometry import distance\n"
                "assert distance((0, 0), (3, 4)) == 5.0\n"
            ),
        ),
        code_task(
            "realistic.config.env_bool",
            "Parse environment booleans",
            "code_repair",
            "platform-utils",
            "app/config.py",
            "def env_bool(value):\n    return bool(value)\n",
            (
                "def env_bool(value):\n"
                "    normalized = str(value).strip().lower()\n"
                "    if normalized in {'1', 'true', 'yes', 'on'}:\n"
                "        return True\n"
                "    if normalized in {'0', 'false', 'no', 'off'}:\n"
                "        return False\n"
                "    raise ValueError(f'not a boolean: {value}')\n"
            ),
            (
                "from app.config import env_bool\n"
                "assert env_bool('YES') is True\n"
                "assert env_bool('off') is False\n"
            ),
        ),
        code_task(
            "realistic.batching.chunks",
            "Chunk sequences",
            "data_pipeline",
            "platform-utils",
            "app/batching.py",
            "def chunks(items, size):\n    return [items]\n",
            (
                "def chunks(items, size):\n"
                "    return [items[i:i + size] for i in range(0, len(items), size)]\n"
            ),
            (
                "from app.batching import chunks\n"
                "assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            ),
        ),
        code_task(
            "realistic.retry.backoff",
            "Generate exponential backoff schedule",
            "numeric_reasoning",
            "platform-utils",
            "app/retry.py",
            "def backoff(base, retries):\n    return [base] * retries\n",
            "def backoff(base, retries):\n    return [base * (2 ** i) for i in range(retries)]\n",
            (
                "from app.retry import backoff\n"
                "assert backoff(0.5, 4) == [0.5, 1.0, 2.0, 4.0]\n"
            ),
        ),
        code_task(
            "realistic.schema.required",
            "Report missing required fields",
            "benchmark_authoring",
            "authoring-core",
            "app/schema.py",
            "def missing_required(payload, required):\n    return []\n",
            (
                "def missing_required(payload, required):\n"
                "    return [key for key in required if key not in payload]\n"
            ),
            (
                "from app.schema import missing_required\n"
                "assert missing_required({'a': 1}, ['a', 'b', 'c']) == ['b', 'c']\n"
            ),
        ),
        code_task(
            "realistic.logging.redact_email",
            "Redact email local part",
            "code_repair",
            "security-utils",
            "app/logging.py",
            "def redact_email(value):\n    return value\n",
            (
                "def redact_email(value):\n"
                "    if '@' not in value:\n"
                "        return value\n"
                "    _, domain = value.split('@', 1)\n"
                "    return '***@' + domain\n"
            ),
            (
                "from app.logging import redact_email\n"
                "assert redact_email('ada@example.com') == '***@example.com'\n"
                "assert redact_email('no-email') == 'no-email'\n"
            ),
        ),
        code_task(
            "realistic.tokens.count_words",
            "Count non-empty words",
            "data_pipeline",
            "analytics-core",
            "app/tokens.py",
            "def count_words(text):\n    return len(text)\n",
            "def count_words(text):\n    return len([part for part in text.split() if part])\n",
            (
                "from app.tokens import count_words\n"
                "assert count_words('one  two\\nthree') == 3\n"
            ),
        ),
        {
            "task_id": "realistic.config.service_json",
            "title": "Write service runtime config",
            "category": "benchmark_authoring",
            "prompt": "Write config/service.json with retries=3 and enabled=true.",
            "fixtures": {
                "toy_repo": {
                    "repo_id": "service-config",
                    "files": {"README.md": "# service config\n"},
                    "agent_files": {
                        "config/service.json": "{\"retries\": 3, \"enabled\": true}\n"
                    },
                    "hidden_judge": [
                        {
                            "kind": "json_value",
                            "path": "config/service.json",
                            "json_path": "retries",
                            "expected": 3,
                        },
                        {
                            "kind": "json_value",
                            "path": "config/service.json",
                            "json_path": "enabled",
                            "expected": True,
                        },
                    ],
                }
            },
            "expected_artifacts": ["config/service.json"],
            "allowed_paths": ["config/"],
            "scoring": hidden_score(),
            "tags": ["simulated_real_repo", "json", "true_loop", "round3"],
            "metadata": {
                "difficulty": "easy-medium",
                "failure_type_if_wrong": "format_error",
                "repo_style": "config_artifact",
            },
        },
    ]
    return tasks


def main() -> None:
    args = parse_args()
    manifest = {
        "dataset_id": "sebench-realistic-v0.1",
        "version": "0.1.0",
        "source": "simulated_real_repo_true_loop",
        "references": [
            "https://arxiv.org/abs/2310.06770",
            "https://github.com/SWE-bench/SWE-bench",
            "https://github.com/SWE-agent/SWE-agent",
        ],
        "tasks": build_tasks(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest['tasks'])} tasks to {args.out}")


if __name__ == "__main__":
    main()
