from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from sebench_infra.benchmark.schemas import TaskSpec
from sebench_infra.orchestrator.whitelist import PathWhitelist
from sebench_infra.orchestrator.workspace import _string_map, _write_files


@dataclass(frozen=True)
class JudgeRunResult:
    passed: bool
    failure_type: str | None
    details: list[dict[str, Any]]
    patch_apply_time_sec: float = 0.0
    hidden_test_time_sec: float = 0.0
    touched_paths: set[str] = field(default_factory=set)


class LocalJudgeRunner:
    """Apply allowlisted submissions in a clean judge workspace and run hidden rules."""

    def __init__(
        self,
        *,
        pytest_diagnostics: bool = False,
        pytest_timeout_sec: float | None = None,
        pytest_plugin_policy: str = "auto",
    ) -> None:
        self.pytest_diagnostics = pytest_diagnostics
        self.pytest_timeout_sec = pytest_timeout_sec
        self.pytest_plugin_policy = pytest_plugin_policy

    def resolve_plugin_policy(self, judge_dir: Path, rules: Any) -> dict[str, Any]:
        return _resolve_pytest_plugin_policy(judge_dir, rules, self.pytest_plugin_policy)

    def run(
        self,
        *,
        judge_dir: Path,
        task: TaskSpec,
        fixture: dict[str, Any],
        safe_artifacts: dict[str, str],
        uses_patch: bool,
        plugin_scan: dict[str, Any],
    ) -> JudgeRunResult:
        patch_apply_time = 0.0
        hidden_test_time = 0.0
        touched_paths: set[str] = set()
        try:
            details_prefix: list[dict[str, Any]] = []
            if uses_patch:
                patch_text = safe_artifacts.get("submission/model.patch")
                if patch_text is None:
                    raise ValueError("patch artifact rejected by path whitelist")
                patch_apply_start = time.perf_counter()
                touched_paths = _apply_unified_diff(
                    judge_dir,
                    patch_text,
                    _patch_whitelist(task, fixture),
                )
                patch_apply_time = time.perf_counter() - patch_apply_start
                details_prefix.append(
                    {
                        "kind": "patch_apply",
                        "passed": True,
                        "paths": sorted(touched_paths),
                        "time_sec": patch_apply_time,
                        "failure_type": None,
                    }
                )
            else:
                _write_files(judge_dir, safe_artifacts)

            hidden_test_start = time.perf_counter()
            passed, failure_type, details = _run_hidden_rules(
                judge_dir,
                fixture.get("hidden_judge", []),
                pytest_diagnostics=self.pytest_diagnostics,
                pytest_timeout_sec=self.pytest_timeout_sec,
                disable_pytest_plugin_autoload=plugin_scan["disable_autoload"],
                pytest_plugin_scan=plugin_scan,
            )
            hidden_test_time = time.perf_counter() - hidden_test_start
            return JudgeRunResult(
                passed=passed,
                failure_type=failure_type,
                details=details_prefix + details,
                patch_apply_time_sec=patch_apply_time,
                hidden_test_time_sec=hidden_test_time,
                touched_paths=touched_paths,
            )
        except Exception as exc:
            return JudgeRunResult(
                passed=False,
                failure_type="patch_apply_error",
                details=[
                    {
                        "kind": "patch_apply",
                        "passed": False,
                        "error": repr(exc),
                        "failure_type": "patch_apply_error",
                    }
                ],
                patch_apply_time_sec=patch_apply_time,
                hidden_test_time_sec=hidden_test_time,
                touched_paths=touched_paths,
            )


def _normalize_pytest_plugin_policy(
    policy: str | None,
    disable_pytest_plugin_autoload: bool,
) -> str:
    if policy is None:
        return "enabled" if not disable_pytest_plugin_autoload else "auto"
    if policy not in {"auto", "disabled", "enabled"}:
        raise ValueError("pytest_plugin_policy must be one of: auto, disabled, enabled")
    return policy


def _resolve_pytest_plugin_policy(
    repo_root: Path,
    rules: Any,
    policy: str,
) -> dict[str, Any]:
    scan = _scan_pytest_plugin_dependencies(repo_root, rules)
    if policy == "disabled":
        disable_autoload = True
        effective_policy = "disabled"
    elif policy == "enabled":
        disable_autoload = False
        effective_policy = "enabled"
    else:
        disable_autoload = not scan["plugin_required"]
        effective_policy = "auto_disabled" if disable_autoload else "auto_enabled"
    return {
        "policy": policy,
        "effective_policy": effective_policy,
        "disable_autoload": disable_autoload,
        "plugin_required": scan["plugin_required"],
        "reasons": scan["reasons"],
        "scanned_files": scan["scanned_files"],
    }


def _scan_pytest_plugin_dependencies(repo_root: Path, rules: Any) -> dict[str, Any]:
    reasons: list[str] = []
    scanned_files: list[str] = []
    for path in _candidate_pytest_config_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        scanned_files.append(rel)
        _extend_plugin_reasons(reasons, rel, _read_text_limited(path))
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict) or rule.get("kind") != "pytest":
                continue
            files = _string_map(rule.get("files", {}))
            for rel, content in files.items():
                label = f"hidden_judge[{index}]:{rel}"
                scanned_files.append(label)
                _extend_plugin_reasons(reasons, label, content)
            args = " ".join(str(arg) for arg in rule.get("args", []))
            if args:
                _extend_plugin_reasons(reasons, f"hidden_judge[{index}]:args", args)
    return {
        "plugin_required": bool(reasons),
        "reasons": sorted(set(reasons)),
        "scanned_files": sorted(set(scanned_files)),
    }


def _candidate_pytest_config_files(repo_root: Path) -> list[Path]:
    names = {"pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "setup.py"}
    paths: list[Path] = []
    for path in repo_root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        name = path.name
        if (
            name in names
            or name == "conftest.py"
            or (name.startswith("requirements") and path.suffix in {".txt", ".in"})
        ):
            paths.append(path)
    return paths


def _read_text_limited(path: Path, limit_bytes: int = 512_000) -> str:
    try:
        payload = path.read_bytes()[:limit_bytes]
    except OSError:
        return ""
    return payload.decode("utf-8", errors="ignore")


def _extend_plugin_reasons(reasons: list[str], source: str, text: str) -> None:
    lowered = text.lower()
    if "pytest_plugins" in lowered:
        reasons.append(f"{source}: pytest_plugins declaration")
    for match in re.finditer(r"(?i)\bpytest[-_][a-z0-9_.-]+", text):
        plugin = match.group(0).lower().replace("_", "-")
        if plugin not in {"pytest", "pytest-disable-plugin-autoload"}:
            reasons.append(f"{source}: dependency {plugin}")
    flag_patterns = {
        "--asyncio-mode": "pytest-asyncio option",
        "pytest.mark.asyncio": "pytest-asyncio mark",
        "--ds": "pytest-django option",
        "django_settings_module": "pytest-django setting",
        "pytest.mark.django_db": "pytest-django mark",
        "--cov": "pytest-cov option",
        "--cov-report": "pytest-cov option",
        "--numprocesses": "pytest-xdist option",
        " -n ": "pytest-xdist short option",
        "--dist": "pytest-xdist option",
        "--benchmark": "pytest-benchmark option",
        "--mypy": "pytest-mypy option",
    }
    padded = f" {lowered} "
    for marker, reason in flag_patterns.items():
        haystack = padded if marker.startswith(" ") else lowered
        if marker in haystack:
            reasons.append(f"{source}: {reason}")


def _patch_whitelist(task: TaskSpec, fixture: dict[str, Any]) -> PathWhitelist:
    raw_paths = fixture.get("patch_allowed_paths")
    if isinstance(raw_paths, list) and raw_paths:
        return PathWhitelist([str(path) for path in raw_paths])
    return PathWhitelist(task.expected_artifacts)


def _apply_unified_diff(judge_dir: Path, patch_text: str, whitelist: PathWhitelist) -> set[str]:
    touched_paths = _paths_in_unified_diff(patch_text)
    if not touched_paths:
        raise ValueError("patch does not modify any tracked path")
    for path in touched_paths:
        if not whitelist.is_allowed(path):
            raise ValueError(f"patch modifies disallowed path: {path}")

    apply_commands = [
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        ["git", "apply", "--whitespace=nowarn", "-"],
    ]
    for args in apply_commands:
        completed = subprocess.run(
            args,
            cwd=judge_dir,
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "git apply failed: "
                f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            )
    return touched_paths


def _paths_in_unified_diff(patch_text: str) -> set[str]:
    paths: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            path = _normalize_diff_path(line[4:].strip())
        elif line.startswith("--- "):
            path = _normalize_diff_path(line[4:].strip())
        else:
            continue
        if path is not None:
            paths.add(path)
    return paths


def _normalize_diff_path(raw_path: str) -> str | None:
    path = raw_path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe patch path: {raw_path}")
    return path


def _run_hidden_rules(
    judge_dir: Path,
    rules: Any,
    pytest_diagnostics: bool = False,
    pytest_timeout_sec: float | None = None,
    disable_pytest_plugin_autoload: bool = False,
    pytest_plugin_scan: dict[str, Any] | None = None,
) -> tuple[bool, str | None, list[dict[str, Any]]]:
    if not isinstance(rules, list):
        return False, "judge_config_error", [{"error": "hidden_judge must be a list"}]

    details: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            return False, "judge_config_error", [{"error": "hidden rule must be an object"}]
        try:
            passed, detail = _run_hidden_rule(
                judge_dir,
                rule,
                pytest_diagnostics=pytest_diagnostics,
                pytest_timeout_sec=pytest_timeout_sec,
                disable_pytest_plugin_autoload=disable_pytest_plugin_autoload,
                pytest_plugin_scan=pytest_plugin_scan,
            )
        except subprocess.TimeoutExpired as exc:
            passed = False
            detail = {
                "kind": rule.get("kind"),
                "error": repr(exc),
                "timeout_sec": (
                    pytest_timeout_sec
                    if rule.get("kind") == "pytest" and pytest_timeout_sec is not None
                    else rule.get("timeout_sec")
                ),
                "failure_type": "hidden_judge_timeout",
            }
        except AssertionError as exc:
            passed = False
            detail = {
                "kind": rule.get("kind"),
                "error": str(exc) or repr(exc),
                "failure_type": "wrong_edit",
            }
        except Exception as exc:
            passed = False
            detail = {
                "kind": rule.get("kind"),
                "error": repr(exc),
                "failure_type": "hidden_judge_error",
            }
        details.append(detail)
        if not passed:
            return False, str(detail.get("failure_type", "hidden_judge_failed")), details
    return True, None, details


def _run_hidden_rule(
    judge_dir: Path,
    rule: dict[str, Any],
    pytest_diagnostics: bool = False,
    pytest_timeout_sec: float | None = None,
    disable_pytest_plugin_autoload: bool = False,
    pytest_plugin_scan: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    kind = rule.get("kind")
    if kind == "python_function":
        rel_path = str(rule["path"])
        module = _load_python_module(judge_dir / rel_path)
        function = getattr(module, str(rule["function"]))
        actual = function(*rule.get("args", []))
        expected = rule.get("expected")
        passed = actual == expected
        return passed, {
            "kind": kind,
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "failure_type": None if passed else "wrong_edit",
        }

    if kind == "cli_stdout_contains":
        command = [str(part) for part in rule["command"]]
        if command and command[0] == "python":
            command[0] = sys.executable
        completed = subprocess.run(
            command,
            cwd=judge_dir,
            text=True,
            capture_output=True,
            timeout=float(rule.get("timeout_sec", 5)),
            check=False,
        )
        expected = str(rule["expected"])
        passed = completed.returncode == 0 and expected in completed.stdout
        return passed, {
            "kind": kind,
            "passed": passed,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "expected": expected,
            "failure_type": None if passed else "wrong_edit",
        }

    if kind == "python_script_stdout_contains":
        rel_path = str(rule["path"])
        args = [str(arg) for arg in rule.get("args", [])]
        expected = str(rule["expected"])
        stdout = _run_python_script_in_process(judge_dir, judge_dir / rel_path, args)
        passed = expected in stdout
        return passed, {
            "kind": kind,
            "passed": passed,
            "stdout": stdout.strip(),
            "expected": expected,
            "failure_type": None if passed else "wrong_edit",
        }

    if kind == "python_inline_tests":
        code = str(rule["code"])
        completed = _run_python_inline_tests(judge_dir, code, float(rule.get("timeout_sec", 5)))
        passed = completed.returncode == 0
        failure_type = None
        if not passed:
            failure_type = (
                "wrong_edit" if "AssertionError" in completed.stderr else "hidden_judge_error"
            )
        return passed, {
            "kind": kind,
            "passed": passed,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "failure_type": failure_type,
        }

    if kind == "pytest":
        test_files = _string_map(rule.get("files", {}))
        file_write_start = time.perf_counter()
        _write_files(judge_dir, test_files)
        file_write_time = time.perf_counter() - file_write_start
        args = _normalized_pytest_args(rule.get("args", ["-q"]))
        timeout_sec = (
            float(pytest_timeout_sec)
            if pytest_timeout_sec is not None
            else float(rule.get("timeout_sec", 10))
        )
        python_startup_time = 0.0
        pytest_startup_time = 0.0
        pytest_collection_time = 0.0
        pytest_subprocess_count = 1
        pytest_env = _pytest_env_overrides(disable_pytest_plugin_autoload)
        if pytest_diagnostics:
            pytest_subprocess_count += 3
            _, python_startup_time = _run_timed_command(
                [sys.executable, "-c", "pass"],
                judge_dir,
                timeout_sec,
            )
            _, pytest_startup_time = _run_timed_command(
                [sys.executable, "-m", "pytest", "--version"],
                judge_dir,
                timeout_sec,
                extra_env=pytest_env,
            )
            _, pytest_collection_time = _run_timed_command(
                [sys.executable, "-m", "pytest", *args, "--collect-only"],
                judge_dir,
                timeout_sec,
                extra_env=pytest_env,
            )
        completed, pytest_execution_time = _run_pytest(
            judge_dir,
            args,
            timeout_sec,
            extra_env=pytest_env,
        )
        passed = completed.returncode == 0
        failure_type = None
        if not passed:
            failure_type = "wrong_edit" if completed.returncode == 1 else "hidden_judge_error"
        pytest_total_time = (
            python_startup_time
            + pytest_startup_time
            + pytest_collection_time
            + pytest_execution_time
        )
        return passed, {
            "kind": kind,
            "passed": passed,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "args": args,
            "timeout_sec": timeout_sec,
            "disable_pytest_plugin_autoload": disable_pytest_plugin_autoload,
            "pytest_plugin_scan": pytest_plugin_scan or {},
            "hidden_test_file_write_time_sec": file_write_time,
            "python_subprocess_startup_time_sec": python_startup_time,
            "pytest_process_startup_time_sec": pytest_startup_time,
            "pytest_collection_time_sec": pytest_collection_time,
            "pytest_execution_time_sec": pytest_execution_time,
            "pytest_total_time_sec": pytest_total_time,
            "pytest_subprocess_count": pytest_subprocess_count,
            "failure_type": failure_type,
        }

    if kind == "json_value":
        payload = json.loads((judge_dir / str(rule["path"])).read_text(encoding="utf-8"))
        actual: Any = payload
        for part in str(rule["json_path"]).split("."):
            actual = actual[part]
        expected = rule.get("expected")
        passed = actual == expected
        return passed, {
            "kind": kind,
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "failure_type": None if passed else "format_error",
        }

    return False, {"kind": kind, "passed": False, "failure_type": "judge_config_error"}


def _load_python_module(path: Path) -> Any:
    module_name = f"sebench_toy_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with _python_execution_context(path.parent):
        spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _python_execution_context(root: Path):
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_path = sys.path[:]
    try:
        _clear_repo_package_modules(root)
        os.chdir(root)
        sys.path.insert(0, str(root))
        yield
    finally:
        sys.argv = old_argv
        sys.path = old_path
        os.chdir(old_cwd)


def _clear_repo_package_modules(root: Path) -> None:
    package_names = [
        path.name for path in root.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    ]
    for package_name in package_names:
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                del sys.modules[module_name]


def _run_python_script_in_process(root: Path, path: Path, args: list[str]) -> str:
    buffer = io.StringIO()
    with _python_execution_context(root):
        sys.argv = [str(path), *args]
        code = path.read_text(encoding="utf-8")
        namespace = {"__name__": "__main__", "__file__": str(path)}
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, str(path), "exec"), namespace)
    return buffer.getvalue()


def _run_python_inline_tests(
    root: Path,
    code: str,
    timeout_sec: float,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root) if not existing_pythonpath else f"{root}{os.pathsep}{existing_pythonpath}"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def _normalized_pytest_args(raw_args: Any) -> list[str]:
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else ["-q"]
    if not any(arg in {"-q", "--quiet"} for arg in args):
        args.insert(0, "-q")
    if not any(arg == "--tb" or arg.startswith("--tb=") for arg in args):
        args.append("--tb=short")
    if "--disable-warnings" not in args:
        args.append("--disable-warnings")
    if not any(arg == "--maxfail" or arg.startswith("--maxfail=") or arg == "-x" for arg in args):
        args.append("--maxfail=1")
    return args


def _run_timed_command(
    command: list[str],
    root: Path,
    timeout_sec: float,
    extra_env: dict[str, str | None] | None = None,
) -> tuple[subprocess.CompletedProcess, float]:
    env = os.environ.copy()
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root) if not existing_pythonpath else f"{root}{os.pathsep}{existing_pythonpath}"
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    return completed, time.perf_counter() - start


def _run_pytest(
    root: Path,
    args: list[str],
    timeout_sec: float,
    extra_env: dict[str, str | None] | None = None,
) -> tuple[subprocess.CompletedProcess, float]:
    return _run_timed_command(
        [sys.executable, "-m", "pytest", *args],
        root,
        timeout_sec,
        extra_env=extra_env,
    )


def _pytest_env_overrides(disable_plugin_autoload: bool) -> dict[str, str | None]:
    if disable_plugin_autoload:
        return {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    return {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": None}


def _aggregate_detail_timings(details: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "hidden_test_file_write_time_sec",
        "python_subprocess_startup_time_sec",
        "pytest_process_startup_time_sec",
        "pytest_collection_time_sec",
        "pytest_execution_time_sec",
        "pytest_total_time_sec",
    ]
    return {
        key: sum(float(detail.get(key, 0.0)) for detail in details if isinstance(detail, dict))
        for key in keys
    }


def _aggregate_pytest_subprocess_count(details: list[dict[str, Any]]) -> int:
    return sum(
        int(detail.get("pytest_subprocess_count", 0))
        for detail in details
        if isinstance(detail, dict)
    )
