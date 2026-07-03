"""Compatibility exports for orchestrator sandbox components.

The implementation lives in smaller modules so callers can depend on the
checkout, cache, agent, and judge layers independently. This module preserves the
original import path used by tests and scripts.
"""

from sebench_infra.orchestrator.agents import (
    AgentRunResult,
    CodexCliPatchAgentRunner,
    FixturePatchAgentRunner,
    PatchAgentRunner,
)
from sebench_infra.orchestrator.cache import (
    JudgeVerdictCache,
    SnapshotHitTracker,
    _benchmark_environment,
    _hash_artifacts,
    _judge_cache_key,
    _policy_aware_judge_cache_key,
    _stable_key,
)
from sebench_infra.orchestrator.docker_sandbox import DockerSandbox
from sebench_infra.orchestrator.judge import (
    JudgeRunResult,
    LocalJudgeRunner,
    _aggregate_detail_timings,
    _aggregate_pytest_subprocess_count,
    _apply_unified_diff,
    _normalize_diff_path,
    _normalize_pytest_plugin_policy,
    _normalized_pytest_args,
    _patch_whitelist,
    _paths_in_unified_diff,
    _pytest_env_overrides,
    _resolve_pytest_plugin_policy,
    _run_hidden_rule,
    _run_hidden_rules,
    _run_pytest,
    _run_timed_command,
)
from sebench_infra.orchestrator.local_sandbox import LocalSandbox
from sebench_infra.orchestrator.sandbox_types import SandboxResult
from sebench_infra.orchestrator.workspace import (
    CheckoutResult,
    WorkspaceManager,
    _cleanup_hidden_judge_artifacts,
    _clone_git_repo,
    _empty_checkout_timings,
    _file_lock,
    _merge_checkout_timings,
    _remove_empty_parents,
    _resolve_shared_cache_root,
    _run_git,
    _string_map,
    _write_files,
    shared_checkout_cache_key,
)

__all__ = [
    "AgentRunResult",
    "CheckoutResult",
    "CodexCliPatchAgentRunner",
    "DockerSandbox",
    "FixturePatchAgentRunner",
    "JudgeRunResult",
    "JudgeVerdictCache",
    "LocalJudgeRunner",
    "LocalSandbox",
    "PatchAgentRunner",
    "SandboxResult",
    "SnapshotHitTracker",
    "WorkspaceManager",
    "_aggregate_detail_timings",
    "_aggregate_pytest_subprocess_count",
    "_apply_unified_diff",
    "_benchmark_environment",
    "_cleanup_hidden_judge_artifacts",
    "_clone_git_repo",
    "_empty_checkout_timings",
    "_file_lock",
    "_hash_artifacts",
    "_judge_cache_key",
    "_merge_checkout_timings",
    "_normalize_diff_path",
    "_normalize_pytest_plugin_policy",
    "_normalized_pytest_args",
    "_patch_whitelist",
    "_paths_in_unified_diff",
    "_policy_aware_judge_cache_key",
    "_pytest_env_overrides",
    "_remove_empty_parents",
    "_resolve_pytest_plugin_policy",
    "_resolve_shared_cache_root",
    "_run_git",
    "_run_hidden_rule",
    "_run_hidden_rules",
    "_run_pytest",
    "_run_timed_command",
    "_stable_key",
    "_string_map",
    "_write_files",
    "shared_checkout_cache_key",
]
