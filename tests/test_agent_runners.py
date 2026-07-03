import subprocess
from pathlib import Path

from sebench_infra.benchmark.schemas import TaskCategory, TaskSpec
from sebench_infra.orchestrator.agents import CodexCliPatchAgentRunner, FixturePatchAgentRunner


def _patch_task() -> TaskSpec:
    return TaskSpec(
        task_id="agent.patch",
        title="Patch task",
        category=TaskCategory.CODE_REPAIR,
        prompt="Fix add.",
        expected_artifacts=["app/math_utils.py"],
        allowed_paths=["submission/model.patch"],
    )


def test_fixture_patch_agent_collects_fixture_patch(tmp_path: Path) -> None:
    patch = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n"
    result = FixturePatchAgentRunner().run(
        task=_patch_task(),
        work_dir=tmp_path,
        fixture={"agent_patch": patch},
        patch_required=True,
    )

    assert result.artifacts == {"submission/model.patch": patch}
    assert result.metadata["agent_backend"] == "fixture"
    assert result.latency_sec >= 0


def test_codex_cli_patch_agent_uses_fake_subprocess_output(tmp_path: Path) -> None:
    patch = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n"

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"```diff\n{patch}```", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = CodexCliPatchAgentRunner(run_command=fake_run).run(
        task=_patch_task(),
        work_dir=tmp_path,
        fixture={},
        patch_required=True,
    )

    assert result.artifacts["submission/model.patch"] == patch
    assert result.metadata["agent_backend"] == "codex_cli"
    assert result.metadata["patch_validation_status"] == "collected"
    assert result.metadata["codex_returncode"] == 0
