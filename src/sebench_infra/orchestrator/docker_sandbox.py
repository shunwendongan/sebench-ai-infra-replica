from __future__ import annotations

from sebench_infra.benchmark.schemas import TaskSpec
from sebench_infra.orchestrator.cache import _hash_artifacts
from sebench_infra.orchestrator.sandbox_types import SandboxResult
from sebench_infra.orchestrator.whitelist import PathWhitelist


class DockerSandbox:
    """Container runner placeholder following the Work/Judge split."""

    def __init__(self, work_image: str, judge_image: str, timeout_seconds: int = 120) -> None:
        self.work_image = work_image
        self.judge_image = judge_image
        self.timeout_seconds = timeout_seconds

    def run_task(self, task: TaskSpec) -> SandboxResult:
        try:
            import docker
        except ImportError as exc:
            raise RuntimeError("Install docker SDK to use docker runner mode") from exc

        client = docker.from_env()
        command = ["python", "-m", "sebench_work", task.model_dump_json()]
        output = client.containers.run(
            self.work_image,
            command=command,
            detach=False,
            remove=True,
            network_disabled=True,
            stdout=True,
            stderr=True,
            mem_limit="1g",
        )
        answer = output.decode("utf-8", errors="replace")
        artifacts = {"submission/answer.txt": answer}
        whitelist = PathWhitelist(task.allowed_paths)
        return SandboxResult(
            artifacts={
                path: content
                for path, content in artifacts.items()
                if whitelist.is_allowed(path)
            },
            metadata={
                "docker_runner": True,
                "judge_image": self.judge_image,
                "patch_hash": _hash_artifacts(artifacts),
            },
        )
