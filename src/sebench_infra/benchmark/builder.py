import hashlib
from collections.abc import Iterable

from sebench_infra.benchmark.schemas import DatasetSpec, TaskSpec


class DatasetBuilder:
    """Build stable synthetic benchmark datasets from authored tasks."""

    def build(
        self,
        requirement: str,
        tasks: Iterable[TaskSpec],
        references: list[str],
    ) -> DatasetSpec:
        task_list = list(tasks)
        digest = hashlib.sha256(requirement.encode("utf-8")).hexdigest()[:12]
        return DatasetSpec(
            dataset_id=f"sebench-replica-{digest}",
            tasks=task_list,
            references=references,
        )
