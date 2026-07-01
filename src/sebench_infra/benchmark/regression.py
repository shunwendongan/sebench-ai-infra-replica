from sebench_infra.benchmark.schemas import EvaluationReport


class RegressionGate:
    """Compare a new report against a baseline score threshold."""

    def __init__(self, min_aggregate_score: float = 0.8) -> None:
        self.min_aggregate_score = min_aggregate_score

    def check(self, report: EvaluationReport) -> bool:
        return report.aggregate_score >= self.min_aggregate_score and all(
            result.status != "error" for result in report.task_results
        )
