from sebench_infra.benchmark.schemas import ScoringRule, TaskCategory, TaskSpec
from sebench_infra.benchmark.scoring import ScoreEngine


def test_score_engine_scores_weighted_rules() -> None:
    task = TaskSpec(
        task_id="score",
        title="score",
        category=TaskCategory.BENCHMARK_AUTHORING,
        prompt="x",
        scoring=[
            ScoringRule(
                name="exists",
                kind="file_exists",
                path="submission/answer.txt",
                weight=0.5,
            ),
            ScoringRule(
                name="contains",
                kind="contains",
                path="submission/answer.txt",
                expected="reproducible",
                weight=0.5,
            ),
        ],
    )

    result = ScoreEngine().score_task(
        task,
        {"artifacts": {"submission/answer.txt": "this is reproducible"}, "metadata": {}},
    )

    assert result.status == "passed"
    assert result.score == 1.0


def test_numeric_close_rule() -> None:
    rule = ScoringRule(name="mae", kind="numeric_close", expected=0.25, tolerance=0.01)
    task = TaskSpec(
        task_id="numeric",
        title="numeric",
        category=TaskCategory.NUMERIC_REASONING,
        prompt="x",
        scoring=[rule],
    )

    result = ScoreEngine().score_task(task, {"answer": "0.251", "artifacts": {}, "metadata": {}})

    assert result.score == 1.0
