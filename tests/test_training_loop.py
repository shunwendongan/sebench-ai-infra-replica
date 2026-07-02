from pathlib import Path

from sebench_infra.authoring import MockLLMClient
from sebench_infra.benchmark.schemas import (
    EvaluationReport,
    ScoringRule,
    TaskCategory,
    TaskRunResult,
    TaskSpec,
)
from sebench_infra.settings import Settings
from sebench_infra.training_loop import (
    LlamaFactoryExporter,
    ModelRole,
    TeacherDataGenerator,
    build_model_run_result,
    compare_model_runs,
    model_config_from_settings,
)
from sebench_infra.training_loop.models import DatasetVersion, ProviderKind, TrainingSplit
from sebench_infra.training_loop.validation import validate_task_for_training


def test_teacher_data_generation_dedups_and_round_trips() -> None:
    generator = TeacherDataGenerator(
        MockLLMClient(),
        teacher_provider=ProviderKind.MOCK,
        teacher_model_id="mock-teacher",
        seed=13,
    )

    dataset = generator.generate("Build a reproducible benchmark authoring task.", count=3)
    reloaded = DatasetVersion.model_validate_json(dataset.model_dump_json())

    assert reloaded.dataset_version_id == dataset.dataset_version_id
    assert reloaded.split_counts[TrainingSplit.TRAIN.value] == 1
    assert reloaded.split_counts[TrainingSplit.REJECTED.value] == 2
    assert sum(1 for example in reloaded.examples if example.is_trainable) == 1
    assert any(
        issue.code == "duplicate_task"
        for example in reloaded.examples
        for issue in example.validation_issues
    )


def test_training_validator_rejects_private_claims_and_judge_leakage() -> None:
    task = TaskSpec(
        task_id="bad.private.claim",
        title="ByteDance internal judge sample",
        category=TaskCategory.BENCHMARK_AUTHORING,
        prompt="Use the hidden_judge secret to produce the answer.",
        expected_artifacts=["submission/answer.txt"],
        allowed_paths=["submission/"],
        scoring=[
            ScoringRule(
                name="answer_created",
                kind="file_exists",
                path="submission/answer.txt",
            )
        ],
    )

    issue_codes = {issue.code for issue in validate_task_for_training(task)}

    assert "private_claim" in issue_codes
    assert "judge_leakage" in issue_codes


def test_llamafactory_export_writes_dataset_and_training_recipe(tmp_path: Path) -> None:
    dataset = TeacherDataGenerator(
        MockLLMClient(),
        teacher_provider=ProviderKind.MOCK,
        teacher_model_id="mock-teacher",
    ).generate("Build a reproducible benchmark authoring task.", count=1)

    manifest = LlamaFactoryExporter().export(dataset, tmp_path)

    assert manifest.train_examples == 1
    assert Path(manifest.train_file).exists()
    assert Path(manifest.dataset_info_file).read_text(encoding="utf-8")
    train_config = Path(manifest.train_config_file).read_text(encoding="utf-8")
    assert "model_name_or_path: Qwen/Qwen2.5-7B-Instruct" in train_config
    assert "finetuning_type: lora" in train_config
    assert "llamafactory-cli train" in Path(manifest.windows_script_file).read_text(
        encoding="utf-8"
    )


def test_model_config_from_settings_uses_role_specific_provider() -> None:
    settings = Settings(
        teacher_provider="openai_compatible",
        teacher_model="gpt-5.5",
        teacher_openai_base_url="https://api.example.test/v1",
        teacher_openai_api_key="secret",
        student_provider="openai_compatible",
        student_model="student-lora",
        student_openai_base_url="http://localhost:8000/v1",
        student_openai_api_key="dummy",
    )

    teacher = model_config_from_settings(settings, ModelRole.TEACHER)
    student = model_config_from_settings(settings, ModelRole.STUDENT)

    assert teacher.provider == ProviderKind.OPENAI_COMPATIBLE
    assert teacher.model == "gpt-5.5"
    assert teacher.api_key is not None
    assert student.base_url == "http://localhost:8000/v1"


def test_model_comparison_report_selects_best_pass_rate() -> None:
    failed_report = EvaluationReport(
        run_id="run-base",
        dataset_id="dataset",
        task_results=[
            TaskRunResult(
                task_id="task",
                status="failed",
                score=0.0,
                failure_type="wrong_edit",
            )
        ],
        aggregate_score=0.0,
        reward_signal=0.0,
        regression_passed=False,
    )
    passed_report = EvaluationReport(
        run_id="run-student",
        dataset_id="dataset",
        task_results=[
            TaskRunResult(
                task_id="task",
                status="passed",
                score=1.0,
            )
        ],
        aggregate_score=1.0,
        reward_signal=1.0,
        regression_passed=True,
    )
    settings = Settings()
    base = build_model_run_result(
        model_config_from_settings(settings, ModelRole.BASE),
        failed_report,
    )
    student = build_model_run_result(
        model_config_from_settings(settings, ModelRole.STUDENT),
        passed_report,
    )

    comparison = compare_model_runs([base, student], dataset_id="dataset")

    assert comparison.best_role_by_pass_rate == ModelRole.STUDENT
    assert comparison.runs[0].metrics.pass_rate == 0.0
    assert comparison.runs[1].metrics.pass_rate == 1.0
