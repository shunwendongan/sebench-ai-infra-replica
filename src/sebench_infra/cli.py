import json
import shlex
from pathlib import Path

import typer

from sebench_infra.authoring import AuthoringAgent, MockLLMClient
from sebench_infra.benchmark import DatasetBuilder
from sebench_infra.benchmark.schemas import DatasetSpec
from sebench_infra.observability import configure_logging
from sebench_infra.orchestrator import EvaluationOrchestrator
from sebench_infra.settings import get_settings
from sebench_infra.spatial import SpatialDiagnosisEngine
from sebench_infra.storage import ArtifactStore
from sebench_infra.training_loop import (
    DatasetVersion,
    ExternalBenchmarkSource,
    LlamaFactoryExporter,
    ModelRole,
    SWEHarnessRunner,
    SWEIssueInstance,
    build_model_run_result,
    compare_model_runs,
    load_swe_instances_from_hf,
    load_swe_instances_from_json,
    load_swe_instances_from_jsonl,
    model_config_from_settings,
    model_configs_from_settings,
    swe_instances_to_dataset_version,
    write_swe_predictions,
)
from sebench_infra.training_loop.comparison import comparison_to_markdown
from sebench_infra.training_loop.generation import TeacherDataGenerator
from sebench_infra.training_loop.patch_agent import ModelPatchAgent
from sebench_infra.training_loop.providers import create_llm_client

app = typer.Typer(no_args_is_help=True)


@app.command()
def author(
    requirements: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("artifacts/dataset.json")),
) -> None:
    """Generate a synthetic benchmark dataset from a requirement file."""

    payload = json.loads(requirements.read_text(encoding="utf-8"))
    requirement = payload["requirement"]
    references = payload.get("references", [])
    tasks = AuthoringAgent(MockLLMClient()).author_tasks(requirement)
    dataset = DatasetBuilder().build(requirement, tasks, references)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"wrote dataset: {out}")


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("reports/evaluation_report.json")),
    runner_mode: str = typer.Option("local"),
    max_tasks: int | None = typer.Option(None, min=1),
    agent_backend: str = typer.Option("fixture"),
    codex_binary: str = typer.Option("codex"),
    codex_model: str | None = typer.Option(None),
    codex_timeout_sec: float = typer.Option(300.0, min=1.0),
) -> None:
    """Evaluate a dataset with the local mock runner or Docker runner."""

    configure_logging()
    dataset_spec = DatasetSpec.model_validate_json(dataset.read_text(encoding="utf-8"))
    report = EvaluationOrchestrator(
        runner_mode=runner_mode,
        agent_backend=agent_backend,
        codex_binary=codex_binary,
        codex_model=codex_model,
        codex_timeout_sec=codex_timeout_sec,
    ).run(dataset_spec, max_tasks=max_tasks)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"aggregate_score={report.aggregate_score:.3f} report={out}")


@app.command()
def diagnose_spatial(
    scene: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("reports/spatial_diagnosis.json")),
) -> None:
    """Run centroid and Kabsch diagnostics on a synthetic 3D scene."""

    diagnosis = SpatialDiagnosisEngine().diagnose_scene_file(scene)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"wrote spatial diagnosis: {out}")


@app.command()
def reproduce(
    requirements: Path = typer.Option(..., exists=True, readable=True),
    spatial_scene: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("reports/demo_run.json")),
) -> None:
    """Run the full public-paper reproduction prototype."""

    configure_logging()
    settings = get_settings()
    requirement_payload = json.loads(requirements.read_text(encoding="utf-8"))
    requirement = requirement_payload["requirement"]
    references = requirement_payload.get("references", [])
    tasks = AuthoringAgent(MockLLMClient()).author_tasks(requirement)
    dataset = DatasetBuilder().build(requirement, tasks, references)
    report = EvaluationOrchestrator(runner_mode=settings.runner_mode).run(dataset)
    spatial = SpatialDiagnosisEngine().diagnose_scene_file(spatial_scene)

    store = ArtifactStore(settings.artifact_db)
    store.put_artifact(dataset.dataset_id, "dataset", dataset.model_dump(mode="json"))
    store.put_report(report.run_id, report.model_dump(mode="json"))

    payload = {
        "dataset": dataset.model_dump(mode="json"),
        "evaluation_report": report.model_dump(mode="json"),
        "spatial_diagnosis": spatial,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"wrote reproduction run: {out}")


@app.command()
def generate_data(
    requirements: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("artifacts/training/dataset_version.json")),
    count: int = typer.Option(1, min=1),
    seed: int = typer.Option(13),
) -> None:
    """Generate validated teacher-authored SFT data for the student loop."""

    settings = get_settings()
    payload = json.loads(requirements.read_text(encoding="utf-8"))
    requirement = str(payload["requirement"])
    references = list(payload.get("references", []))
    teacher_config = model_config_from_settings(settings, ModelRole.TEACHER)
    generator = TeacherDataGenerator(
        create_llm_client(teacher_config),
        teacher_provider=teacher_config.provider,
        teacher_model_id=teacher_config.model,
        prompt_version=teacher_config.prompt_version or settings.teacher_prompt_version,
        seed=seed,
    )
    dataset = generator.generate(requirement, references=references, count=count)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        " ".join(
            [
                f"dataset_version={dataset.dataset_version_id}",
                f"accepted={dataset.metadata['accepted_count']}",
                f"rejected={dataset.metadata['rejected_count']}",
                f"out={out}",
            ]
        )
    )


@app.command()
def export_llamafactory(
    dataset_version: Path = typer.Option(..., exists=True, readable=True),
    out_dir: Path = typer.Option(Path("artifacts/llamafactory/sebench_student_sft")),
    dataset_name: str = typer.Option("sebench_student_sft"),
    base_model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct"),
    template: str = typer.Option("qwen"),
    lora_rank: int = typer.Option(16, min=1),
) -> None:
    """Export accepted teacher data to LLaMA-Factory SFT/QLoRA files."""

    dataset = DatasetVersion.model_validate_json(dataset_version.read_text(encoding="utf-8"))
    manifest = LlamaFactoryExporter().export(
        dataset,
        out_dir,
        dataset_name=dataset_name,
        base_model=base_model,
        template=template,
        lora_rank=lora_rank,
    )
    typer.echo(
        " ".join(
            [
                f"dataset={manifest.dataset_name}",
                f"train={manifest.train_examples}",
                f"validation={manifest.validation_examples}",
                f"test={manifest.test_examples}",
                f"out={out_dir}",
            ]
        )
    )


@app.command()
def evaluate_models(
    dataset: Path = typer.Option(..., exists=True, readable=True),
    out_dir: Path = typer.Option(Path("reports/model_evals")),
    roles: str = typer.Option("base,student,teacher"),
    runner_mode: str = typer.Option("local"),
) -> None:
    """Evaluate configured model roles on the same benchmark dataset."""

    configure_logging()
    settings = get_settings()
    dataset_spec = DatasetSpec.model_validate_json(dataset.read_text(encoding="utf-8"))
    selected_roles = _parse_roles(roles)
    configs = model_configs_from_settings(settings, selected_roles)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for config in configs:
        report = EvaluationOrchestrator(runner_mode=runner_mode).run(dataset_spec)
        report_path = out_dir / f"{config.role.value}_evaluation_report.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        runs.append(
            build_model_run_result(
                config,
                report,
                report_path=str(report_path),
            )
        )

    comparison = compare_model_runs(
        runs,
        dataset_id=dataset_spec.dataset_id,
        metadata={"runner_mode": runner_mode, "source_dataset": str(dataset)},
    )
    comparison_path = out_dir / "model_comparison.json"
    markdown_path = out_dir / "model_comparison.md"
    comparison_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    typer.echo(f"comparison={comparison_path} markdown={markdown_path}")


@app.command()
def compare_models(
    runs: list[str] = typer.Option(..., "--run"),
    out: Path = typer.Option(Path("reports/model_comparison.json")),
    markdown_out: Path | None = typer.Option(Path("reports/model_comparison.md")),
    dataset_id: str = typer.Option("manual-comparison"),
) -> None:
    """Compare existing EvaluationReport files as role:path pairs."""

    settings = get_settings()
    model_runs = []
    for item in runs:
        role_value, path_value = _parse_role_path(item)
        config = model_config_from_settings(settings, ModelRole(role_value))
        path = Path(path_value)
        evaluation_report = _load_evaluation_report(path)
        model_runs.append(
            build_model_run_result(config, evaluation_report, report_path=str(path))
        )

    comparison = compare_model_runs(model_runs, dataset_id=dataset_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    typer.echo(f"comparison={out}")


@app.command()
def import_swe_dataset(
    out: Path = typer.Option(Path("artifacts/public_swe/dataset.json")),
    training_out: Path | None = typer.Option(
        Path("artifacts/public_swe/dataset_version.json")
    ),
    input_file: Path | None = typer.Option(None, exists=True, readable=True),
    hf_dataset: str | None = typer.Option(None),
    split: str = typer.Option("test"),
    source_name: str = typer.Option("public_swe"),
    source_url: str | None = typer.Option(None),
    license_note: str = typer.Option("public_dataset_license_check_required"),
    limit: int | None = typer.Option(20, min=1),
) -> None:
    """Import public SWE-bench-style rows into TaskSpec and training data."""

    source = ExternalBenchmarkSource(
        name=source_name,
        dataset=hf_dataset or (input_file.name if input_file else "public_swe"),
        split=split,
        url=source_url or (f"https://huggingface.co/datasets/{hf_dataset}" if hf_dataset else None),
        license=license_note,
    )
    instances = _load_swe_instances(
        input_file=input_file,
        hf_dataset=hf_dataset,
        split=split,
        source=source,
        limit=limit,
    )
    dataset = DatasetSpec(
        dataset_id=f"public-swe-{source.dataset.replace('/', '.')}-{split}",
        version="0.1.0",
        source="public_swe_ecosystem",
        tasks=[instance.to_task_spec() for instance in instances],
        references=[source.url] if source.url else [],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")

    training_count = 0
    if training_out is not None:
        version = swe_instances_to_dataset_version(instances)
        training_out.parent.mkdir(parents=True, exist_ok=True)
        training_out.write_text(version.model_dump_json(indent=2), encoding="utf-8")
        training_count = len([example for example in version.examples if example.is_trainable])

    typer.echo(
        f"tasks={len(dataset.tasks)} training_examples={training_count} out={out}"
    )


@app.command()
def swe_predict(
    dataset: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("artifacts/public_swe/predictions.jsonl")),
    role: str = typer.Option("student"),
    model_name: str | None = typer.Option(None),
) -> None:
    """Generate SWE-bench-style prediction rows with a configured model role."""

    settings = get_settings()
    config = model_config_from_settings(settings, ModelRole(role))
    client = create_llm_client(config)
    agent = ModelPatchAgent(client)
    dataset_spec = DatasetSpec.model_validate_json(dataset.read_text(encoding="utf-8"))
    predictions: dict[str, str] = {}
    errors: dict[str, str] = {}
    for task in dataset_spec.tasks:
        public_swe = task.fixtures.get("public_swe", {})
        instance_id = str(public_swe.get("instance_id", task.task_id))
        try:
            predictions[instance_id] = agent.generate_patch(task)
        except Exception as exc:
            predictions[instance_id] = ""
            errors[instance_id] = repr(exc)

    write_swe_predictions(
        predictions,
        out,
        model_name_or_path=model_name or config.model,
    )
    if errors:
        error_path = out.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"predictions={out} errors={error_path}")
    else:
        typer.echo(f"predictions={out}")


@app.command()
def run_swe_harness(
    predictions: Path = typer.Option(..., exists=True, readable=True),
    dataset_name: str = typer.Option("SWE-bench/SWE-bench_Lite"),
    split: str = typer.Option("test"),
    out: Path = typer.Option(Path("reports/swebench_harness_report.json")),
    output_dir: Path = typer.Option(Path("artifacts/swebench_harness")),
    results_path: Path | None = typer.Option(None, exists=True, readable=True),
    harness_command: str | None = typer.Option(None),
    timeout_sec: float = typer.Option(3600.0, min=1.0),
) -> None:
    """Run or stage the official SWE-bench harness for prediction evaluation."""

    runner = SWEHarnessRunner(
        command=_split_command(harness_command),
        timeout_sec=timeout_sec,
    )
    report = runner.run(
        predictions_path=predictions,
        dataset_name=dataset_name,
        split=split,
        output_dir=output_dir,
        results_path=results_path,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"aggregate_score={report.aggregate_score:.3f} report={out}")


def _parse_roles(value: str) -> list[ModelRole]:
    roles = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        roles.append(ModelRole(stripped))
    return roles


def _parse_role_path(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise typer.BadParameter("--run must be formatted as role:path")
    role, path = value.split(":", 1)
    return role, path


def _load_evaluation_report(path: Path):
    from sebench_infra.benchmark.schemas import EvaluationReport

    return EvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))


def _load_swe_instances(
    *,
    input_file: Path | None,
    hf_dataset: str | None,
    split: str,
    source: ExternalBenchmarkSource,
    limit: int | None,
) -> list[SWEIssueInstance]:
    if input_file is None and hf_dataset is None:
        raise typer.BadParameter("provide either --input-file or --hf-dataset")
    if input_file is not None and hf_dataset is not None:
        raise typer.BadParameter("use only one of --input-file or --hf-dataset")
    if input_file is not None:
        if input_file.suffix == ".jsonl":
            return load_swe_instances_from_jsonl(input_file, source=source, limit=limit)
        return load_swe_instances_from_json(input_file, source=source, limit=limit)
    assert hf_dataset is not None
    return load_swe_instances_from_hf(
        hf_dataset,
        split=split,
        source=source,
        limit=limit,
    )


def _split_command(command: str | None) -> list[str] | None:
    if not command:
        return None
    return shlex.split(command)


if __name__ == "__main__":
    app()
