import json
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
) -> None:
    """Evaluate a dataset with the local mock runner or Docker runner."""

    configure_logging()
    dataset_spec = DatasetSpec.model_validate_json(dataset.read_text(encoding="utf-8"))
    report = EvaluationOrchestrator(runner_mode=runner_mode).run(dataset_spec)
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


if __name__ == "__main__":
    app()
