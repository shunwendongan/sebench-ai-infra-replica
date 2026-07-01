import json
from pathlib import Path

from sebench_infra.authoring import AuthoringAgent, MockLLMClient
from sebench_infra.benchmark import DatasetBuilder
from sebench_infra.orchestrator import EvaluationOrchestrator
from sebench_infra.spatial import SpatialDiagnosisEngine


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    requirement_payload = json.loads(
        (root / "examples/requirements.json").read_text(encoding="utf-8")
    )
    tasks = AuthoringAgent(MockLLMClient()).author_tasks(requirement_payload["requirement"])
    dataset = DatasetBuilder().build(
        requirement_payload["requirement"],
        tasks,
        requirement_payload["references"],
    )
    report = EvaluationOrchestrator().run(dataset)
    spatial = SpatialDiagnosisEngine().diagnose_scene_file(
        root / "examples/synthetic_spatial_scene.json"
    )
    output = {
        "dataset": dataset.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "spatial": spatial,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
