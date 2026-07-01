from fastapi import FastAPI

from sebench_infra.authoring import AuthoringAgent, MockLLMClient
from sebench_infra.benchmark import DatasetBuilder
from sebench_infra.benchmark.schemas import DatasetSpec, EvaluationReport, RunRequest
from sebench_infra.orchestrator import EvaluationOrchestrator

app = FastAPI(title="SE-Bench AI Infra Replica", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/author", response_model=DatasetSpec)
def author(payload: dict) -> DatasetSpec:
    requirement = str(payload["requirement"])
    references = list(payload.get("references", []))
    tasks = AuthoringAgent(MockLLMClient()).author_tasks(requirement)
    return DatasetBuilder().build(requirement, tasks, references)


@app.post("/evaluate", response_model=EvaluationReport)
def evaluate(request: RunRequest) -> EvaluationReport:
    return EvaluationOrchestrator(runner_mode=request.runner_mode).run(
        request.dataset,
        max_tasks=request.max_tasks,
    )
