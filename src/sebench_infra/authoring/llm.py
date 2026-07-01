import json
from typing import Protocol

from sebench_infra.benchmark.schemas import TaskCategory


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict:
        """Return a JSON-compatible object."""


class MockLLMClient:
    """Deterministic authoring client for tests and local demos."""

    def complete_json(self, system: str, user: str) -> dict:
        spatial = "3d" in user.lower() or "spatial" in user.lower() or "空间" in user
        category = TaskCategory.SPATIAL_REASONING if spatial else TaskCategory.BENCHMARK_AUTHORING
        task_id = "spatial_geometry_bridge" if spatial else "benchmark_contract_check"
        return {
            "tasks": [
                {
                    "task_id": task_id,
                    "title": (
                        "Validate geometry bridge"
                        if spatial
                        else "Validate benchmark contract"
                    ),
                    "category": category.value,
                    "prompt": (
                        "Use the provided synthetic fixtures and write a concise answer in "
                        "submission/answer.txt."
                    ),
                    "fixtures": {
                        "requirement": user,
                        "public_reproduction": True,
                    },
                    "expected_artifacts": ["submission/answer.txt"],
                    "allowed_paths": ["submission/"],
                    "scoring": [
                        {
                            "name": "answer_created",
                            "kind": "file_exists",
                            "path": "submission/answer.txt",
                            "weight": 0.4,
                        },
                        {
                            "name": "mentions_reproducible",
                            "kind": "contains",
                            "path": "submission/answer.txt",
                            "expected": "reproducible",
                            "weight": 0.6,
                        },
                    ],
                    "tags": ["synthetic", "ai-infra", "public-paper-replica"],
                    "metadata": {"author": "mock_llm", "paper_replication": True},
                }
            ]
        }


class OpenAICompatibleClient:
    """Adapter for OpenAI-compatible endpoints such as vLLM serving."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> dict:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the llm extra to use OpenAI-compatible inference") from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
