from pydantic import ValidationError

from sebench_infra.authoring.llm import LLMClient
from sebench_infra.authoring.prompts import AUTHORING_SYSTEM_PROMPT, REPAIR_SYSTEM_PROMPT
from sebench_infra.benchmark.schemas import TaskSpec


class AuthoringAgent:
    """Generate, validate, and repair benchmark tasks."""

    def __init__(self, llm: LLMClient, max_repairs: int = 1) -> None:
        self.llm = llm
        self.max_repairs = max_repairs

    def author_tasks(self, requirement: str) -> list[TaskSpec]:
        payload = self.llm.complete_json(AUTHORING_SYSTEM_PROMPT, requirement)
        return self._validate_or_repair(payload, requirement)

    def _validate_or_repair(self, payload: dict, requirement: str) -> list[TaskSpec]:
        last_error: ValidationError | None = None
        current_payload = payload

        for _ in range(self.max_repairs + 1):
            try:
                return [TaskSpec.model_validate(item) for item in current_payload.get("tasks", [])]
            except ValidationError as exc:
                last_error = exc
                current_payload = self.llm.complete_json(
                    REPAIR_SYSTEM_PROMPT,
                    "\n\n".join(
                        [
                            f"Requirement:\n{requirement}",
                            f"Validation error:\n{exc}",
                            f"Payload:\n{current_payload}",
                        ]
                    ),
                )

        raise ValueError(f"unable to author valid tasks: {last_error}")
