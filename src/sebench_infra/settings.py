from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe local defaults."""

    model_config = SettingsConfigDict(env_prefix="SEBENCH_", env_file=".env", extra="ignore")

    artifact_db: Path = Field(default=Path("artifacts/sebench.sqlite"))
    runner_mode: str = Field(default="local", pattern="^(local|docker)$")
    llm_provider: str = Field(default="mock", pattern="^(mock|openai_compatible)$")
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "mock"
    work_image: str = "sebench-work:latest"
    judge_image: str = "sebench-judge:latest"
    run_timeout_seconds: int = Field(default=120, ge=1)

    teacher_provider: str = Field(default="mock", pattern="^(mock|openai_compatible)$")
    teacher_model: str = "gpt-5.5"
    teacher_openai_base_url: str | None = None
    teacher_openai_api_key: str | None = None
    teacher_prompt_version: str = "teacher-data-v1"
    teacher_cost_per_1k_input_tokens: float = Field(default=0.0, ge=0.0)
    teacher_cost_per_1k_output_tokens: float = Field(default=0.0, ge=0.0)

    base_provider: str = Field(default="mock", pattern="^(mock|openai_compatible)$")
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_openai_base_url: str | None = None
    base_openai_api_key: str | None = None
    base_prompt_version: str = "base-authoring-v1"
    base_cost_per_1k_input_tokens: float = Field(default=0.0, ge=0.0)
    base_cost_per_1k_output_tokens: float = Field(default=0.0, ge=0.0)

    student_provider: str = Field(default="mock", pattern="^(mock|openai_compatible)$")
    student_model: str = "sebench-student-lora"
    student_openai_base_url: str | None = None
    student_openai_api_key: str | None = None
    student_prompt_version: str = "student-authoring-v1"
    student_cost_per_1k_input_tokens: float = Field(default=0.0, ge=0.0)
    student_cost_per_1k_output_tokens: float = Field(default=0.0, ge=0.0)


def get_settings() -> Settings:
    return Settings()
