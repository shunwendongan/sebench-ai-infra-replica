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
    work_image: str = "sebench-work:latest"
    judge_image: str = "sebench-judge:latest"
    run_timeout_seconds: int = Field(default=120, ge=1)


def get_settings() -> Settings:
    return Settings()
