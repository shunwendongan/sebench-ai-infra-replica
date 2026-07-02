from __future__ import annotations

from typing import TYPE_CHECKING

from sebench_infra.authoring import LLMClient, MockLLMClient, OpenAICompatibleClient
from sebench_infra.training_loop.models import ModelRole, ModelRunConfig, ProviderKind

if TYPE_CHECKING:
    from sebench_infra.settings import Settings


def create_llm_client(config: ModelRunConfig) -> LLMClient:
    if config.provider == ProviderKind.MOCK:
        return MockLLMClient()
    if config.provider == ProviderKind.OPENAI_COMPATIBLE:
        if not config.base_url:
            raise ValueError(f"{config.role} openai_compatible provider requires base_url")
        if not config.api_key:
            raise ValueError(f"{config.role} openai_compatible provider requires api_key")
        return OpenAICompatibleClient(
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value(),
            model=config.model,
        )
    raise ValueError(f"unsupported provider: {config.provider}")


def model_config_from_settings(settings: Settings, role: ModelRole | str) -> ModelRunConfig:
    resolved_role = ModelRole(role)
    prefix = resolved_role.value
    provider = ProviderKind(getattr(settings, f"{prefix}_provider"))
    api_key = getattr(settings, f"{prefix}_openai_api_key")
    return ModelRunConfig(
        role=resolved_role,
        provider=provider,
        model=getattr(settings, f"{prefix}_model"),
        base_url=getattr(settings, f"{prefix}_openai_base_url"),
        api_key=api_key,
        api_key_env=f"SEBENCH_{prefix.upper()}_OPENAI_API_KEY" if api_key else None,
        prompt_version=getattr(settings, f"{prefix}_prompt_version"),
        cost_per_1k_input_tokens=getattr(settings, f"{prefix}_cost_per_1k_input_tokens"),
        cost_per_1k_output_tokens=getattr(settings, f"{prefix}_cost_per_1k_output_tokens"),
        metadata={"configured_from": "settings"},
    )


def model_configs_from_settings(
    settings: Settings,
    roles: list[ModelRole | str] | None = None,
) -> list[ModelRunConfig]:
    selected = roles or [ModelRole.BASE, ModelRole.STUDENT, ModelRole.TEACHER]
    return [model_config_from_settings(settings, role) for role in selected]
