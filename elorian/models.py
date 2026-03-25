"""VLM model registry for multimodal evaluation.

Register vision-language models here. Each ModelSpec wraps a DataDesigner
ModelConfig with metadata so the pipeline can generate one response column
per model automatically.

To add a new model:
    registry = ModelRegistry()
    registry.register(ModelSpec(
        alias="my_new_model",
        model_id="claude-sonnet-4-20250514",  # model name without provider prefix
        provider="anthropic",
        description="Short description",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass

import data_designer.config as dd

# Pre-built providers for common LLM APIs.
# The provider_type is used by the LiteLLM bridge as the routing prefix,
# so it must match what LiteLLM expects (e.g. "anthropic", "openai").
# api_key uses env var names that the secret resolver looks up at runtime.
PROVIDERS: dict[str, dd.ModelProvider] = {
    "anthropic": dd.ModelProvider(
        name="anthropic",
        endpoint="https://api.anthropic.com",
        provider_type="anthropic",
        api_key="ANTHROPIC_API_KEY",
    ),
    "openai": dd.ModelProvider(
        name="openai",
        endpoint="https://api.openai.com/v1",
        provider_type="openai",
        api_key="OPENAI_API_KEY",
    ),
}


def get_provider(name: str) -> dd.ModelProvider:
    """Get a provider by name, or raise KeyError with available names."""
    if name not in PROVIDERS:
        raise KeyError(f"Unknown provider {name!r}. Available: {list(PROVIDERS)}")
    return PROVIDERS[name]


def register_provider(provider: dd.ModelProvider) -> None:
    """Register a custom provider globally."""
    PROVIDERS[provider.name] = provider


@dataclass
class ModelSpec:
    """Specification for a vision-language model.

    Attributes:
        alias: Unique short name used as column suffix and ModelConfig alias.
            Must use underscores, not hyphens (Jinja2 template compatibility).
        model_id: Model name (e.g. "claude-sonnet-4-20250514", "gpt-4o").
            Do NOT include the provider prefix — that is handled by ``provider``.
        provider: Provider name matching a key in PROVIDERS (e.g. "anthropic", "openai").
        description: Human-readable description of the model.
        max_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        skip_health_check: Whether to skip the DataDesigner health check.
    """

    alias: str
    model_id: str
    provider: str = "anthropic"
    description: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    skip_health_check: bool = True

    def to_model_config(self) -> dd.ModelConfig:
        """Convert to a DataDesigner ModelConfig."""
        return dd.ModelConfig(
            alias=self.alias,
            model=self.model_id,
            provider=self.provider,
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
            skip_health_check=self.skip_health_check,
        )


class ModelRegistry:
    """Registry of VLM models available for evaluation.

    Ships with Claude and GPT-4o by default. Call ``register()`` to add more.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        """Register a model spec (overwrites if alias already exists)."""
        self._specs[spec.alias] = spec

    def unregister(self, alias: str) -> None:
        """Remove a model by alias."""
        self._specs.pop(alias, None)

    def get(self, alias: str) -> ModelSpec:
        """Get a model spec by alias."""
        return self._specs[alias]

    @property
    def specs(self) -> list[ModelSpec]:
        """All registered model specs in insertion order."""
        return list(self._specs.values())

    @property
    def aliases(self) -> list[str]:
        """All registered model aliases."""
        return list(self._specs.keys())

    def to_model_configs(self) -> list[dd.ModelConfig]:
        """Convert all specs to DataDesigner ModelConfig objects."""
        return [spec.to_model_config() for spec in self._specs.values()]

    def get_unique_providers(self) -> list[dd.ModelProvider]:
        """Get deduplicated list of ModelProviders referenced by registered specs."""
        seen: dict[str, dd.ModelProvider] = {}
        for spec in self._specs.values():
            if spec.provider not in seen:
                seen[spec.provider] = get_provider(spec.provider)
        return list(seen.values())


def get_default_model_registry() -> ModelRegistry:
    """Create a registry pre-loaded with Claude and GPT-4o vision models."""
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            alias="claude_vision",
            model_id="claude-sonnet-4-20250514",
            provider="anthropic",
            description="Anthropic Claude Sonnet 4 (vision)",
        )
    )
    registry.register(
        ModelSpec(
            alias="gpt4o_vision",
            model_id="gpt-4o",
            provider="openai",
            description="OpenAI GPT-4o (vision)",
        )
    )
    return registry
