# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.cli.repositories.persona_repository import PersonaRepository
from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.cli.services.agent_introspection import (
    AgentIntrospectionError,
    get_builder_api,
    get_family_catalog,
    get_family_counts,
    get_family_names,
    get_family_schema,
    get_family_schemas,
    get_library_version,
    get_resolved_family_name,
)
from data_designer.cli.services.download_service import DownloadService
from data_designer.config.default_model_settings import get_providers_with_missing_api_keys


class AgentController:
    """Controller for the agent-only JSON introspection interface."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.model_repository = ModelRepository(config_dir)
        self.provider_repository = ProviderRepository(config_dir)
        self.persona_repository = PersonaRepository()
        self.download_service = DownloadService(config_dir, self.persona_repository)

    def get_library_version(self) -> str:
        """Return the installed Data Designer package version."""
        return get_library_version()

    def get_context(self) -> dict[str, Any]:
        """Return a self-describing bootstrap payload for agents."""
        return {
            "operations": self._get_operations(),
            "families": get_family_counts(),
            "types": {family: get_family_catalog(family) for family in get_family_names()},
            "state": {
                "model_aliases": self.get_model_aliases_state(),
                "persona_datasets": self.get_persona_datasets_state(),
            },
            "builder": get_builder_api(include_docstrings=False),
        }

    def get_types(self, family: str | None) -> dict[str, Any]:
        """Return available type catalogs for one family or all families."""
        if family is None:
            return {
                "families": get_family_counts(),
                "items": {family_name: get_family_catalog(family_name) for family_name in get_family_names()},
            }
        return {
            "family": get_resolved_family_name(family),
            "items": get_family_catalog(family),
        }

    def get_schema(self, family: str, type_name: str | None, *, all_types: bool) -> dict[str, Any]:
        """Return JSON schema data for one family member or an entire family."""
        if all_types and type_name is not None:
            raise AgentIntrospectionError(
                code="invalid_schema_request",
                message="Provide either a type name or --all, but not both.",
                details={"family": family, "type_name": type_name, "all": all_types},
            )
        if all_types:
            return get_family_schemas(family)
        if type_name is None:
            raise AgentIntrospectionError(
                code="missing_type_name",
                message="A type name is required unless --all is provided.",
                details={"family": family},
            )
        return get_family_schema(family, type_name)

    def get_builder(self) -> dict[str, Any]:
        """Return the config-builder API surface."""
        return get_builder_api(include_docstrings=True)

    def get_model_aliases_state(self) -> dict[str, Any]:
        """Return the current configured model aliases with provider status."""
        model_registry = self._load_model_registry()
        provider_registry = self._load_provider_registry()

        items: list[dict[str, Any]] = []
        if model_registry is None:
            return {
                "model_config_present": False,
                "provider_config_present": provider_registry is not None,
                "default_provider": None if provider_registry is None else provider_registry.default,
                "items": items,
            }

        providers_by_name = {}
        missing_api_key_provider_names: set[str] = set()
        default_provider: str | None = None
        if provider_registry is not None:
            providers_by_name = {provider.name: provider for provider in provider_registry.providers}
            default_provider = provider_registry.default or provider_registry.providers[0].name
            missing_api_key_provider_names = {
                provider.name for provider in get_providers_with_missing_api_keys(provider_registry.providers)
            }

        for model_config in sorted(model_registry.model_configs, key=lambda item: item.alias):
            configured_provider = model_config.provider
            effective_provider = configured_provider or default_provider
            usable = True
            reason: str | None = None

            if effective_provider is None:
                usable = False
                reason = "No model provider is configured."
            elif effective_provider not in providers_by_name:
                usable = False
                reason = f"Provider {effective_provider!r} is not configured."
            elif effective_provider in missing_api_key_provider_names:
                usable = False
                reason = f"Provider {effective_provider!r} is missing an API key."

            items.append(
                {
                    "model_alias": model_config.alias,
                    "model": model_config.model,
                    "generation_type": self._get_enum_value(model_config.generation_type),
                    "configured_provider": configured_provider,
                    "effective_provider": effective_provider,
                    "usable": usable,
                    "reason": reason,
                }
            )

        return {
            "model_config_present": True,
            "provider_config_present": provider_registry is not None,
            "default_provider": default_provider,
            "items": items,
        }

    def get_persona_datasets_state(self) -> dict[str, Any]:
        """Return built-in persona dataset availability and local install state."""
        return {
            "managed_assets_directory": str(self.download_service.get_managed_assets_directory()),
            "items": [
                {
                    "locale": locale.code,
                    "dataset_name": locale.dataset_name,
                    "size": locale.size,
                    "installed": self.download_service.is_locale_downloaded(locale.code),
                }
                for locale in sorted(self.persona_repository.list_all(), key=lambda item: item.code)
            ],
        }

    def _load_model_registry(self) -> ModelConfigRegistry | None:
        if not self.model_repository.exists():
            return None
        model_registry = self.model_repository.load()
        if model_registry is None:
            raise AgentIntrospectionError(
                code="invalid_model_registry",
                message=f"Failed to load the model registry from {str(self.model_repository.config_file)!r}.",
                details={"config_file": str(self.model_repository.config_file)},
            )
        return model_registry

    def _load_provider_registry(self) -> ModelProviderRegistry | None:
        if not self.provider_repository.exists():
            return None
        provider_registry = self.provider_repository.load()
        if provider_registry is None:
            raise AgentIntrospectionError(
                code="invalid_provider_registry",
                message=f"Failed to load the provider registry from {str(self.provider_repository.config_file)!r}.",
                details={"config_file": str(self.provider_repository.config_file)},
            )
        return provider_registry

    def _get_enum_value(self, value: Any) -> str:
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)

    def _get_operations(self) -> list[dict[str, str]]:
        return [
            {
                "name": "context",
                "command_pattern": "data-designer agent context",
                "description": "Return a bootstrap payload with operations, families, types, local state, and builder summary.",
                "returns": "agent_context",
            },
            {
                "name": "types",
                "command_pattern": "data-designer agent types [family]",
                "description": "Return available type names and import paths for one schema family or all families.",
                "returns": "agent_types",
            },
            {
                "name": "schema",
                "command_pattern": "data-designer agent schema <family> <type-name> | --all",
                "description": "Return JSON schema for a specific type or every type in a family.",
                "returns": "agent_schema",
            },
            {
                "name": "builder",
                "command_pattern": "data-designer agent builder",
                "description": "Return the DataDesignerConfigBuilder method surface with signatures and docstrings.",
                "returns": "agent_builder",
            },
            {
                "name": "state.model-aliases",
                "command_pattern": "data-designer agent state model-aliases",
                "description": "Return configured model aliases and whether each one is currently usable.",
                "returns": "agent_state_model_aliases",
            },
            {
                "name": "state.persona-datasets",
                "command_pattern": "data-designer agent state persona-datasets",
                "description": "Return built-in persona locales and whether each dataset is installed locally.",
                "returns": "agent_state_persona_datasets",
            },
        ]
