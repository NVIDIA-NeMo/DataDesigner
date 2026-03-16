# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import data_designer.config as dd
from data_designer.cli.agent_command_defs import AGENT_COMMANDS
from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.persona_repository import PersonaRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.services.download_service import DownloadService
from data_designer.cli.utils.agent_schema_view import describe_pydantic_model
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.default_model_settings import get_providers_with_missing_api_keys
from data_designer.config.processor_types import ProcessorConfigT
from data_designer.config.sampler_constraints import ColumnConstraintT
from data_designer.config.sampler_params import SamplerParamsT
from data_designer.config.validator_params import ValidatorParamsT


@dataclass(frozen=True)
class FamilySpec:
    name: str
    type_union: Any
    discriminator_field: str


class AgentContextError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


FAMILY_SPECS: dict[str, FamilySpec] = {
    "columns": FamilySpec(name="columns", type_union=ColumnConfigT, discriminator_field="column_type"),
    "samplers": FamilySpec(name="samplers", type_union=SamplerParamsT, discriminator_field="sampler_type"),
    "validators": FamilySpec(name="validators", type_union=ValidatorParamsT, discriminator_field="validator_type"),
    "processors": FamilySpec(name="processors", type_union=ProcessorConfigT, discriminator_field="processor_type"),
    "constraints": FamilySpec(name="constraints", type_union=ColumnConstraintT, discriminator_field="constraint_type"),
}


# --- Public API ---


def get_family_names() -> list[str]:
    return sorted(FAMILY_SPECS)


def get_family_spec(family: str) -> FamilySpec:
    spec = FAMILY_SPECS.get(_normalize_family_name(family))
    if spec is None:
        raise AgentContextError(
            code="unknown_family",
            message=f"Unknown family {family!r}.",
            details={"available_families": get_family_names()},
        )
    return spec


def discover_family_types(family: str) -> dict[str, type]:
    spec = get_family_spec(family)
    discovered: dict[str, type] = {}
    for model in get_args(spec.type_union):
        type_name = _extract_literal_value(model.model_fields[spec.discriminator_field].annotation)
        if type_name in discovered and discovered[type_name] is not model:
            raise AgentContextError(
                code="duplicate_discriminator_value",
                message=f"Duplicate discriminator {type_name!r} in family {family!r}.",
                details={"family": family, "type_name": type_name},
            )
        discovered[type_name] = model
    return dict(sorted(discovered.items()))


def get_import_path(cls: type) -> str:
    exported = getattr(dd, cls.__name__, None)
    if exported is cls:
        return f"data_designer.config.{cls.__name__}"
    return f"{cls.__module__}.{cls.__qualname__}"


def get_family_catalog(family: str) -> list[dict[str, str]]:
    return [
        {"type_name": type_name, "class_name": cls.__name__, "import_path": get_import_path(cls)}
        for type_name, cls in discover_family_types(family).items()
    ]


def get_family_schema(family: str, type_name: str) -> dict[str, Any]:
    spec = get_family_spec(family)
    types_map = discover_family_types(family)
    cls = types_map.get(type_name)
    if cls is None:
        raise AgentContextError(
            code="unknown_type",
            message=f"Unknown type {type_name!r} for family {family!r}.",
            details={"family": family, "available_types": list(types_map)},
        )
    return _build_schema_dict(spec, type_name, cls)


def get_family_schemas(family: str) -> dict[str, Any]:
    spec = get_family_spec(family)
    types_map = discover_family_types(family)
    items = [_build_schema_dict(spec, type_name, cls) for type_name, cls in types_map.items()]
    return {"family": spec.name, "items": items}


def get_builder_api() -> dict[str, Any]:
    return {
        "class_name": DataDesignerConfigBuilder.__name__,
        "import_path": get_import_path(DataDesignerConfigBuilder),
        "methods": _get_builder_methods(),
    }


def get_operations() -> list[dict[str, str]]:
    return [
        {
            "name": command.name,
            "command_pattern": command.command_pattern,
            "description": command.help,
            "returns": command.returns,
        }
        for command in AGENT_COMMANDS
    ]


def get_context(config_dir: Path) -> dict[str, Any]:
    catalogs = {family: get_family_catalog(family) for family in get_family_names()}
    return {
        "operations": get_operations(),
        "families": [{"family": family, "count": len(items)} for family, items in catalogs.items()],
        "types": catalogs,
        "state": {
            "model_aliases": get_model_aliases_state(config_dir),
            "persona_datasets": get_persona_datasets_state(config_dir),
        },
        "builder": get_builder_api(),
    }


def get_types(family: str | None) -> dict[str, Any]:
    if family is None:
        catalogs = {name: get_family_catalog(name) for name in get_family_names()}
        return {
            "families": [{"family": name, "count": len(items)} for name, items in catalogs.items()],
            "items": catalogs,
        }
    return {"family": get_family_spec(family).name, "items": get_family_catalog(family)}


def get_schema(family: str, type_name: str | None, *, all_types: bool) -> dict[str, Any]:
    if all_types and type_name is not None:
        raise AgentContextError(
            code="invalid_schema_request",
            message=(
                f"Cannot use --all with a type name. "
                f"Use 'agent schema {family} {type_name}' for a single type, "
                f"or 'agent schema {family} --all' for all types."
            ),
            details={"family": family, "type_name": type_name, "all": all_types},
        )
    if all_types:
        return get_family_schemas(family)
    if type_name is None:
        raise AgentContextError(
            code="missing_type_name",
            message="A type name is required unless --all is provided.",
            details={"family": family},
        )
    return get_family_schema(family, type_name)


def get_model_aliases_state(config_dir: Path) -> dict[str, Any]:
    model_registry = _load_registry(ModelRepository(config_dir))
    provider_registry = _load_registry(ProviderRepository(config_dir))

    items: list[dict[str, Any]] = []
    if model_registry is None:
        return {
            "model_config_present": False,
            "provider_config_present": provider_registry is not None,
            "default_provider": None if provider_registry is None else provider_registry.default,
            "items": items,
        }

    providers_by_name: dict[str, Any] = {}
    missing_key_names: set[str] = set()
    default_provider: str | None = None
    if provider_registry is not None:
        providers_by_name = {provider.name: provider for provider in provider_registry.providers}
        default_provider = provider_registry.default or (
            provider_registry.providers[0].name if provider_registry.providers else None
        )
        missing_key_names = {
            provider.name for provider in get_providers_with_missing_api_keys(provider_registry.providers)
        }

    for model_config in sorted(model_registry.model_configs, key=lambda model: model.alias):
        effective = model_config.provider or default_provider
        usable = True
        reason: str | None = None
        if effective is None:
            usable, reason = False, "No model provider is configured."
        elif effective not in providers_by_name:
            usable, reason = False, f"Provider {effective!r} is not configured."
        elif effective in missing_key_names:
            usable, reason = False, f"Provider {effective!r} is missing an API key."

        items.append(
            {
                "model_alias": model_config.alias,
                "model": model_config.model,
                "generation_type": getattr(model_config.generation_type, "value", str(model_config.generation_type)),
                "configured_provider": model_config.provider,
                "effective_provider": effective,
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


def get_persona_datasets_state(config_dir: Path) -> dict[str, Any]:
    persona_repo = PersonaRepository()
    download_service = DownloadService(config_dir, persona_repo)
    return {
        "managed_assets_directory": str(download_service.get_managed_assets_directory()),
        "items": [
            {
                "locale": locale.code,
                "dataset_name": locale.dataset_name,
                "size": locale.size,
                "installed": download_service.is_locale_downloaded(locale.code),
            }
            for locale in sorted(persona_repo.list_all(), key=lambda locale: locale.code)
        ],
    }


# --- Private helpers ---


_FAMILY_HIDDEN_FIELDS: dict[str, frozenset[str]] = {
    "columns": frozenset({"allow_resize"}),
}

_TYPE_VISIBLE_OVERRIDES: dict[tuple[str, str], frozenset[str]] = {
    ("columns", "custom"): frozenset({"allow_resize"}),
}


def _get_hidden_fields(spec: FamilySpec, type_name: str) -> frozenset[str]:
    hidden = {spec.discriminator_field}
    hidden |= _FAMILY_HIDDEN_FIELDS.get(spec.name, frozenset())
    hidden -= _TYPE_VISIBLE_OVERRIDES.get((spec.name, type_name), frozenset())
    return frozenset(hidden)


def _build_schema_dict(spec: FamilySpec, type_name: str, cls: type) -> dict[str, Any]:
    return {
        "family": spec.name,
        "type_name": type_name,
        "class_name": cls.__name__,
        "import_path": get_import_path(cls),
        "schema": cls.model_json_schema(),
        "schema_view": describe_pydantic_model(cls, hidden_fields=_get_hidden_fields(spec, type_name)),
    }


def _normalize_family_name(family: str) -> str:
    normalized = family.strip().lower()
    if normalized in FAMILY_SPECS:
        return normalized
    plural = f"{normalized}s"
    if plural in FAMILY_SPECS:
        return plural
    return normalized


def _extract_literal_value(annotation: Any) -> str:
    if get_origin(annotation) is not Literal or not get_args(annotation):
        raise AgentContextError(
            code="invalid_discriminator_annotation",
            message=f"Expected non-empty Literal annotation, got {annotation!r}.",
        )
    value = get_args(annotation)[0]
    return str(value.value) if isinstance(value, Enum) else str(value)


def _get_builder_methods() -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []
    for name, attr in inspect.getmembers(DataDesignerConfigBuilder):
        if name.startswith("_") and name != "__init__":
            continue
        if not callable(attr):
            continue
        try:
            signature = inspect.signature(attr)
        except (TypeError, ValueError):
            continue

        docstring = inspect.getdoc(attr)
        methods.append(
            {
                "name": name,
                "signature": _format_signature(name, signature),
                "summary": _get_first_line(docstring),
                "docstring": docstring,
            }
        )

    return methods


def _format_signature(method_name: str, signature: inspect.Signature) -> str:
    params = [param for param in signature.parameters.values() if param.name not in {"self", "cls"}]
    signature_text = str(signature.replace(parameters=params))
    signature_text = re.sub(
        r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+",
        lambda match: match.group().rsplit(".", 1)[-1],
        signature_text,
    )
    signature_text = re.sub(r"(?<=: )'([^']+)'", r"\1", signature_text)
    signature_text = re.sub(r"(?<=-> )'([^']+)'", r"\1", signature_text)
    return f"{method_name}{signature_text}"


def _get_first_line(text: str | None) -> str | None:
    return next((line.strip() for line in text.strip().splitlines() if line.strip()), None) if text else None


def _load_registry(repository: Any) -> Any:
    if not repository.exists():
        return None
    registry = repository.load()
    if registry is None:
        raise AgentContextError(
            code="invalid_registry",
            message=f"Failed to load registry from {str(repository.config_file)!r}.",
            details={"config_file": str(repository.config_file)},
        )
    return registry
