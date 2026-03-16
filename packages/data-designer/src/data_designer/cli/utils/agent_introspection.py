# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import inspect
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
from data_designer.config.column_types import ColumnConfigT
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


class AgentIntrospectionError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


_FAMILY_SPECS: dict[str, FamilySpec] = {
    "columns": FamilySpec(name="columns", type_union=ColumnConfigT, discriminator_field="column_type"),
    "samplers": FamilySpec(name="samplers", type_union=SamplerParamsT, discriminator_field="sampler_type"),
    "validators": FamilySpec(name="validators", type_union=ValidatorParamsT, discriminator_field="validator_type"),
    "processors": FamilySpec(name="processors", type_union=ProcessorConfigT, discriminator_field="processor_type"),
    "constraints": FamilySpec(name="constraints", type_union=ColumnConstraintT, discriminator_field="constraint_type"),
}


# --- Public API ---


def get_family_names() -> list[str]:
    return sorted(_FAMILY_SPECS)


def get_library_version() -> str:
    try:
        return importlib.metadata.version("data-designer")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_config_package_path() -> str:
    return str(Path(inspect.getfile(dd)).parent)


def get_family_spec(family: str) -> FamilySpec:
    spec = _FAMILY_SPECS.get(_normalize_family_name(family))
    if spec is None:
        raise AgentIntrospectionError(
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
            raise AgentIntrospectionError(
                code="duplicate_discriminator_value",
                message=f"Duplicate discriminator {type_name!r} in family {family!r}.",
                details={"family": family, "type_name": type_name},
            )
        discovered[type_name] = model
    return dict(sorted(discovered.items()))


def get_family_catalog(family: str) -> list[dict[str, str]]:
    return [
        {
            "type_name": type_name,
            "description": _get_first_paragraph(cls.__doc__) or "",
        }
        for type_name, cls in discover_family_types(family).items()
    ]


def get_family_source_file(family: str) -> str:
    members = get_args(get_family_spec(family).type_union)
    return _get_source_file(members[0]) if members else ""


def get_operations() -> list[dict[str, str]]:
    return [
        {"name": c.name, "command_pattern": c.command_pattern, "description": c.help, "returns": c.returns}
        for c in AGENT_COMMANDS
    ]


def get_context(config_dir: Path) -> dict[str, Any]:
    families = get_family_names()
    catalogs = {f: get_family_catalog(f) for f in families}
    return {
        "library_version": get_library_version(),
        "config_package_path": get_config_package_path(),
        "operations": get_operations(),
        "families": [{"family": f, "count": len(catalogs[f]), "file": get_family_source_file(f)} for f in families],
        "types": catalogs,
        "state": {
            "model_aliases": get_model_aliases_state(config_dir),
            "persona_datasets": get_persona_datasets_state(config_dir),
        },
    }


def get_types(family: str | None) -> dict[str, Any]:
    if family is None:
        families = get_family_names()
        catalogs = {f: get_family_catalog(f) for f in families}
        return {
            "families": [{"family": f, "count": len(catalogs[f]), "file": get_family_source_file(f)} for f in families],
            "items": catalogs,
        }
    spec = get_family_spec(family)
    return {"family": spec.name, "file": get_family_source_file(spec.name), "items": get_family_catalog(family)}


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
        providers_by_name = {p.name: p for p in provider_registry.providers}
        default_provider = provider_registry.default or (
            provider_registry.providers[0].name if provider_registry.providers else None
        )
        missing_key_names = {p.name for p in get_providers_with_missing_api_keys(provider_registry.providers)}

    for mc in sorted(model_registry.model_configs, key=lambda m: m.alias):
        effective = mc.provider or default_provider
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
                "model_alias": mc.alias,
                "model": mc.model,
                "generation_type": getattr(mc.generation_type, "value", str(mc.generation_type)),
                "configured_provider": mc.provider,
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
    download_svc = DownloadService(config_dir, persona_repo)
    return {
        "managed_assets_directory": str(download_svc.get_managed_assets_directory()),
        "items": [
            {
                "locale": loc.code,
                "dataset_name": loc.dataset_name,
                "size": loc.size,
                "installed": download_svc.is_locale_downloaded(loc.code),
            }
            for loc in sorted(persona_repo.list_all(), key=lambda loc: loc.code)
        ],
    }


# --- Private helpers ---


def _normalize_family_name(family: str) -> str:
    normalized = family.strip().lower()
    if normalized in _FAMILY_SPECS:
        return normalized
    plural = f"{normalized}s"
    if plural in _FAMILY_SPECS:
        return plural
    return normalized


def _extract_literal_value(annotation: Any) -> str:
    if get_origin(annotation) is not Literal or not get_args(annotation):
        raise AgentIntrospectionError(
            code="invalid_discriminator_annotation",
            message=f"Expected non-empty Literal annotation, got {annotation!r}.",
        )
    value = get_args(annotation)[0]
    return str(value.value) if isinstance(value, Enum) else str(value)


def _get_first_paragraph(docstring: str | None) -> str | None:
    """Extract the first paragraph of a docstring, before any blank line or section header."""
    if not docstring:
        return None
    lines: list[str] = []
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped.lower() in _SECTION_HEADERS:
            break
        if not stripped and lines:
            break
        if stripped:
            lines.append(stripped)
    return " ".join(lines) if lines else None


def _get_source_file(cls: type) -> str:
    """Return the source file path relative to the data_designer package (e.g. 'data_designer/config/foo.py')."""
    try:
        full_path = Path(inspect.getfile(cls))
    except (TypeError, OSError):
        return ""
    parts = full_path.parts
    try:
        idx = parts.index("data_designer")
    except ValueError:
        return str(full_path)
    return str(Path(*parts[idx:]))


def _load_registry(repo: Any) -> Any:
    if not repo.exists():
        return None
    registry = repo.load()
    if registry is None:
        raise AgentIntrospectionError(
            code="invalid_registry",
            message=f"Failed to load registry from {str(repo.config_file)!r}.",
            details={"config_file": str(repo.config_file)},
        )
    return registry


_SECTION_HEADERS = frozenset(
    {
        "args:",
        "arguments:",
        "attributes:",
        "example:",
        "examples:",
        "keyword args:",
        "keyword arguments:",
        "note:",
        "notes:",
        "raises:",
        "references:",
        "returns:",
        "see also:",
        "todo:",
        "warns:",
        "yields:",
    }
)
