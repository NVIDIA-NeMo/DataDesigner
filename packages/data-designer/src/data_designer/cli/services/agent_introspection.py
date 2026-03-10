# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, get_args, get_origin

import data_designer.config as dd
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
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


def get_family_names() -> list[str]:
    return sorted(_FAMILY_SPECS)


def get_library_version() -> str:
    try:
        return importlib.metadata.version("data-designer")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_resolved_family_name(family: str) -> str:
    return _get_family_spec(family).name


def get_family_counts() -> list[dict[str, str | int]]:
    return [{"family": family, "count": len(get_family_catalog(family))} for family in get_family_names()]


def get_family_catalog(family: str) -> list[dict[str, str]]:
    return [
        {
            "type_name": type_name,
            "class_name": cls.__name__,
            "import_path": get_import_path(cls),
        }
        for type_name, cls in _discover_family_types(family).items()
    ]


def get_family_schema(family: str, type_name: str) -> dict[str, Any]:
    cls = _discover_family_types(family).get(type_name)
    if cls is None:
        raise AgentIntrospectionError(
            code="unknown_type",
            message=f"Unknown type {type_name!r} for family {family!r}.",
            details={"family": family, "available_types": list(_discover_family_types(family))},
        )

    return {
        "family": get_resolved_family_name(family),
        "type_name": type_name,
        "class_name": cls.__name__,
        "import_path": get_import_path(cls),
        "schema": cls.model_json_schema(),
    }


def get_family_schemas(family: str) -> dict[str, Any]:
    items = [get_family_schema(family, type_name) for type_name in _discover_family_types(family)]
    return {"family": get_resolved_family_name(family), "items": items}


def get_builder_api(*, include_docstrings: bool) -> dict[str, Any]:
    return {
        "class_name": DataDesignerConfigBuilder.__name__,
        "import_path": get_import_path(DataDesignerConfigBuilder),
        "methods": get_builder_methods(include_docstrings=include_docstrings),
    }


def get_builder_methods(*, include_docstrings: bool) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []

    for name, attr in inspect.getmembers(DataDesignerConfigBuilder):
        if name.startswith("_") and name != "__init__":
            continue
        if name.startswith("__") and name != "__init__":
            continue
        if not callable(attr):
            continue

        try:
            signature = inspect.signature(attr)
        except (TypeError, ValueError):
            continue

        docstring = inspect.getdoc(attr)
        method_info: dict[str, Any] = {
            "name": name,
            "signature": _format_signature(name, signature),
            "summary": _get_first_docstring_line(docstring),
        }
        if include_docstrings:
            method_info["docstring"] = docstring

        methods.append(method_info)

    return methods


def get_import_path(cls: type) -> str:
    exported = getattr(dd, cls.__name__, None)
    if exported is cls:
        return f"data_designer.config.{cls.__name__}"
    return f"{cls.__module__}.{cls.__qualname__}"


def _discover_family_types(family: str) -> dict[str, type]:
    spec = _get_family_spec(family)
    discovered: dict[str, type] = {}
    for model in get_args(spec.type_union):
        if not hasattr(model, "model_fields"):
            raise AgentIntrospectionError(
                code="invalid_family_model",
                message=f"Model {model!r} in family {family!r} does not expose Pydantic model_fields.",
                details={"family": family, "model": repr(model)},
            )
        if spec.discriminator_field not in model.model_fields:
            raise AgentIntrospectionError(
                code="missing_discriminator_field",
                message=f"Model {model.__name__!r} in family {family!r} is missing discriminator field "
                f"{spec.discriminator_field!r}.",
                details={
                    "family": family,
                    "model": model.__name__,
                    "discriminator_field": spec.discriminator_field,
                },
            )

        type_name = _extract_literal_value(model.model_fields[spec.discriminator_field].annotation)
        if type_name in discovered and discovered[type_name] is not model:
            raise AgentIntrospectionError(
                code="duplicate_discriminator_value",
                message=f"Multiple models in family {family!r} resolve to discriminator value {type_name!r}.",
                details={
                    "family": family,
                    "type_name": type_name,
                    "models": [discovered[type_name].__name__, model.__name__],
                },
            )

        discovered[type_name] = model

    return dict(sorted(discovered.items()))


def _get_family_spec(family: str) -> FamilySpec:
    spec = _FAMILY_SPECS.get(_normalize_family_name(family))
    if spec is None:
        raise AgentIntrospectionError(
            code="unknown_family",
            message=f"Unknown family {family!r}.",
            details={"available_families": get_family_names()},
        )
    return spec


def _normalize_family_name(family: str) -> str:
    normalized = family.strip().lower()
    if normalized in _FAMILY_SPECS:
        return normalized
    plural = f"{normalized}s"
    if plural in _FAMILY_SPECS:
        return plural
    return normalized


def _extract_literal_value(annotation: Any) -> str:
    if get_origin(annotation) is not Literal:
        raise AgentIntrospectionError(
            code="invalid_discriminator_annotation",
            message=f"Expected Literal discriminator annotation, received {annotation!r}.",
        )

    args = get_args(annotation)
    if not args:
        raise AgentIntrospectionError(
            code="missing_discriminator_value",
            message=f"Literal discriminator annotation {annotation!r} does not contain any values.",
        )

    value = args[0]
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _format_signature(method_name: str, signature: inspect.Signature) -> str:
    rendered_params: list[str] = []
    seen_keyword_only = False
    has_var_positional = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values())

    for param in signature.parameters.values():
        if param.name in {"self", "cls"}:
            continue

        if param.kind == inspect.Parameter.KEYWORD_ONLY and not seen_keyword_only and not has_var_positional:
            seen_keyword_only = True
            rendered_params.append("*")

        annotation = _format_annotation(param.annotation)
        default = ""
        if param.default is not inspect.Parameter.empty:
            default = f" = {param.default!r}"

        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            part = f"*{param.name}"
            if annotation:
                part += f": {annotation}"
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            part = f"**{param.name}"
            if annotation:
                part += f": {annotation}"
        else:
            part = param.name
            if annotation:
                part += f": {annotation}"
            part += default

        rendered_params.append(part)

    rendered = f"{method_name}({', '.join(rendered_params)})"
    return_annotation = _format_annotation(signature.return_annotation)
    if return_annotation:
        rendered += f" -> {return_annotation}"
    return rendered


def _format_annotation(annotation: Any) -> str | None:
    if annotation is inspect.Signature.empty:
        return None
    if isinstance(annotation, str):
        value = annotation
    elif hasattr(annotation, "__name__"):
        value = annotation.__name__
    else:
        value = str(annotation)
    return value.replace("typing.", "").replace("typing_extensions.", "")


def _get_first_docstring_line(docstring: str | None) -> str | None:
    if not docstring:
        return None
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None
