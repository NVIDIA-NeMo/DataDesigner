# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.cli.services.introspection.discovery import (
    discover_column_configs,
    discover_constraint_types,
    discover_mcp_types,
    discover_model_configs,
    discover_processor_configs,
    discover_sampler_types,
    discover_seed_types,
    discover_validator_types,
)


def _collect_models_with_fields() -> list[tuple[str, str, type]]:
    """Collect all discovered model classes and their fields.

    Returns:
        List of (source_label, field_name, model_class) tuples.
    """
    items: list[tuple[str, str, type]] = []

    discovery_sources: list[tuple[str, dict[str, type]]] = [
        ("column_configs", discover_column_configs()),
        ("sampler_types", discover_sampler_types()),
        ("validator_types", discover_validator_types()),
        ("processor_configs", discover_processor_configs()),
        ("model_configs", discover_model_configs()),
        ("constraint_types", discover_constraint_types()),
        ("seed_types", discover_seed_types()),
        ("mcp_types", discover_mcp_types()),
    ]

    for source_label, discovered in discovery_sources:
        for type_name, cls in discovered.items():
            if not hasattr(cls, "model_fields"):
                continue
            for field_name in cls.model_fields:
                items.append((f"{source_label}:{type_name}", field_name, cls))

    return items


_ALL_FIELDS = _collect_models_with_fields()


@pytest.mark.parametrize(
    "source_label,field_name,cls",
    _ALL_FIELDS,
    ids=[f"{src}.{field}" for src, field, _ in _ALL_FIELDS],
)
def test_all_discovered_fields_have_descriptions(source_label: str, field_name: str, cls: type) -> None:
    """Every field in discovered config models must have a non-empty description."""
    field_info = cls.model_fields[field_name]
    assert field_info.description, (
        f"{cls.__name__}.{field_name} (from {source_label}) has no Field(description=...). "
        f"Add a description to this field in the source model."
    )
