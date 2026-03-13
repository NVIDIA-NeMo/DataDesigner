# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# IMPORTANT: This module must NOT import from any data_designer submodules (i.e., data_designer.*).
# These base abstractions are foundational and should only depend on pydantic and Python builtins.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        json_schema_mode_override="validation",
    )

    @classmethod
    def schema_text(cls) -> str:
        """Return a human-readable summary of the model's fields."""
        lines: list[str] = [f"{cls.__name__}:"]
        docstring = _get_docstring_summary(cls.__doc__)
        if docstring:
            lines.append(f"  {docstring}")
        lines.append("")
        for name, field_info in cls.model_fields.items():
            annotation = _format_annotation(field_info.annotation)
            if field_info.is_required():
                lines.append(f"  {name}: {annotation}  [required]")
            else:
                if field_info.default_factory is not None:
                    factory_name = getattr(field_info.default_factory, "__name__", repr(field_info.default_factory))
                    lines.append(f"  {name}: {annotation} = {factory_name}()")
                else:
                    default = field_info.default
                    if isinstance(default, Enum):
                        default = default.value
                    lines.append(f"  {name}: {annotation} = {default!r}")
            if field_info.description:
                lines.append(f"      {field_info.description}")
        return "\n".join(lines)


class SingleColumnConfig(ConfigBase, ABC):
    """Abstract base class for all single-column configuration types.

    This class serves as the foundation for all column configurations in DataDesigner,
    defining shared fields and properties across all column types.

    Attributes:
        name: Unique name of the column to be generated.
        drop: If True, the column will be generated but removed from the final dataset.
            Useful for intermediate columns that are dependencies for other columns.
        column_type: Discriminator field that identifies the specific column type.
            Subclasses must override this field to specify the column type with a `Literal` value.
    """

    name: str
    drop: bool = False
    allow_resize: bool = False
    column_type: str

    @staticmethod
    def get_column_emoji() -> str:
        return "🎨"

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Returns a list of column names that must exist before this column can be generated.

        Returns:
            List of column names that this column depends on. Empty list indicates
            no dependencies. Override in subclasses to specify dependencies.
        """

    @property
    @abstractmethod
    def side_effect_columns(self) -> list[str]:
        """Returns a list of additional columns that this column will create as a side effect.

        Some column types generate additional metadata or auxiliary columns alongside
        the primary column (e.g., reasoning traces for LLM columns).

        Returns:
            List of column names that this column will create as a side effect. Empty list
            indicates no side effect columns. Override in subclasses to specify side effects.
        """


class ProcessorConfig(ConfigBase, ABC):
    """Abstract base class for all processor configuration types.

    Processors are transformations that run at different stages of the generation
    pipeline. They can modify, reshape, or augment the dataset.

    Attributes:
        name: Unique name of the processor, used to identify the processor in results
            and to name output artifacts on disk.
    """

    name: str = Field(
        description="The name of the processor, used to identify the processor in the results and to write the artifacts to disk.",
    )
    processor_type: str


def _format_annotation(annotation: Any) -> str:
    """Convert a type annotation to a readable string, stripping module paths."""
    raw = str(annotation) if not hasattr(annotation, "__name__") else annotation.__name__
    return re.sub(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+", lambda m: m.group().rsplit(".", 1)[-1], raw)


def _get_docstring_summary(docstring: str | None) -> str | None:
    """Extract the first paragraph of a docstring, up to the Attributes section."""
    if not docstring:
        return None
    lines: list[str] = []
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("attributes:"):
            break
        if not stripped and lines:
            break
        if stripped:
            lines.append(stripped)
    return " ".join(lines) if lines else None
