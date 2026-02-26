# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# IMPORTANT: This module must NOT import from any data_designer submodules (i.e., data_designer.*).
# These base abstractions are foundational and should only depend on pydantic and Python builtins.

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        json_schema_mode_override="validation",
    )


class SingleColumnConfig(ConfigBase, ABC):
    """Abstract base class for all single-column configuration types.

    This class serves as the foundation for all column configurations in DataDesigner,
    defining shared fields and properties across all column types.

    Attributes:
        name: Unique name of the column to be generated.
        drop: If True, the column will be generated but removed from the final dataset.
            Useful for intermediate columns that are dependencies for other columns.
        allow_resize: If True, the column is allowed to be resized during generation.
        column_type: Discriminator field that identifies the specific column type.
            Subclasses must override this field to specify the column type with a `Literal` value.
    """

    name: str = Field(description="Unique name of the column to be generated")
    drop: bool = Field(
        default=False, description="If True, the column will be generated but removed from the final dataset"
    )
    allow_resize: bool = Field(
        default=False, description="If True, the column is allowed to be resized during generation"
    )
    column_type: str = Field(description="Discriminator field that identifies the specific column type")

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
