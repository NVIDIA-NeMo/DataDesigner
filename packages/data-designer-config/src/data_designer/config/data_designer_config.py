# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.analysis.column_profilers import ColumnProfilerConfigT
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.exportable_config import ExportableConfigBase
from data_designer.config.fingerprint import fingerprint_config
from data_designer.config.mcp import ToolConfig
from data_designer.config.models import ModelConfig
from data_designer.config.processor_types import ProcessorConfigT
from data_designer.config.record_selection import RecordSelectionConfig
from data_designer.config.sampler_constraints import ColumnConstraintInputT
from data_designer.config.sampler_params import SamplerType
from data_designer.config.seed import SeedConfig


class DataDesignerConfig(ExportableConfigBase):
    """Configuration for NeMo Data Designer.

    This class defines the main configuration structure for NeMo Data Designer,
    which the engine consumes when generating synthetic data.

    Attributes:
        columns: Required list of column configurations defining how each column
            should be generated. May be empty for seeded processor-only configs.
        model_configs: Optional list of model configurations for LLM-based generation.
            Each model config defines the model, provider, and inference parameters.
        tool_configs: Optional list of tool configurations for MCP tool calling.
            Each tool config defines the provider, allowed tools, and execution limits.
        seed_config: Optional seed dataset settings to use for generation.
        constraints: Optional list of column constraints.
        profilers: Optional list of column profilers for analyzing generated data characteristics.
        processors: Optional list of processor configurations for post-generation transformations.
        record_selection: Optional policy for selecting an exact target of accepted records.
    """

    columns: list[Annotated[ColumnConfigT, Field(discriminator="column_type")]]
    model_configs: list[ModelConfig] | None = None
    tool_configs: list[ToolConfig] | None = None
    seed_config: SeedConfig | None = None
    constraints: list[ColumnConstraintInputT] | None = None
    profilers: list[ColumnProfilerConfigT] | None = None
    processors: list[Annotated[ProcessorConfigT, Field(discriminator="processor_type")]] | None = None
    record_selection: RecordSelectionConfig | None = None

    @model_validator(mode="after")
    def _validate_subcategory_parents(self) -> Self:
        by_name = {c.name: c for c in self.columns}
        for col in self.columns:
            if col.column_type != "sampler" or col.sampler_type != SamplerType.SUBCATEGORY:
                continue
            parent = by_name.get(col.params.category)
            if parent is not None and (parent.column_type != "sampler" or parent.sampler_type != SamplerType.CATEGORY):
                if parent.column_type == "sampler":
                    parent_sampler_type = getattr(parent.sampler_type, "value", parent.sampler_type)
                    parent_type = f"sampler column with sampler_type='{parent_sampler_type}'"
                else:
                    parent_type = f"'{parent.column_type}' column"
                raise ValueError(
                    f"Subcategory column '{col.name}' has parent '{parent.name}', which is a {parent_type}. "
                    f"Subcategory parents must be sampler columns "
                    f"with sampler_type='category'."
                )
        return self

    def fingerprint(self) -> dict[str, str | int]:
        """Compute a deterministic content-addressable fingerprint of this config.

        See `data_designer.config.fingerprint.fingerprint_config` for the full
        list of identity-relevant and excluded fields, and how custom column
        generators are identified.

        Returns:
            A dict with `config_hash`, `config_hash_algo`, and
            `config_hash_version`.
        """
        return fingerprint_config(self)
