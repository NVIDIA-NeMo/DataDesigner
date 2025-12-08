# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import Field

from data_designer.config.analysis.column_profilers import ColumnProfilerConfigT
from data_designer.config.base import ExportableConfigBase
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.config.processors import ProcessorConfig
from data_designer.config.sampler_constraints import ColumnConstraintT
from data_designer.config.seed import SeedConfig


class DataDesignerConfig(ExportableConfigBase):
    """Configuration for NeMo Data Designer.

    This class defines the main configuration structure for NeMo Data Designer,
    which orchestrates the generation of synthetic data.

    Attributes:
        columns: Required list of column configurations defining how each column
            should be generated. Must contain at least one column.
        model_configs: Optional list of model configurations for LLM-based generation.
            Each model config defines the model, provider, and inference parameters.
        seed_config: Optional seed dataset settings to use for generation.
        constraints: Optional list of column constraints.
        profilers: Optional list of column profilers for analyzing generated data characteristics.
    """

    columns: list[Annotated[ColumnConfigT, Field(discriminator="column_type")]] = Field(min_length=1)
    model_configs: Optional[list[ModelConfig]] = None
    seed_config: Optional[SeedConfig] = None
    constraints: Optional[list[ColumnConstraintT]] = None
    profilers: Optional[list[ColumnProfilerConfigT]] = None
    processors: Optional[list[ProcessorConfig]] = None
