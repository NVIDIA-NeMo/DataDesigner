# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import FromScratchColumnGenerator
from data_designer_e2e_tests.plugins.column_generator.config import DemoColumnGeneratorConfig


class DemoColumnGeneratorImpl(FromScratchColumnGenerator[DemoColumnGeneratorConfig]):
    """Produces a constant column with the configured text uppercased.

    Modeled as a from-scratch generator because it produces values without
    needing any upstream column data. The async engine routes ``no-upstream``
    columns through the from-scratch path; declaring the strategy explicitly
    matches that contract.
    """

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        return lazy.pd.DataFrame({self.config.name: [self.config.text.upper()] * num_records})
