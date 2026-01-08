# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer.plugins.testing.e2e.column_generator_plugin.config import TestColumnGeneratorConfig


class TestColumnGeneratorImpl(ColumnGenerator[TestColumnGeneratorConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="test-column-generator",
            description="Shouts at you",
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            required_resources=None,
        )

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.config.name] = self.config.text.upper()

        return data
