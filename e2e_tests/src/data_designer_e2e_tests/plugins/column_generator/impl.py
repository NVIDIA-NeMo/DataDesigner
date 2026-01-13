# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer_e2e_tests.plugins.column_generator.config import DemoColumnGeneratorConfig


class DemoColumnGeneratorImpl(ColumnGenerator[DemoColumnGeneratorConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="demo-column-generator",
            description="Shouts at you",
            generation_strategy=GenerationStrategy.FULL_COLUMN,
        )

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.config.name] = self.config.text.upper()

        return data
