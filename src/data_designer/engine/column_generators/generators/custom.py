# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from data_designer.config.columns import CustomColumnConfig
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer.engine.errors import DataDesignerRuntimeError

logger = logging.getLogger(__name__)


class CustomColumnGenerator(ColumnGenerator[CustomColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="custom",
            description="Generate a custom column.",
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            required_resources=None,
        )

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"🛠️ Generating custom column {self.config.name!r} with {len(data)} records")
        logger.info(f"  |-- generator function: {self.config.generator_function.__name__}")

        original_columns = set(data.columns)
        try:
            result = self.config.generator_function(data)

            # Check if custom column is the only one that was added
            diff_columns = set(result.columns) - original_columns
            if len(diff_columns) == 0:
                raise DataDesignerRuntimeError(
                    f"Custom column generator {self.config.generator_function.__name__} added no columns. "
                    f"Expected column {self.config.name!r} to be added by this generator."
                )
            elif diff_columns != {self.config.name}:
                diff_columns_str = ", ".join(diff_columns - {self.config.name})
                raise DataDesignerRuntimeError(
                    f"Custom column generator {self.config.generator_function.__name__} added unexpected columns: {diff_columns_str}. "
                    f"Expected only column {self.config.name!r} to be added by this generator."
                )
        except Exception as e:
            raise DataDesignerRuntimeError(f"Error generating custom column {self.config.name!r}: {e}")

        return result
