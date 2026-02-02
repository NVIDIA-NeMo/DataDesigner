# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom column generator using user-provided callable functions."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from data_designer.config.column_configs import CustomColumnConfig, GenerationStrategy
from data_designer.config.custom_column import CustomColumnContext
from data_designer.engine.column_generators.generators.base import ColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class CustomColumnGenerator(ColumnGenerator[CustomColumnConfig]):
    """Column generator that uses a user-provided callable function.

    Supports two strategies based on config.strategy:
        - cell_by_cell: Processes rows one at a time (dict -> dict), parallelized by framework.
        - full_column: Processes entire batch (DataFrame -> DataFrame) for vectorized ops.

    The context provides access to kwargs, column_name, generate_text(),
    and get_model() for LLM integration.
    """

    def get_generation_strategy(self) -> GenerationStrategy:
        """Return strategy based on config."""
        if self.config.generation_strategy == GenerationStrategy.FULL_COLUMN:
            return GenerationStrategy.FULL_COLUMN
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict | pd.DataFrame) -> dict | pd.DataFrame:
        """Generate column value(s).

        Args:
            data: A dict (cell_by_cell) or DataFrame (full_column).

        Returns:
            The data with new column(s) added.
        """
        if self.config.generation_strategy == GenerationStrategy.FULL_COLUMN:
            return self._generate_full_column(data)
        return self._generate_cell_by_cell(data)

    def _generate_cell_by_cell(self, data: dict) -> dict:
        """Generate column for a single row."""
        missing_columns = set(self.config.required_columns) - set(data.keys())
        if missing_columns:
            raise CustomColumnGenerationError(
                f"Missing required columns for custom generator '{self.config.name}': {sorted(missing_columns)}"
            )

        keys_before = set(data.keys())

        try:
            result = self._invoke_generation_function(data)
        except Exception as e:
            logger.error(f"Custom column generator failed for '{self.config.name}': {e}")
            raise CustomColumnGenerationError(
                f"Custom generator function failed for column '{self.config.name}': {e}"
            ) from e

        if not isinstance(result, dict):
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' must return a dict, got {type(result).__name__}"
            )

        result = self._validate_output_columns(result, keys_before)
        return result

    def _generate_full_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate column for entire batch."""
        from data_designer.lazy_heavy_imports import pd

        missing_columns = set(self.config.required_columns) - set(data.columns)
        if missing_columns:
            raise CustomColumnGenerationError(
                f"Missing required columns for custom generator '{self.config.name}': {sorted(missing_columns)}"
            )

        columns_before = set(data.columns)

        try:
            result = self._invoke_generation_function(data)
        except Exception as e:
            logger.error(f"Custom column generator failed for '{self.config.name}': {e}")
            raise CustomColumnGenerationError(
                f"Custom generator function failed for column '{self.config.name}': {e}"
            ) from e

        if not isinstance(result, pd.DataFrame):
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' must return a DataFrame, got {type(result).__name__}"
            )

        result = self._validate_output_columns_df(result, columns_before)
        return result

    def _validate_output_columns(self, result: dict, keys_before: set[str]) -> dict:
        """Validate expected columns were created, no pre-existing columns removed, and remove undeclared columns."""
        expected_new_keys = {self.config.name} | set(self.config.side_effect_columns)

        if self.config.name not in result:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create the expected column. "
                f"The generation_function must add a key named '{self.config.name}' to the row dict."
            )

        missing_output_columns = set(self.config.side_effect_columns) - set(result.keys())
        if missing_output_columns:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create declared output_columns: "
                f"{sorted(missing_output_columns)}. Declared output_columns must be added to the row."
            )

        removed_keys = keys_before - set(result.keys())
        if removed_keys:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' removed pre-existing columns: "
                f"{sorted(removed_keys)}. The generation_function must not remove any existing columns from the row."
            )

        actual_new_keys = set(result.keys()) - keys_before
        undeclared_keys = actual_new_keys - expected_new_keys

        if undeclared_keys:
            logger.warning(
                f"⚠️ Custom generator for column '{self.config.name}' created undeclared columns: "
                f"{sorted(undeclared_keys)}. These columns will be removed. "
                f"To keep additional columns, declare them in 'output_columns'."
            )
            for key in undeclared_keys:
                del result[key]

        return result

    def _validate_output_columns_df(self, result: pd.DataFrame, columns_before: set[str]) -> pd.DataFrame:
        """Validate expected columns were created, no pre-existing columns removed, and remove undeclared columns."""
        expected_new_cols = {self.config.name} | set(self.config.side_effect_columns)

        if self.config.name not in result.columns:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create the expected column. "
                f"The generation_function must add a column named '{self.config.name}' to the DataFrame."
            )

        missing_output_columns = set(self.config.side_effect_columns) - set(result.columns)
        if missing_output_columns:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create declared output_columns: "
                f"{sorted(missing_output_columns)}. Declared output_columns must be added to the DataFrame."
            )

        removed_cols = columns_before - set(result.columns)
        if removed_cols:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' removed pre-existing columns: "
                f"{sorted(removed_cols)}. The generation_function must not remove any existing columns from the DataFrame."
            )

        actual_new_cols = set(result.columns) - columns_before
        undeclared_cols = actual_new_cols - expected_new_cols

        if undeclared_cols:
            logger.warning(
                f"⚠️ Custom generator for column '{self.config.name}' created undeclared columns: "
                f"{sorted(undeclared_cols)}. These columns will be removed. "
                f"To keep additional columns, declare them in 'output_columns'."
            )
            result = result.drop(columns=list(undeclared_cols))

        return result

    def _invoke_generation_function(self, data: dict | pd.DataFrame) -> dict | pd.DataFrame:
        """Invoke the user's generate function with appropriate arguments."""
        if self._function_accepts_context():
            ctx = CustomColumnContext(
                resource_provider=self.resource_provider,
                config=self.config,
            )
            return self.config.generation_function(data, ctx)
        return self.config.generation_function(data)

    def _function_accepts_context(self) -> bool:
        """Check if the user's generation_function accepts a context parameter (2+ args)."""
        sig = inspect.signature(self.config.generation_function)
        positional_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional_params) >= 2

    def log_pre_generation(self) -> None:
        logger.info(f"{self.config.get_column_emoji()} Custom column config for column '{self.config.name}'")
        logger.info(f"  |-- generation_function: {self.config.generation_function.__name__!r}")
        logger.info(f"  |-- generation_strategy: {self.config.generation_strategy!r}")
        logger.info(f"  |-- required_columns: {self.config.required_columns}")
        if self.config.side_effect_columns:
            logger.info(f"  |-- side_effect_columns: {self.config.side_effect_columns}")
        if self.config.model_aliases:
            logger.info(f"  |-- model_aliases: {self.config.model_aliases}")
        if self.config.kwargs:
            logger.info(f"  |-- kwargs: {self.config.kwargs}")
