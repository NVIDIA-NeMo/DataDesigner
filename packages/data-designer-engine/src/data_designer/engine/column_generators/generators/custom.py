# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom column generator using user-provided callable functions."""

from __future__ import annotations

import inspect
import logging

from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.custom_column import CustomColumnContext
from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError

logger = logging.getLogger(__name__)


class CustomColumnGenerator(ColumnGeneratorCellByCell[CustomColumnConfig]):
    """Column generator that uses a user-provided callable function.

    This generator processes rows one at a time, allowing the framework to
    parallelize across rows automatically. The function signature can be either:
        - fn(row: dict) -> dict
        - fn(row: dict, ctx: CustomColumnContext) -> dict

    The context provides access to kwargs, column_name, generate_text(),
    and get_model() for LLM integration.
    """

    def generate(self, data: dict) -> dict:
        """Generate column value for a single row.

        Args:
            data: A dictionary representing a single row.

        Returns:
            The row dictionary with the new column(s) added.
        """
        missing_columns = set(self.config.required_columns) - set(data.keys())
        if missing_columns:
            raise CustomColumnGenerationError(
                f"Missing required columns for custom generator '{self.config.name}': {sorted(missing_columns)}"
            )

        keys_before = set(data.keys())

        try:
            result = self._invoke_generate_fn(data)
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

    def _validate_output_columns(self, result: dict, keys_before: set[str]) -> dict:
        """Validate expected columns were created and remove undeclared columns."""
        expected_new_keys = {self.config.name} | set(self.config.output_columns)

        if self.config.name not in result:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create the expected column. "
                f"The generate_fn must add a key named '{self.config.name}' to the row dict."
            )

        missing_output_columns = set(self.config.output_columns) - set(result.keys())
        if missing_output_columns:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create declared output columns: "
                f"{sorted(missing_output_columns)}. Declared output_columns must be added to the row."
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

    def _invoke_generate_fn(self, data: dict) -> dict:
        """Invoke the user's generate function with appropriate arguments.

        Supports two function signatures:
        - fn(row: dict) -> dict  # Simple, no context needed
        - fn(row: dict, ctx: CustomColumnContext) -> dict  # With LLM/resource access
        """
        if self._function_accepts_context():
            ctx = CustomColumnContext(
                resource_provider=self.resource_provider,
                config=self.config,
            )
            return self.config.generate_fn(data, ctx)
        return self.config.generate_fn(data)

    def _function_accepts_context(self) -> bool:
        """Check if the user's generate_fn accepts a context parameter (2+ args)."""
        sig = inspect.signature(self.config.generate_fn)
        positional_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional_params) >= 2

    def log_pre_generation(self) -> None:
        logger.info(f"{self.config.get_column_emoji()} Custom column config for column '{self.config.name}'")
        logger.info(f"  |-- generate_fn: {self.config.generate_fn.__name__!r}")
        logger.info(f"  |-- input_columns: {self.config.input_columns}")
        if self.config.output_columns:
            logger.info(f"  |-- output_columns: {self.config.output_columns}")
        if self.config.kwargs:
            logger.info(f"  |-- kwargs: {self.config.kwargs}")
