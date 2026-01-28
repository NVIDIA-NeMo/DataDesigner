# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.custom_column import CustomColumnContext
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class CustomColumnGenerator(ColumnGeneratorFullColumn[CustomColumnConfig]):
    """Column generator that uses a user-provided callable function.

    The function signature can be either:
        - fn(df: pd.DataFrame) -> pd.DataFrame
        - fn(df: pd.DataFrame, ctx: CustomColumnContext) -> pd.DataFrame

    The context provides access to kwargs, column_name, generate_text(), generate_text_batch(),
    and get_model() for LLM integration.
    """

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"🔧 Generating custom column `{self.config.name}`")

        missing_columns = list(set(self.config.required_columns) - set(data.columns))
        if len(missing_columns) > 0:
            raise CustomColumnGenerationError(
                f"Missing required columns for custom generator '{self.config.name}': {missing_columns}"
            )

        columns_before = set(data.columns)

        try:
            result = self._invoke_generate_fn(data)
        except Exception as e:
            logger.error(f"Custom column generator failed for '{self.config.name}': {e}")
            raise CustomColumnGenerationError(
                f"Custom generator function failed for column '{self.config.name}': {e}"
            ) from e

        if not isinstance(result, pd.DataFrame):
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' must return a pandas DataFrame, "
                f"got {type(result).__name__}"
            )

        result = self._validate_output_columns(result, columns_before)

        return result

    def _validate_output_columns(self, result: pd.DataFrame, columns_before: set[str]) -> pd.DataFrame:
        """Validate expected columns were created and remove undeclared columns."""
        expected_new_columns = {self.config.name} | set(self.config.output_columns)

        if self.config.name not in result.columns:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create the expected column. "
                f"The generate_fn must add a column named '{self.config.name}' to the DataFrame."
            )

        missing_output_columns = set(self.config.output_columns) - set(result.columns)
        if missing_output_columns:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create declared output columns: "
                f"{sorted(missing_output_columns)}. Declared output_columns must be added to the DataFrame."
            )

        actual_new_columns = set(result.columns) - columns_before
        undeclared_columns = actual_new_columns - expected_new_columns

        if undeclared_columns:
            logger.warning(
                f"⚠️ Custom generator for column '{self.config.name}' created undeclared columns: "
                f"{sorted(undeclared_columns)}. These columns will be removed. "
                f"To keep additional columns, declare them in 'output_columns'."
            )
            result = result.drop(columns=list(undeclared_columns))

        return result

    def _invoke_generate_fn(self, data: pd.DataFrame) -> pd.DataFrame:
        """Invoke generate function, passing context if function accepts two parameters."""
        sig = inspect.signature(self.config.generate_fn)
        params = list(sig.parameters.values())

        positional_params = [p for p in params if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )]

        if len(positional_params) >= 2:
            ctx = CustomColumnContext(
                resource_provider=self.resource_provider,
                config=self.config,
                batch_generator_fn=self._generate_text_batch,
            )
            return self.config.generate_fn(data, ctx)
        return self.config.generate_fn(data)

    def _generate_text_batch(
        self,
        model_alias: str,
        prompts: list[str],
        system_prompt: str | None,
        max_workers: int | None,
    ) -> list[str]:
        """Generate text for multiple prompts in parallel."""
        if not prompts:
            return []

        if max_workers is None:
            model_config = self.resource_provider.model_registry.get_model_config(model_alias=model_alias)
            max_workers = getattr(model_config.inference_parameters, "max_parallel_requests", 4)

        model = self.resource_provider.model_registry.get_model(model_alias=model_alias)
        logger.debug(f"Model ready: {model.model_name}")

        if len(prompts) == 1:
            logger.info("🚀 Generating 1 text")
            try:
                response, _ = model.generate(
                    prompt=prompts[0],
                    parser=lambda x: x,
                    system_prompt=system_prompt,
                    max_correction_steps=0,
                    max_conversation_restarts=0,
                )
                return [response]
            except Exception as e:
                logger.error(f"Failed to generate text: {e}")
                return [f"[ERROR: {e}]"]

        logger.info(f"🚀 Generating {len(prompts)} texts in parallel with {max_workers} workers")

        results: list[str | None] = [None] * len(prompts)

        def result_callback(result: tuple[int, str], *, context: dict | None = None) -> None:
            index, response = result
            results[index] = response

        def error_callback(exc: Exception, *, context: dict | None = None) -> None:
            if context is not None:
                idx = context.get("index", 0)
                logger.error(f"Failed to generate text for prompt {idx}: {exc}")
                results[idx] = f"[ERROR: {exc}]"

        def generate_single(prompt: str, index: int) -> tuple[int, str]:
            response, _ = model.generate(
                prompt=prompt,
                parser=lambda x: x,
                system_prompt=system_prompt,
                max_correction_steps=0,
                max_conversation_restarts=0,
            )
            return index, response

        run_config = self.resource_provider.run_config
        with ConcurrentThreadExecutor(
            max_workers=max_workers,
            column_name=self.config.name,
            result_callback=result_callback,
            error_callback=error_callback,
            shutdown_error_rate=run_config.shutdown_error_rate,
            shutdown_error_window=run_config.shutdown_error_window,
            disable_early_shutdown=run_config.disable_early_shutdown,
        ) as executor:
            for i, prompt in enumerate(prompts):
                executor.submit(generate_single, prompt, i, context={"index": i})

        return [r if r is not None else "[ERROR: Unknown]" for r in results]

    def log_pre_generation(self) -> None:
        logger.info(
            f"{self.config.get_column_emoji()} Custom column config for column '{self.config.name}'"
        )
        logger.info(f"  |-- generate_fn: {self.config.generate_fn.__name__!r}")
        logger.info(f"  |-- input_columns: {self.config.input_columns}")
        if self.config.output_columns:
            logger.info(f"  |-- output_columns: {self.config.output_columns}")
        if self.config.kwargs:
            logger.info(f"  |-- kwargs: {self.config.kwargs}")
