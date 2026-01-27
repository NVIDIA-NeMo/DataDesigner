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

    This generator provides a flexible way for users to implement custom column
    generation logic without creating a full plugin. The user's generate function
    receives a pandas DataFrame and must return a DataFrame with the new column(s) added.

    The function signature can be either:
        - `fn(df: pd.DataFrame) -> pd.DataFrame` - Simple case, just receives the DataFrame
        - `fn(df: pd.DataFrame, ctx: CustomColumnContext) -> pd.DataFrame` - Advanced case,
          receives a context object for accessing resources like LLM models

    When the context is passed, you have access to:
        - `ctx.kwargs`: Custom parameters from the config
        - `ctx.column_name`: The name of the column being generated
        - `ctx.generate_text(model_alias, prompt)`: Easy text generation with LLMs
        - `ctx.generate_text_batch(model_alias, prompts)`: Parallel batch text generation
        - `ctx.get_model(model_alias)`: Direct model access for advanced use

    Example - Simple usage:
        ```python
        def my_custom_generator(df: pd.DataFrame) -> pd.DataFrame:
            df["result"] = df["input_value"] * 2
            return df

        config = CustomColumnConfig(
            name="result",
            generate_fn=my_custom_generator,
            input_columns=["input_value"],
        )
        ```

    Example - With LLM access:
        ```python
        def generate_with_llm(df: pd.DataFrame, ctx: CustomColumnContext) -> pd.DataFrame:
            prompts = [f"Summarize: {row['text']}" for _, row in df.iterrows()]
            results = ctx.generate_text_batch(
                model_alias="nvidia-text",
                prompts=prompts,
            )
            df["summary"] = results
            return df

        config = CustomColumnConfig(
            name="summary",
            generate_fn=generate_with_llm,
            input_columns=["text"],
        )
        ```
    """

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"🔧 Generating custom column `{self.config.name}`")

        missing_columns = list(set(self.config.required_columns) - set(data.columns))
        if len(missing_columns) > 0:
            raise CustomColumnGenerationError(
                f"Missing required columns for custom generator '{self.config.name}': {missing_columns}"
            )

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

        if self.config.name not in result.columns:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create the expected column. "
                f"The generate_fn must add a column named '{self.config.name}' to the DataFrame."
            )

        return result

    def _invoke_generate_fn(self, data: pd.DataFrame) -> pd.DataFrame:
        """Invoke the generate function, passing a context if the function accepts it.

        If the function signature accepts two parameters (df, ctx), pass a CustomColumnContext.
        Otherwise, only pass the DataFrame.
        """
        sig = inspect.signature(self.config.generate_fn)
        params = list(sig.parameters.values())

        # Filter out *args and **kwargs
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
        """Generate text for multiple prompts in parallel using ConcurrentThreadExecutor."""
        if not prompts:
            return []

        # Determine max workers from model config if not specified
        if max_workers is None:
            model_config = self.resource_provider.model_registry.get_model_config(model_alias=model_alias)
            max_workers = getattr(model_config.inference_parameters, "max_parallel_requests", 4)

        # Get the model on the main thread before starting the thread pool
        model = self.resource_provider.model_registry.get_model(model_alias=model_alias)
        logger.debug(f"Model ready: {model.model_name}")

        # For a single prompt, run sequentially to avoid thread overhead
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

        # Replace any None values (shouldn't happen unless executor failed silently)
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
