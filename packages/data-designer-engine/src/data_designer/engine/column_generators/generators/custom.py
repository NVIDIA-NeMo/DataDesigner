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
            results = []
            for _, row in df.iterrows():
                response = ctx.generate_text(
                    model_alias="nvidia-text",
                    prompt=f"Summarize: {row['text']}",
                )
                results.append(response)

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
            )
            return self.config.generate_fn(data, ctx)
        return self.config.generate_fn(data)

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
