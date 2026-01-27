# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing utilities for custom column generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from data_designer.config.column_configs import CustomColumnConfig

logger = logging.getLogger(__name__)

# Type alias for the batch generation function injected by the engine
BatchGeneratorFn = Callable[
    [str, list[str], str | None, int | None],  # model_alias, prompts, system_prompt, max_workers
    list[str],  # results
]


class CustomColumnContext:
    """A facade providing easy access to resources for custom column generation.

    This class provides a clean API for custom column generators to access
    resources like LLM models without needing to understand the internal
    structure of the generator.

    Attributes:
        kwargs: Custom parameters passed via the CustomColumnConfig.
        column_name: The name of the column being generated.

    Example:
        ```python
        from data_designer.essentials import CustomColumnConfig, CustomColumnContext

        def my_generator(df: pd.DataFrame, ctx: CustomColumnContext) -> pd.DataFrame:
            # Access custom parameters
            temperature = ctx.kwargs.get("temperature", 0.7)

            # Generate text using an LLM
            for idx, row in df.iterrows():
                result = ctx.generate_text(
                    model_alias="nvidia-text",
                    prompt=f"Summarize: {row['text']}",
                )
                df.at[idx, "summary"] = result

            return df
        ```
    """

    def __init__(
        self,
        resource_provider: Any,
        config: CustomColumnConfig,
        batch_generator_fn: BatchGeneratorFn | None = None,
    ):
        """Initialize the context.

        Args:
            resource_provider: The resource provider for accessing models and other resources.
            config: The CustomColumnConfig for this column.
            batch_generator_fn: Optional function for parallel batch generation (injected by generator).
        """
        self._resource_provider = resource_provider
        self._config = config
        self._batch_generator_fn = batch_generator_fn

    @property
    def kwargs(self) -> dict[str, Any]:
        """Custom parameters passed via the CustomColumnConfig."""
        return self._config.kwargs

    @property
    def column_name(self) -> str:
        """The name of the column being generated."""
        return self._config.name

    @property
    def model_registry(self) -> Any:
        """Access to the model registry for advanced use cases.

        Returns:
            The ModelRegistry instance for accessing model configurations.
        """
        return self._resource_provider.model_registry

    def get_model(self, model_alias: str) -> Any:
        """Get a model facade for direct access.

        Args:
            model_alias: The alias of the model to retrieve.

        Returns:
            The ModelFacade for the specified model.
        """
        return self._resource_provider.model_registry.get_model(model_alias=model_alias)

    def generate_text(
        self,
        model_alias: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using an LLM model.

        This is a convenience method that handles the common case of generating
        simple text output from an LLM.

        Args:
            model_alias: The alias of the model to use (e.g., "nvidia-text").
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt to set model behavior.

        Returns:
            The generated text as a string.

        Example:
            ```python
            result = ctx.generate_text(
                model_alias="nvidia-text",
                prompt="Write a haiku about coding",
                system_prompt="You are a creative poet.",
            )
            ```
        """
        model = self.get_model(model_alias)
        response, _ = model.generate(
            prompt=prompt,
            parser=lambda x: x,  # Return raw text
            system_prompt=system_prompt,
            max_correction_steps=0,
            max_conversation_restarts=0,
        )
        return response

    def generate_text_batch(
        self,
        model_alias: str,
        prompts: list[str],
        system_prompt: str | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        """Generate text for multiple prompts in parallel.

        This method parallelizes LLM calls across multiple prompts, significantly
        improving performance when generating text for many rows.

        Args:
            model_alias: The alias of the model to use (e.g., "nvidia-text").
            prompts: List of prompts to send to the model.
            system_prompt: Optional system prompt to set model behavior (shared across all prompts).
            max_workers: Maximum number of parallel workers. Defaults to the model's
                max_parallel_requests setting (typically 4-10).

        Returns:
            List of generated texts, in the same order as the input prompts.

        Example:
            ```python
            def my_generator(df: pd.DataFrame, ctx: CustomColumnContext) -> pd.DataFrame:
                # Build prompts for each row
                prompts = [
                    f"Write a greeting for {row['name']}"
                    for _, row in df.iterrows()
                ]

                # Generate all at once in parallel
                results = ctx.generate_text_batch(
                    model_alias="nvidia-text",
                    prompts=prompts,
                    system_prompt="You are friendly.",
                    max_workers=4,
                )

                df["greeting"] = results
                return df
            ```
        """
        if not prompts:
            return []

        if self._batch_generator_fn is None:
            raise RuntimeError(
                "Batch generation requires a batch_generator_fn. "
                "This is typically injected by the CustomColumnGenerator."
            )

        return self._batch_generator_fn(model_alias, prompts, system_prompt, max_workers)
