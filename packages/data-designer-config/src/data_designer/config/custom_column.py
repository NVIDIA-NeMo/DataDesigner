# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing utilities for custom column generation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_designer.config.column_configs import CustomColumnConfig
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.registry import ModelRegistry
    from data_designer.engine.resources.resource_provider import ResourceProvider

logger = logging.getLogger(__name__)


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
        resource_provider: ResourceProvider,
        config: CustomColumnConfig,
    ):
        self._resource_provider = resource_provider
        self._config = config

    @property
    def kwargs(self) -> dict[str, Any]:
        """Custom parameters passed via the CustomColumnConfig."""
        return self._config.kwargs

    @property
    def column_name(self) -> str:
        """The name of the column being generated."""
        return self._config.name

    @property
    def model_registry(self) -> ModelRegistry:
        """Access to the model registry for advanced use cases."""
        return self._resource_provider.model_registry

    def get_model(self, model_alias: str) -> ModelFacade:
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

        # Determine max workers from model config if not specified
        if max_workers is None:
            model_config = self._resource_provider.model_registry.get_model_config(model_alias=model_alias)
            max_workers = getattr(model_config.inference_parameters, "max_parallel_requests", 4)

        # IMPORTANT: Get the model on the main thread BEFORE starting the thread pool.
        # This ensures lazy initialization happens on the main thread, avoiding potential
        # thread-safety issues with model/router initialization.
        model = self.get_model(model_alias)

        # Warm up the model by making a test call if needed (triggers any lazy init)
        # This is a no-op if the model is already warmed up from a health check
        logger.debug(f"Model ready: {model.model_name}")

        results: list[str | None] = [None] * len(prompts)
        errors: list[Exception] = []

        # For a single prompt, run sequentially to avoid thread overhead
        if len(prompts) == 1:
            logger.info("🚀 Generating 1 text (sequential)")
            logger.info(f"  Model: {model.model_name}")
            logger.info(f"  Prompt: {prompts[0][:50]}...")
            try:
                logger.info("  Calling model.generate()...")
                response, _ = model.generate(
                    prompt=prompts[0],
                    parser=lambda x: x,
                    system_prompt=system_prompt,
                    max_correction_steps=0,
                    max_conversation_restarts=0,
                )
                logger.info(f"  Got response: {response[:50] if response else 'None'}...")
                return [response]
            except Exception as e:
                logger.error(f"Failed to generate text: {e}")
                import traceback
                traceback.print_exc()
                return [f"[ERROR: {e}]"]

        logger.info(
            f"🚀 Generating {len(prompts)} texts in parallel with {max_workers} workers"
        )

        def generate_single(index: int, prompt: str) -> tuple[int, str]:
            # Use the model reference from the outer scope (already initialized)
            logger.debug(f"  Thread starting for prompt {index}")
            response, _ = model.generate(
                prompt=prompt,
                parser=lambda x: x,
                system_prompt=system_prompt,
                max_correction_steps=0,
                max_conversation_restarts=0,
            )
            logger.debug(f"  Thread completed for prompt {index}")
            return index, response

        results: list[str | None] = [None] * len(prompts)
        errors: list[Exception] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_single, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            for future in as_completed(futures):
                try:
                    index, response = future.result()
                    results[index] = response
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Failed to generate text for prompt {idx}: {e}")
                    errors.append(e)
                    results[idx] = f"[ERROR: {e}]"

        if errors:
            logger.warning(f"Completed with {len(errors)} errors out of {len(prompts)} prompts")

        return results
