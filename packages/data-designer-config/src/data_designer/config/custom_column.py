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
    """Facade providing access to resources for custom column generation.

    Attributes:
        kwargs: Custom parameters passed via the CustomColumnConfig.
        column_name: The name of the column being generated.
    """

    def __init__(
        self,
        resource_provider: Any,
        config: CustomColumnConfig,
        batch_generator_fn: BatchGeneratorFn | None = None,
    ):
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
        """Access to the model registry for advanced use cases."""
        return self._resource_provider.model_registry

    def get_model(self, model_alias: str) -> Any:
        """Get a model facade for direct access."""
        return self._resource_provider.model_registry.get_model(model_alias=model_alias)

    def generate_text(
        self,
        model_alias: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using an LLM model.

        Args:
            model_alias: The alias of the model to use.
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt to set model behavior.

        Returns:
            The generated text as a string.
        """
        model = self.get_model(model_alias)
        response, _ = model.generate(
            prompt=prompt,
            parser=lambda x: x,
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

        Args:
            model_alias: The alias of the model to use.
            prompts: List of prompts to send to the model.
            system_prompt: Optional system prompt (shared across all prompts).
            max_workers: Maximum parallel workers. Defaults to model's max_parallel_requests.

        Returns:
            List of generated texts, in the same order as the input prompts.
        """
        if not prompts:
            return []

        if self._batch_generator_fn is None:
            raise RuntimeError(
                "Batch generation requires a batch_generator_fn. "
                "This is typically injected by the CustomColumnGenerator."
            )

        return self._batch_generator_fn(model_alias, prompts, system_prompt, max_workers)
