# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing utilities for custom column generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_designer.config.column_configs import CustomColumnConfig

logger = logging.getLogger(__name__)


class CustomColumnContext:
    """Facade providing access to resources for custom column generation.

    This context is passed to user-defined generator functions, providing
    access to LLM models and custom parameters without exposing internal
    implementation details.

    Attributes:
        kwargs: Custom parameters passed via the CustomColumnConfig.
        column_name: The name of the column being generated.
    """

    def __init__(
        self,
        resource_provider: Any,
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
    def model_registry(self) -> Any:
        """Access to the model registry for advanced use cases."""
        return self._resource_provider.model_registry

    def get_model(self, model_alias: str) -> Any:
        """Get a model facade for direct access.

        Args:
            model_alias: The alias of the model to use.

        Returns:
            A ModelFacade for generating text with full control over parameters.
        """
        return self._resource_provider.model_registry.get_model(model_alias=model_alias)

    def generate_text(
        self,
        model_alias: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using an LLM model.

        This is a convenience method for simple text generation. For more control
        over generation parameters, use get_model() and call generate() directly.

        Args:
            model_alias: The alias of the model to use (e.g., "openai-text").
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
        max_workers: int = 8,
    ) -> list[str]:
        """Generate text for multiple prompts in parallel.

        Use this method in full_column strategy to parallelize LLM calls across rows.

        Args:
            model_alias: The alias of the model to use.
            prompts: List of prompts to send to the model.
            system_prompt: Optional system prompt to set model behavior.
            max_workers: Maximum number of parallel requests (default: 8).

        Returns:
            List of generated texts in the same order as the input prompts.
        """
        from concurrent.futures import ThreadPoolExecutor

        model = self.get_model(model_alias)

        def generate_single(prompt: str) -> str:
            response, _ = model.generate(
                prompt=prompt,
                parser=lambda x: x,
                system_prompt=system_prompt,
                max_correction_steps=0,
                max_conversation_restarts=0,
            )
            return response

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(generate_single, prompts))

        return results
