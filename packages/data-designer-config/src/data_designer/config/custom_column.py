# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing utilities for custom column generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from data_designer.config.column_configs import CustomColumnConfig

logger = logging.getLogger(__name__)


class CustomColumnContext:
    """Facade providing access to resources for custom column generation.

    This context is passed to user-defined generator functions, providing
    access to LLM models and custom parameters without exposing internal
    implementation details.

    Attributes:
        generator_config: Typed configuration object passed via the CustomColumnConfig.
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
    def generator_config(self) -> BaseModel | None:
        """Typed configuration object passed via the CustomColumnConfig."""
        return self._config.generator_config

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
        return_trace: bool = False,
    ) -> str | tuple[str, list[Any]]:
        """Generate text using an LLM model.

        This is a convenience method for simple text generation. For more control
        over generation parameters, use get_model() and call generate() directly.

        Args:
            model_alias: The alias of the model to use (e.g., "openai-text").
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt to set model behavior.
            return_trace: If True, returns a tuple of (response, trace) where trace
                is the full conversation history including any corrections or tool calls.

        Returns:
            If return_trace is False: The generated text as a string.
            If return_trace is True: A tuple of (text, trace) where trace is
                a list of ChatMessage objects representing the conversation.
        """
        model = self.get_model(model_alias)
        response, trace = model.generate(
            prompt=prompt,
            parser=lambda x: x,
            system_prompt=system_prompt,
            max_correction_steps=0,
            max_conversation_restarts=0,
        )
        if return_trace:
            return response, trace
        return response

    def generate_text_batch(
        self,
        model_alias: str,
        prompts: list[str],
        system_prompt: str | None = None,
        max_workers: int = 8,
        return_trace: bool = False,
    ) -> list[str] | list[tuple[str, list[Any]]]:
        """Generate text for multiple prompts in parallel.

        Use this method in full_column strategy to parallelize LLM calls across rows.

        Args:
            model_alias: The alias of the model to use.
            prompts: List of prompts to send to the model.
            system_prompt: Optional system prompt to set model behavior.
            max_workers: Maximum number of parallel requests (default: 8).
            return_trace: If True, returns list of (response, trace) tuples.

        Returns:
            If return_trace is False: List of generated texts in the same order as the input prompts.
            If return_trace is True: List of (text, trace) tuples where each trace is
                a list of ChatMessage objects representing that conversation.
        """
        from concurrent.futures import ThreadPoolExecutor

        model = self.get_model(model_alias)

        def generate_single(prompt: str) -> str | tuple[str, list[Any]]:
            response, trace = model.generate(
                prompt=prompt,
                parser=lambda x: x,
                system_prompt=system_prompt,
                max_correction_steps=0,
                max_conversation_restarts=0,
            )
            if return_trace:
                return response, trace
            return response

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(generate_single, prompts))

        return results
