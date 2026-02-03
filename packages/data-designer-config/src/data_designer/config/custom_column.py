# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing utilities for custom column generation."""

from __future__ import annotations

import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from data_designer.config.config_builder import DataDesignerConfigBuilder

if TYPE_CHECKING:
    from data_designer.config.interface import DataDesignerInterface

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def custom_column_generator(
    required_columns: list[str] | None = None,
    side_effect_columns: list[str] | None = None,
    model_aliases: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to define metadata for a custom column generator function.

    Args:
        required_columns: Columns that must exist before this column runs (DAG ordering).
        side_effect_columns: Additional columns the function will create.
        model_aliases: Model aliases used (enables health checks).
    """

    def decorator(fn: F) -> F:
        fn._custom_column_metadata = {  # type: ignore[attr-defined]
            "required_columns": required_columns or [],
            "side_effect_columns": side_effect_columns or [],
            "model_aliases": model_aliases or [],
        }

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        # Copy metadata to wrapper
        wrapper._custom_column_metadata = fn._custom_column_metadata  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


class CustomColumnContext:
    """Facade providing access to LLM models for custom column generation.

    Created automatically by the engine, or manually via `from_data_designer()` for development.
    """

    def __init__(
        self,
        resource_provider: Any,
        column_name: str = "custom_column",
    ):
        self._resource_provider = resource_provider
        self._column_name = column_name

    @classmethod
    def from_data_designer(
        cls,
        data_designer: DataDesignerInterface,
        column_name: str = "dev_column",
    ) -> CustomColumnContext:
        """Create a context from a DataDesigner instance for development/testing."""
        config_builder = DataDesignerConfigBuilder()
        # Note: uses internal API for dev/testing convenience
        resource_provider = data_designer._create_resource_provider("dev", config_builder)

        return cls(
            resource_provider=resource_provider,
            column_name=column_name,
        )

    @property
    def column_name(self) -> str:
        """The name of the column being generated."""
        return self._column_name

    @property
    def model_registry(self) -> Any:
        """Access to the model registry for advanced use cases."""
        return self._resource_provider.model_registry

    def get_model(self, model_alias: str) -> Any:
        """Get a ModelFacade for direct model access."""
        return self._resource_provider.model_registry.get_model(model_alias=model_alias)

    def generate_text(
        self,
        model_alias: str,
        prompt: str,
        system_prompt: str | None = None,
        return_trace: bool = False,
    ) -> str | tuple[str, list[Any]]:
        """Generate text using an LLM model.

        Returns the generated text, or (text, trace) tuple if return_trace=True.
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
        """Generate text for multiple prompts in parallel. Use in full_column strategy."""

        def generate_single(prompt: str) -> str | tuple[str, list[Any]]:
            return self.generate_text(model_alias, prompt, system_prompt, return_trace)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(generate_single, prompts))
