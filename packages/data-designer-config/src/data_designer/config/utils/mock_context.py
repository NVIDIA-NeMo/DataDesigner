# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock context for developing custom column generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


class MockCustomColumnContext:
    """A mock context for developing and iterating on custom column generators.

    This class provides a lightweight way to test your generator functions
    during development without needing the full DataDesigner framework.

    Example:
        ```python
        from pydantic import BaseModel
        from data_designer.config.utils import MockCustomColumnContext

        class MyConfig(BaseModel):
            tone: str = "friendly"

        def my_generator(row: dict, ctx) -> dict:
            config = ctx.generator_config
            row["greeting"] = f"Hello in a {config.tone} way!"
            return row

        # Create a mock context for development
        ctx = MockCustomColumnContext(
            column_name="greeting",
            generator_config=MyConfig(tone="professional"),
            mock_responses=["This is a mock LLM response"],
        )

        # Iterate on your generator
        result = my_generator({"name": "Alice"}, ctx)
        print(result)
        ```

    Attributes:
        column_name: The name of the column being generated.
        generator_config: Your typed configuration object.
    """

    def __init__(
        self,
        column_name: str = "mock_column",
        generator_config: BaseModel | None = None,
        mock_responses: list[str] | None = None,
    ):
        """Create a mock context for development.

        Args:
            column_name: The name of the column being generated.
            generator_config: Your typed configuration object (Pydantic BaseModel).
            mock_responses: List of mock responses for generate_text() calls.
                Responses are returned in order, cycling if more calls are made.
        """
        self._column_name = column_name
        self._generator_config = generator_config
        self._mock_responses = mock_responses or ["[mock LLM response]"]
        self._response_index = 0
        self._call_history: list[dict[str, Any]] = []

    @property
    def column_name(self) -> str:
        """The name of the column being generated."""
        return self._column_name

    @property
    def generator_config(self) -> BaseModel | None:
        """Your typed configuration object."""
        return self._generator_config

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """History of all generate_text() calls made during development."""
        return self._call_history

    def generate_text(
        self,
        model_alias: str,
        prompt: str,
        system_prompt: str | None = None,
        return_trace: bool = False,
    ) -> str | tuple[str, list[Any]]:
        """Mock LLM text generation.

        Returns mock responses in order. Useful for testing your generator
        logic without making actual LLM calls.

        Args:
            model_alias: The model alias (logged but not used).
            prompt: The prompt (logged for inspection).
            system_prompt: Optional system prompt (logged).
            return_trace: If True, returns (response, []) tuple.

        Returns:
            The next mock response, or (response, []) if return_trace=True.
        """
        call_info = {
            "model_alias": model_alias,
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
        self._call_history.append(call_info)

        response = self._mock_responses[self._response_index % len(self._mock_responses)]
        self._response_index += 1

        if return_trace:
            return response, []
        return response

    def generate_text_batch(
        self,
        model_alias: str,
        prompts: list[str],
        system_prompt: str | None = None,
        max_workers: int = 8,
        return_trace: bool = False,
    ) -> list[str] | list[tuple[str, list[Any]]]:
        """Mock batch LLM text generation.

        Args:
            model_alias: The model alias (logged but not used).
            prompts: List of prompts.
            system_prompt: Optional system prompt.
            max_workers: Ignored in mock.
            return_trace: If True, returns list of (response, []) tuples.

        Returns:
            List of mock responses.
        """
        results = []
        for prompt in prompts:
            result = self.generate_text(
                model_alias=model_alias,
                prompt=prompt,
                system_prompt=system_prompt,
                return_trace=return_trace,
            )
            results.append(result)
        return results

    def get_model(self, model_alias: str) -> Any:
        """Mock get_model that raises an informative error.

        For development, use generate_text() instead. If you need direct
        model access, use the full DataDesigner framework.
        """
        raise NotImplementedError(
            f"MockCustomColumnContext.get_model('{model_alias}') is not implemented. "
            "Use ctx.generate_text() for development, or use the full DataDesigner "
            "framework for direct model access."
        )

    def reset(self) -> None:
        """Reset the mock state (response index and call history)."""
        self._response_index = 0
        self._call_history.clear()
