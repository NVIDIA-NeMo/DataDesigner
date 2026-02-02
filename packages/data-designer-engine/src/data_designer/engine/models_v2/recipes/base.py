# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class ResponseRecipe(abc.ABC, Generic[T]):
    """Base class for defining response recipes.

    Response recipes contain all necessary information for
    getting an LLM to perform a particular common task,
    like outputting code in a desired format or following
    structured output.
    """

    @abc.abstractmethod
    def _build_parser_fn(self) -> Callable[[str], T]:
        """Build the recipe's output parser function."""
        ...

    @property
    @abc.abstractmethod
    def example_template(self) -> str: ...

    @abc.abstractmethod
    def serialize_output(self, output: T) -> str:
        """Serialize an instance of the parser output."""
        ...

    @abc.abstractmethod
    def deserialize_output(self, serialized_output: str) -> T:
        """Deserialize a serialized instance of the parser output."""
        ...

    def __init__(self):
        self._parse_fn = self._build_parser_fn()

    @property
    def task_instructions(self) -> str | None:
        """Specifies task instructions.

        These instructions lay out the particular task information the
        LLM requires in order to carry out the function of the recipe.
        """
        return None

    def parse(self, response: str) -> T:
        """Apply the recipe's parser to a raw model output."""
        return self._parse_fn(response)

    def generate_response_example(self, example: T) -> str:
        """Create a serialized response example that the parser would admit."""
        return self.example_template.format(example=example)

    def apply_recipe_to_user_prompt(self, user_prompt: str) -> str:
        """Appends recipe specific task instructions if applicable.

        Args:
            user_prompt (str): User prompt to be appended with recipe specific task instructions if applicable.

        Returns:
            str: Final user prompt
        """
        return f"{user_prompt}\n\n{self.task_instructions}" if self.task_instructions else user_prompt

    def apply_recipe_to_system_prompt(self, system_prompt: str | None) -> str:
        """Appends recipe specific task instructions if applicable.

        Args:
            system_prompt (str): System prompt to be appended with recipe specific task instructions if applicable.

        Returns:
            str: Final system prompt
        """
        return system_prompt
