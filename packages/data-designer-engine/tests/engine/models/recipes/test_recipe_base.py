# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.models.recipes.base import ResponseRecipe


class MockResponseRecipe(ResponseRecipe[str]):
    def _build_parser_fn(self):
        def parser(response: str) -> str:
            if not response:
                raise ValueError("Empty response")
            return response.strip()

        return parser

    @property
    def example_template(self) -> str:
        return "Example: {example}"

    def serialize_output(self, output: str) -> str:
        if not output:
            raise ValueError("Cannot serialize empty output")
        return output

    def deserialize_output(self, serialized_output: str) -> str:
        if not serialized_output:
            raise ValueError("Cannot deserialize empty output")
        return serialized_output.strip()


class MockRecipeWithInstructions(MockResponseRecipe):
    @property
    def task_instructions(self) -> str:
        return "Please follow these instructions."


def test_response_recipe_initialization():
    recipe = MockResponseRecipe()
    assert recipe._parse_fn is not None
    assert callable(recipe._parse_fn)
    assert recipe.task_instructions is None


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("  hello world  ", "hello world"),
        ("test", "test"),
        ("  ", ""),
    ],
)
def test_response_recipe_parse_success(input_text, expected):
    recipe = MockResponseRecipe()
    result = recipe.parse(input_text)
    assert result == expected


def test_response_recipe_parse_error():
    recipe = MockResponseRecipe()
    with pytest.raises(ValueError, match="Empty response"):
        recipe.parse("")


def test_response_recipe_generate_response_example():
    recipe = MockResponseRecipe()
    example = "test example"
    result = recipe.generate_response_example(example)
    assert result == "Example: test example"


@pytest.mark.parametrize(
    "recipe_class,user_prompt,expected",
    [
        (MockResponseRecipe, "Test prompt", "Test prompt"),
        (MockRecipeWithInstructions, "Test prompt", "Test prompt\n\nPlease follow these instructions."),
    ],
)
def test_response_recipe_apply_to_user_prompt(recipe_class, user_prompt, expected):
    recipe = recipe_class()
    result = recipe.apply_recipe_to_user_prompt(user_prompt)
    assert result == expected


@pytest.mark.parametrize(
    "system_prompt,expected",
    [
        ("System prompt", "System prompt"),
        (None, None),
    ],
)
def test_response_recipe_apply_to_system_prompt(system_prompt, expected):
    recipe = MockResponseRecipe()
    result = recipe.apply_recipe_to_system_prompt(system_prompt)
    assert result == expected


@pytest.mark.parametrize(
    "output,expected",
    [
        ("test output", "test output"),
        ("", ValueError("Cannot serialize empty output")),
    ],
)
def test_response_recipe_serialize_output(output, expected):
    recipe = MockResponseRecipe()
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            recipe.serialize_output(output)
    else:
        result = recipe.serialize_output(output)
        assert result == expected


@pytest.mark.parametrize(
    "serialized_output,expected",
    [
        ("  test output  ", "test output"),
        ("", ValueError("Cannot deserialize empty output")),
    ],
)
def test_response_recipe_deserialize_output(serialized_output, expected):
    recipe = MockResponseRecipe()
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            recipe.deserialize_output(serialized_output)
    else:
        result = recipe.deserialize_output(serialized_output)
        assert result == expected
