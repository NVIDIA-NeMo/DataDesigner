# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import (
    CodeResponseRecipe,
    PydanticResponseRecipe,
    StructuredResponseRecipe,
    TextResponseRecipe,
)

UNICODE_MATH_STRING = "Equation: ∑_{𝑛=1}^{∞} 𝑛² + ∫₀^{∞} 𝑒^{-𝑥} 𝑑𝑥 = π²/6 + 1"


### Test Default Recipe ###
def test_default_response():
    recipe = TextResponseRecipe()
    example = "hello world!"

    assert recipe.generate_response_example(example) == example

    assert recipe.parse(recipe.generate_response_example(example)) == example

    assert recipe.deserialize_output(recipe.serialize_output(example)) == example


def test_default_response_with_unicode_math():
    recipe = TextResponseRecipe()
    assert recipe.generate_response_example(UNICODE_MATH_STRING) == UNICODE_MATH_STRING
    assert recipe.parse(recipe.generate_response_example(UNICODE_MATH_STRING)) == UNICODE_MATH_STRING
    assert recipe.deserialize_output(recipe.serialize_output(UNICODE_MATH_STRING)) == UNICODE_MATH_STRING


### Test Pydantic & Structured Recipes ###
# Nested test type
class Bar(BaseModel):
    baz: int = Field(..., description="Baz!!")
    unicode_math: str = Field(default=UNICODE_MATH_STRING, description="Unicode math stirng")


class Foo(BaseModel):
    bar: Bar


@pytest.fixture(params=["pydantic", "structured"])
def json_recipe(request: pytest.FixtureRequest) -> PydanticResponseRecipe | StructuredResponseRecipe:
    if request.param == "pydantic":
        return PydanticResponseRecipe(Foo)
    return StructuredResponseRecipe(Foo.model_json_schema())


@pytest.mark.parametrize("padding", ["", " \n\t", "    "])
def test_json_recipe_accepts_bare_json(
    json_recipe: PydanticResponseRecipe | StructuredResponseRecipe, padding: str
) -> None:
    example = Foo(bar=Bar(baz=42, unicode_math=f"<b>**value**</b> &amp; `code` {UNICODE_MATH_STRING}"))
    response = padding + example.model_dump_json() + padding
    result = json_recipe.parse(response)
    assert (result.model_dump() if isinstance(result, BaseModel) else result) == example.model_dump()


@pytest.mark.parametrize("response", ["{}", '{"bar": {"baz": "hello"}}'])
def test_json_recipe_validates_bare_json(
    json_recipe: PydanticResponseRecipe | StructuredResponseRecipe, response: str
) -> None:
    with pytest.raises(ParserException, match="Response doesn't match requested <response_schema>"):
        json_recipe.parse(response)


@pytest.mark.parametrize(
    "response",
    [
        '{"bar": {"baz": 42}',
        '{"bar": {"baz": 42,}}',
        "{bar: {baz: 42}}",
        'Here is the JSON: {"bar": {"baz": 42}}',
        '{"bar": {"baz": 42}} trailing text',
        '{"bar": {"baz": 42}} {"bar": {"baz": 43}}',
    ],
)
def test_json_recipe_rejects_malformed_or_embedded_bare_json(
    json_recipe: PydanticResponseRecipe | StructuredResponseRecipe, response: str
) -> None:
    with pytest.raises(ParserException, match="No parsable JSON found"):
        json_recipe.parse(response)


def test_json_recipe_preserves_fenced_json_precedence(
    json_recipe: PydanticResponseRecipe | StructuredResponseRecipe,
) -> None:
    example = Foo(bar=Bar(baz=42))
    response = f'{example.model_dump_json()}\n\n```json\n{{"bar": {{"baz": "invalid"}}}}\n```'
    with pytest.raises(ParserException, match="Response doesn't match requested <response_schema>"):
        json_recipe.parse(response)


def test_structured_response_prunes_bare_json_extra_fields() -> None:
    recipe = StructuredResponseRecipe(Foo.model_json_schema())
    example = Foo(bar=Bar(baz=42)).model_dump()
    assert recipe.parse(json.dumps({**example, "extra": "drop me"})) == example


def test_pydantic_response():
    recipe = PydanticResponseRecipe(Foo)
    example = Foo(bar=Bar(baz=42))

    assert recipe.deserialize_output(recipe.serialize_output(example)) == example

    # Should be able to recover from its own example
    assert recipe.parse(recipe.generate_response_example(example)) == example

    # Should be able to handle extraneous text
    response = f"This is early text\n{recipe.generate_response_example(example)}\nThis is late text\n"
    assert recipe.parse(response) == example

    # Should be able to handle extraneous code blocks
    response = (
        f"```python\nimport foo\n```\n{recipe.generate_response_example(example)}\n```sql\nselect * from table\n```\n"
    )
    assert recipe.parse(response) == example

    # In the case of multiple, this is going to return the last item in the list
    response = (
        f"{recipe.generate_response_example(Foo(bar=Bar(baz=0, unicode_math='a')))}\n"
        f"{recipe.generate_response_example(example)}\n"
    )
    assert recipe.parse(response) == example

    # Should have some limited recoverability from malformed json
    response = "```json\n{bar: {baz: 42}}\n```"
    assert recipe.parse(response) == example


def test_structured_response():
    recipe = StructuredResponseRecipe(Foo.model_json_schema())
    example = Foo(bar=Bar(baz=42, unicode_math=UNICODE_MATH_STRING)).model_dump()

    assert recipe.deserialize_output(recipe.serialize_output(example)) == example

    # Should be able to recover from its own example
    assert recipe.parse(recipe.generate_response_example(example)) == example

    # Should be able to handle extraneous text
    response = f"This is early text\n{recipe.generate_response_example(example)}\nThis is late text\n"
    assert recipe.parse(response) == example

    # Should be able to handle extraneous code blocks
    response = (
        f"```python\nimport foo\n```\n{recipe.generate_response_example(example)}\n```sql\nselect * from table\n```\n"
    )
    assert recipe.parse(response) == example

    # In the case of multiple, this is going to return the last item in the list
    response = (
        f"{recipe.generate_response_example(Foo(bar=Bar(baz=0)).model_dump())}\n"
        f"{recipe.generate_response_example(example)}\n"
    )
    assert recipe.parse(response) == example

    # Should have some limited recoverability from malformed json
    example = Foo(bar=Bar(baz=42, unicode_math="a")).model_dump()
    response = "```json\n{bar: {baz: 42, unicode_math: a}}\n```"
    assert recipe.parse(response) == example


def test_structured_response_extra_fields():
    recipe = StructuredResponseRecipe(Foo.model_json_schema())
    ## Make an example with an extra field in it -- should be pruned out.
    valid_output = Foo(bar=Bar(baz=42, unicode_math=UNICODE_MATH_STRING)).model_dump()
    extra_field_response = {
        "bar": {
            "baz": 42,
            "unicode_math": UNICODE_MATH_STRING,
            "extra_field": "I don't belong here!",
        },
        "extra_field": "I don't belong here!",
    }

    # Should be able to handle extraneous text
    response = f"This is early text\n{recipe.generate_response_example(extra_field_response)}\nThis is late text\n"
    assert recipe.parse(response) == valid_output


def test_pydantic_response_task_instructions():
    recipe = PydanticResponseRecipe(Foo)

    # Ensure that the schema is being given to the instructions
    assert recipe.schema in recipe.task_instructions

    # Also ensure that the free-text descriptions are as well.
    assert "Baz!!" in recipe.task_instructions


def test_structured_response_task_instructions():
    recipe = StructuredResponseRecipe(Foo.model_json_schema())

    # Ensure that the schema is being given to the instructions
    assert recipe.schema in recipe.task_instructions

    # Also ensure that the free-text descriptions are as well.
    assert "Baz!!" in recipe.task_instructions


def test_pydantic_response_patched_case_single_lb():
    recipe = PydanticResponseRecipe(Foo)
    example = Foo(bar=Bar(baz=42))
    example_str = recipe.generate_response_example(example)

    response = f"<think>\nhi\n</think>\n{example_str}"

    assert recipe.parse(response) == example


def test_structured_response_patched_case_single_lb():
    recipe = StructuredResponseRecipe(Foo.model_json_schema())
    example = Foo(bar=Bar(baz=42)).model_dump()
    example_str = recipe.generate_response_example(example)

    response = f"<think>\nhi\n</think>\n{example_str}"

    assert recipe.parse(response) == example


@pytest.mark.parametrize(
    "bad_response",
    [
        "There is no json here!",
        "```",
        "```\n```",
        "```\n```\n",
        "```json\n```\n",
        "```json\n{}\n```\n",
        '```json\n{"foo": 2}\n```\n',
        '```json\n{"bar": {"baz": "hello"}}\n```\n',
    ],
)
def test_pydantic_response_failure_cases(bad_response):
    recipe = PydanticResponseRecipe(Foo)

    with pytest.raises(ParserException):
        recipe.parse(bad_response)


@pytest.mark.parametrize(
    "bad_response",
    [
        "",
        "There is no json here!",
        "```",
        "```\n```",
        "```\n```\n",
        "```json\n```\n",
        "```json\n{}\n```\n",
        '```json\n{"foo": 2}\n```\n',
        '```json\n{"bar": {"baz": "hello"}}\n```\n',
    ],
)
def test_structured_response_failure_cases(bad_response):
    recipe = StructuredResponseRecipe(Foo.model_json_schema())

    with pytest.raises(ParserException):
        recipe.parse(bad_response)


### Test CodeResponseLLMRecipe Recipe ###
def test_code_response():
    recipe = CodeResponseRecipe(syntax="python")
    example = "import pandas as pd"

    assert recipe.deserialize_output(recipe.serialize_output(example)) == example

    # Can recover from its example
    example_str = recipe.generate_response_example(example)
    assert recipe.parse(example_str) == example

    # In the case of multiple, this is going to return the last item in the list
    response = (
        f"{recipe.generate_response_example('import numpy as np')}\n{recipe.generate_response_example(example)}\n"
    )
    assert recipe.parse(response) == example

    # Should be able to handle extraneous text
    response = f"This is early text\n{recipe.generate_response_example(example)}\nThis is late text\n"
    assert recipe.parse(response) == example

    # If no syntax specified, should still be valid
    response = "```\nimport pandas as pd\n```\n"
    assert recipe.parse(response) == example

    # Marko should have some self healing
    response = "```\nimport pandas as pd\n"
    assert recipe.parse(response) == example


def test_code_response_sys_prompt():
    assert "hooplah" in CodeResponseRecipe(syntax="hooplah").task_instructions


@pytest.mark.parametrize(
    "bad_response",
    [
        "There is no code here!",
        "```sql\nselect * from table;\n```\n",
    ],
)
def test_code_response_failures(bad_response):
    recipe = CodeResponseRecipe(syntax="python")

    with pytest.raises(ParserException):
        recipe.parse(bad_response)
