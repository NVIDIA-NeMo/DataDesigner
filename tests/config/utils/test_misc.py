# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from jinja2 import TemplateSyntaxError

from data_designer.config.utils.errors import UserJinjaTemplateSyntaxError
from data_designer.config.utils.misc import (
    assert_valid_jinja2_template,
    extract_keywords_from_jinja2_template,
    json_indent_list_of_strings,
    kebab_to_snake,
    template_error_handler,
)


def test_kebab_to_snake():
    assert kebab_to_snake("kebab-case-string") == "kebab_case_string"
    assert kebab_to_snake("simple") == "simple"


def test_template_error_handler():
    val_set = False
    with template_error_handler():
        val_set = True
    # shouldn't raise any errors
    assert val_set


def test_template_error_handler_invalid_template():
    with pytest.raises(UserJinjaTemplateSyntaxError):
        with template_error_handler():
            raise TemplateSyntaxError("Invalid template", 1)


def test_template_error_handler_catch_non_template_error():
    with pytest.raises(ValueError):
        with template_error_handler():
            raise ValueError("Invalid value")


def test_assert_valid_jinja2_template():
    assert_valid_jinja2_template("{% if name %}Hello, {{ name }}!{% endif %}")
    assert_valid_jinja2_template("{% for item in items %}{{ item }}{% endfor %}")

    with pytest.raises(UserJinjaTemplateSyntaxError):
        assert_valid_jinja2_template("{{ name }")

    with patch("data_designer.config.utils.misc.ImmutableSandboxedEnvironment.parse") as mock_sandbox_env_parse:
        mock_sandbox_env_parse.side_effect = ValueError("Invalid Value")
        with pytest.raises(ValueError):
            assert_valid_jinja2_template("{hello}")


def test_get_prompt_template_keywords():
    assert extract_keywords_from_jinja2_template("{{ first_name }} {{last_name}}") == {"first_name", "last_name"}
    assert extract_keywords_from_jinja2_template("{% if first_name %}Hello, {{ last_name }}!{% endif %}") == {
        "first_name",
        "last_name",
    }

    with pytest.raises(UserJinjaTemplateSyntaxError):
        extract_keywords_from_jinja2_template("{{ name }")


def test_json_indent_list_of_strings():
    assert json_indent_list_of_strings([]) is None
    assert json_indent_list_of_strings(["hello", "world"]) == ["hello", "world"]
    assert json_indent_list_of_strings(["hello", "world"], indent=2) == ["hello", "world"]
    assert (
        json_indent_list_of_strings(["hello", "world", "foo", "bar"], indent=2)
        == '[\n  "hello",\n  "world",\n  "foo",\n  "bar"\n]'
    )
