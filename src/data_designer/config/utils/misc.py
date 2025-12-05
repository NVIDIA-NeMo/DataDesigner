# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Optional, Union

from jinja2 import TemplateSyntaxError, meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

from data_designer.config.utils.errors import UserJinjaTemplateSyntaxError

REPR_LIST_LENGTH_USE_JSON = 4


def kebab_to_snake(s: str) -> str:
    return s.replace("-", "_")


@contextmanager
def template_error_handler():
    try:
        yield
    except TemplateSyntaxError as exception:
        exception_string = (
            f"Encountered a syntax error in the provided Jinja2 template:\n{str(exception)}\n"
            "For more information on writing Jinja2 templates, "
            "refer to https://jinja.palletsprojects.com/en/stable/templates"
        )
        raise UserJinjaTemplateSyntaxError(exception_string)
    except Exception:
        raise


def assert_valid_jinja2_template(template: str) -> None:
    """Raises an error if the template cannot be parsed."""
    with template_error_handler():
        meta.find_undeclared_variables(ImmutableSandboxedEnvironment().parse(template))


def can_run_data_designer_locally() -> bool:
    """Returns True if Data Designer can be run locally, False otherwise."""
    try:
        from ... import engine  # noqa: F401, TID252
    except ImportError:
        return False
    return True


def get_prompt_template_keywords(template: str) -> set[str]:
    """Extract all keywords from a valid string template."""
    with template_error_handler():
        ast = ImmutableSandboxedEnvironment().parse(template)
        keywords = set(meta.find_undeclared_variables(ast))

    return keywords


def json_indent_list_of_strings(
    column_names: list[str], *, indent: Optional[Union[int, str]] = None
) -> Optional[Union[list[str], str]]:
    """Convert a list of column names to a JSON string if the list is long.

    This function helps keep Data Designer's __repr__ output clean and readable.

    Args:
        column_names: List of column names.
        indent: Indentation for the JSON string.

    Returns:
        A list of column names or a JSON string if the list is long.
    """
    return (
        None
        if len(column_names) == 0
        else (
            column_names if len(column_names) < REPR_LIST_LENGTH_USE_JSON else json.dumps(column_names, indent=indent)
        )
    )
