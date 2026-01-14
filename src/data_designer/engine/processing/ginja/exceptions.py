# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from jinja2 import TemplateAssertionError


class UserTemplateError(Exception):
    """Exception for user-induced template flaws, intentional or not."""


class UserTemplateUnsupportedFiltersError(UserTemplateError):
    """Specific exception for the case of unsupported filters."""


class RecordContentsError(Exception):
    """Exception for cases involving the record providing template context."""


def maybe_handle_missing_filter_exception(exception: BaseException, available_jinja_filters: list[str]) -> None:
    """Interpret and handle the possible case of a missing filter exception.

    If this wasn't a missing filter exception, then this function will do
    nothing.

    Args:
        exception (BaseException): The caught exception.
        available_jinja_filters (list[str]): The list of Jinja filters that
            are known to be available within the environment.

    Raises:
        UserTemplateUnsupportedFiltersError: If the exception was specifically for an unknown
            or unsupported Jinja2 filter.
    """
    if not isinstance(exception, TemplateAssertionError):
        return

    exc_message = exception.message or ""

    ## The missing filter message has the format:
    ## "No filter named '____'"
    match = re.search(r"No filter named '([^']+)'", exc_message)
    if not match:
        return
    else:
        missing_filter_name = match.group(1)
        available_filter_str = ", ".join(available_jinja_filters)
        raise UserTemplateUnsupportedFiltersError(
            (
                f"The Jinja2 filter `{{{{ ... | {missing_filter_name} }}}}` "
                f"is not a permitted operation. Available filters: {available_filter_str}"
            )
        ) from exception
