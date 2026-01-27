# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.processing.ginja.environment import UserTemplateSandboxEnvironment
from data_designer.engine.processing.ginja.exceptions import UserTemplateUnsupportedFiltersError


def test_maybe_handle_missing_filter_exception():
    env = UserTemplateSandboxEnvironment(allowed_references=["foo"])

    with pytest.raises(UserTemplateUnsupportedFiltersError) as exc_info:
        env.safe_render("{{ foo | asdf }}", record={"foo": 42})

    exc_message = str(exc_info.value)
    assert "Available filters" in exc_message

    with pytest.raises(UserTemplateUnsupportedFiltersError) as exc_info:
        env.validate_template("{{ foo | asdf }}")
    assert "Available filters" in exc_message
