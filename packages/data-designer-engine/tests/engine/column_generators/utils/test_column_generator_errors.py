# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.column_generators.utils.errors import PromptTemplateRenderError


def test_prompt_template_render_error_message():
    message = "Template rendering failed"
    error = PromptTemplateRenderError(message)
    assert str(error) == message


def test_prompt_template_render_error_without_message():
    error = PromptTemplateRenderError()
    assert str(error) == ""
