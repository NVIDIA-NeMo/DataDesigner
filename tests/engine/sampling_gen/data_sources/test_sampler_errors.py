# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.sampling_gen.data_sources.errors import InvalidSamplerParamsError


def test_invalid_sampler_params_error_message():
    message = "Invalid sampler parameters"
    error = InvalidSamplerParamsError(message)
    assert str(error) == message


def test_invalid_sampler_params_error_raising():
    with pytest.raises(InvalidSamplerParamsError, match="Test error"):
        raise InvalidSamplerParamsError("Test error")
