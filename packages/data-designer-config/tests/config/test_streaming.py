# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig


@pytest.mark.parametrize("stream", [False, True])
def test_streaming_config_round_trip_and_request_parameters(stream: bool) -> None:
    """A stream flag survives config serialization and enables incremental chat transport."""
    config = ModelConfig(
        alias="test", model="test", provider="test", inference_parameters=ChatCompletionInferenceParams(stream=stream)
    )
    restored = ModelConfig.model_validate_json(config.model_dump_json())
    assert restored.inference_parameters.stream is stream
    assert restored.inference_parameters.generate_kwargs.get("stream", False) is stream


def test_streaming_is_opt_in() -> None:
    """Default inference kwargs preserve the existing provider request body."""
    assert "stream" not in ChatCompletionInferenceParams().generate_kwargs
