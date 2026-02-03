# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import (
    CodeValidatorParams,
    LocalCallableValidatorParams,
    RemoteValidatorParams,
)
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


def test_code_validator_params():
    assert CodeValidatorParams(code_lang=CodeLang.PYTHON).code_lang == CodeLang.PYTHON

    with pytest.raises(ValidationError):
        CodeValidatorParams(code_lang=CodeLang.RUBY)


def test_remote_validator_params():
    stub_url = "https://example.com"
    params = RemoteValidatorParams(endpoint_url=stub_url)
    assert params.endpoint_url == stub_url
    assert params.output_schema is None
    assert params.timeout == 30.0
    assert params.max_retries == 3
    assert params.retry_backoff == 2.0
    assert params.max_parallel_requests == 4

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        RemoteValidatorParams(endpoint_url=stub_url, timeout=0.0)
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        RemoteValidatorParams(endpoint_url=stub_url, max_retries=-1)
    with pytest.raises(ValidationError, match="Input should be greater than 1"):
        RemoteValidatorParams(endpoint_url=stub_url, retry_backoff=1.0)
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
        RemoteValidatorParams(endpoint_url=stub_url, max_parallel_requests=0)


def test_callback_validator_params():
    def stub_callback(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([{"is_valid": True, "confidence": "0.98"}])

    params = LocalCallableValidatorParams(validation_function=stub_callback)
    assert params.validation_function == stub_callback
    assert params.output_schema is None

    params_model_dump = params.model_dump(mode="json")
    assert params_model_dump["validation_function"] == "stub_callback"
