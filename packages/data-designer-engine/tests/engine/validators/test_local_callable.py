# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_designer.config.validator_params import LocalCallableValidatorParams
from data_designer.engine.validators.local_callable import LocalCallableValidator
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture()
def stub_data() -> list[dict]:
    return [{"text": "Sample text", "id": 1}]


def test_validate_with_callback_validator(stub_data: list[dict]):
    def callback_fn(df: pd.DataFrame) -> pd.DataFrame:
        if df.iloc[0]["text"] == "Sample text":
            return pd.DataFrame([{"is_valid": True, "confidence": "0.98"}])
        else:
            return pd.DataFrame([{"is_valid": False, "confidence": "0.0"}])

    validator = LocalCallableValidator(
        LocalCallableValidatorParams(
            validation_function=callback_fn,
        )
    )

    results = validator.run_validation(stub_data)

    assert len(results) == 1
    assert results[0].is_valid is True
    assert results[0].confidence == "0.98"
