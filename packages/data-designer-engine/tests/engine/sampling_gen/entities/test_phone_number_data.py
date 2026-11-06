# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_designer.engine.sampling_gen.entities import phone_number
from data_designer.engine.sampling_gen.entities.phone_number import get_area_code


@pytest.fixture(scope="module")
def phone_data() -> pd.DataFrame:
    return pd.read_parquet(Path(phone_number.__file__).parent / "assets/zip_area_code_map.parquet")


def test_phone_data_coverage_and_weights(phone_data: pd.DataFrame) -> None:
    assert list(phone_data.columns) == ["zipcode", "area_code", "count"]
    assert len(phone_data) == 33772
    assert phone_data.zipcode.is_unique
    assert phone_data.zipcode.str.fullmatch(r"[0-9]{5}").all()
    assert phone_data.area_code.dtype == np.dtype("int64")
    assert phone_data.area_code.between(200, 999).all()
    assert phone_data["count"].dtype == np.dtype("float64")
    assert np.isfinite(phone_data["count"]).all()
    assert phone_data["count"].ge(0).all()
    assert phone_data["count"].sum() == 335555510.0


def test_all_packaged_zip_codes_resolve(phone_data: pd.DataFrame) -> None:
    assert phone_data.zipcode.map(get_area_code).equals(phone_data.area_code.astype(str))


@pytest.mark.parametrize(
    ("zipcode", "expected"),
    [("00901", "787"), ("02108", "617"), ("06360", "860"), ("10001", "212"), ("90210", "310"), ("98101", "206")],
)
def test_representative_regional_area_codes(zipcode: str, expected: str) -> None:
    assert get_area_code(zipcode) == expected
