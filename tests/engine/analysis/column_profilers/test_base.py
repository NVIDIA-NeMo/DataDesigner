# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.analysis.column_profilers.base import ColumnConfigWithDataFrame
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


def test_column_config_with_dataframe_valid_column_config_with_dataframe():
    df = pd.DataFrame({"test_column": [1, 2, 3]})
    column_config = SamplerColumnConfig(
        name="test_column", sampler_type=SamplerType.CATEGORY, params={"values": [1, 2, 3]}
    )

    config_with_df = ColumnConfigWithDataFrame(column_config=column_config, df=df)

    assert config_with_df.column_config.name == "test_column"
    assert "test_column" in config_with_df.df.columns
    assert config_with_df.df["test_column"].tolist() == [1, 2, 3]


def test_column_config_with_dataframe_column_not_found_validation_error():
    df = pd.DataFrame({"other_column": [1, 2, 3]})
    column_config = SamplerColumnConfig(
        name="test_column", sampler_type=SamplerType.CATEGORY, params={"values": [1, 2, 3]}
    )

    with pytest.raises(ValidationError, match="Column 'test_column' not found in DataFrame"):
        ColumnConfigWithDataFrame(column_config=column_config, df=df)


def test_column_config_with_dataframe_as_tuple_method():
    df = pd.DataFrame({"test_column": [1, 2, 3]})
    column_config = SamplerColumnConfig(
        name="test_column", sampler_type=SamplerType.CATEGORY, params={"values": [1, 2, 3]}
    )

    config_with_df = ColumnConfigWithDataFrame(column_config=column_config, df=df)
    column_config_result, df_result = config_with_df.as_tuple()

    assert column_config_result == column_config

    assert df_result["test_column"].tolist() == df["test_column"].tolist()
