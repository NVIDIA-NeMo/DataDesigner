# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_designer.config.seed_source import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    LocalFileSeedReader,
    SeedReaderError,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


def test_one_reader_per_seed_type():
    local_1 = LocalFileSeedReader()
    local_2 = LocalFileSeedReader()

    with pytest.raises(SeedReaderError):
        SeedReaderRegistry([local_1, local_2])

    registry = SeedReaderRegistry([local_1])

    with pytest.raises(SeedReaderError):
        registry.add_reader(local_2)


def test_get_reader_basic():
    local_reader = LocalFileSeedReader()
    df_reader = DataFrameSeedReader()
    registry = SeedReaderRegistry([local_reader, df_reader])

    df = pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    reader = registry.get_reader(local_seed_config, PlaintextResolver())

    assert reader == df_reader


def test_get_reader_missing():
    local_reader = LocalFileSeedReader()
    registry = SeedReaderRegistry([local_reader])

    df = pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    with pytest.raises(SeedReaderError):
        registry.get_reader(local_seed_config, PlaintextResolver())
