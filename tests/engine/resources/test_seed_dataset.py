# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from data_designer.config.seed_dataset import DataFrameSeedConfig
from data_designer.engine.resources.seed_dataset import (
    DataFrameSeedReader,
    LocalFileSeedReader,
    SeedDatasetReaderError,
    SeedDatasetReaderRegistry,
)
from data_designer.engine.secret_resolver import PlaintextResolver


def test_one_reader_per_seed_type():
    local_1 = LocalFileSeedReader()
    local_2 = LocalFileSeedReader()

    with pytest.raises(SeedDatasetReaderError):
        SeedDatasetReaderRegistry([local_1, local_2])

    registry = SeedDatasetReaderRegistry([local_1])

    with pytest.raises(SeedDatasetReaderError):
        registry.add_reader(local_2)


def test_get_reader_basic():
    local_reader = LocalFileSeedReader()
    df_reader = DataFrameSeedReader()
    registry = SeedDatasetReaderRegistry([local_reader, df_reader])

    df = pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedConfig(df=df)

    reader = registry.get_reader(local_seed_config, PlaintextResolver())

    assert reader == df_reader


def test_get_reader_missing():
    local_reader = LocalFileSeedReader()
    registry = SeedDatasetReaderRegistry([local_reader])

    df = pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedConfig(df=df)

    with pytest.raises(SeedDatasetReaderError):
        registry.get_reader(local_seed_config, PlaintextResolver())
