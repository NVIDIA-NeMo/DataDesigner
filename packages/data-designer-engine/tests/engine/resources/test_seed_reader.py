# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    LocalFileSeedReader,
    SeedReaderError,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import PlaintextResolver


def test_one_reader_per_seed_type() -> None:
    local_1 = LocalFileSeedReader
    local_2 = LocalFileSeedReader

    with pytest.raises(SeedReaderError):
        SeedReaderRegistry([local_1, local_2])

    registry = SeedReaderRegistry([local_1])

    with pytest.raises(SeedReaderError):
        registry.add_reader(local_2)


def test_get_reader_basic() -> None:
    registry = SeedReaderRegistry([LocalFileSeedReader, DataFrameSeedReader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    reader = registry.get_reader(local_seed_config, PlaintextResolver())

    assert isinstance(reader, DataFrameSeedReader)


def test_get_reader_creates_fresh_reader_instances_from_registered_types() -> None:
    registry = SeedReaderRegistry([DataFrameSeedReader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    seed_config = DataFrameSeedSource(df=df)

    reader_1 = registry.get_reader(seed_config, PlaintextResolver())
    reader_2 = registry.get_reader(seed_config, PlaintextResolver())

    assert isinstance(reader_1, DataFrameSeedReader)
    assert isinstance(reader_2, DataFrameSeedReader)
    assert reader_1 is not reader_2


def test_get_reader_reuses_registered_reader_instances() -> None:
    df_reader = DataFrameSeedReader()
    registry = SeedReaderRegistry([df_reader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    seed_config = DataFrameSeedSource(df=df)

    reader = registry.get_reader(seed_config, PlaintextResolver())

    assert reader is df_reader


def test_get_reader_missing() -> None:
    registry = SeedReaderRegistry([LocalFileSeedReader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    with pytest.raises(SeedReaderError):
        registry.get_reader(local_seed_config, PlaintextResolver())
