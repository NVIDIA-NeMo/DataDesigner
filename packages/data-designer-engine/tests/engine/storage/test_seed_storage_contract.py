# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.config.utils.io_helpers import read_parquet_dataset
from data_designer.config.utils.storage_encoding import read_storage_plan
from data_designer.config.utils.storage_schema import JSON_STORAGE_RUN_KEY, JsonStoragePlan
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.resources.seed_reader import DataFrameSeedReader, LocalFileSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage


@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_dataframe_to_local_seed_preserves_json_storage(tmp_path: Path, nested: bool) -> None:
    values = [None, {}, {"alpha": 1}, {"beta": None}, {"mixed": [1, "two", True, None]}]
    expected = [{"arguments": value, "tool": "search"} for value in values] if nested else values
    plan = JsonStoragePlan(columns={"payload": [["arguments"]] if nested else [[]]})
    storage = ArtifactStorage(artifact_path=tmp_path, json_storage=plan)
    storage.write_batch_to_parquet_file(
        0, pd.DataFrame({"payload": pd.Series(expected, dtype=object)}), BatchStage.FINAL_RESULT
    )

    source = LocalFileSeedSource.from_dataframe(storage.load_dataset(), str(tmp_path / "seed.parquet"))
    assert read_storage_plan(pq.read_schema(source.path)) == plan
    assert read_parquet_dataset(Path(source.path))["payload"].tolist() == expected
    reader = LocalFileSeedReader()
    reader.attach(source, PlaintextResolver())
    batch = reader.create_batch_reader(batch_size=len(values), index_range=None, shuffle=False).read_next_batch()
    assert batch.to_pandas()["payload"].tolist() == expected


def test_dataframe_to_local_seed_keeps_unannotated_native_storage(tmp_path: Path) -> None:
    frame = pd.DataFrame({"payload": [{"alpha": 1}, {"beta": 2}]})
    source = LocalFileSeedSource.from_dataframe(frame, str(tmp_path / "seed.parquet"))
    table = pq.read_table(source.path)
    assert not read_storage_plan(table.schema).columns
    assert table.to_pylist() == [
        {"payload": {"alpha": 1, "beta": None}},
        {"payload": {"alpha": None, "beta": 2}},
    ]


@pytest.mark.parametrize("local_file", [False, True], ids=["dataframe-reader", "local-helper"])
def test_seed_rejects_malformed_dataframe_storage_contract(tmp_path: Path, local_file: bool) -> None:
    frame = pd.DataFrame({"payload": [{"alpha": 1}]})
    frame.attrs[JSON_STORAGE_RUN_KEY] = {"version": 2, "columns": {"payload": [[]]}}
    with pytest.raises(InvalidFileFormatError):
        if local_file:
            LocalFileSeedSource.from_dataframe(frame, str(tmp_path / "seed.parquet"))
        else:
            reader = DataFrameSeedReader()
            reader.attach(DataFrameSeedSource(df=frame), PlaintextResolver())
            reader.get_storage_plan()
    assert not (tmp_path / "seed.parquet").exists()


def test_dataframe_seed_projection_restricts_storage_contract() -> None:
    plan = JsonStoragePlan(columns={"payload": [[]], "kept": [["arguments"]]})
    original = pd.DataFrame({"payload": [{"alpha": 1}], "kept": [{"arguments": [1, "two"]}]})
    original.attrs[JSON_STORAGE_RUN_KEY] = plan.model_dump()
    reader = DataFrameSeedReader()
    reader.attach(DataFrameSeedSource(df=original[["kept"]]), PlaintextResolver())

    assert reader.get_storage_plan() == plan.for_columns(["kept"])
    batch = reader.create_batch_reader(batch_size=1, index_range=None, shuffle=False).read_next_batch().to_pandas()
    assert batch.to_dict("records") == [{"kept": {"arguments": [1, "two"]}}]
    assert batch.attrs[JSON_STORAGE_RUN_KEY] == plan.for_columns(["kept"]).model_dump()
    assert original.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


class StalePlanSeedReader(DataFrameSeedReader):
    """Plugin reader reporting a contract that includes a column outside its output."""

    def get_storage_plan(self) -> JsonStoragePlan:
        return JsonStoragePlan(columns={"payload": [[]]})


def test_builder_inherits_storage_only_for_actual_seed_columns(stub_resource_provider: Mock) -> None:
    frame = pd.DataFrame({"question": ["hi"]})
    reader = StalePlanSeedReader()
    source = DataFrameSeedSource(df=frame)
    reader.attach(source, PlaintextResolver())
    provider = stub_resource_provider
    provider.seed_reader = reader
    provider.run_config = RunConfig(buffer_size=2, max_concurrent_row_groups=1)
    config = DataDesignerConfigBuilder().with_seed_dataset(source)
    config.add_column(ExpressionColumnConfig(name="payload", expr="new", dtype="str"))

    builder = DatasetBuilder(config.build(), provider)
    builder.build(num_records=1)
    assert not builder.artifact_storage.json_storage.columns
    assert pq.read_table(builder.artifact_storage.final_dataset_path).to_pylist() == [
        {"question": "hi", "payload": "new"}
    ]
    assert JSON_STORAGE_RUN_KEY not in builder.artifact_storage.read_metadata()
