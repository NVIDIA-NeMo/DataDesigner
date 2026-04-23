# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.interface.data_designer import DataDesigner


class FakeRayDataset:
    def __init__(self, blocks: list[lazy.pd.DataFrame]) -> None:
        self.blocks = blocks
        self.map_batches_kwargs: dict[str, Any] | None = None

    def map_batches(self, fn, **kwargs) -> FakeRayDataset:
        self.map_batches_kwargs = kwargs
        fn_kwargs = kwargs.get("fn_kwargs") or {}
        return FakeRayDataset([fn(block, **fn_kwargs) for block in self.blocks])

    def to_arrow_refs(self) -> list[str]:
        return [f"arrow-ref-{i}" for i, _ in enumerate(self.blocks)]

    def to_pandas(self) -> lazy.pd.DataFrame:
        return lazy.pd.concat(self.blocks, ignore_index=True)


class FakeRayDataModule:
    Dataset = FakeRayDataset

    def __init__(self) -> None:
        self.from_arrow_refs_input: list[Any] | None = None
        self.from_pandas_refs_input: list[Any] | None = None

    def range(self, num_records: int) -> FakeRayDataset:
        return FakeRayDataset([lazy.pd.DataFrame({"id": list(range(num_records))})])

    def from_arrow_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_arrow_refs_input = refs
        return FakeRayDataset(list(refs))

    def from_pandas_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_pandas_refs_input = refs
        return FakeRayDataset(list(refs))


def _install_fake_ray(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_ray = types.ModuleType("ray")
    fake_ray.data = FakeRayDataModule()
    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda: None
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return fake_ray


def _managed_assets_path(tmp_path) -> Any:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_ray_backend_uses_input_dataset_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stub_model_configs,
    stub_model_providers,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})])
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        ExpressionColumnConfig(
            name="x_label",
            expr="{{ x }}-{{ label }}",
        )
    )

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, ray_remote_args={"num_cpus": 0.5}),
    )
    designer.set_run_config(RunConfig(buffer_size=2))

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert input_dataset.map_batches_kwargs is not None
    assert input_dataset.map_batches_kwargs["num_cpus"] == 0.5
    assert "ray_remote_args" not in input_dataset.map_batches_kwargs
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_uses_pandas_object_refs_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stub_model_configs,
    stub_model_providers,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    input_refs = [lazy.pd.DataFrame({"x": [1], "label": ["a"]}), lazy.pd.DataFrame({"x": [2], "label": ["b"]})]
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        ExpressionColumnConfig(
            name="x_label",
            expr="{{ x }}-{{ label }}",
        )
    )

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, object_ref_format="pandas"),
    )

    results = designer.create(config_builder, input_dataset=input_refs)
    output_df = results.load_dataset().to_pandas()

    assert fake_ray.data.from_pandas_refs_input == input_refs
    assert fake_ray.data.from_arrow_refs_input is None
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_can_return_arrow_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stub_sampler_only_config_builder,
    stub_model_providers,
) -> None:
    _install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, output="arrow_refs"),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)

    assert results.output == ["arrow-ref-0"]
    assert results.to_arrow_refs() == ["arrow-ref-0"]


def test_ray_backend_import_is_lazy_when_ray_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = ray_backend_module.importlib.import_module

    def import_module(name: str) -> Any:
        if name == "ray":
            raise ImportError
        return real_import_module(name)

    monkeypatch.setattr(ray_backend_module.importlib, "import_module", import_module)
    backend = RayBackend()

    with pytest.raises(ImportError, match="data-designer\\[ray\\]"):
        backend.create(
            data_designer=object(),
            config_builder=DataDesignerConfigBuilder(model_configs=[]),
            num_records=1,
            dataset_name="dataset",
        )


def test_input_dataset_requires_backend(tmp_path, stub_sampler_only_config_builder, stub_model_providers) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
    )

    with pytest.raises(ValueError, match="input_dataset requires an execution backend"):
        designer.create(stub_sampler_only_config_builder, input_dataset=object())


def test_seed_readers_are_cloned_without_attachment_state() -> None:
    reader = DataFrameSeedReader()
    reader.attach(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [1]})),
        PlaintextResolver(),
    )
    assert reader.get_seed_dataset_size() == 1
    assert getattr(reader, "_duckdb_conn") is not None

    clones = ray_backend_module._clone_seed_readers_for_worker([reader])

    assert len(clones) == 1
    assert clones[0] is not reader
    assert getattr(clones[0], "_duckdb_conn") is None
    assert not hasattr(clones[0], "source")
    assert not hasattr(clones[0], "secret_resolver")
