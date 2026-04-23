# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.model_provider import resolve_model_provider_registry
from data_designer.engine.resources.person_reader import PersonReader, create_person_reader
from data_designer.engine.resources.resource_provider import create_resource_provider
from data_designer.engine.resources.seed_reader import SeedReader, SeedReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage

if TYPE_CHECKING:
    from data_designer.config.mcp import MCPProviderT
    from data_designer.config.models import ModelProvider


RayOutputMode = Literal["dataset", "arrow_refs"]
RayObjectRefInputFormat = Literal["arrow", "pandas"]


@dataclass(frozen=True)
class _RayWorkerOptions:
    model_providers: list[ModelProvider]
    default_provider_name: str
    secret_resolver: SecretResolver
    seed_readers: list[SeedReader]
    managed_assets_path: str
    person_reader: PersonReader | None
    mcp_providers: list[MCPProviderT]
    run_config: RunConfig


class RayDatasetCreationResults:
    """Results wrapper for Ray-resident Data Designer outputs."""

    def __init__(
        self,
        *,
        dataset: Any,
        config_builder: DataDesignerConfigBuilder,
        output: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self._config_builder = config_builder
        self._output = output

    def load_dataset(self) -> Any:
        """Return the Ray Dataset without materializing it on the driver."""
        return self.dataset

    def load_analysis(self) -> None:
        """Ray-resident jobs do not produce local profiler artifacts."""
        return None

    def to_arrow_refs(self) -> list[Any]:
        """Return Ray ObjectRefs containing PyArrow tables, one per Ray block."""
        if self._output is not None:
            return self._output
        return self.dataset.to_arrow_refs()

    @property
    def output(self) -> Any:
        """Backend-selected output object."""
        return self._output if self._output is not None else self.dataset


class RayBackend:
    """Ray Data execution backend for in-memory Data Designer jobs.

    The backend maps Data Designer generation over Ray Data blocks and returns
    Ray-resident outputs. Ray is imported lazily so base Data Designer installs
    do not require the optional dependency. input_dataset may be a Ray Dataset
    or a sequence of ObjectRefs containing Arrow tables or pandas DataFrames.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        output: RayOutputMode = "dataset",
        object_ref_format: RayObjectRefInputFormat = "arrow",
        auto_init: bool = False,
        zero_copy_batch: bool = True,
        ray_remote_args: dict[str, Any] | None = None,
    ) -> None:
        if output not in ("dataset", "arrow_refs"):
            raise ValueError("RayBackend output must be 'dataset' or 'arrow_refs'.")
        if object_ref_format not in ("arrow", "pandas"):
            raise ValueError("RayBackend object_ref_format must be 'arrow' or 'pandas'.")
        self.batch_size = batch_size
        self.output = output
        self.object_ref_format = object_ref_format
        self.auto_init = auto_init
        self.zero_copy_batch = zero_copy_batch
        self.ray_remote_args = ray_remote_args

    def create(
        self,
        *,
        data_designer: Any,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        input_dataset: Any | None = None,
    ) -> RayDatasetCreationResults:
        del dataset_name
        ray = _import_ray()
        if not ray.is_initialized():
            if not self.auto_init:
                raise RuntimeError(
                    "Ray is not initialized. Call ray.init(...) before using RayBackend, "
                    "or construct RayBackend(auto_init=True)."
                )
            ray.init()

        use_input_dataset = input_dataset is not None
        if use_input_dataset and config_builder.get_seed_config() is not None:
            raise ValueError("RayBackend input_dataset is used as the seed dataset; remove the existing seed config.")

        dataset = self._resolve_input_dataset(ray, input_dataset=input_dataset, num_records=num_records)
        batch_size = self.batch_size or data_designer._run_config.buffer_size
        worker_options = _RayWorkerOptions(
            model_providers=list(data_designer._model_providers),
            default_provider_name=data_designer._model_provider_registry.get_default_provider_name(),
            secret_resolver=data_designer._secret_resolver,
            seed_readers=_clone_seed_readers_for_worker(data_designer._seed_reader_registry._readers.values()),
            managed_assets_path=str(data_designer._managed_assets_path),
            person_reader=data_designer._person_reader,
            mcp_providers=list(data_designer._mcp_providers),
            run_config=data_designer._run_config,
        )

        map_batches_kwargs: dict[str, Any] = {
            "fn_kwargs": {
                "config_builder": config_builder,
                "worker_options": worker_options,
                "use_input_dataset": use_input_dataset,
            },
            "batch_size": batch_size,
            "batch_format": "pandas",
            "zero_copy_batch": self.zero_copy_batch,
        }
        if self.ray_remote_args is not None:
            map_batches_kwargs.update(self.ray_remote_args)

        mapped = dataset.map_batches(_generate_batch, **map_batches_kwargs)
        output = mapped.to_arrow_refs() if self.output == "arrow_refs" else None
        return RayDatasetCreationResults(dataset=mapped, config_builder=config_builder, output=output)

    def _resolve_input_dataset(self, ray: Any, *, input_dataset: Any | None, num_records: int) -> Any:
        if input_dataset is None:
            return ray.data.range(num_records)
        if hasattr(input_dataset, "map_batches"):
            return input_dataset
        if isinstance(input_dataset, (list, tuple)):
            refs = list(input_dataset)
            if self.object_ref_format == "pandas":
                return ray.data.from_pandas_refs(refs)
            return ray.data.from_arrow_refs(refs)
        raise TypeError(
            "RayBackend input_dataset must be a ray.data.Dataset or a sequence of Ray ObjectRefs "
            "containing PyArrow tables or pandas DataFrames."
        )


def _clone_seed_readers_for_worker(readers: Iterable[SeedReader]) -> list[SeedReader]:
    return [_clone_seed_reader_for_worker(reader) for reader in readers]


def _clone_seed_reader_for_worker(reader: SeedReader) -> SeedReader:
    clone = copy.copy(reader)
    clone._reset_attachment_state()
    for attr in ("source", "secret_resolver"):
        if hasattr(clone, attr):
            delattr(clone, attr)
    return clone


def _generate_batch(
    batch: Any,
    *,
    config_builder: DataDesignerConfigBuilder,
    worker_options: _RayWorkerOptions,
    use_input_dataset: bool,
) -> Any:
    os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
    dataframe = _coerce_pandas_dataframe(batch)
    num_records = len(dataframe)
    if num_records == 0:
        return dataframe

    block_builder = copy.deepcopy(config_builder)
    if use_input_dataset:
        block_builder.with_seed_dataset(DataFrameSeedSource(df=dataframe.copy()))

    with tempfile.TemporaryDirectory(prefix="data-designer-ray-") as artifact_dir:
        ArtifactStorage.mkdir_if_needed(Path(artifact_dir))
        seed_readers = copy.deepcopy(worker_options.seed_readers)
        resource_provider = create_resource_provider(
            artifact_storage=ArtifactStorage(artifact_path=artifact_dir, dataset_name="ray-block"),
            model_configs=block_builder.model_configs,
            secret_resolver=worker_options.secret_resolver,
            model_provider_registry=resolve_model_provider_registry(
                worker_options.model_providers,
                worker_options.default_provider_name,
            ),
            seed_reader_registry=SeedReaderRegistry(readers=seed_readers),
            person_reader=worker_options.person_reader or create_person_reader(worker_options.managed_assets_path),
            seed_dataset_source=(
                block_builder.get_seed_config().source if block_builder.get_seed_config() is not None else None
            ),
            run_config=copy.deepcopy(worker_options.run_config),
            mcp_providers=worker_options.mcp_providers,
            tool_configs=block_builder.tool_configs,
        )
        builder = DatasetBuilder(
            data_designer_config=block_builder.build(),
            resource_provider=resource_provider,
            use_async=True,
        )
        raw_dataset = builder.build_preview(num_records=num_records)
        return builder.process_preview(raw_dataset)


def _coerce_pandas_dataframe(batch: Any) -> Any:
    if isinstance(batch, lazy.pd.DataFrame):
        return batch
    if isinstance(batch, dict):
        return lazy.pd.DataFrame(batch)
    return lazy.pd.DataFrame(batch)


def _import_ray() -> Any:
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        raise ImportError("RayBackend requires Ray. Install Data Designer with `data-designer[ray]`.") from exc
