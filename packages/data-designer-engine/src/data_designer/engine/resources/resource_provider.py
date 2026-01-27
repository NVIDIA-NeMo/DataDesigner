# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.base import ConfigBase
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source import SeedSource
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.factory import create_model_registry
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.resources.managed_storage import ManagedBlobStorage, init_managed_blob_storage
from data_designer.engine.resources.seed_reader import SeedReader, SeedReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver


class ResourceType(StrEnum):
    BLOB_STORAGE = "blob_storage"
    MODEL_REGISTRY = "model_registry"
    SEED_READER = "seed_reader"


class ResourceProvider(ConfigBase):
    artifact_storage: ArtifactStorage
    blob_storage: ManagedBlobStorage | None = None
    model_registry: ModelRegistry | None = None
    run_config: RunConfig = RunConfig()
    seed_reader: SeedReader | None = None

    def get_dataset_metadata(self) -> DatasetMetadata:
        """Get metadata about the dataset being generated.

        Returns:
            DatasetMetadata with seed column names and other metadata.
        """
        seed_column_names = []
        if self.seed_reader is not None:
            seed_column_names = self.seed_reader.get_column_names()
        return DatasetMetadata(seed_column_names=seed_column_names)


def create_resource_provider(
    *,
    artifact_storage: ArtifactStorage,
    model_configs: list[ModelConfig],
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    seed_reader_registry: SeedReaderRegistry,
    blob_storage: ManagedBlobStorage | None = None,
    seed_dataset_source: SeedSource | None = None,
    run_config: RunConfig | None = None,
) -> ResourceProvider:
    """Factory function for creating a ResourceProvider instance.
    This function triggers lazy loading of heavy dependencies like litellm.
    """
    seed_reader = None
    if seed_dataset_source:
        seed_reader = seed_reader_registry.get_reader(
            seed_dataset_source,
            secret_resolver,
        )

    return ResourceProvider(
        artifact_storage=artifact_storage,
        model_registry=create_model_registry(
            model_configs=model_configs,
            secret_resolver=secret_resolver,
            model_provider_registry=model_provider_registry,
        ),
        blob_storage=blob_storage or init_managed_blob_storage(),
        seed_reader=seed_reader,
        run_config=run_config or RunConfig(),
    )
