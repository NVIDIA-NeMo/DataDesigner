# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.base import ConfigBase
from data_designer.config.models import ModelConfig
from data_designer.config.seed_dataset import SeedDatasetConfig
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.registry import ModelRegistry, create_model_registry
from data_designer.engine.resources.managed_storage import ManagedBlobStorage, init_managed_blob_storage
from data_designer.engine.resources.seed_dataset import SeedDatasetReader, SeedDatasetReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver


class ResourceType(StrEnum):
    BLOB_STORAGE = "blob_storage"
    MODEL_REGISTRY = "model_registry"
    SEED_DATASET_READER = "seed_dataset_reader"


class ResourceProvider(ConfigBase):
    artifact_storage: ArtifactStorage
    blob_storage: ManagedBlobStorage | None = None
    model_registry: ModelRegistry | None = None
    seed_dataset_reader: SeedDatasetReader | None = None


def create_resource_provider(
    *,
    artifact_storage: ArtifactStorage,
    model_configs: list[ModelConfig],
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    seed_dataset_reader_registry: SeedDatasetReaderRegistry,
    blob_storage: ManagedBlobStorage | None = None,
    seed_dataset_config: SeedDatasetConfig | None = None,
) -> ResourceProvider:
    seed_dataset_reader = None
    if seed_dataset_config:
        seed_dataset_reader = seed_dataset_reader_registry.get_reader(
            seed_dataset_config,
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
        seed_dataset_reader=seed_dataset_reader,
    )
