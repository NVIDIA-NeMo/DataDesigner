# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from data_designer.config.base import ConfigBase
from data_designer.config.models import ModelConfig
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.registry import ModelRegistry, create_model_registry
from data_designer.engine.resources.managed_storage import ManagedBlobStorage
from data_designer.engine.resources.seed_dataset_source import SeedDatasetRepository, SeedDatasetSourceRegistry
from data_designer.engine.secret_resolver import SecretResolver


class ResourceType(StrEnum):
    BLOB_STORAGE = "blob_storage"
    SEED_DATASET_REPOSITORY = "seed_dataset_repository"
    MODEL_REGISTRY = "model_registry"


class ResourceProvider(ConfigBase):
    artifact_storage: ArtifactStorage
    blob_storage: ManagedBlobStorage
    seed_dataset_repository: SeedDatasetRepository
    model_registry: ModelRegistry


def create_resource_provider(
    *,
    artifact_storage: ArtifactStorage,
    model_configs: list[ModelConfig],
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    seed_dataset_source_registry: SeedDatasetSourceRegistry,
    blob_storage: ManagedBlobStorage,
) -> ResourceProvider:
    return ResourceProvider(
        artifact_storage=artifact_storage,
        seed_dataset_repository=SeedDatasetRepository(
            registry=seed_dataset_source_registry,
            secret_resolver=secret_resolver,
        ),
        model_registry=create_model_registry(
            model_configs=model_configs,
            secret_resolver=secret_resolver,
            model_provider_registry=model_provider_registry,
        ),
        blob_storage=blob_storage,
    )
