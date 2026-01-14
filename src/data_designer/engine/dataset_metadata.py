# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.engine.resources.resource_provider import ResourceProvider


def create_dataset_metadata(resource_provider: ResourceProvider) -> DatasetMetadata:
    """Create DatasetMetadata from a ResourceProvider.

    Args:
        resource_provider: The resource provider containing seed reader and other resources.

    Returns:
        A DatasetMetadata instance with resolved metadata.
    """
    seed_column_names = []
    if resource_provider.seed_reader is not None:
        seed_column_names = resource_provider.seed_reader.get_column_names()

    return DatasetMetadata(seed_column_names=seed_column_names)
