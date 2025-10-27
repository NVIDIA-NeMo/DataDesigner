# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum

from pydantic import field_validator

from .base import ConfigBase
from .datastore import DatastoreSettings
from .utils.io_helpers import validate_dataset_file_path


class SamplingStrategy(str, Enum):
    ORDERED = "ordered"
    SHUFFLE = "shuffle"


class SeedConfig(ConfigBase):
    dataset: str
    sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED


class SeedDatasetReference(ABC, ConfigBase):
    dataset: str


class DatastoreSeedDatasetReference(SeedDatasetReference):
    datastore_settings: DatastoreSettings

    @property
    def repo_id(self) -> str:
        return "/".join(self.dataset.split("/")[:-1])

    @property
    def filename(self) -> str:
        return self.dataset.split("/")[-1]


class LocalSeedDatasetReference(SeedDatasetReference):
    @field_validator("dataset", mode="after")
    def validate_dataset_is_file(cls, v: str) -> str:
        return str(validate_dataset_file_path(v))
