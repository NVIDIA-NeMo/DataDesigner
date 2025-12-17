# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, get_origin

import pandas as pd

from data_designer.config.base import ConfigBase
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.resources.resource_provider import ResourceProvider, ResourceType

DataT = TypeVar("DataT", dict, pd.DataFrame)
TaskConfigT = TypeVar("ConfigT", bound=ConfigBase)


class ConfigurableTaskMetadata(ConfigBase):
    name: str
    description: str
    required_resources: list[ResourceType] | None


class ConfigurableTask(ABC, Generic[TaskConfigT]):
    def __init__(self, config: TaskConfigT, *, resource_provider: ResourceProvider | None):
        self._config = self.get_config_type().model_validate(config)
        self._resource_provider = resource_provider
        self._validate_resources()
        self._validate()
        self._initialize()

    @classmethod
    def get_config_type(cls) -> type[TaskConfigT]:
        for base in cls.__orig_bases__:
            if hasattr(base, "__args__") and len(base.__args__) == 1:
                arg = base.__args__[0]
                origin = get_origin(arg) or arg
                if isinstance(origin, type) and issubclass(origin, ConfigBase):
                    return base.__args__[0]
        raise TypeError(
            f"Could not determine config type for `{cls.__name__}`. Please ensure that the "
            "`ConfigurableTask` is defined with a generic type argument, where the type argument "
            "is a subclass of `ConfigBase`."
        )

    @property
    def artifact_path(self) -> Path:
        return self.artifact_storage.artifact_path

    @property
    def artifact_storage(self) -> ArtifactStorage:
        return self.resource_provider.artifact_storage

    @property
    def base_dataset_path(self) -> Path:
        return self.artifact_storage.base_dataset_path

    @property
    def config(self) -> TaskConfigT:
        return self._config

    @property
    def resource_provider(self) -> ResourceProvider:
        if self._resource_provider is None:
            raise ValueError(f"No resource provider provided for the `{self.metadata().name}` task.")
        return self._resource_provider

    @staticmethod
    @abstractmethod
    def metadata() -> ConfigurableTaskMetadata: ...

    def _initialize(self) -> None:
        """An internal method for custom initialization logic, which will be called in the constructor."""

    def _validate(self) -> None:
        """An internal method for custom validation logic, which will be called in the constructor."""

    def _validate_resources(self) -> None:
        for resource in self.metadata().required_resources or []:
            if resource is not None:
                if getattr(self.resource_provider, ResourceType(resource).value) is None:
                    raise ValueError(f"Resource {resource} is required for the `{self.metadata().name}`")
