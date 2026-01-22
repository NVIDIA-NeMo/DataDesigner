# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, get_origin

from data_designer.config.base import ConfigBase
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

DataT = TypeVar("DataT", dict, pd.DataFrame)
TaskConfigT = TypeVar("ConfigT", bound=ConfigBase)


class ConfigurableTask(ABC, Generic[TaskConfigT]):
    def __init__(self, config: TaskConfigT, resource_provider: ResourceProvider):
        self._config = self.get_config_type().model_validate(config)
        self._resource_provider = resource_provider
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
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def resource_provider(self) -> ResourceProvider:
        return self._resource_provider

    def _initialize(self) -> None:
        """An internal method for custom initialization logic, which will be called in the constructor."""

    def _validate(self) -> None:
        """An internal method for custom validation logic, which will be called in the constructor."""
