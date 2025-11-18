# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Any, Generic, Type, TypeVar

from data_designer.config.base import ConfigBase
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.engine.registry.errors import NotFoundInRegistryError, RegistryItemNotTypeError

EnumNameT = TypeVar("EnumNameT", bound=StrEnum)
TaskT = TypeVar("TaskT", bound=ConfigurableTask)
TaskConfigT = TypeVar("TaskConfigT", bound=ConfigBase)


class TaskRegistry(Generic[EnumNameT, TaskT, TaskConfigT]):
    # registered type name -> type
    _registry: dict[EnumNameT, Type[TaskT]] = {}
    # type -> registered type name
    _reverse_registry: dict[Type[TaskT], EnumNameT] = {}

    # registered type name -> config type
    _config_registry: dict[EnumNameT, Type[TaskConfigT]] = {}
    # config type -> registered type name
    _reverse_config_registry: dict[Type[TaskConfigT], EnumNameT] = {}

    # all registries are singletons
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def register(
        cls,
        name: EnumNameT,
        task: Type[TaskT],
        config: Type[TaskConfigT],
        raise_on_collision: bool = False,
    ) -> None:
        if cls._has_been_registered(name):
            if not raise_on_collision:
                return
            raise ValueError(f"{name} has already been registered!")

        cls._raise_if_not_type(task)
        cls._raise_if_not_type(config)

        with cls._lock:
            cls._registry[name] = task
            cls._reverse_registry[task] = name
            cls._config_registry[name] = config
            cls._reverse_config_registry[config] = name

    @classmethod
    def get_task_type(cls, name: EnumNameT) -> Type[TaskT]:
        cls._raise_if_not_registered(name, cls._registry)
        return cls._registry[name]

    @classmethod
    def get_config_type(cls, name: EnumNameT) -> Type[TaskConfigT]:
        cls._raise_if_not_registered(name, cls._config_registry)
        return cls._config_registry[name]

    @classmethod
    def get_registered_name(cls, task: Type[TaskT]) -> EnumNameT:
        cls._raise_if_not_registered(task, cls._reverse_registry)
        return cls._reverse_registry[task]

    @classmethod
    def get_for_config_type(cls, config: Type[TaskConfigT]) -> Type[TaskT]:
        cls._raise_if_not_registered(config, cls._reverse_config_registry)
        name = cls._reverse_config_registry[config]
        return cls.get_task_type(name)

    @classmethod
    def _has_been_registered(cls, name: EnumNameT) -> bool:
        return name in cls._registry

    @classmethod
    def _raise_if_not_registered(cls, key: EnumNameT | Type[TaskT] | Type[TaskConfigT], mapping: dict) -> None:
        if not (isinstance(key, StrEnum) or isinstance(key, str)):
            cls._raise_if_not_type(key)
        if key not in mapping:
            raise NotFoundInRegistryError(f"{key} not found in registry")

    @classmethod
    def _raise_if_not_type(cls, obj: Any) -> None:
        if not isinstance(obj, type):
            raise RegistryItemNotTypeError(f"{obj} is not a class!")

    def __new__(cls, *args, **kwargs):
        """Registry is a singleton."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
