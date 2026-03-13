# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel
from typing_extensions import Self

from data_designer.engine.registry.errors import NotFoundInRegistryError, RegistryItemNotTypeError

ConfigT = TypeVar("ConfigT", bound=BaseModel)
HandlerT = TypeVar("HandlerT", bound="Handler[BaseModel]")


class Handler(ABC, Generic[ConfigT]):
    """Base class for lightweight handlers selected by a config discriminator."""

    @classmethod
    def get_config_type(cls) -> type[ConfigT]:
        for base in getattr(cls, "__orig_bases__", []):
            origin = get_origin(base)
            if not isinstance(origin, type) or not issubclass(origin, Handler):
                continue

            args = get_args(base)
            if not args:
                continue

            config_type = args[0]
            origin_type = get_origin(config_type) or config_type
            if isinstance(origin_type, type) and issubclass(origin_type, BaseModel):
                return config_type

        raise TypeError(
            f"Could not determine config type for {cls.__name__!r}. "
            "Please ensure the handler is defined with a generic config type argument."
        )

    @classmethod
    def get_registered_name(cls, discriminator_field: str) -> str:
        config_type = cls.get_config_type()
        field = config_type.model_fields.get(discriminator_field)
        if field is None:
            raise ValueError(
                f"Config type {config_type.__name__!r} does not define discriminator field {discriminator_field!r}."
            )
        if not isinstance(field.default, str):
            raise ValueError(
                f"Config type {config_type.__name__!r} must define a string default for {discriminator_field!r}."
            )
        return field.default


class HandlerRegistry(Generic[ConfigT, HandlerT]):
    """Instance-scoped registry for lightweight config-driven handlers."""

    def __init__(
        self,
        *,
        discriminator_field: str,
        handler_factory: Callable[[type[HandlerT], ConfigT], HandlerT],
        error_type: type[Exception] = ValueError,
        handler_label: str = "handler",
        handlers: Sequence[type[HandlerT]] | None = None,
    ) -> None:
        self._discriminator_field = discriminator_field
        self._handler_factory = handler_factory
        self._error_type = error_type
        self._handler_label = handler_label
        self._handlers: dict[str, type[HandlerT]] = {}
        self._reverse_handlers: dict[type[HandlerT], str] = {}

        for handler_type in handlers or []:
            self.register(handler_type)

    def register(self, handler_type: type[HandlerT], raise_on_collision: bool = True) -> Self:
        self._raise_if_not_type(handler_type)
        if not issubclass(handler_type, Handler):
            raise self._error_type(f"{handler_type!r} is not a subclass of Handler")

        try:
            registered_name = handler_type.get_registered_name(self._discriminator_field)
        except (TypeError, ValueError) as error:
            raise self._error_type(str(error)) from error
        if registered_name in self._handlers:
            if not raise_on_collision:
                return self
            raise self._error_type(
                f"A {self._handler_label} for {self._discriminator_field} {registered_name!r} already exists"
            )

        self._handlers[registered_name] = handler_type
        self._reverse_handlers[handler_type] = registered_name
        return self

    def has_registered_name(self, registered_name: str) -> bool:
        return registered_name in self._handlers

    def create_for_config(self, config: ConfigT) -> HandlerT:
        return self._handler_factory(self.get_handler_type_for_config(config), config)

    def get_handler_type(self, registered_name: str) -> type[HandlerT]:
        try:
            return self._handlers[registered_name]
        except KeyError as error:
            raise NotFoundInRegistryError(
                f"No {self._handler_label} found for {self._discriminator_field} {registered_name!r}"
            ) from error

    def get_handler_type_for_config(self, config: ConfigT) -> type[HandlerT]:
        return self.get_handler_type(self.get_name_for_config(config))

    def get_name_for_config(self, config: ConfigT) -> str:
        registered_name = getattr(config, self._discriminator_field, None)
        if not isinstance(registered_name, str):
            raise self._error_type(
                f"Config {type(config).__name__!r} must define a string {self._discriminator_field!r} field."
            )
        return registered_name

    def get_registered_name(self, handler_type: type[HandlerT]) -> str:
        try:
            return self._reverse_handlers[handler_type]
        except KeyError as error:
            raise NotFoundInRegistryError(f"{handler_type} not found in registry") from error

    @staticmethod
    def _raise_if_not_type(obj: object) -> None:
        if not isinstance(obj, type):
            raise RegistryItemNotTypeError(f"{obj} is not a class!")
