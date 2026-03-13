# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from data_designer.engine.registry.errors import NotFoundInRegistryError, RegistryItemNotTypeError
from data_designer.engine.registry.handler import Handler, HandlerRegistry


class ExampleConfig(BaseModel):
    handler_type: str


class LocalExampleConfig(ExampleConfig):
    handler_type: Literal["local"] = "local"
    path: str


class RemoteExampleConfig(ExampleConfig):
    handler_type: Literal["remote"] = "remote"
    endpoint: str


class ExampleHandler(Handler[ExampleConfig]):
    def __init__(self, config: ExampleConfig) -> None:
        self.config = config


class LocalExampleHandler(Handler[LocalExampleConfig]):
    def __init__(self, config: LocalExampleConfig) -> None:
        self.config = config


class RemoteExampleHandler(Handler[RemoteExampleConfig]):
    def __init__(self, config: RemoteExampleConfig) -> None:
        self.config = config


def test_register_and_create_handler_from_config() -> None:
    registry = HandlerRegistry[ExampleConfig, ExampleHandler](
        discriminator_field="handler_type",
        handler_factory=lambda handler_type, config: handler_type(config),
    )
    registry.register(LocalExampleHandler)
    registry.register(RemoteExampleHandler)

    local_handler = registry.create_for_config(LocalExampleConfig(path="local.txt"))
    remote_handler = registry.create_for_config(RemoteExampleConfig(endpoint="https://example.com"))

    assert isinstance(local_handler, LocalExampleHandler)
    assert isinstance(remote_handler, RemoteExampleHandler)
    assert registry.get_registered_name(LocalExampleHandler) == "local"
    assert registry.get_registered_name(RemoteExampleHandler) == "remote"


def test_register_collision_raises() -> None:
    registry = HandlerRegistry[ExampleConfig, ExampleHandler](
        discriminator_field="handler_type",
        handler_factory=lambda handler_type, config: handler_type(config),
    )
    registry.register(LocalExampleHandler)

    with pytest.raises(ValueError, match="already exists"):
        registry.register(LocalExampleHandler)


def test_register_requires_class() -> None:
    registry = HandlerRegistry[ExampleConfig, ExampleHandler](
        discriminator_field="handler_type",
        handler_factory=lambda handler_type, config: handler_type(config),
    )

    with pytest.raises(RegistryItemNotTypeError, match="is not a class"):
        registry.register("not-a-class")  # type: ignore[arg-type]


def test_get_handler_type_missing_raises() -> None:
    registry = HandlerRegistry[ExampleConfig, ExampleHandler](
        discriminator_field="handler_type",
        handler_factory=lambda handler_type, config: handler_type(config),
    )

    with pytest.raises(NotFoundInRegistryError, match="No handler found"):
        registry.get_handler_type("missing")
