# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib
import importlib.util
from enum import Enum
from functools import cached_property
from typing import Literal, get_origin

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.plugins.errors import PluginLoadError


class PluginType(str, Enum):
    COLUMN_GENERATOR = "column-generator"
    SEED_READER = "seed-reader"

    @property
    def discriminator_field(self) -> str:
        if self == PluginType.COLUMN_GENERATOR:
            return "column_type"
        elif self == PluginType.SEED_READER:
            return "seed_type"
        else:
            raise ValueError(f"Invalid plugin type: {self.value}")

    @property
    def display_name(self) -> str:
        return self.value.replace("-", " ")


def _get_module_and_object_names(fully_qualified_object: str) -> tuple[str, str]:
    try:
        module_name, object_name = fully_qualified_object.rsplit(".", 1)
    except ValueError:
        # If fully_qualified_object does not have any periods, the rsplit call will return
        # a list of length 1 and the variable assignment above will raise ValueError
        raise PluginLoadError("Expected a fully-qualified object name, e.g. 'my_plugin.config.MyConfig'")

    return module_name, object_name


def _check_class_exists_in_file(filepath: str, class_name: str) -> None:
    try:
        with open(filepath, "r") as file:
            source = file.read()
    except FileNotFoundError:
        raise PluginLoadError(f"Could not read source code at {filepath!r}")

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return None

    raise PluginLoadError(f"Could not find class named {class_name!r} in {filepath!r}")


class Plugin(BaseModel):
    impl_qualified_name: str = Field(
        ...,
        description="The fully-qualified name of the implementation class object, e.g. 'my_plugin.generator.MyColumnGenerator'",
    )
    config_qualified_name: str = Field(
        ..., description="The fully-qualified name o the config class object, e.g. 'my_plugin.config.MyConfig'"
    )
    plugin_type: PluginType = Field(..., description="The type of plugin")

    @property
    def config_type_as_class_name(self) -> str:
        return self.enum_key_name.title().replace("_", "")

    @property
    def enum_key_name(self) -> str:
        return self.name.replace("-", "_").upper()

    @property
    def name(self) -> str:
        return self.config_cls.model_fields[self.discriminator_field].default

    @property
    def discriminator_field(self) -> str:
        return self.plugin_type.discriminator_field

    @field_validator("impl_qualified_name", "config_qualified_name", mode="after")
    @classmethod
    def validate_class_name(cls, value: str) -> str:
        module_name, object_name = _get_module_and_object_names(value)
        try:
            spec = importlib.util.find_spec(module_name)
        except:
            raise PluginLoadError(f"Could not find module {module_name!r}")

        if spec is None or spec.origin is None:
            raise PluginLoadError(f"Error finding source for module {module_name!r}")

        _check_class_exists_in_file(spec.origin, object_name)

        return value

    @model_validator(mode="after")
    def validate_discriminator_field(self) -> Self:
        _, cfg = _get_module_and_object_names(self.config_qualified_name)
        field = self.plugin_type.discriminator_field
        if field not in self.config_cls.model_fields:
            raise ValueError(f"Discriminator field {field!r} not found in config class {cfg!r}")
        field_info = self.config_cls.model_fields[field]
        if get_origin(field_info.annotation) is not Literal:
            raise ValueError(f"Field {field!r} of {cfg!r} must be a Literal type, not {field_info.annotation!r}.")
        if not isinstance(field_info.default, str):
            raise ValueError(f"The default of {field!r} must be a string, not {type(field_info.default)!r}.")
        enum_key = field_info.default.replace("-", "_").upper()
        if not enum_key.isidentifier():
            raise ValueError(
                f"The default value {field_info.default!r} for discriminator field {field!r} "
                f"cannot be converted to a valid enum key. The converted key {enum_key!r} "
                f"must be a valid Python identifier."
            )
        return self

    @cached_property
    def config_cls(self) -> type[ConfigBase]:
        return self._load(self.config_qualified_name)

    @cached_property
    def impl_cls(self) -> type:
        return self._load(self.impl_qualified_name)

    @staticmethod
    def _load(fully_qualified_object: str) -> type:
        module_name, object_name = _get_module_and_object_names(fully_qualified_object)
        module = importlib.import_module(module_name)
        try:
            return getattr(module, object_name)
        except AttributeError:
            raise PluginLoadError(f"Could not find class {object_name!r} in module {module_name!r}")
