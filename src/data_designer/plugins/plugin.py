# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal, get_origin

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.engine.configurable_task import ConfigurableTask


class PluginType(str, Enum):
    COLUMN_GENERATOR = "column-generator"

    @property
    def discriminator_field(self) -> str:
        if self == PluginType.COLUMN_GENERATOR:
            return "column_type"
        else:
            raise ValueError(f"Invalid plugin type: {self.value}")

    @property
    def display_name(self) -> str:
        return self.value.replace("-", " ")


class Plugin(BaseModel):
    task_cls: type[ConfigurableTask]
    config_cls: type[ConfigBase]
    plugin_type: PluginType
    emoji: str = "🔌"

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

    @model_validator(mode="after")
    def validate_discriminator_field(self) -> Self:
        cfg = self.config_cls.__name__
        field = self.plugin_type.discriminator_field
        if field not in self.config_cls.model_fields:
            raise ValueError(f"Discriminator field '{field}' not found in config class {cfg}")
        field_info = self.config_cls.model_fields[field]
        if get_origin(field_info.annotation) is not Literal:
            raise ValueError(f"Field '{field}' of {cfg} must be a Literal type, not {field_info.annotation}.")
        if not isinstance(field_info.default, str):
            raise ValueError(f"The default of '{field}' must be a string, not {type(field_info.default)}.")
        enum_key = field_info.default.replace("-", "_").upper()
        if not enum_key.isidentifier():
            raise ValueError(
                f"The default value '{field_info.default}' for discriminator field '{field}' "
                f"cannot be converted to a valid enum key. The converted key '{enum_key}' "
                f"must be a valid Python identifier."
            )
        return self
