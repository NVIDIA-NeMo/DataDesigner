# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, TypeVar

from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.type_helpers import get_sampler_params
from data_designer.config.utils.visualization import (
    display_model_configs_table,
    display_model_providers_table,
    display_sampler_table,
)


class InfoType(str, Enum):
    SAMPLERS = "samplers"
    MODEL_CONFIGS = "model_configs"
    MODEL_PROVIDERS = "model_providers"


ConfigBuilderInfoType = Literal[InfoType.SAMPLERS, InfoType.MODEL_CONFIGS]
DataDesignerInfoType = Literal[InfoType.MODEL_PROVIDERS]
InfoTypeT = TypeVar("InfoTypeT", bound=InfoType)


class InfoDisplay(ABC):
    """Base class for info display classes that provide type-safe display methods."""

    @abstractmethod
    def display(self, info_type: InfoTypeT, **kwargs) -> None:
        """Display information based on the provided info type.

        Args:
            info_type: Type of information to display.
        """
        ...


class ConfigBuilderInfo(InfoDisplay):
    def __init__(self, model_configs: list[ModelConfig]):
        self._sampler_params = get_sampler_params()
        self._model_configs = model_configs

    def display(self, info_type: ConfigBuilderInfoType, **kwargs) -> None:
        """Display information based on the provided info type.

        Args:
            info_type: Type of information to display. Only SAMPLERS and MODEL_CONFIGS are supported.

        Raises:
            ValueError: If an unsupported info_type is provided.
        """
        if info_type == InfoType.SAMPLERS:
            self._display_sampler_info(sampler_type=kwargs.get("sampler_type"))
        elif info_type == InfoType.MODEL_CONFIGS:
            display_model_configs_table(self._model_configs)
        else:
            raise ValueError(
                f"Unsupported info_type: {str(info_type)!r}. "
                f"ConfigBuilderInfo only supports {InfoType.SAMPLERS.value!r} and {InfoType.MODEL_CONFIGS.value!r}."
            )

    def _display_sampler_info(self, sampler_type: SamplerType | None) -> None:
        if sampler_type is not None:
            title = f"{SamplerType(sampler_type).value.replace('_', ' ').title()} Sampler"
            display_sampler_table({sampler_type: self._sampler_params[sampler_type]}, title=title)
        else:
            display_sampler_table(self._sampler_params)


class InterfaceInfo(InfoDisplay):
    def __init__(self, model_providers: list[ModelProvider]):
        self._model_providers = model_providers

    def display(self, info_type: DataDesignerInfoType, **kwargs) -> None:
        """Display information based on the provided info type.

        Args:
            info_type: Type of information to display. Only MODEL_PROVIDERS is supported.

        Raises:
            ValueError: If an unsupported info_type is provided.
        """
        if info_type == InfoType.MODEL_PROVIDERS:
            display_model_providers_table(self._model_providers)
        else:
            raise ValueError(
                f"Unsupported info_type: {str(info_type)!r}. InterfaceInfo only supports {InfoType.MODEL_PROVIDERS.value!r}."
            )
