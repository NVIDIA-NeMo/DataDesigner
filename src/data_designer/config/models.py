# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self, TypeAlias

from .base import ConfigBase
from .errors import InvalidConfigError
from .utils.constants import (
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
    NVIDIA_API_KEY_ENV_VAR_NAME,
    NVIDIA_PROVIDER_NAME,
    OPENAI_API_KEY_ENV_VAR_NAME,
    OPENAI_PROVIDER_NAME,
)
from .utils.io_helpers import smart_load_yaml

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    IMAGE = "image"


class ModalityDataType(str, Enum):
    URL = "url"
    BASE64 = "base64"


class ImageFormat(str, Enum):
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"


class DistributionType(str, Enum):
    UNIFORM = "uniform"
    MANUAL = "manual"


class ModalityContext(ABC, BaseModel):
    modality: Modality
    column_name: str
    data_type: ModalityDataType

    @abstractmethod
    def get_context(self, record: dict) -> dict[str, Any]: ...


class ImageContext(ModalityContext):
    modality: Modality = Modality.IMAGE
    image_format: Optional[ImageFormat] = None

    def get_context(self, record: dict) -> dict[str, Any]:
        context = dict(type="image_url")
        context_value = record[self.column_name]
        if self.data_type == ModalityDataType.URL:
            context["image_url"] = context_value
        else:
            context["image_url"] = {
                "url": f"data:image/{self.image_format.value};base64,{context_value}",
                "format": self.image_format.value,
            }
        return context

    @model_validator(mode="after")
    def _validate_image_format(self) -> Self:
        if self.data_type == ModalityDataType.BASE64 and self.image_format is None:
            raise ValueError(f"image_format is required when data_type is {self.data_type.value}")
        return self


DistributionParamsT = TypeVar("DistributionParamsT", bound=ConfigBase)


class Distribution(ABC, ConfigBase, Generic[DistributionParamsT]):
    distribution_type: DistributionType
    params: DistributionParamsT

    @abstractmethod
    def sample(self) -> float: ...


class ManualDistributionParams(ConfigBase):
    values: List[float] = Field(min_length=1)
    weights: Optional[List[float]] = None

    @model_validator(mode="after")
    def _normalize_weights(self) -> Self:
        if self.weights is not None:
            self.weights = [w / sum(self.weights) for w in self.weights]
        return self

    @model_validator(mode="after")
    def _validate_equal_lengths(self) -> Self:
        if self.weights and len(self.values) != len(self.weights):
            raise ValueError("`values` and `weights` must have the same length")
        return self


class ManualDistribution(Distribution[ManualDistributionParams]):
    distribution_type: Optional[DistributionType] = "manual"
    params: ManualDistributionParams

    def sample(self) -> float:
        return float(np.random.choice(self.params.values, p=self.params.weights))


class UniformDistributionParams(ConfigBase):
    low: float
    high: float

    @model_validator(mode="after")
    def _validate_low_lt_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError("`low` must be less than `high`")
        return self


class UniformDistribution(Distribution[UniformDistributionParams]):
    distribution_type: Optional[DistributionType] = "uniform"
    params: UniformDistributionParams

    def sample(self) -> float:
        return float(np.random.uniform(low=self.params.low, high=self.params.high, size=1)[0])


DistributionT: TypeAlias = Union[UniformDistribution, ManualDistribution]


class InferenceParameters(ConfigBase):
    temperature: Optional[Union[float, DistributionT]] = None
    top_p: Optional[Union[float, DistributionT]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_parallel_requests: int = Field(default=4, ge=1)
    timeout: Optional[int] = Field(default=None, ge=1)
    extra_body: Optional[dict[str, Any]] = None

    @property
    def generate_kwargs(self) -> dict[str, Union[float, int]]:
        result = {}
        if self.temperature is not None:
            result["temperature"] = (
                self.temperature.sample() if hasattr(self.temperature, "sample") else self.temperature
            )
        if self.top_p is not None:
            result["top_p"] = self.top_p.sample() if hasattr(self.top_p, "sample") else self.top_p
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.extra_body is not None and self.extra_body != {}:
            result["extra_body"] = self.extra_body
        return result

    @model_validator(mode="after")
    def _validate_temperature(self) -> Self:
        return self._run_validation(
            value=self.temperature,
            param_name="temperature",
            min_value=MIN_TEMPERATURE,
            max_value=MAX_TEMPERATURE,
        )

    @model_validator(mode="after")
    def _validate_top_p(self) -> Self:
        return self._run_validation(
            value=self.top_p,
            param_name="top_p",
            min_value=MIN_TOP_P,
            max_value=MAX_TOP_P,
        )

    def _run_validation(
        self,
        value: Union[float, DistributionT, None],
        param_name: str,
        min_value: float,
        max_value: float,
    ) -> Self:
        if value is None:
            return self
        value_err = ValueError(f"{param_name} defined in model config must be between {min_value} and {max_value}")
        if isinstance(value, Distribution):
            if value.distribution_type == DistributionType.UNIFORM:
                if value.params.low < min_value or value.params.high > max_value:
                    raise value_err
            elif value.distribution_type == DistributionType.MANUAL:
                if any(not self._is_value_in_range(v, min_value, max_value) for v in value.params.values):
                    raise value_err
        else:
            if not self._is_value_in_range(value, min_value, max_value):
                raise value_err
        return self

    def _is_value_in_range(self, value: float, min_value: float, max_value: float) -> bool:
        return min_value <= value <= max_value


class ModelConfig(ConfigBase):
    alias: str
    model: str
    inference_parameters: InferenceParameters
    provider: Optional[str] = None


class ModelProvider(ConfigBase):
    name: str
    endpoint: str
    provider_type: str = "openai"
    api_key: Optional[str] = None
    extra_body: Optional[dict[str, Any]] = None


def load_model_configs(model_configs: Union[list[ModelConfig], str, Path]) -> list[ModelConfig]:
    if isinstance(model_configs, list) and all(isinstance(mc, ModelConfig) for mc in model_configs):
        return model_configs
    json_config = smart_load_yaml(model_configs)
    if "model_configs" not in json_config:
        raise InvalidConfigError(
            "The list of model configs must be provided under model_configs in the configuration file."
        )
    return [ModelConfig.model_validate(mc) for mc in json_config["model_configs"]]


def get_default_text_alias_inference_parameters() -> InferenceParameters:
    return InferenceParameters(
        temperature=0.85,
        top_p=0.95,
    )


def get_default_reasoning_alias_inference_parameters() -> InferenceParameters:
    return InferenceParameters(
        temperature=0.35,
        top_p=0.95,
    )


def get_default_vision_alias_inference_parameters() -> InferenceParameters:
    return InferenceParameters(
        temperature=0.85,
        top_p=0.95,
    )


def get_default_nvidia_model_configs() -> list[ModelConfig]:
    if not get_nvidia_api_key():
        logger.warning(
            f"🔑 {NVIDIA_API_KEY_ENV_VAR_NAME!r} environment variable is not set. Please set it to your API key from 'https://build.nvidia.com' if you want to use the default NVIDIA model configs."
        )
        return []
    return [
        ModelConfig(
            alias=f"{NVIDIA_PROVIDER_NAME}-text",
            model="nvidia/nvidia-nemotron-nano-9b-v2",
            provider=NVIDIA_PROVIDER_NAME,
            inference_parameters=get_default_text_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{NVIDIA_PROVIDER_NAME}-reasoning",
            model="openai/gpt-oss-20b",
            provider=NVIDIA_PROVIDER_NAME,
            inference_parameters=get_default_reasoning_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{NVIDIA_PROVIDER_NAME}-vision",
            model="nvidia/nemotron-nano-12b-v2-vl",
            provider=NVIDIA_PROVIDER_NAME,
            inference_parameters=get_default_vision_alias_inference_parameters(),
        ),
    ]


def get_default_openai_model_configs() -> list[ModelConfig]:
    if not get_openai_api_key():
        logger.warning(
            f"🔑 {OPENAI_API_KEY_ENV_VAR_NAME!r} environment variable is not set. Please set it to your API key from 'https://platform.openai.com/api-keys' if you want to use the default OpenAI model configs."
        )
        return []
    return [
        ModelConfig(
            alias=f"{OPENAI_PROVIDER_NAME}-text",
            model="gpt-4.1",
            provider=OPENAI_PROVIDER_NAME,
            inference_parameters=get_default_text_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{OPENAI_PROVIDER_NAME}-reasoning",
            model="gpt-5",
            provider=OPENAI_PROVIDER_NAME,
            inference_parameters=get_default_reasoning_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{OPENAI_PROVIDER_NAME}-vision",
            model="gpt-5",
            provider=OPENAI_PROVIDER_NAME,
            inference_parameters=get_default_vision_alias_inference_parameters(),
        ),
    ]


def get_default_model_configs() -> list[ModelConfig]:
    return get_default_nvidia_model_configs() + get_default_openai_model_configs()


def get_default_providers() -> list[ModelProvider]:
    return [
        ModelProvider(
            name=NVIDIA_PROVIDER_NAME,
            endpoint="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY_ENV_VAR_NAME,
        ),
        ModelProvider(
            name=OPENAI_PROVIDER_NAME,
            endpoint="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY_ENV_VAR_NAME,
        ),
    ]


def get_nvidia_api_key() -> Optional[str]:
    return os.getenv(NVIDIA_API_KEY_ENV_VAR_NAME)


def get_openai_api_key() -> Optional[str]:
    return os.getenv(OPENAI_API_KEY_ENV_VAR_NAME)
