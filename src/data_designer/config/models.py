# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self, TypeAlias

from data_designer.config.base import ConfigBase
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.constants import (
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
)
from data_designer.config.utils.io_helpers import smart_load_yaml

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported modality types for multimodal model data."""

    IMAGE = "image"


class ModalityDataType(str, Enum):
    """Data type formats for multimodal data."""

    URL = "url"
    BASE64 = "base64"


class ImageFormat(str, Enum):
    """Supported image formats for image modality."""

    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"


class DistributionType(str, Enum):
    """Types of distributions for sampling inference parameters."""

    UNIFORM = "uniform"
    MANUAL = "manual"


class ModalityContext(ABC, BaseModel):
    modality: Modality
    column_name: str
    data_type: ModalityDataType

    @abstractmethod
    def get_context(self, record: dict) -> dict[str, Any]: ...


class ImageContext(ModalityContext):
    """Configuration for providing image context to multimodal models.

    Attributes:
        modality: The modality type (always "image").
        column_name: Name of the column containing image data.
        data_type: Format of the image data ("url" or "base64").
        image_format: Image format (required for base64 data).
    """

    modality: Modality = Modality.IMAGE
    image_format: Optional[ImageFormat] = None

    def get_context(self, record: dict) -> dict[str, Any]:
        """Get the context for the image modality.

        Args:
            record: The record containing the image data.

        Returns:
            The context for the image modality.
        """
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
    """Parameters for manual distribution sampling.

    Attributes:
        values: List of possible values to sample from.
        weights: Optional list of weights for each value. If not provided, all values have equal probability.
    """

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
    """Manual (discrete) distribution for sampling inference parameters.

    Samples from a discrete set of values with optional weights. Useful for testing
    specific values or creating custom probability distributions for temperature or top_p.

    Attributes:
        distribution_type: Type of distribution ("manual").
        params: Distribution parameters (values, weights).
    """

    distribution_type: Optional[DistributionType] = "manual"
    params: ManualDistributionParams

    def sample(self) -> float:
        """Sample a value from the manual distribution.

        Returns:
            A float value sampled from the manual distribution.
        """
        return float(np.random.choice(self.params.values, p=self.params.weights))


class UniformDistributionParams(ConfigBase):
    """Parameters for uniform distribution sampling.

    Attributes:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
    """

    low: float
    high: float

    @model_validator(mode="after")
    def _validate_low_lt_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError("`low` must be less than `high`")
        return self


class UniformDistribution(Distribution[UniformDistributionParams]):
    """Uniform distribution for sampling inference parameters.

    Samples values uniformly between low and high bounds. Useful for exploring
    a continuous range of values for temperature or top_p.

    Attributes:
        distribution_type: Type of distribution ("uniform").
        params: Distribution parameters (low, high).
    """

    distribution_type: Optional[DistributionType] = "uniform"
    params: UniformDistributionParams

    def sample(self) -> float:
        """Sample a value from the uniform distribution.

        Returns:
            A float value sampled from the uniform distribution.
        """
        return float(np.random.uniform(low=self.params.low, high=self.params.high, size=1)[0])


DistributionT: TypeAlias = Union[UniformDistribution, ManualDistribution]


class InferenceParameters(ConfigBase):
    """Configuration for LLM inference parameters.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Can be a fixed value or a distribution for dynamic sampling.
        top_p: Nucleus sampling probability (0.0-1.0). Can be a fixed value or a distribution for dynamic sampling.
        max_tokens: Maximum number of tokens (includes both input and output tokens).
        max_parallel_requests: Maximum number of parallel requests to the model API.
        timeout: Timeout in seconds for each request.
        extra_body: Additional parameters to pass to the model API.
    """

    temperature: Optional[Union[float, DistributionT]] = None
    top_p: Optional[Union[float, DistributionT]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_parallel_requests: int = Field(default=4, ge=1)
    timeout: Optional[int] = Field(default=None, ge=1)
    extra_body: Optional[dict[str, Any]] = None

    @property
    def generate_kwargs(self) -> dict[str, Union[float, int]]:
        """Get the generate kwargs for the inference parameters.

        Returns:
            A dictionary of the generate kwargs.
        """
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
    """Configuration for a model used for generation.

    Attributes:
        alias: User-defined alias to reference in column configurations.
        model: Model identifier (e.g., from build.nvidia.com or other providers).
        inference_parameters: Inference parameters for the model (temperature, top_p, max_tokens, etc.).
        provider: Optional model provider name if using custom providers.
    """

    alias: str
    model: str
    inference_parameters: InferenceParameters = Field(default_factory=InferenceParameters)
    provider: Optional[str] = None


class ModelProvider(ConfigBase):
    """Configuration for a custom model provider.

    Attributes:
        name: Name of the model provider.
        endpoint: API endpoint URL for the provider.
        provider_type: Provider type (default: "openai"). Determines the API format to use.
        api_key: Optional API key for authentication.
        extra_body: Additional parameters to pass in API requests.
    """

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
