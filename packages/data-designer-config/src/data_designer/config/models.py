# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator
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
from data_designer.lazy_heavy_imports import np

if TYPE_CHECKING:
    import numpy as np

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
    def get_contexts(self, record: dict) -> list[dict[str, Any]]: ...


class ImageContext(ModalityContext):
    """Configuration for providing image context to multimodal models.

    Attributes:
        modality: The modality type (always "image").
        column_name: Name of the column containing image data.
        data_type: Format of the image data ("url" or "base64").
        image_format: Image format (required for base64 data).
    """

    modality: Modality = Modality.IMAGE
    image_format: ImageFormat | None = None

    def get_contexts(self, record: dict) -> list[dict[str, Any]]:
        """Get the contexts for the image modality.

        Args:
            record: The record containing the image data. The data can be:
                - A JSON serialized list of strings
                - A list of strings
                - A single string

        Returns:
            A list of image contexts.
        """
        raw_value = record[self.column_name]

        # Normalize to list of strings
        if isinstance(raw_value, str):
            # Try to parse as JSON first
            try:
                parsed_value = json.loads(raw_value)
                if isinstance(parsed_value, list):
                    context_values = parsed_value
                else:
                    context_values = [raw_value]
            except (json.JSONDecodeError, TypeError):
                context_values = [raw_value]
        elif isinstance(raw_value, list):
            context_values = raw_value
        elif hasattr(raw_value, "__iter__") and not isinstance(raw_value, (str, bytes, dict)):
            # Handle array-like objects (numpy arrays, pandas Series, etc.)
            context_values = list(raw_value)
        else:
            context_values = [raw_value]

        # Build context list
        contexts = []
        for context_value in context_values:
            context = dict(type="image_url")
            if self.data_type == ModalityDataType.URL:
                context["image_url"] = context_value
            else:
                context["image_url"] = {
                    "url": f"data:image/{self.image_format.value};base64,{context_value}",
                    "format": self.image_format.value,
                }
            contexts.append(context)

        return contexts

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

    values: list[float] = Field(min_length=1)
    weights: list[float] | None = None

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

    distribution_type: DistributionType | None = "manual"
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

    distribution_type: DistributionType | None = "uniform"
    params: UniformDistributionParams

    def sample(self) -> float:
        """Sample a value from the uniform distribution.

        Returns:
            A float value sampled from the uniform distribution.
        """
        return float(np.random.uniform(low=self.params.low, high=self.params.high, size=1)[0])


DistributionT: TypeAlias = UniformDistribution | ManualDistribution


class GenerationType(str, Enum):
    CHAT_COMPLETION = "chat-completion"
    EMBEDDING = "embedding"


class BaseInferenceParams(ConfigBase, ABC):
    """Base configuration for inference parameters.

    Attributes:
        generation_type: Type of generation (chat-completion or embedding). Acts as discriminator.
        max_parallel_requests: Maximum number of parallel requests to the model API.
        timeout: Timeout in seconds for each request.
        extra_body: Additional parameters to pass to the model API.
    """

    generation_type: GenerationType
    max_parallel_requests: int = Field(default=4, ge=1)
    timeout: int | None = Field(default=None, ge=1)
    extra_body: dict[str, Any] | None = None

    @property
    def generate_kwargs(self) -> dict[str, Any]:
        """Get the generate kwargs for the inference parameters.

        Returns:
            A dictionary of the generate kwargs.
        """
        result = {}
        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.extra_body is not None and self.extra_body != {}:
            result["extra_body"] = self.extra_body
        return result

    def format_for_display(self) -> str:
        """Format inference parameters for display.

        Returns:
            Formatted string of inference parameters
        """
        params_dict = self.model_dump(exclude_none=True, mode="json")

        if not params_dict:
            return "(none)"

        parts = []
        for key, value in params_dict.items():
            formatted_value = self._format_value(key, value)
            parts.append(f"{key}={formatted_value}")
        return ", ".join(parts)

    def _format_value(self, key: str, value: Any) -> str:
        """Format a single parameter value. Override in subclasses for custom formatting.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            Formatted string representation of the value
        """
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)


class ChatCompletionInferenceParams(BaseInferenceParams):
    """Configuration for LLM inference parameters.

    Attributes:
        generation_type: Type of generation, always "chat-completion" for this class.
        temperature: Sampling temperature (0.0-2.0). Can be a fixed value or a distribution for dynamic sampling.
        top_p: Nucleus sampling probability (0.0-1.0). Can be a fixed value or a distribution for dynamic sampling.
        max_tokens: Maximum number of tokens to generate in the response.
    """

    generation_type: Literal[GenerationType.CHAT_COMPLETION] = GenerationType.CHAT_COMPLETION
    temperature: float | DistributionT | None = None
    top_p: float | DistributionT | None = None
    max_tokens: int | None = Field(default=None, ge=1)

    @property
    def generate_kwargs(self) -> dict[str, Any]:
        result = super().generate_kwargs
        if self.temperature is not None:
            result["temperature"] = (
                self.temperature.sample() if hasattr(self.temperature, "sample") else self.temperature
            )
        if self.top_p is not None:
            result["top_p"] = self.top_p.sample() if hasattr(self.top_p, "sample") else self.top_p
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
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
        value: float | DistributionT | None,
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

    def _format_value(self, key: str, value: Any) -> str:
        """Format chat completion parameter values, including distributions.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            Formatted string representation of the value
        """
        if isinstance(value, dict) and "distribution_type" in value:
            return "dist"
        return super()._format_value(key, value)


class EmbeddingInferenceParams(BaseInferenceParams):
    """Configuration for embedding generation parameters.

    Attributes:
        generation_type: Type of generation, always "embedding" for this class.
        encoding_format: Format of the embedding encoding ("float" or "base64").
        dimensions: Number of dimensions for the embedding.
    """

    generation_type: Literal[GenerationType.EMBEDDING] = GenerationType.EMBEDDING
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None

    @property
    def generate_kwargs(self) -> dict[str, float | int]:
        result = super().generate_kwargs
        if self.encoding_format is not None:
            result["encoding_format"] = self.encoding_format
        if self.dimensions is not None:
            result["dimensions"] = self.dimensions
        return result


InferenceParamsT: TypeAlias = Annotated[
    ChatCompletionInferenceParams | EmbeddingInferenceParams, Field(discriminator="generation_type")
]


class ModelConfig(ConfigBase):
    """Configuration for a model used for generation.

    Attributes:
        alias: User-defined alias to reference in column configurations.
        model: Model identifier (e.g., from build.nvidia.com or other providers).
        inference_parameters: Inference parameters for the model (temperature, top_p, max_tokens, etc.).
            The generation_type is determined by the type of inference_parameters.
        provider: Optional model provider name if using custom providers.
        skip_health_check: Whether to skip the health check for this model. Defaults to False.
    """

    alias: str
    model: str
    inference_parameters: InferenceParamsT = Field(default_factory=ChatCompletionInferenceParams)
    provider: str | None = None
    skip_health_check: bool = False

    @property
    def generation_type(self) -> GenerationType:
        """Get the generation type from the inference parameters."""
        return self.inference_parameters.generation_type

    @field_validator("inference_parameters", mode="before")
    @classmethod
    def _convert_inference_parameters(cls, value: Any) -> Any:
        """Convert raw dict to appropriate inference parameters type based on field presence."""
        if isinstance(value, dict):
            # Infer type from presence of embedding-specific fields
            if "encoding_format" in value or "dimensions" in value:
                return EmbeddingInferenceParams(**value)
            else:
                return ChatCompletionInferenceParams(**value)
        return value


class ModelProvider(ConfigBase):
    """Configuration for a custom model provider.

    Attributes:
        name: Name of the model provider.
        endpoint: API endpoint URL for the provider.
        provider_type: Provider type (default: "openai"). Determines the API format to use.
        api_key: Optional API key for authentication.
        extra_body: Additional parameters to pass in API requests.
        extra_headers: Additional headers to pass in API requests.
    """

    name: str
    endpoint: str
    provider_type: str = "openai"
    api_key: str | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None


def load_model_configs(model_configs: list[ModelConfig] | str | Path) -> list[ModelConfig]:
    if isinstance(model_configs, list) and all(isinstance(mc, ModelConfig) for mc in model_configs):
        return model_configs
    json_config = smart_load_yaml(model_configs)
    if "model_configs" not in json_config:
        raise InvalidConfigError(
            "The list of model configs must be provided under model_configs in the configuration file."
        )
    return [ModelConfig.model_validate(mc) for mc in json_config["model_configs"]]
