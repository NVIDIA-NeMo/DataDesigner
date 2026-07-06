# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self, TypeAlias

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import ConfigBase
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.constants import (
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
)
from data_designer.config.utils.io_helpers import smart_load_yaml
from data_designer.config.utils.media_helpers import (
    AudioFormat,
    ImageFormat,
    VideoFormat,
    audio_format_from_mime_type,
    audio_mime_type,
    decode_base64_image,
    detect_image_format,
    get_media_base64_context,
    get_media_url_context,
    image_format_from_mime_type,
    is_audio_path,
    is_image_path,
    is_image_url,
    is_media_url,
    is_video_path,
    load_image_path_to_base64,
    normalize_media_context_values,
    parse_base64_data_uri,
    video_format_from_mime_type,
    video_mime_type,
)

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported modality types for multimodal model data."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ModalityDataType(str, Enum):
    """Data type formats for multimodal data."""

    URL = "url"
    BASE64 = "base64"


class DistributionType(str, Enum):
    """Types of distributions for sampling inference parameters."""

    UNIFORM = "uniform"
    MANUAL = "manual"


class ModalityContext(ABC, BaseModel):
    modality: Modality
    column_name: str
    data_type: ModalityDataType | None = None

    @abstractmethod
    def get_contexts(self, record: dict, *, base_path: str | None = None) -> list[dict[str, Any]]: ...


class ImageContext(ModalityContext):
    """Configuration for providing image context to multimodal models.

    Attributes:
        modality: The modality type (always "image").
        column_name: Name of the column containing image data.
        data_type: Format of the image data ("url", "base64", or None for auto-detection).
            When None, the format is auto-detected: URLs are passed through, file paths that
            exist under base_path are loaded as base64, and other values are assumed to be base64.
        image_format: Image format (required when data_type is explicitly "base64").
    """

    modality: Literal[Modality.IMAGE] = Modality.IMAGE
    image_format: ImageFormat | None = None

    def get_contexts(self, record: dict, *, base_path: str | None = None) -> list[dict[str, Any]]:
        """Get the contexts for the image modality.

        Args:
            record: The record containing the image data. The data can be:
                - A JSON serialized list of strings
                - A list of strings
                - A single string
            base_path: Optional base path for resolving relative file paths.
                When provided, file paths that exist under this directory are loaded
                and converted to base64. This enables generated images (stored as relative
                paths in create mode) to be sent to remote model endpoints.

        Returns:
            A list of image contexts.
        """
        return [
            self._build_context(value, base_path=base_path)
            for value in normalize_media_context_values(record[self.column_name])
        ]

    def _build_context(self, context_value: Any, *, base_path: str | None) -> dict[str, Any]:
        if self.data_type == ModalityDataType.URL:
            return get_media_url_context(Modality.IMAGE.value, context_value)
        if self.data_type == ModalityDataType.BASE64:
            return self._format_base64_context(context_value)
        return self._auto_resolve_context_value(context_value, base_path)

    def _auto_resolve_context_value(self, context_value: Any, base_path: str | None) -> dict[str, Any]:
        """Auto-detect the format of a context value and resolve it.

        Resolution rules:
        - File path that exists under base_path → load to base64 (generated artifact)
        - URL (http/https) → pass through as-is
        - Otherwise → assume base64 data
        """
        if base_path is not None and is_image_path(context_value):
            base64_data = load_image_path_to_base64(context_value, base_path=base_path)
            if base64_data is not None:
                return self._format_base64_context(base64_data)

        if is_image_url(context_value):
            return get_media_url_context(Modality.IMAGE.value, context_value)

        return self._format_base64_context(context_value)

    def _format_base64_context(self, base64_data: str) -> dict[str, Any]:
        """Format base64 image data as a canonical image source dict.

        Uses self.image_format if set, otherwise detects from the image bytes.
        """
        parsed = parse_base64_data_uri(base64_data)
        if parsed is not None:
            media_type, data = parsed
            detected_format = image_format_from_mime_type(media_type)
            if detected_format is None:
                raise ValueError(f"Unsupported image media type {media_type!r}")
            if self.image_format is not None and not _image_formats_match(self.image_format, detected_format):
                raise ValueError(
                    f"image_format {self.image_format.value!r} does not match data URI media type {media_type!r}"
                )
            return get_media_base64_context(Modality.IMAGE.value, media_type, data)

        image_format = self.image_format
        if image_format is None:
            image_bytes = decode_base64_image(base64_data)
            image_format = detect_image_format(image_bytes)
        return get_media_base64_context(Modality.IMAGE.value, f"image/{image_format.value}", base64_data)

    @model_validator(mode="after")
    def _validate_image_format(self) -> Self:
        if self.data_type == ModalityDataType.BASE64 and self.image_format is None:
            raise ValueError(f"image_format is required when data_type is {self.data_type.value}")
        return self


def _image_formats_match(configured_format: ImageFormat, detected_format: ImageFormat) -> bool:
    if configured_format == detected_format:
        return True
    return {configured_format, detected_format} == {ImageFormat.JPG, ImageFormat.JPEG}


class AudioContext(ModalityContext):
    """Configuration for providing audio context to multimodal models.

    Audio context values are URL or base64 media values. Local paths may be
    passed through only in explicit URL mode so colocated model endpoints can
    read them directly. ``audio_format`` is consulted only for base64 sources.
    """

    modality: Literal[Modality.AUDIO] = Modality.AUDIO
    audio_format: AudioFormat | None = None

    def get_contexts(self, record: dict, *, base_path: str | None = None) -> list[dict[str, Any]]:
        """Get audio contexts.

        ``base_path`` is accepted for signature compatibility with ``ImageContext``
        but unused; audio contexts do not resolve local files to base64.
        """
        return [self._build_context(value) for value in normalize_media_context_values(record[self.column_name])]

    def _build_context(self, context_value: Any) -> dict[str, Any]:
        if self.data_type == ModalityDataType.URL:
            self._validate_url_context_value(context_value)
            return get_media_url_context(Modality.AUDIO.value, context_value)

        if self.data_type is None and is_media_url(context_value):
            return get_media_url_context(Modality.AUDIO.value, context_value)

        media_type, data = self._resolve_base64_parts(context_value)
        return get_media_base64_context(Modality.AUDIO.value, media_type, data)

    def _resolve_base64_parts(self, context_value: Any) -> tuple[str, Any]:
        parsed = parse_base64_data_uri(context_value)
        if parsed is not None:
            media_type, data = parsed
            detected_format = audio_format_from_mime_type(media_type)
            if detected_format is None:
                raise ValueError(f"Unsupported audio media type {media_type!r}")
            if self.audio_format is not None and self.audio_format != detected_format:
                raise ValueError(
                    f"audio_format {self.audio_format.value!r} does not match data URI media type {media_type!r}"
                )
            return media_type, data

        if is_audio_path(context_value):
            raise ValueError(
                "audio context values that look like local paths must use data_type=url; "
                "otherwise provide base64 audio data"
            )

        if self.audio_format is None:
            raise ValueError("audio_format is required for base64 audio context values")
        return audio_mime_type(self.audio_format), context_value

    def _validate_url_context_value(self, context_value: Any) -> None:
        if not is_media_url(context_value) and not is_audio_path(context_value):
            raise ValueError("audio URL context values must be HTTP(S) URLs or local audio paths")

    @model_validator(mode="after")
    def _validate_audio_format(self) -> Self:
        if self.data_type == ModalityDataType.BASE64 and self.audio_format is None:
            raise ValueError(f"audio_format is required when data_type is {self.data_type.value}")
        return self


class VideoContext(ModalityContext):
    """Configuration for providing video context to multimodal models.

    Video context values are URL or base64 media values. Local paths may be
    passed through only in explicit URL mode so colocated model endpoints can
    read them directly. ``video_format`` is consulted only for base64 sources.
    """

    modality: Literal[Modality.VIDEO] = Modality.VIDEO
    video_format: VideoFormat | None = None

    def get_contexts(self, record: dict, *, base_path: str | None = None) -> list[dict[str, Any]]:
        """Get video contexts.

        ``base_path`` is accepted for signature compatibility with ``ImageContext``
        but unused; video contexts do not resolve local files to base64.
        """
        return [self._build_context(value) for value in normalize_media_context_values(record[self.column_name])]

    def _build_context(self, context_value: Any) -> dict[str, Any]:
        if self.data_type == ModalityDataType.URL:
            self._validate_url_context_value(context_value)
            return get_media_url_context(Modality.VIDEO.value, context_value)

        if self.data_type is None and is_media_url(context_value):
            return get_media_url_context(Modality.VIDEO.value, context_value)

        media_type, data = self._resolve_base64_parts(context_value)
        return get_media_base64_context(Modality.VIDEO.value, media_type, data)

    def _resolve_base64_parts(self, context_value: Any) -> tuple[str, Any]:
        parsed = parse_base64_data_uri(context_value)
        if parsed is not None:
            media_type, data = parsed
            detected_format = video_format_from_mime_type(media_type)
            if detected_format is None:
                raise ValueError(f"Unsupported video media type {media_type!r}")
            if self.video_format is not None and self.video_format != detected_format:
                raise ValueError(
                    f"video_format {self.video_format.value!r} does not match data URI media type {media_type!r}"
                )
            return media_type, data

        if is_video_path(context_value):
            raise ValueError(
                "video context values that look like local paths must use data_type=url; "
                "otherwise provide base64 video data"
            )

        if self.video_format is None:
            raise ValueError("video_format is required for base64 video context values")
        return video_mime_type(self.video_format), context_value

    def _validate_url_context_value(self, context_value: Any) -> None:
        if not is_media_url(context_value) and not is_video_path(context_value):
            raise ValueError("video URL context values must be HTTP(S) URLs or local video paths")

    @model_validator(mode="after")
    def _validate_video_format(self) -> Self:
        if self.data_type == ModalityDataType.BASE64 and self.video_format is None:
            raise ValueError(f"video_format is required when data_type is {self.data_type.value}")
        return self


MultiModalContextT: TypeAlias = Annotated[
    ImageContext | AudioContext | VideoContext,
    Field(discriminator="modality"),
]


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
            total_weight = sum(self.weights)
            if total_weight == 0:
                raise ValueError("`weights` must sum to a non-zero value")
            self.weights = [w / total_weight for w in self.weights]
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
        return float(lazy.np.random.choice(self.params.values, p=self.params.weights))


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
        return float(lazy.np.random.uniform(low=self.params.low, high=self.params.high, size=1)[0])


DistributionT: TypeAlias = UniformDistribution | ManualDistribution


class GenerationType(str, Enum):
    CHAT_COMPLETION = "chat-completion"
    EMBEDDING = "embedding"
    IMAGE = "image"


class BaseInferenceParams(ConfigBase, ABC):
    """Base configuration for inference parameters.

    Attributes:
        generation_type: Type of generation (chat-completion, embedding, or image). Acts as discriminator.
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
        """Format inference parameters for display as a single line.

        Returns:
            Formatted string of inference parameters
        """
        parts = self.get_formatted_params()
        if not parts:
            return "(none)"
        return ", ".join(parts)

    def get_formatted_params(self) -> list[str]:
        """Get a list of formatted parameter strings.

        Returns:
            List of formatted parameter strings (e.g., ["temperature=0.70", "max_tokens=100"])
        """
        params_dict = self.model_dump(exclude_none=True, mode="json")
        parts = []
        for key, value in params_dict.items():
            formatted_value = self._format_value(key, value)
            parts.append(f"{key}={formatted_value}")
        return parts

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


class ImageInferenceParams(BaseInferenceParams):
    """Configuration for image generation models.

    Works for both diffusion and autoregressive image generation models. Pass all model-specific image options via `extra_body`.

    Attributes:
        generation_type: Type of generation, always "image" for this class.

    Example:
        ```python
        # OpenAI-style (DALL·E): quality and size in extra_body or as top-level kwargs
        dd.ImageInferenceParams(
            extra_body={"size": "1024x1024", "quality": "hd"}
        )

        # Gemini-style: generationConfig.imageConfig
        dd.ImageInferenceParams(
            extra_body={
                "generationConfig": {
                    "imageConfig": {
                        "aspectRatio": "1:1",
                        "imageSize": "1024"
                    }
                }
            }
        )
        ```
    """

    generation_type: Literal[GenerationType.IMAGE] = GenerationType.IMAGE


InferenceParamsT: TypeAlias = Annotated[
    ChatCompletionInferenceParams | EmbeddingInferenceParams | ImageInferenceParams,
    Field(discriminator="generation_type"),
]


class ModelConfig(ConfigBase):
    """Configuration for a model used for generation.

    Attributes:
        alias: User-defined alias to reference in column configurations.
        model: Model identifier (e.g., from build.nvidia.com or other providers).
        inference_parameters: Inference parameters for the model (temperature, top_p, max_tokens, etc.).
            The generation_type is determined by the type of inference_parameters.
        provider: Name of the model provider. Must match the ``name`` field of a
            ``ModelProvider`` registered with the surrounding ``DataDesigner`` instance.
        skip_health_check: Whether to skip the health check for this model. Defaults to False.
    """

    alias: str
    model: str
    inference_parameters: InferenceParamsT = Field(default_factory=ChatCompletionInferenceParams)
    provider: str
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
            # Check for explicit generation_type first
            gen_type = value.get("generation_type")

            # Infer type from generation_type or field presence
            if gen_type == "image":
                return ImageInferenceParams(**value)
            elif gen_type == "embedding" or "encoding_format" in value or "dimensions" in value:
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

    @field_validator("provider_type", mode="after")
    @classmethod
    def normalize_provider_type(cls, v: str) -> str:
        return v.lower()


def load_model_configs(model_configs: list[ModelConfig] | str | Path) -> list[ModelConfig]:
    if isinstance(model_configs, list) and all(isinstance(mc, ModelConfig) for mc in model_configs):
        return model_configs
    json_config = smart_load_yaml(model_configs)
    if "model_configs" not in json_config:
        raise InvalidConfigError(
            "The list of model configs must be provided under model_configs in the configuration file."
        )
    return [ModelConfig.model_validate(mc) for mc in json_config["model_configs"]]
