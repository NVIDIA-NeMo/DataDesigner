# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from collections import Counter
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    GenerationType,
    ImageContext,
    ImageFormat,
    ManualDistribution,
    ManualDistributionParams,
    ModalityDataType,
    ModelConfig,
    UniformDistribution,
    UniformDistributionParams,
    load_model_configs,
)
from data_designer.lazy_heavy_imports import np

if TYPE_CHECKING:
    import numpy as np


def test_image_context_get_contexts_single_string():
    """Test get_contexts with a single string value."""
    image_context = ImageContext(
        column_name="image_base64", data_type=ModalityDataType.BASE64, image_format=ImageFormat.PNG
    )
    assert image_context.get_contexts({"image_base64": "somebase64encodedimagestring"}) == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,somebase64encodedimagestring", "format": "png"},
        }
    ]

    image_context = ImageContext(column_name="image_url", data_type=ModalityDataType.URL)
    assert image_context.get_contexts({"image_url": "https://example.com/examle_image.png"}) == [
        {
            "type": "image_url",
            "image_url": "https://example.com/examle_image.png",
        }
    ]


def test_image_context_get_contexts_list_of_strings():
    """Test get_contexts with a list of strings."""
    image_context = ImageContext(
        column_name="image_base64", data_type=ModalityDataType.BASE64, image_format=ImageFormat.PNG
    )
    assert image_context.get_contexts({"image_base64": ["image1base64", "image2base64", "image3base64"]}) == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,image1base64", "format": "png"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,image2base64", "format": "png"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,image3base64", "format": "png"},
        },
    ]

    image_context = ImageContext(column_name="image_url", data_type=ModalityDataType.URL)
    assert image_context.get_contexts(
        {"image_url": ["https://example.com/image1.png", "https://example.com/image2.png"]}
    ) == [
        {
            "type": "image_url",
            "image_url": "https://example.com/image1.png",
        },
        {
            "type": "image_url",
            "image_url": "https://example.com/image2.png",
        },
    ]


def test_image_context_get_contexts_numpy_array():
    """Test get_contexts with numpy arrays (happens after parquet serialization)."""
    image_context = ImageContext(
        column_name="image_base64", data_type=ModalityDataType.BASE64, image_format=ImageFormat.PNG
    )
    numpy_array = np.array(["image1base64", "image2base64"])
    assert image_context.get_contexts({"image_base64": numpy_array}) == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,image1base64", "format": "png"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,image2base64", "format": "png"},
        },
    ]

    image_context = ImageContext(column_name="image_url", data_type=ModalityDataType.URL)
    numpy_array = np.array(["https://example.com/image1.png", "https://example.com/image2.png"])
    assert image_context.get_contexts({"image_url": numpy_array}) == [
        {
            "type": "image_url",
            "image_url": "https://example.com/image1.png",
        },
        {
            "type": "image_url",
            "image_url": "https://example.com/image2.png",
        },
    ]


def test_image_context_get_contexts_json_serialized_list():
    """Test get_contexts with a JSON serialized list of strings."""
    image_context = ImageContext(
        column_name="image_base64", data_type=ModalityDataType.BASE64, image_format=ImageFormat.PNG
    )
    json_str = json.dumps(["image1base64", "image2base64"])
    assert image_context.get_contexts({"image_base64": json_str}) == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,image1base64", "format": "png"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,image2base64", "format": "png"},
        },
    ]

    image_context = ImageContext(column_name="image_url", data_type=ModalityDataType.URL)
    json_str = json.dumps(["https://example.com/image1.png", "https://example.com/image2.png"])
    assert image_context.get_contexts({"image_url": json_str}) == [
        {
            "type": "image_url",
            "image_url": "https://example.com/image1.png",
        },
        {
            "type": "image_url",
            "image_url": "https://example.com/image2.png",
        },
    ]


def test_image_context_get_contexts_json_string_not_list():
    """Test get_contexts with a JSON string that isn't a list (should treat as single string)."""
    image_context = ImageContext(column_name="image_url", data_type=ModalityDataType.URL)
    json_str = json.dumps({"nested": "object"})
    # Should treat the entire JSON string as a single image URL
    assert image_context.get_contexts({"image_url": json_str}) == [
        {
            "type": "image_url",
            "image_url": json_str,
        }
    ]


def test_image_context_get_contexts_invalid_json():
    """Test get_contexts with invalid JSON string (should treat as single string)."""
    image_context = ImageContext(column_name="image_url", data_type=ModalityDataType.URL)
    invalid_json = "not a valid json string"
    assert image_context.get_contexts({"image_url": invalid_json}) == [
        {
            "type": "image_url",
            "image_url": invalid_json,
        }
    ]


def test_image_context_get_contexts_empty_list():
    """Test get_contexts with an empty list."""
    image_context = ImageContext(column_name="image_url", data_type=ModalityDataType.URL)
    assert image_context.get_contexts({"image_url": []}) == []


def test_image_context_validate_image_format():
    with pytest.raises(ValueError, match="image_format is required when data_type is base64"):
        ImageContext(column_name="image_base64", data_type=ModalityDataType.BASE64)


def test_inference_parameters_default_construction():
    empty_inference_parameters = ChatCompletionInferenceParams()
    assert empty_inference_parameters.generate_kwargs == {}
    assert empty_inference_parameters.max_parallel_requests == 4


def test_inference_parameters_generate_kwargs():
    assert ChatCompletionInferenceParams(
        temperature=0.95,
        top_p=0.95,
        max_tokens=100,
        max_parallel_requests=40,
        timeout=10,
        extra_body={"reasoning_effort": "high"},
    ).generate_kwargs == {
        "temperature": 0.95,
        "top_p": 0.95,
        "max_tokens": 100,
        "timeout": 10,
        "extra_body": {"reasoning_effort": "high"},
    }

    assert ChatCompletionInferenceParams().generate_kwargs == {}

    inference_parameters_kwargs = ChatCompletionInferenceParams(
        temperature=UniformDistribution(params=UniformDistributionParams(low=0.0, high=1.0)),
        top_p=ManualDistribution(params=ManualDistributionParams(values=[0.0, 1.0], weights=[0.5, 0.5])),
    ).generate_kwargs
    assert inference_parameters_kwargs["temperature"] is not None
    assert inference_parameters_kwargs["top_p"] is not None


def test_uniform_distribution_low_lt_high_validation():
    with pytest.raises(ValueError, match="`low` must be less than `high`"):
        UniformDistribution(params=UniformDistributionParams(low=0.8, high=0.8))


def test_uniform_distribution_sampling():
    dist = UniformDistribution(params=UniformDistributionParams(low=0.2, high=0.8))
    samples = [dist.sample() for _ in range(1000)]
    assert all(0.2 <= s <= 0.8 for s in samples)


def test_manual_distribution_sampling():
    values = [0.1, 0.5, 0.9]
    weights = [0.2, 0.5, 0.3]
    dist = ManualDistribution(params=ManualDistributionParams(values=values, weights=weights))
    samples = [dist.sample() for _ in range(10000)]
    assert set(samples) == set(values)

    sample_counts = Counter(samples)
    total_samples = sum(sample_counts.values())
    for value, weight in zip(values, weights):
        observed_freq = sample_counts[value] / total_samples
        expected_freq = weight
        # Allow small margin for randomness
        assert abs(observed_freq - expected_freq) < 0.05


def test_manual_distribution_equal_length_validation():
    with pytest.raises(ValueError, match="`values` and `weights` must have the same length"):
        ManualDistribution(params=ManualDistributionParams(values=[0.1, 0.2], weights=[0.5]))


def test_manual_distribution_weight_normalization():
    dist = ManualDistribution(params=ManualDistributionParams(values=[0.2, 0.8], weights=[2, 2]))
    assert dist.params.weights == [0.5, 0.5]


def test_invalid_manual_distribution():
    # Empty list should fail
    with pytest.raises(ValidationError):
        ManualDistribution(params=ManualDistributionParams(values=[], weights=[]))


def test_manual_distribution_sampling_without_weights():
    dist = ManualDistribution(params=ManualDistributionParams(values=[0.1, 0.4, 0.7]))
    samples = [dist.sample() for _ in range(1000)]
    assert all(0.0 <= s <= 1.0 for s in samples)


def test_inference_parameters_temperature_validation():
    expected_error_msg = "temperature defined in model config must be between 0.0 and 2.0"

    # All temp values provide in a manual destribution should be valid
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(
            temperature=ManualDistribution(params=ManualDistributionParams(values=[0.5, 2.5], weights=[0.5, 0.5]))
        )

    # High and low values of uniform distribution should be valid
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(
            temperature=UniformDistribution(params=UniformDistributionParams(low=0.5, high=2.5))
        )

    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(
            temperature=UniformDistribution(params=UniformDistributionParams(low=-0.5, high=2.0))
        )

    # Static values should be valid
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(temperature=3.0)
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(temperature=-1.0)

    # Valid temperature values shouldn't raise validation errors
    try:
        ChatCompletionInferenceParams(temperature=0.1)
        ChatCompletionInferenceParams(
            temperature=UniformDistribution(params=UniformDistributionParams(low=0.5, high=2.0))
        )
        ChatCompletionInferenceParams(
            temperature=ManualDistribution(params=ManualDistributionParams(values=[0.5, 2.0], weights=[0.5, 0.5]))
        )
    except Exception:
        pytest.fail("Unexpected exception raised during CompletionInferenceParameters temperature validation")


def test_generation_parameters_top_p_validation():
    expected_error_msg = "top_p defined in model config must be between 0.0 and 1.0"

    # All top_p values provide in a manual destribution should be valid
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(
            top_p=ManualDistribution(params=ManualDistributionParams(values=[0.5, 1.5], weights=[0.5, 0.5]))
        )

    # High and low values of uniform distribution should be valid
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(top_p=UniformDistribution(params=UniformDistributionParams(low=0.5, high=1.5)))
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(top_p=UniformDistribution(params=UniformDistributionParams(low=-0.5, high=1.0)))

    # Static values should be valid
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(top_p=1.5)
    with pytest.raises(ValidationError, match=expected_error_msg):
        ChatCompletionInferenceParams(top_p=-0.1)

    # Valid top_p values shouldn't raise validation errors
    try:
        ChatCompletionInferenceParams(top_p=0.1)
        ChatCompletionInferenceParams(top_p=UniformDistribution(params=UniformDistributionParams(low=0.5, high=1.0)))
        ChatCompletionInferenceParams(
            top_p=ManualDistribution(params=ManualDistributionParams(values=[0.5, 1.0], weights=[0.5, 0.5]))
        )
    except Exception:
        pytest.fail("Unexpected exception raised during CompletionInferenceParameters top_p validation")


def test_generation_parameters_max_tokens_validation():
    with pytest.raises(
        ValidationError,
        match="Input should be greater than or equal to 1",
    ):
        ChatCompletionInferenceParams(max_tokens=0)

    # Valid max_tokens values shouldn't raise validation errors
    try:
        ChatCompletionInferenceParams(max_tokens=128_000)
        ChatCompletionInferenceParams(max_tokens=4096)
        ChatCompletionInferenceParams(max_tokens=1)
    except Exception:
        pytest.fail("Unexpected exception raised during CompletionInferenceParameters max_tokens validation")


def test_load_model_configs():
    stub_model_configs = [
        ModelConfig(alias="test", model="test"),
        ModelConfig(alias="test2", model="test2"),
    ]
    stub_model_configs_dict_list = [mc.model_dump(mode="json") for mc in stub_model_configs]
    assert load_model_configs([]) == []
    assert load_model_configs(stub_model_configs) == stub_model_configs

    with tempfile.NamedTemporaryFile(prefix="model_configs", suffix=".yaml") as tmp_file:
        model_configs = {"model_configs": stub_model_configs_dict_list}
        tmp_file.write(yaml.safe_dump(model_configs).encode("utf-8"))
        tmp_file.flush()
        assert load_model_configs(tmp_file.name) == stub_model_configs

    with tempfile.NamedTemporaryFile(prefix="model_configs", suffix=".json") as tmp_file:
        model_configs = {"model_configs": stub_model_configs_dict_list}
        tmp_file.write(json.dumps(model_configs).encode("utf-8"))
        tmp_file.flush()
        assert load_model_configs(tmp_file.name) == stub_model_configs

    with pytest.raises(
        InvalidConfigError,
        match="The list of model configs must be provided under model_configs in the configuration file.",
    ):
        # test failure when the list is not grouped under model_configs in the yaml file
        with tempfile.NamedTemporaryFile(prefix="model_configs", suffix=".yaml") as tmp_file:
            model_configs = {"some_other_key": "invalid_config"}
            tmp_file.write(yaml.safe_dump(model_configs).encode("utf-8"))
            tmp_file.flush()
            load_model_configs(tmp_file.name)

    with pytest.raises(ValidationError):
        # test failure when model config validation fails (tests line 220)
        with tempfile.NamedTemporaryFile(prefix="model_configs", suffix=".json") as tmp_file:
            invalid_model_configs = {"model_configs": [{"invalid": "config"}]}
            tmp_file.write(json.dumps(invalid_model_configs).encode("utf-8"))
            tmp_file.flush()
            load_model_configs(tmp_file.name)


def test_model_config_construction():
    # test default construction
    model_config = ModelConfig(alias="test", model="test")
    assert model_config.inference_parameters == ChatCompletionInferenceParams()
    assert model_config.generation_type == GenerationType.CHAT_COMPLETION

    # test construction with completion inference parameters
    completion_params = ChatCompletionInferenceParams(temperature=0.5, top_p=0.5, max_tokens=100)
    model_config = ModelConfig(alias="test", model="test", inference_parameters=completion_params)
    assert model_config.inference_parameters == completion_params
    assert model_config.generation_type == GenerationType.CHAT_COMPLETION

    # test construction with embedding inference parameters
    embedding_params = EmbeddingInferenceParams(dimensions=100)
    model_config = ModelConfig(alias="test", model="test", inference_parameters=embedding_params)
    assert model_config.inference_parameters == embedding_params
    assert model_config.generation_type == GenerationType.EMBEDDING


def test_model_config_generation_type_from_dict():
    # Test that generation_type in dict is used to create the right inference params type
    model_config = ModelConfig.model_validate(
        {
            "alias": "test",
            "model": "test",
            "inference_parameters": {"generation_type": "embedding", "dimensions": 100},
        }
    )
    assert isinstance(model_config.inference_parameters, EmbeddingInferenceParams)
    assert model_config.generation_type == GenerationType.EMBEDDING

    model_config = ModelConfig.model_validate(
        {
            "alias": "test",
            "model": "test",
            "inference_parameters": {"generation_type": "chat-completion", "temperature": 0.5},
        }
    )
    assert isinstance(model_config.inference_parameters, ChatCompletionInferenceParams)
    assert model_config.generation_type == GenerationType.CHAT_COMPLETION


def test_chat_completion_params_format_for_display_all_params():
    """Test formatting chat completion model with all parameters."""
    params = ChatCompletionInferenceParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        max_parallel_requests=4,
        timeout=60,
    )
    result = params.format_for_display()
    assert "generation_type=chat-completion" in result
    assert "temperature=0.70" in result
    assert "top_p=0.90" in result
    assert "max_tokens=2048" in result
    assert "max_parallel_requests=4" in result
    assert "timeout=60" in result


def test_chat_completion_params_format_for_display_partial_params():
    """Test formatting chat completion model with partial parameters (some None)."""
    params = ChatCompletionInferenceParams(
        temperature=0.5,
        max_tokens=1024,
    )
    result = params.format_for_display()
    assert "generation_type=chat-completion" in result
    assert "temperature=0.50" in result
    assert "max_tokens=1024" in result
    # None values should be excluded
    assert "top_p" not in result
    assert "timeout" not in result


def test_embedding_params_format_for_display():
    """Test formatting embedding model parameters."""
    params = EmbeddingInferenceParams(
        encoding_format="float",
        dimensions=1024,
        max_parallel_requests=8,
    )
    result = params.format_for_display()
    assert "generation_type=embedding" in result
    assert "encoding_format=float" in result
    assert "dimensions=1024" in result
    assert "max_parallel_requests=8" in result
    # Chat completion params should not appear
    assert "temperature" not in result
    assert "top_p" not in result


def test_chat_completion_params_format_for_display_with_distribution():
    """Test formatting parameters with distribution (should show 'dist')."""
    params = ChatCompletionInferenceParams(
        temperature=UniformDistribution(
            distribution_type="uniform",
            params=UniformDistributionParams(low=0.5, high=0.9),
        ),
        max_tokens=2048,
    )
    result = params.format_for_display()
    assert "generation_type=chat-completion" in result
    assert "temperature=dist" in result
    assert "max_tokens=2048" in result


def test_inference_params_format_for_display_float_formatting():
    """Test that float values are formatted to 2 decimal places."""
    params = ChatCompletionInferenceParams(
        temperature=0.123456,
        top_p=0.987654,
    )
    result = params.format_for_display()
    assert "temperature=0.12" in result
    assert "top_p=0.99" in result


def test_inference_params_format_for_display_minimal_params():
    """Test formatting with only required parameters."""
    params = ChatCompletionInferenceParams()
    result = params.format_for_display()
    assert "generation_type=chat-completion" in result
    assert "max_parallel_requests=4" in result  # Default value
    # Optional params should not appear when None
    assert "temperature" not in result
    assert "top_p" not in result
    assert "max_tokens" not in result
    assert "timeout" not in result
