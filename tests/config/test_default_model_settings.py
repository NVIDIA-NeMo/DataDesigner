# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from data_designer.config.default_model_settings import (
    get_builtin_model_configs,
    get_builtin_model_providers,
    get_default_inference_parameters,
    get_default_model_configs,
    get_default_provider_name,
    get_default_providers,
    get_providers_with_missing_api_keys,
    resolve_seed_default_model_settings,
)
from data_designer.config.models import ChatCompletionInferenceParams, EmbeddingInferenceParams, ModelProvider
from data_designer.config.utils.visualization import get_nvidia_api_key, get_openai_api_key


def test_get_default_inference_parameters():
    assert get_default_inference_parameters(
        "text", {"temperature": 0.85, "top_p": 0.95}
    ) == ChatCompletionInferenceParams(
        temperature=0.85,
        top_p=0.95,
    )
    assert get_default_inference_parameters(
        "reasoning", {"temperature": 0.35, "top_p": 0.95}
    ) == ChatCompletionInferenceParams(
        temperature=0.35,
        top_p=0.95,
    )
    assert get_default_inference_parameters(
        "vision", {"temperature": 0.85, "top_p": 0.95}
    ) == ChatCompletionInferenceParams(
        temperature=0.85,
        top_p=0.95,
    )
    assert get_default_inference_parameters(
        "embedding", {"encoding_format": "float", "extra_body": {"input_type": "query"}}
    ) == EmbeddingInferenceParams(
        encoding_format="float",
        extra_body={"input_type": "query"},
    )
    assert get_default_inference_parameters("embedding", {"encoding_format": "float"}) == EmbeddingInferenceParams(
        encoding_format="float",
    )


def test_get_builtin_model_configs():
    builtin_model_configs = get_builtin_model_configs()
    assert len(builtin_model_configs) == 12
    assert builtin_model_configs[0].alias == "nvidia-text"
    assert builtin_model_configs[0].model == "nvidia/nemotron-3-nano-30b-a3b"
    assert builtin_model_configs[0].provider == "nvidia"
    assert builtin_model_configs[1].alias == "nvidia-reasoning"
    assert builtin_model_configs[1].model == "openai/gpt-oss-20b"
    assert builtin_model_configs[1].provider == "nvidia"
    assert builtin_model_configs[2].alias == "nvidia-vision"
    assert builtin_model_configs[2].model == "nvidia/nemotron-nano-12b-v2-vl"
    assert builtin_model_configs[2].provider == "nvidia"
    assert builtin_model_configs[3].alias == "nvidia-embedding"
    assert builtin_model_configs[3].model == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert builtin_model_configs[3].provider == "nvidia"
    assert builtin_model_configs[4].alias == "openai-text"
    assert builtin_model_configs[4].model == "gpt-4.1"
    assert builtin_model_configs[4].provider == "openai"
    assert builtin_model_configs[5].alias == "openai-reasoning"
    assert builtin_model_configs[5].model == "gpt-5"
    assert builtin_model_configs[5].provider == "openai"
    assert builtin_model_configs[6].alias == "openai-vision"
    assert builtin_model_configs[6].model == "gpt-5"
    assert builtin_model_configs[6].provider == "openai"
    assert builtin_model_configs[7].alias == "openai-embedding"
    assert builtin_model_configs[7].model == "text-embedding-3-large"
    assert builtin_model_configs[7].provider == "openai"
    assert builtin_model_configs[8].alias == "openrouter-text"
    assert builtin_model_configs[8].model == "nvidia/nemotron-3-nano-30b-a3b"
    assert builtin_model_configs[8].provider == "openrouter"
    assert builtin_model_configs[9].alias == "openrouter-reasoning"
    assert builtin_model_configs[9].model == "openai/gpt-oss-20b"
    assert builtin_model_configs[9].provider == "openrouter"
    assert builtin_model_configs[10].alias == "openrouter-vision"
    assert builtin_model_configs[10].model == "nvidia/nemotron-nano-12b-v2-vl"
    assert builtin_model_configs[10].provider == "openrouter"
    assert builtin_model_configs[11].alias == "openrouter-embedding"
    assert builtin_model_configs[11].model == "openai/text-embedding-3-large"
    assert builtin_model_configs[11].provider == "openrouter"


def test_get_builtin_model_providers():
    builtin_model_providers = get_builtin_model_providers()
    assert len(builtin_model_providers) == 3
    assert builtin_model_providers[0].name == "nvidia"
    assert builtin_model_providers[0].endpoint == "https://integrate.api.nvidia.com/v1"
    assert builtin_model_providers[0].provider_type == "openai"
    assert builtin_model_providers[0].api_key == "NVIDIA_API_KEY"
    assert builtin_model_providers[1].name == "openai"
    assert builtin_model_providers[1].endpoint == "https://api.openai.com/v1"
    assert builtin_model_providers[1].provider_type == "openai"
    assert builtin_model_providers[1].api_key == "OPENAI_API_KEY"
    assert builtin_model_providers[2].name == "openrouter"
    assert builtin_model_providers[2].endpoint == "https://openrouter.ai/api/v1"
    assert builtin_model_providers[2].provider_type == "openai"
    assert builtin_model_providers[2].api_key == "OPENROUTER_API_KEY"


def test_get_default_model_configs_path_exists(tmp_path: Path):
    model_configs_file_path = tmp_path / "model_configs.yaml"
    model_configs_file_path.write_text(
        json.dumps(dict(model_configs=[mc.model_dump() for mc in get_builtin_model_configs()]))
    )
    with patch("data_designer.config.default_model_settings.MODEL_CONFIGS_FILE_PATH", new=model_configs_file_path):
        assert get_default_model_configs() == get_builtin_model_configs()


def test_get_default_model_configs_path_does_not_exist():
    with patch("data_designer.config.default_model_settings.MODEL_CONFIGS_FILE_PATH", new=Path("non_existent_path")):
        assert get_default_model_configs() == []


def test_get_default_providers_path_exists(tmp_path: Path):
    providers_file_path = tmp_path / "providers.yaml"
    providers_file_path.write_text(json.dumps(dict(providers=[p.model_dump() for p in get_builtin_model_providers()])))
    with patch("data_designer.config.default_model_settings.MODEL_PROVIDERS_FILE_PATH", new=providers_file_path):
        assert get_default_providers() == get_builtin_model_providers()


def test_get_default_providers_path_does_not_exist():
    with patch("data_designer.config.default_model_settings.MODEL_PROVIDERS_FILE_PATH", new=Path("non_existent_path")):
        with pytest.raises(FileNotFoundError, match=r"Default model providers file not found at 'non_existent_path'"):
            get_default_providers()


def test_get_default_provider_name_with_default_key(tmp_path: Path):
    providers_file_path = tmp_path / "providers.yaml"
    providers_file_path.write_text(
        json.dumps(dict(providers=[p.model_dump() for p in get_builtin_model_providers()], default="nvidia"))
    )
    with patch("data_designer.config.default_model_settings.MODEL_PROVIDERS_FILE_PATH", new=providers_file_path):
        assert get_default_provider_name() == "nvidia"


def test_get_default_provider_name_without_default_key(tmp_path: Path):
    providers_file_path = tmp_path / "providers.yaml"
    providers_file_path.write_text(json.dumps({"providers": [p.model_dump() for p in get_builtin_model_providers()]}))
    with patch("data_designer.config.default_model_settings.MODEL_PROVIDERS_FILE_PATH", new=providers_file_path):
        assert get_default_provider_name() is None


def test_get_default_provider_name_path_does_not_exist():
    with patch("data_designer.config.default_model_settings.MODEL_PROVIDERS_FILE_PATH", new=Path("non_existent_path")):
        with pytest.raises(FileNotFoundError, match=r"Default model providers file not found at 'non_existent_path'"):
            get_default_provider_name()


def test_get_nvidia_api_key():
    with patch("data_designer.config.utils.visualization.os.getenv", return_value="nvidia_api_key"):
        assert get_nvidia_api_key() == "nvidia_api_key"


def test_get_openai_api_key():
    with patch("data_designer.config.utils.visualization.os.getenv", return_value="openai_api_key"):
        assert get_openai_api_key() == "openai_api_key"


def test_resolve_seed_default_model_settings(tmp_path: Path):
    model_configs_file_path = tmp_path / "model_configs.yaml"
    model_providers_file_path = tmp_path / "providers.yaml"
    with patch("data_designer.config.default_model_settings.MODEL_CONFIGS_FILE_PATH", new=model_configs_file_path):
        with patch(
            "data_designer.config.default_model_settings.MODEL_PROVIDERS_FILE_PATH", new=model_providers_file_path
        ):
            resolve_seed_default_model_settings()
            assert model_configs_file_path.exists()
            assert model_providers_file_path.exists()

            # Validate YAML format (not JSON)
            with open(model_configs_file_path) as f:
                model_configs_data = yaml.safe_load(f)
            assert model_configs_data == {"model_configs": [mc.model_dump() for mc in get_builtin_model_configs()]}

            with open(model_providers_file_path) as f:
                providers_data = yaml.safe_load(f)
            assert providers_data == {"providers": [p.model_dump() for p in get_builtin_model_providers()]}


def test_get_providers_with_missing_api_keys():
    """Test detection of providers with missing API keys."""
    # Test providers with various API key configurations
    providers = [
        ModelProvider(name="provider1", endpoint="http://test1.com", api_key="NVIDIA_API_KEY"),  # env var
        ModelProvider(name="provider2", endpoint="http://test2.com", api_key="sk-actual-key-12345"),  # actual key
        ModelProvider(name="provider3", endpoint="http://test3.com", api_key=None),  # no key
    ]

    with patch("data_designer.config.default_model_settings.os.environ.get") as mock_env:
        # Mock env to have NVIDIA_API_KEY set but not MISSING_VAR
        def mock_get(key: str) -> str | None:
            return "test-key" if key == "NVIDIA_API_KEY" else None

        mock_env.side_effect = mock_get

        missing = get_providers_with_missing_api_keys(providers)

        # provider1 has env var set -> OK
        # provider2 has actual API key -> OK
        # provider3 has no API key -> MISSING
        assert len(missing) == 1
        assert missing[0].name == "provider3"


def test_get_providers_with_missing_api_keys_env_var_not_set():
    """Test detection when environment variable is not set."""
    providers = [
        ModelProvider(name="provider1", endpoint="http://test1.com", api_key="MISSING_ENV_VAR"),
    ]

    with patch("data_designer.config.default_model_settings.os.environ.get", return_value=None):
        missing = get_providers_with_missing_api_keys(providers)
        assert len(missing) == 1
        assert missing[0].name == "provider1"


def test_get_providers_with_missing_api_keys_all_valid():
    """Test when all providers have valid API keys."""
    providers = [
        ModelProvider(name="provider1", endpoint="http://test1.com", api_key="sk-actual-key-1"),
        ModelProvider(name="provider2", endpoint="http://test2.com", api_key="sk-actual-key-2"),
    ]

    missing = get_providers_with_missing_api_keys(providers)
    assert len(missing) == 0


def test_get_providers_with_missing_api_keys_all_missing():
    """Test when all providers have missing API keys."""
    providers = [
        ModelProvider(name="provider1", endpoint="http://test1.com", api_key="MISSING_VAR_1"),
        ModelProvider(name="provider2", endpoint="http://test2.com", api_key=None),
    ]

    with patch("data_designer.config.default_model_settings.os.environ.get", return_value=None):
        missing = get_providers_with_missing_api_keys(providers)
        assert len(missing) == 2
        assert {p.name for p in missing} == {"provider1", "provider2"}


def test_get_providers_with_missing_api_keys_mixed_case():
    """Test that lowercase API keys are treated as actual keys, not env vars."""
    providers = [
        ModelProvider(name="provider1", endpoint="http://test1.com", api_key="lowercase_key"),
        ModelProvider(name="provider2", endpoint="http://test2.com", api_key="UPPERCASE_KEY"),
    ]

    with patch("data_designer.config.default_model_settings.os.environ.get", return_value=None):
        missing = get_providers_with_missing_api_keys(providers)
        # provider1 has lowercase key (treated as actual key) -> OK
        # provider2 has uppercase key but env var not set -> MISSING
        assert len(missing) == 1
        assert missing[0].name == "provider2"
