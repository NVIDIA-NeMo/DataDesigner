# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from data_designer.config.default_model_settings import (
    get_builtin_model_configs,
    get_builtin_model_providers,
    get_defaul_model_providers_missing_api_keys,
    get_default_inference_parameters,
    get_default_model_configs,
    get_default_provider_name,
    get_default_providers,
    resolve_seed_default_model_settings,
)
from data_designer.config.models import InferenceParameters
from data_designer.config.utils.visualization import get_nvidia_api_key, get_openai_api_key


def test_get_default_inference_parameters():
    assert get_default_inference_parameters("text") == InferenceParameters(
        temperature=0.85,
        top_p=0.95,
    )
    assert get_default_inference_parameters("reasoning") == InferenceParameters(
        temperature=0.35,
        top_p=0.95,
    )
    assert get_default_inference_parameters("vision") == InferenceParameters(
        temperature=0.85,
        top_p=0.95,
    )


def test_get_builtin_model_configs():
    builtin_model_configs = get_builtin_model_configs()
    assert len(builtin_model_configs) == 6
    assert builtin_model_configs[0].alias == "nvidia-text"
    assert builtin_model_configs[0].model == "nvidia/nvidia-nemotron-nano-9b-v2"
    assert builtin_model_configs[0].provider == "nvidia"
    assert builtin_model_configs[1].alias == "nvidia-reasoning"
    assert builtin_model_configs[1].model == "openai/gpt-oss-20b"
    assert builtin_model_configs[1].provider == "nvidia"
    assert builtin_model_configs[2].alias == "nvidia-vision"
    assert builtin_model_configs[2].model == "nvidia/nemotron-nano-12b-v2-vl"
    assert builtin_model_configs[2].provider == "nvidia"
    assert builtin_model_configs[3].alias == "openai-text"
    assert builtin_model_configs[3].model == "gpt-4.1"
    assert builtin_model_configs[3].provider == "openai"
    assert builtin_model_configs[4].alias == "openai-reasoning"
    assert builtin_model_configs[4].model == "gpt-5"


def test_get_builtin_model_providers():
    builtin_model_providers = get_builtin_model_providers()
    assert len(builtin_model_providers) == 2
    assert builtin_model_providers[0].name == "nvidia"
    assert builtin_model_providers[0].endpoint == "https://integrate.api.nvidia.com/v1"
    assert builtin_model_providers[0].provider_type == "openai"
    assert builtin_model_providers[0].api_key == "NVIDIA_API_KEY"
    assert builtin_model_providers[1].name == "openai"
    assert builtin_model_providers[1].endpoint == "https://api.openai.com/v1"
    assert builtin_model_providers[1].provider_type == "openai"
    assert builtin_model_providers[1].api_key == "OPENAI_API_KEY"


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


@patch("data_designer.config.default_model_settings.os.environ.get")
def test_get_default_model_providers_missing_api_keys(mock_environ_get):
    mock_environ_get.return_value = None
    assert get_defaul_model_providers_missing_api_keys() == ["NVIDIA_API_KEY", "OPENAI_API_KEY"]
