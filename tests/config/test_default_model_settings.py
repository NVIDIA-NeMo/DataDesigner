from pathlib import Path
from unittest.mock import patch

from data_designer.config.default_model_settings import (
    get_default_model_configs,
    get_default_nvidia_model_configs,
    get_default_openai_model_configs,
    get_default_providers,
    get_user_defined_default_model_configs,
    get_user_defined_default_providers,
)
from data_designer.config.models import InferenceParameters, ModelConfig, ModelProvider


@patch("data_designer.config.default_model_settings.get_nvidia_api_key")
def test_get_default_nvidia_model_configs(mock_get_nvidia_api_key):
    mock_get_nvidia_api_key.return_value = "nv-some-api-key"
    nvidia_model_configs = get_default_nvidia_model_configs()
    assert len(nvidia_model_configs) == 3
    assert nvidia_model_configs[0].alias == "nvidia-text"
    assert nvidia_model_configs[0].model == "nvidia/nvidia-nemotron-nano-9b-v2"
    assert nvidia_model_configs[0].provider == "nvidia"
    assert nvidia_model_configs[0].inference_parameters is not None

    assert nvidia_model_configs[1].alias == "nvidia-reasoning"
    assert nvidia_model_configs[1].model == "openai/gpt-oss-20b"
    assert nvidia_model_configs[1].provider == "nvidia"
    assert nvidia_model_configs[1].inference_parameters is not None

    assert nvidia_model_configs[2].alias == "nvidia-vision"
    assert nvidia_model_configs[2].model == "nvidia/nemotron-nano-12b-v2-vl"
    assert nvidia_model_configs[2].provider == "nvidia"
    assert nvidia_model_configs[2].inference_parameters is not None


@patch("data_designer.config.default_model_settings.get_nvidia_api_key")
def test_get_default_nvidia_model_configs_no_api_key(mock_get_nvidia_api_key):
    mock_get_nvidia_api_key.return_value = None
    nvidia_model_configs = get_default_nvidia_model_configs()
    assert len(nvidia_model_configs) == 0


@patch("data_designer.config.default_model_settings.get_openai_api_key")
def test_get_default_openai_model_configs(mock_get_openai_api_key):
    mock_get_openai_api_key.return_value = "sk-some-api-key"
    openai_model_configs = get_default_openai_model_configs()
    assert len(openai_model_configs) == 3
    assert openai_model_configs[0].alias == "openai-text"
    assert openai_model_configs[0].model == "gpt-4.1"
    assert openai_model_configs[0].provider == "openai"
    assert openai_model_configs[0].inference_parameters is not None

    assert openai_model_configs[1].alias == "openai-reasoning"
    assert openai_model_configs[1].model == "gpt-5"
    assert openai_model_configs[1].provider == "openai"
    assert openai_model_configs[1].inference_parameters is not None

    assert openai_model_configs[2].alias == "openai-vision"
    assert openai_model_configs[2].model == "gpt-5"
    assert openai_model_configs[2].provider == "openai"
    assert openai_model_configs[2].inference_parameters is not None


@patch("data_designer.config.default_model_settings.get_openai_api_key")
def test_get_default_openai_model_configs_no_api_key(mock_get_openai_api_key):
    mock_get_openai_api_key.return_value = None
    openai_model_configs = get_default_openai_model_configs()
    assert len(openai_model_configs) == 0


def test_get_user_defined_default_model_configs(tmp_path: Path) -> None:
    """Test getting user-defined model configs from custom config directory."""
    (tmp_path / "model_configs.yaml").write_text(
        """
        model_configs:
        - alias: test-model-1
          model: test/model-id
          provider: model-provider
          inference_parameters:
            temperature: 0.8
            top_p: 0.9
        - alias: test-model-2
          model: test/model-id-2
          provider: model-provider-2
          inference_parameters:
            temperature: 0.8
            top_p: 0.9
        """
    )
    user_defined_model_configs = get_user_defined_default_model_configs(tmp_path)
    assert len(user_defined_model_configs) == 2
    assert user_defined_model_configs[0].alias == "test-model-1"
    assert user_defined_model_configs[0].model == "test/model-id"
    assert user_defined_model_configs[0].provider == "model-provider"
    assert user_defined_model_configs[0].inference_parameters is not None
    assert user_defined_model_configs[1].alias == "test-model-2"
    assert user_defined_model_configs[1].model == "test/model-id-2"
    assert user_defined_model_configs[1].provider == "model-provider-2"
    assert user_defined_model_configs[1].inference_parameters is not None


def test_get_user_defined_default_model_configs_no_user_defined_configs(tmp_path: Path) -> None:
    """Test getting user-defined model configs when file doesn't exist."""
    assert len(get_user_defined_default_model_configs(tmp_path)) == 0


@patch("data_designer.config.default_model_settings.get_default_nvidia_model_configs")
@patch("data_designer.config.default_model_settings.get_default_openai_model_configs")
@patch("data_designer.config.default_model_settings.get_user_defined_default_model_configs")
def test_get_default_model_configs_no_user_defined_configs(
    mock_get_user_defined_default_model_configs,
    mock_get_default_openai_model_configs,
    mock_get_default_nvidia_model_configs,
):
    mock_get_default_nvidia_model_configs.return_value = [
        ModelConfig(
            alias="test-model-1",
            model="test/model-id",
            provider="nvidia",
            inference_parameters=InferenceParameters(temperature=0.8, top_p=0.9),
        ),
    ]
    mock_get_default_openai_model_configs.return_value = [
        ModelConfig(
            alias="test-model-2",
            model="test/model-id-2",
            provider="openai",
            inference_parameters=InferenceParameters(temperature=0.8, top_p=0.9),
        ),
    ]
    mock_get_user_defined_default_model_configs.return_value = []
    model_configs = get_default_model_configs()
    assert len(model_configs) == 2
    assert model_configs[0].alias == "test-model-1"
    assert model_configs[0].provider == "nvidia"
    assert model_configs[1].alias == "test-model-2"
    assert model_configs[1].provider == "openai"


@patch("data_designer.config.default_model_settings.get_user_defined_default_model_configs")
def test_get_default_model_configs_with_user_defined_configs(mock_get_user_defined_default_model_configs):
    mock_get_user_defined_default_model_configs.return_value = [
        ModelConfig(
            alias="test-model-1",
            model="test/model-id-1",
            provider="model-provider",
            inference_parameters=InferenceParameters(temperature=0.8, top_p=0.9),
        ),
    ]
    model_configs = get_default_model_configs()
    assert len(model_configs) == 1
    assert model_configs[0].alias == "test-model-1"
    assert model_configs[0].provider == "model-provider"


def test_get_user_defined_default_providers(tmp_path: Path) -> None:
    """Test getting user-defined providers from custom config directory."""
    (tmp_path / "model_providers.yaml").write_text(
        """
        providers:
        - name: test-provider-1
          endpoint: https://api.test-provider-1.com/v1
          api_key: test-api-key-1
        - name: test-provider-2
          endpoint: https://api.test-provider-2.com/v1
          api_key: test-api-key-2
        """
    )
    user_defined_providers = get_user_defined_default_providers(tmp_path)
    assert len(user_defined_providers) == 2
    assert user_defined_providers[0].name == "test-provider-1"
    assert user_defined_providers[0].endpoint == "https://api.test-provider-1.com/v1"
    assert user_defined_providers[0].api_key == "test-api-key-1"
    assert user_defined_providers[1].name == "test-provider-2"
    assert user_defined_providers[1].endpoint == "https://api.test-provider-2.com/v1"
    assert user_defined_providers[1].api_key == "test-api-key-2"


def test_get_user_defined_default_providers_no_user_defined_providers(tmp_path: Path) -> None:
    """Test getting user-defined providers when file doesn't exist."""
    assert len(get_user_defined_default_providers(tmp_path)) == 0


@patch("data_designer.config.default_model_settings.get_user_defined_default_providers")
def test_get_default_providers_no_user_defined_providers(mock_get_user_defined_default_providers):
    mock_get_user_defined_default_providers.return_value = []
    default_providers = get_default_providers()
    assert len(default_providers) == 2
    assert default_providers[0].name == "nvidia"
    assert default_providers[0].endpoint == "https://integrate.api.nvidia.com/v1"
    assert default_providers[0].api_key == "NVIDIA_API_KEY"
    assert default_providers[1].name == "openai"
    assert default_providers[1].endpoint == "https://api.openai.com/v1"
    assert default_providers[1].api_key == "OPENAI_API_KEY"


@patch("data_designer.config.default_model_settings.get_user_defined_default_providers")
def test_get_default_providers_with_user_defined_providers(mock_get_user_defined_default_providers):
    mock_get_user_defined_default_providers.return_value = [
        ModelProvider(
            name="test-provider-1",
            endpoint="https://api.test-provider-1.com/v1",
            api_key="test-api-key-1",
        ),
    ]
    default_providers = get_default_providers()
    assert len(default_providers) == 1
    assert default_providers[0].name == "test-provider-1"
    assert default_providers[0].endpoint == "https://api.test-provider-1.com/v1"
    assert default_providers[0].api_key == "test-api-key-1"
