from pathlib import Path

from data_designer.cli.constants import MODEL_PROVIDERS_FILE_NAME
from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.cli.utils import save_config_file
from data_designer.config.models import ModelProvider


def test_config_file(tmp_path: Path):
    repository = ProviderRepository(tmp_path)
    assert repository.config_file == tmp_path / MODEL_PROVIDERS_FILE_NAME


def test_load_does_not_exist(tmp_path: Path):
    repository = ProviderRepository(tmp_path)
    assert repository.load() is None


def test_load_exists(tmp_path: Path, stub_model_providers: list[ModelProvider]):
    config_file_path = tmp_path / MODEL_PROVIDERS_FILE_NAME
    save_config_file(
        config_file_path,
        ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name).model_dump(),
    )
    repository = ProviderRepository(tmp_path)
    assert repository.load() is not None
    assert repository.load().providers == stub_model_providers


def test_save(tmp_path: Path, stub_model_providers: list[ModelProvider]):
    repository = ProviderRepository(tmp_path)
    repository.save(ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name))
    assert repository.load() is not None
    assert repository.load().providers == stub_model_providers
