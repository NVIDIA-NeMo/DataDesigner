from pathlib import Path

from data_designer.cli.constants import MODEL_CONFIGS_FILE_NAME
from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.cli.utils import save_config_file
from data_designer.config.models import ModelConfig


def test_config_file(tmp_path: Path):
    repository = ModelRepository(tmp_path)
    assert repository.config_file == tmp_path / MODEL_CONFIGS_FILE_NAME


def test_load_does_not_exist(tmp_path: Path):
    repository = ModelRepository(tmp_path)
    assert repository.load() is None


def test_load_exists(tmp_path: Path, stub_model_configs: list[ModelConfig]):
    config_file = tmp_path / MODEL_CONFIGS_FILE_NAME
    save_config_file(config_file, ModelConfigRegistry(model_configs=stub_model_configs).model_dump())
    repository = ModelRepository(tmp_path)
    assert repository.load() is not None
    assert repository.load().model_configs == stub_model_configs


def test_save(tmp_path: Path, stub_model_configs: list[ModelConfig]):
    repository = ModelRepository(tmp_path)
    repository.save(ModelConfigRegistry(model_configs=stub_model_configs))
    assert repository.load() is not None
    assert repository.load().model_configs == stub_model_configs
