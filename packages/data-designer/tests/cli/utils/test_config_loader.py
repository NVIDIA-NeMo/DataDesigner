# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.utils.config_loader import (
    ConfigLoadError,
    load_config_builder,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_yaml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a YAML file delegates to from_config."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("data_designer:\n  columns: []\n")

    mock_builder = MagicMock()
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(yaml_file))

    mock_from_config.assert_called_once_with(yaml_file)
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_yml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a .yml file delegates to from_config."""
    yml_file = tmp_path / "config.yml"
    yml_file.write_text("data_designer:\n  columns: []\n")

    mock_builder = MagicMock()
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(yml_file))

    mock_from_config.assert_called_once_with(yml_file)
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_json(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a JSON file delegates to from_config."""
    json_file = tmp_path / "config.json"
    json_file.write_text('{"data_designer": {"columns": []}}')

    mock_builder = MagicMock()
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(json_file))

    mock_from_config.assert_called_once_with(json_file)
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_yaml_url(mock_from_config: MagicMock) -> None:
    """Test loading a config builder from a YAML URL delegates to from_config."""
    config_url = "https://example.com/config.yaml"
    mock_builder = MagicMock()
    mock_from_config.return_value = mock_builder

    result = load_config_builder(config_url)

    mock_from_config.assert_called_once_with(config_url)
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_json_url_with_query(mock_from_config: MagicMock) -> None:
    """Test loading a config builder from a JSON URL with query params delegates to from_config."""
    config_url = "https://example.com/config.json?version=1"
    mock_builder = MagicMock()
    mock_from_config.return_value = mock_builder

    result = load_config_builder(config_url)

    mock_from_config.assert_called_once_with(config_url)
    assert result is mock_builder


def test_load_config_builder_from_python_module(tmp_path: Path) -> None:
    """Test loading a config builder from a Python module with load_config_builder()."""
    py_file = tmp_path / "my_config.py"
    py_file.write_text("def load_config_builder(): pass\n")

    with patch("data_designer.cli.utils.config_loader._load_from_python_module") as mock_load_py:
        mock_builder = MagicMock()
        mock_load_py.return_value = mock_builder

        result = load_config_builder(str(py_file))

        mock_load_py.assert_called_once_with(py_file)
        assert result is mock_builder


def test_load_config_builder_file_not_found() -> None:
    """Test that a non-existent file raises ConfigLoadError."""
    with pytest.raises(ConfigLoadError, match="Config source not found"):
        load_config_builder("/nonexistent/path/config.yaml")


def test_load_config_builder_not_a_file(tmp_path: Path) -> None:
    """Test that a directory path raises ConfigLoadError."""
    with pytest.raises(ConfigLoadError, match="Config source is not a file"):
        load_config_builder(str(tmp_path))


def test_load_config_builder_unsupported_extension(tmp_path: Path) -> None:
    """Test that an unsupported file extension raises ConfigLoadError."""
    txt_file = tmp_path / "config.txt"
    txt_file.write_text("some content")

    with pytest.raises(ConfigLoadError, match="Unsupported file extension"):
        load_config_builder(str(txt_file))


def test_load_config_builder_url_unsupported_extension() -> None:
    """Test that a URL with unsupported extension raises ConfigLoadError."""
    with pytest.raises(ConfigLoadError, match="Unsupported file extension"):
        load_config_builder("https://example.com/config.txt")


def test_load_config_builder_remote_python_module_not_supported() -> None:
    """Test that a Python module URL is rejected."""
    with pytest.raises(ConfigLoadError, match="Remote Python config modules are not supported"):
        load_config_builder("https://example.com/config.py")


def test_load_config_builder_url_no_extension() -> None:
    """Test that a URL with no file extension raises ConfigLoadError."""
    with pytest.raises(ConfigLoadError, match="Unsupported file extension"):
        load_config_builder("https://example.com/config")


def test_load_config_builder_python_module_missing_function(tmp_path: Path) -> None:
    """Test that a Python module without load_config_builder() raises ConfigLoadError."""
    py_file = tmp_path / "no_func_config.py"
    py_file.write_text("x = 42\n")

    with pytest.raises(ConfigLoadError, match="does not define a 'load_config_builder\\(\\)' function"):
        load_config_builder(str(py_file))


def test_load_config_builder_python_module_wrong_return_type(tmp_path: Path) -> None:
    """Test that load_config_builder() returning wrong type raises ConfigLoadError."""
    py_file = tmp_path / "wrong_type_config.py"
    py_file.write_text("def load_config_builder():\n    return {'not': 'a builder'}\n")

    with pytest.raises(ConfigLoadError, match="returned dict, expected DataDesignerConfigBuilder"):
        load_config_builder(str(py_file))


def test_load_config_builder_python_module_syntax_error(tmp_path: Path) -> None:
    """Test that a Python module with syntax errors raises ConfigLoadError."""
    py_file = tmp_path / "syntax_err_config.py"
    py_file.write_text("def load_config_builder(\n")

    with pytest.raises(ConfigLoadError, match="Failed to execute Python module"):
        load_config_builder(str(py_file))


def test_load_config_builder_python_module_function_raises(tmp_path: Path) -> None:
    """Test that load_config_builder() raising an exception is wrapped in ConfigLoadError."""
    py_file = tmp_path / "raising_config.py"
    py_file.write_text("def load_config_builder():\n    raise ValueError('something went wrong')\n")

    with pytest.raises(ConfigLoadError, match="Error calling 'load_config_builder\\(\\)'"):
        load_config_builder(str(py_file))


def test_load_config_builder_python_module_not_callable(tmp_path: Path) -> None:
    """Test that load_config_builder being a non-callable raises ConfigLoadError."""
    py_file = tmp_path / "not_callable_config.py"
    py_file.write_text("load_config_builder = 'not a function'\n")

    with pytest.raises(ConfigLoadError, match="is not callable"):
        load_config_builder(str(py_file))


def test_load_config_builder_python_module_sibling_import(tmp_path: Path) -> None:
    """Test that a Python config can import sibling modules in the same directory."""
    helper_file = tmp_path / "helpers.py"
    helper_file.write_text("DATASET_NAME = 'my_dataset'\n")

    py_file = tmp_path / "my_config.py"
    py_file.write_text(
        "from data_designer.config.config_builder import DataDesignerConfigBuilder\n"
        "from helpers import DATASET_NAME\n\n"
        "def load_config_builder():\n"
        "    builder = DataDesignerConfigBuilder()\n"
        "    builder._test_marker = DATASET_NAME\n"
        "    return builder\n"
    )

    result = load_config_builder(str(py_file))

    assert isinstance(result, DataDesignerConfigBuilder)
    assert result._test_marker == "my_dataset"


def test_load_config_builder_python_module_cleans_sys_path(tmp_path: Path) -> None:
    """Test that the config's parent directory is removed from sys.path after loading."""
    import sys

    py_file = tmp_path / "clean_path_config.py"
    py_file.write_text(
        "from data_designer.config.config_builder import DataDesignerConfigBuilder\n\n"
        "def load_config_builder():\n"
        "    return DataDesignerConfigBuilder()\n"
    )

    parent_dir = str(tmp_path.resolve())
    assert parent_dir not in sys.path

    load_config_builder(str(py_file))

    assert parent_dir not in sys.path


def test_load_config_builder_invalid_yaml(tmp_path: Path) -> None:
    """Test that a YAML file that fails to parse raises ConfigLoadError."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(":\n  - [\n")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(yaml_file))


def test_load_config_builder_invalid_json(tmp_path: Path) -> None:
    """Test that a malformed JSON file raises ConfigLoadError."""
    json_file = tmp_path / "bad.json"
    json_file.write_text(":\n  - [\n")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(json_file))


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_config_validation_error(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test that a valid YAML file with invalid config structure raises ConfigLoadError."""
    yaml_file = tmp_path / "bad_structure.yaml"
    yaml_file.write_text("data_designer:\n  not_a_valid_field: true\n")

    mock_from_config.side_effect = Exception("Validation error")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(yaml_file))


def test_load_config_builder_non_dict_yaml(tmp_path: Path) -> None:
    """Test that a YAML file that parses to a non-dict raises ConfigLoadError."""
    yaml_file = tmp_path / "list.yaml"
    yaml_file.write_text("- item1\n- item2\n")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(yaml_file))


def test_load_config_builder_non_dict_json(tmp_path: Path) -> None:
    """Test that a JSON file containing an array (not an object) raises ConfigLoadError."""
    json_file = tmp_path / "list.json"
    json_file.write_text('[{"name": "col1"}, {"name": "col2"}]')

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(json_file))


def test_load_config_builder_empty_json(tmp_path: Path) -> None:
    """Test that an empty JSON file raises ConfigLoadError."""
    json_file = tmp_path / "empty.json"
    json_file.write_text("")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(json_file))


def test_load_config_builder_empty_yaml(tmp_path: Path) -> None:
    """Test that an empty YAML file raises ConfigLoadError."""
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(yaml_file))
