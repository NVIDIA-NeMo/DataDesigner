# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.utils.config_loader import ConfigLoadError, load_config_builder
from data_designer.config.config_builder import DataDesignerConfigBuilder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_yaml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a YAML file."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("columns: []\n")

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(yaml_file))

    mock_from_config.assert_called_once_with(yaml_file)
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_yml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a .yml file."""
    yml_file = tmp_path / "config.yml"
    yml_file.write_text("columns: []\n")

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(yml_file))

    mock_from_config.assert_called_once_with(yml_file)
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_json(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a JSON file."""
    json_file = tmp_path / "config.json"
    json_file.write_text('{"columns": []}\n')

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(json_file))

    mock_from_config.assert_called_once_with(json_file)
    assert result is mock_builder


def test_load_config_builder_from_python_module(tmp_path: Path) -> None:
    """Test loading a config builder from a Python module with load_config_builder()."""
    py_file = tmp_path / "my_config.py"
    py_file.write_text(
        "from unittest.mock import MagicMock\n"
        "from data_designer.config.config_builder import DataDesignerConfigBuilder\n\n"
        "def load_config_builder():\n"
        "    return MagicMock(spec=DataDesignerConfigBuilder)\n"
    )

    with patch("data_designer.cli.utils.config_loader._load_from_python_module") as mock_load_py:
        mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
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


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_invalid_yaml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test that a malformed YAML file raises ConfigLoadError."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(": :\n  - invalid:: yaml::\n")

    mock_from_config.side_effect = Exception("YAML parse error")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(yaml_file))


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_invalid_json(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test that a malformed JSON file raises ConfigLoadError."""
    json_file = tmp_path / "bad.json"
    json_file.write_text("{not valid json}")

    mock_from_config.side_effect = Exception("JSON decode error")

    with pytest.raises(ConfigLoadError, match="Failed to load config from"):
        load_config_builder(str(json_file))
