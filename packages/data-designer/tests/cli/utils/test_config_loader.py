# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.utils.config_loader import (
    ConfigLoadError,
    _maybe_wrap_data_designer_config,
    load_config_builder,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_yaml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a YAML file in BuilderConfig format."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("data_designer:\n  columns: []\n")

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(yaml_file))

    mock_from_config.assert_called_once_with({"data_designer": {"columns": []}})
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_yml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a .yml file in BuilderConfig format."""
    yml_file = tmp_path / "config.yml"
    yml_file.write_text("data_designer:\n  columns: []\n")

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(yml_file))

    mock_from_config.assert_called_once_with({"data_designer": {"columns": []}})
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_json(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test loading a config builder from a JSON file in BuilderConfig format."""
    json_file = tmp_path / "config.json"
    json_file.write_text(json.dumps({"data_designer": {"columns": []}}))

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(json_file))

    mock_from_config.assert_called_once_with({"data_designer": {"columns": []}})
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


def test_load_config_builder_python_module_sibling_import(tmp_path: Path) -> None:
    """Test that a Python config can import sibling modules in the same directory."""
    helper_file = tmp_path / "helpers.py"
    helper_file.write_text("DATASET_NAME = 'my_dataset'\n")

    py_file = tmp_path / "my_config.py"
    py_file.write_text(
        "from unittest.mock import MagicMock\n"
        "from data_designer.config.config_builder import DataDesignerConfigBuilder\n"
        "from helpers import DATASET_NAME\n\n"
        "def load_config_builder():\n"
        "    builder = MagicMock(spec=DataDesignerConfigBuilder)\n"
        "    builder._dataset_name = DATASET_NAME\n"
        "    return builder\n"
    )

    result = load_config_builder(str(py_file))

    assert isinstance(result, DataDesignerConfigBuilder)  # MagicMock with spec passes this
    assert result._dataset_name == "my_dataset"


def test_load_config_builder_python_module_cleans_sys_path(tmp_path: Path) -> None:
    """Test that the config's parent directory is removed from sys.path after loading."""
    import sys

    py_file = tmp_path / "clean_path_config.py"
    py_file.write_text(
        "from unittest.mock import MagicMock\n"
        "from data_designer.config.config_builder import DataDesignerConfigBuilder\n\n"
        "def load_config_builder():\n"
        "    return MagicMock(spec=DataDesignerConfigBuilder)\n"
    )

    parent_dir = str(tmp_path.resolve())
    assert parent_dir not in sys.path

    load_config_builder(str(py_file))

    assert parent_dir not in sys.path


def test_load_config_builder_invalid_yaml(tmp_path: Path) -> None:
    """Test that a YAML file that fails to parse raises ConfigLoadError."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(":\n  - [\n")

    with pytest.raises(ConfigLoadError, match="Failed to parse config file"):
        load_config_builder(str(yaml_file))


def test_load_config_builder_invalid_json(tmp_path: Path) -> None:
    """Test that a malformed JSON file raises ConfigLoadError with a JSON-specific message."""
    json_file = tmp_path / "bad.json"
    json_file.write_text("{not valid json}")

    with pytest.raises(ConfigLoadError, match="Failed to parse config file"):
        load_config_builder(str(json_file))


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_from_config_validation_error(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test that a valid YAML file with invalid config structure raises ConfigLoadError."""
    yaml_file = tmp_path / "bad_structure.yaml"
    yaml_file.write_text("data_designer:\n  not_a_valid_field: true\n")

    mock_from_config.side_effect = Exception("Validation error")

    with pytest.raises(ConfigLoadError, match="Invalid config structure in"):
        load_config_builder(str(yaml_file))


def test_load_config_builder_non_dict_yaml(tmp_path: Path) -> None:
    """Test that a YAML file that parses to a non-dict raises ConfigLoadError."""
    yaml_file = tmp_path / "list.yaml"
    yaml_file.write_text("- item1\n- item2\n")

    with pytest.raises(ConfigLoadError, match="Failed to parse config file"):
        load_config_builder(str(yaml_file))


def test_load_config_builder_non_dict_json(tmp_path: Path) -> None:
    """Test that a JSON file containing an array (not an object) raises ConfigLoadError."""
    json_file = tmp_path / "list.json"
    json_file.write_text('[{"name": "col1"}, {"name": "col2"}]')

    with pytest.raises(ConfigLoadError, match="Expected a JSON object"):
        load_config_builder(str(json_file))


def test_load_config_builder_empty_json(tmp_path: Path) -> None:
    """Test that an empty JSON file raises ConfigLoadError."""
    json_file = tmp_path / "empty.json"
    json_file.write_text("")

    with pytest.raises(ConfigLoadError, match="Failed to parse config file"):
        load_config_builder(str(json_file))


def test_load_config_builder_empty_yaml(tmp_path: Path) -> None:
    """Test that an empty YAML file raises ConfigLoadError."""
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")

    with pytest.raises(ConfigLoadError, match="Failed to parse config file"):
        load_config_builder(str(yaml_file))


# --- Auto-detection of DataDesignerConfig format ---


def test_maybe_wrap_data_designer_config_wraps_when_columns_present() -> None:
    """Test that a dict with 'columns' but no 'data_designer' is wrapped."""
    config_dict = {"columns": [{"name": "col1"}], "model_configs": None}
    result = _maybe_wrap_data_designer_config(config_dict)
    # Wrapping creates a new dict, so we use equality (==) rather than identity (is)
    assert result == {"data_designer": config_dict}


def test_maybe_wrap_data_designer_config_passthrough_when_builder_config() -> None:
    """Test that a dict with 'data_designer' key is returned as-is."""
    config_dict = {"data_designer": {"columns": [{"name": "col1"}]}, "library_version": "1.0.0"}
    result = _maybe_wrap_data_designer_config(config_dict)
    assert result is config_dict


def test_maybe_wrap_data_designer_config_passthrough_when_no_columns() -> None:
    """Test that a dict without 'columns' or 'data_designer' is returned as-is."""
    config_dict = {"some_other_key": "value"}
    result = _maybe_wrap_data_designer_config(config_dict)
    assert result is config_dict


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_auto_wraps_data_designer_config_yaml(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test that a YAML file in DataDesignerConfig format is auto-wrapped."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("columns:\n  - name: col1\nmodel_configs: null\n")

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(yaml_file))

    expected_dict = {"data_designer": {"columns": [{"name": "col1"}], "model_configs": None}}
    mock_from_config.assert_called_once_with(expected_dict)
    assert result is mock_builder


@patch("data_designer.cli.utils.config_loader.DataDesignerConfigBuilder.from_config")
def test_load_config_builder_auto_wraps_data_designer_config_json(mock_from_config: MagicMock, tmp_path: Path) -> None:
    """Test that a JSON file in DataDesignerConfig format is auto-wrapped."""
    json_file = tmp_path / "config.json"
    json_file.write_text(json.dumps({"columns": [{"name": "col1"}]}))

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_from_config.return_value = mock_builder

    result = load_config_builder(str(json_file))

    expected_dict = {"data_designer": {"columns": [{"name": "col1"}]}}
    mock_from_config.assert_called_once_with(expected_dict)
    assert result is mock_builder
