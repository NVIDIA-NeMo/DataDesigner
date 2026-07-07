# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from data_designer.cli.utils.config_loader import (
    ConfigLoadError,
    load_config_builder,
    load_run_config,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.run_config import JinjaRenderingEngine, RequestAdmissionTuningConfig, RunConfig


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".YAML"])
def test_load_run_config_accepts_yaml_extensions(tmp_path: Path, suffix: str) -> None:
    run_config_file = tmp_path / f"run-config{suffix}"
    run_config_file.write_text("buffer_size: 250\n")

    assert load_run_config(str(run_config_file)).buffer_size == 250


def test_load_run_config_accepts_all_canonical_fields(tmp_path: Path) -> None:
    expected = RunConfig(
        disable_early_shutdown=False,
        shutdown_error_rate=0.25,
        shutdown_error_window=20,
        buffer_size=250,
        max_in_flight_tasks=128,
        non_inference_max_parallel_workers=8,
        max_conversation_restarts=7,
        max_conversation_correction_steps=2,
        async_trace=True,
        display_tui=False,
        progress_interval=10.0,
        preserve_dropped_columns=False,
        jinja_rendering_engine=JinjaRenderingEngine.NATIVE,
        request_admission=RequestAdmissionTuningConfig(
            multiplicative_decrease_factor=0.5,
            additive_increase_step=2,
            successes_until_increase=10,
            cooldown_seconds=1.5,
            startup_ramp_seconds=30.0,
        ),
    )
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text(yaml.safe_dump(expected.model_dump(mode="json"), sort_keys=False))

    loaded = load_run_config(str(run_config_file))

    assert loaded == expected
    assert loaded.model_dump(exclude_unset=True) == expected.model_dump()


def test_load_run_config_accepts_empty_mapping(tmp_path: Path) -> None:
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text("{}\n")

    loaded = load_run_config(str(run_config_file))

    assert loaded.model_dump(exclude_unset=True) == {}


def test_load_run_config_preserves_explicit_partial_fields(tmp_path: Path) -> None:
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text("buffer_size: 250\ndisplay_tui: true\n")

    loaded = load_run_config(str(run_config_file))

    assert loaded.model_dump(exclude_unset=True) == {"buffer_size": 250, "display_tui": True}


def test_load_run_config_preserves_partial_nested_fields(tmp_path: Path) -> None:
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text("request_admission:\n  successes_until_increase: 7\n")

    loaded = load_run_config(str(run_config_file))

    assert loaded.model_dump(exclude_unset=True) == {"request_admission": {"successes_until_increase": 7}}


def test_load_run_config_canonicalizes_deprecated_aliases(tmp_path: Path) -> None:
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text("progress_bar: false\nthrottle:\n  reduce_factor: 0.5\n  success_window: 7\n")

    with pytest.warns(DeprecationWarning) as caught:
        loaded = load_run_config(str(run_config_file))

    explicit = loaded.model_dump(exclude_unset=True)
    assert len(caught) == 2
    assert explicit["display_tui"] is False
    assert explicit["request_admission"]["multiplicative_decrease_factor"] == 0.5
    assert explicit["request_admission"]["successes_until_increase"] == 7
    assert "progress_bar" not in explicit
    assert "throttle" not in explicit


def test_load_run_config_rejects_missing_file(tmp_path: Path) -> None:
    run_config_file = tmp_path / "missing.yaml"

    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(str(run_config_file))

    assert str(run_config_file) in str(exc_info.value)
    assert "file not found" in str(exc_info.value)


def test_load_run_config_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(str(tmp_path))

    assert str(tmp_path) in str(exc_info.value)
    assert "not a file" in str(exc_info.value)


def test_load_run_config_rejects_url() -> None:
    config_url = "https://user:password@example.com/run-config.yaml?token=secret#fragment"

    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(config_url)

    message = str(exc_info.value)
    assert "https://example.com/run-config.yaml" in message
    assert "password" not in message
    assert "secret" not in message
    assert "fragment" not in message
    assert "remote URLs are not supported" in message


def test_load_run_config_wraps_malformed_url() -> None:
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config("http://[invalid/run-config.yaml?token=secret")

    message = str(exc_info.value)
    assert "<invalid remote URL>" in message
    assert "secret" not in message
    assert "remote URLs are not supported" in message


def test_load_run_config_rejects_unsupported_extension(tmp_path: Path) -> None:
    run_config_file = tmp_path / "run-config.json"
    run_config_file.write_text("{}\n")

    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(str(run_config_file))

    assert str(run_config_file) in str(exc_info.value)
    assert "unsupported file extension" in str(exc_info.value)


@patch("data_designer.cli.utils.config_loader.smart_load_yaml", side_effect=PermissionError("Permission denied"))
def test_load_run_config_wraps_unreadable_file(mock_load_yaml: MagicMock, tmp_path: Path) -> None:
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text("{}\n")

    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(str(run_config_file))

    mock_load_yaml.assert_called_once_with(run_config_file)
    assert str(run_config_file) in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)


@patch("data_designer.cli.utils.config_loader.Path.exists", side_effect=OSError("path probe failed"))
def test_load_run_config_wraps_path_probe_error(mock_exists: MagicMock) -> None:
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config("run-config.yaml")

    mock_exists.assert_called_once_with()
    assert "run-config.yaml" in str(exc_info.value)
    assert "path probe failed" in str(exc_info.value)


def test_load_run_config_omits_invalid_values_from_errors(tmp_path: Path) -> None:
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text("api_key: sk-live-secret\n")

    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(str(run_config_file))

    message = str(exc_info.value)
    assert "api_key" in message
    assert "Extra inputs are not permitted" in message
    assert "sk-live-secret" not in message
    assert "input_value" not in message


@pytest.mark.parametrize(
    ("content", "expected_detail"),
    [
        ("", "NoneType"),
        (":\n  - [\n", "while parsing"),
        ("runtime\n", "str"),
        ("- buffer_size\n", "list"),
        ("unknown_field: 1\n", "unknown_field"),
        ("buffer_size: 0\n", "buffer_size"),
        ("request_admission:\n  successes_until_increase: 0\n", "successes_until_increase"),
    ],
    ids=["blank", "malformed", "scalar", "list", "unknown", "invalid-scalar", "invalid-nested"],
)
def test_load_run_config_wraps_invalid_content(tmp_path: Path, content: str, expected_detail: str) -> None:
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text(content)

    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(str(run_config_file))

    assert str(run_config_file) in str(exc_info.value)
    assert expected_detail in str(exc_info.value)


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
