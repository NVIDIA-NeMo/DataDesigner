# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

from data_designer.config.config_builder import DataDesignerConfigBuilder


class ConfigLoadError(Exception):
    """Raised when a configuration source cannot be loaded."""


YAML_EXTENSIONS = {".yaml", ".yml"}
JSON_EXTENSIONS = {".json"}
PYTHON_EXTENSIONS = {".py"}
CONFIG_FILE_EXTENSIONS = YAML_EXTENSIONS | JSON_EXTENSIONS
ALL_SUPPORTED_EXTENSIONS = CONFIG_FILE_EXTENSIONS | PYTHON_EXTENSIONS

USER_MODULE_FUNC_NAME = "load_config_builder"


def load_config_builder(config_source: str) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a file path.

    Auto-detects the file type by extension:
    - .yaml/.yml/.json: Loads as a config file via DataDesignerConfigBuilder.from_config()
    - .py: Loads as a Python module and calls its load_config_builder() function

    Args:
        config_source: Path to the configuration file or Python module.

    Returns:
        A DataDesignerConfigBuilder instance.

    Raises:
        ConfigLoadError: If the file cannot be loaded or is invalid.
    """
    path = Path(config_source)

    if not path.exists():
        raise ConfigLoadError(f"Config source not found: {path}")

    if not path.is_file():
        raise ConfigLoadError(f"Config source is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in ALL_SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(ALL_SUPPORTED_EXTENSIONS))
        raise ConfigLoadError(f"Unsupported file extension '{suffix}'. Supported extensions: {supported}")

    if suffix in CONFIG_FILE_EXTENSIONS:
        return _load_from_config_file(path)

    return _load_from_python_module(path)


def _load_from_config_file(path: Path) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a YAML or JSON config file.

    Automatically detects if the file contains a DataDesignerConfig (without
    the ``data_designer`` wrapper) and wraps it into a BuilderConfig so that
    users can provide either format.

    Args:
        path: Path to the config file.

    Returns:
        A DataDesignerConfigBuilder instance.

    Raises:
        ConfigLoadError: If the file cannot be parsed or validated.
    """
    config_dict = _parse_config_file(path)
    config_dict = _maybe_wrap_data_designer_config(config_dict)

    try:
        return DataDesignerConfigBuilder.from_config(config_dict)
    except Exception as e:
        raise ConfigLoadError(f"Invalid config structure in '{path}': {e}") from e


def _parse_config_file(path: Path) -> dict:
    """Parse a YAML or JSON config file into a dictionary.

    Dispatches to the appropriate parser based on file extension so that
    error messages are specific to the file format the user provided.

    Args:
        path: Path to the config file.

    Returns:
        The parsed config as a dictionary.

    Raises:
        ConfigLoadError: If the file cannot be parsed or is not a dict.
    """
    suffix = path.suffix.lower()

    try:
        with open(path) as f:
            config = json.load(f) if suffix in JSON_EXTENSIONS else yaml.safe_load(f)
        if not isinstance(config, dict):
            file_type = "JSON object" if suffix in JSON_EXTENSIONS else "YAML mapping"
            raise ValueError(f"Expected a {file_type} (dict), got {type(config).__name__}")
        return config
    except Exception as e:
        raise ConfigLoadError(f"Failed to parse config file '{path}': {e}") from e


def _maybe_wrap_data_designer_config(config_dict: dict) -> dict:
    """Detect if a config dict is a DataDesignerConfig and wrap it as a BuilderConfig.

    A DataDesignerConfig has ``columns`` at the top level but no ``data_designer``
    key, whereas a BuilderConfig nests everything under ``data_designer``. This
    function detects the unwrapped format and wraps it so that
    ``DataDesignerConfigBuilder.from_config`` can process it.

    Args:
        config_dict: The parsed config dictionary.

    Returns:
        The original dict if it is already a BuilderConfig, or a new dict with
        the original nested under a ``data_designer`` key.
    """
    if "columns" in config_dict and "data_designer" not in config_dict:
        return {"data_designer": config_dict}
    return config_dict


def _load_from_python_module(path: Path) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a Python module.

    The module must define a load_config_builder() function that returns
    a DataDesignerConfigBuilder instance.

    Args:
        path: Path to the Python module.

    Returns:
        A DataDesignerConfigBuilder instance.

    Raises:
        ConfigLoadError: If the module cannot be loaded, doesn't define the
            expected function, or the function returns an invalid type.
    """
    module_name = f"_dd_config_{path.resolve().as_posix().replace('/', '_').replace('.', '_')}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigLoadError(f"Failed to create module spec from '{path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    parent_dir = str(path.resolve().parent)
    prepended_path = parent_dir not in sys.path
    if prepended_path:
        sys.path.insert(0, parent_dir)

    try:
        spec.loader.exec_module(module)

        if not hasattr(module, USER_MODULE_FUNC_NAME):
            raise ConfigLoadError(
                f"Python module '{path}' does not define a '{USER_MODULE_FUNC_NAME}()' function. "
                f"Please add a function with signature: "
                f"def {USER_MODULE_FUNC_NAME}() -> DataDesignerConfigBuilder"
            )

        func = getattr(module, USER_MODULE_FUNC_NAME)
        if not callable(func):
            raise ConfigLoadError(f"'{USER_MODULE_FUNC_NAME}' in '{path}' is not callable")

        try:
            config_builder = func()
        except Exception as e:
            raise ConfigLoadError(f"Error calling '{USER_MODULE_FUNC_NAME}()' in '{path}': {e}") from e

        if not isinstance(config_builder, DataDesignerConfigBuilder):
            raise ConfigLoadError(
                f"'{USER_MODULE_FUNC_NAME}()' in '{path}' returned "
                f"{type(config_builder).__name__}, expected DataDesignerConfigBuilder"
            )

        return config_builder

    except ConfigLoadError:
        raise
    except Exception as e:
        raise ConfigLoadError(f"Failed to execute Python module '{path}': {e}") from e
    finally:
        sys.modules.pop(module_name, None)
        if prepended_path and len(sys.path) > 0 and sys.path[0] == parent_dir:
            sys.path.pop(0)
