# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from data_designer.config.config_builder import DataDesignerConfigBuilder


class ConfigLoadError(Exception):
    """Raised when a configuration source cannot be loaded."""


YAML_EXTENSIONS = {".yaml", ".yml"}
JSON_EXTENSIONS = {".json"}
PYTHON_EXTENSIONS = {".py"}
CONFIG_FILE_EXTENSIONS = YAML_EXTENSIONS | JSON_EXTENSIONS
ALL_SUPPORTED_EXTENSIONS = CONFIG_FILE_EXTENSIONS | PYTHON_EXTENSIONS

LOAD_CONFIG_BUILDER_FUNC_NAME = "load_config_builder"


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

    Args:
        path: Path to the config file.

    Returns:
        A DataDesignerConfigBuilder instance.

    Raises:
        ConfigLoadError: If the file cannot be parsed or validated.
    """
    try:
        return DataDesignerConfigBuilder.from_config(path)
    except Exception as e:
        raise ConfigLoadError(f"Failed to load config from '{path}': {e}") from e


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

    # Add the config's parent directory to sys.path so sibling imports work
    parent_dir = str(path.resolve().parent)
    prepended_path = parent_dir not in sys.path
    if prepended_path:
        sys.path.insert(0, parent_dir)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise ConfigLoadError(f"Failed to execute Python module '{path}': {e}") from e
    finally:
        if prepended_path and parent_dir in sys.path:
            sys.path.remove(parent_dir)

    if not hasattr(module, LOAD_CONFIG_BUILDER_FUNC_NAME):
        sys.modules.pop(module_name, None)
        raise ConfigLoadError(
            f"Python module '{path}' does not define a '{LOAD_CONFIG_BUILDER_FUNC_NAME}()' function. "
            f"Please add a function with signature: "
            f"def {LOAD_CONFIG_BUILDER_FUNC_NAME}() -> DataDesignerConfigBuilder"
        )

    func = getattr(module, LOAD_CONFIG_BUILDER_FUNC_NAME)
    if not callable(func):
        sys.modules.pop(module_name, None)
        raise ConfigLoadError(f"'{LOAD_CONFIG_BUILDER_FUNC_NAME}' in '{path}' is not callable")

    try:
        config_builder = func()
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise ConfigLoadError(f"Error calling '{LOAD_CONFIG_BUILDER_FUNC_NAME}()' in '{path}': {e}") from e

    if not isinstance(config_builder, DataDesignerConfigBuilder):
        sys.modules.pop(module_name, None)
        raise ConfigLoadError(
            f"'{LOAD_CONFIG_BUILDER_FUNC_NAME}()' in '{path}' returned {type(config_builder).__name__}, "
            f"expected DataDesignerConfigBuilder"
        )

    # Clean up sys.modules to avoid polluting the module namespace
    sys.modules.pop(module_name, None)

    return config_builder
