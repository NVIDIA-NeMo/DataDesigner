# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from urllib.parse import urlparse

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.io_helpers import VALID_CONFIG_FILE_EXTENSIONS, is_http_url


class ConfigLoadError(Exception):
    """Raised when a configuration source cannot be loaded."""


PYTHON_EXTENSIONS = {".py"}
ALL_SUPPORTED_EXTENSIONS = VALID_CONFIG_FILE_EXTENSIONS | PYTHON_EXTENSIONS

USER_MODULE_FUNC_NAME = "load_config_builder"


def load_config_builder(config_source: str) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a file path or URL.

    Auto-detects the file type by extension:
    - .yaml/.yml/.json: Loads as a config file via DataDesignerConfigBuilder.from_config()
      (supports local paths and HTTP(S) URLs)
    - .py: Loads as a Python module and calls its load_config_builder() function

    Args:
        config_source: Path or URL to the configuration file, or path to a Python module.

    Returns:
        A DataDesignerConfigBuilder instance.

    Raises:
        ConfigLoadError: If the file cannot be loaded or is invalid.
    """
    if is_http_url(config_source):
        return _load_from_config_url(config_source)

    path = Path(config_source)

    if not path.exists():
        raise ConfigLoadError(f"Config source not found: {path}")

    if not path.is_file():
        raise ConfigLoadError(f"Config source is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in ALL_SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(ALL_SUPPORTED_EXTENSIONS))
        raise ConfigLoadError(f"Unsupported file extension '{suffix}'. Supported extensions: {supported}")

    if suffix in VALID_CONFIG_FILE_EXTENSIONS:
        return _load_from_config_file(path)

    return _load_from_python_module(path)


def _load_from_config_url(config_source: str) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a remote YAML or JSON config URL."""
    suffix = Path(urlparse(config_source).path).suffix.lower()

    if suffix in PYTHON_EXTENSIONS:
        raise ConfigLoadError(
            f"Remote Python config modules are not supported: {config_source}. "
            "Please provide a local '.py' file instead."
        )

    if suffix not in VALID_CONFIG_FILE_EXTENSIONS:
        supported = ", ".join(sorted(VALID_CONFIG_FILE_EXTENSIONS))
        raise ConfigLoadError(f"Unsupported file extension '{suffix}'. Supported extensions: {supported}")

    return _load_from_config_file(config_source)


def _load_from_config_file(path: Path | str) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a YAML or JSON config file.

    Delegates to ``DataDesignerConfigBuilder.from_config`` which handles file
    parsing and accepts both the full ``BuilderConfig`` format and the
    shorthand ``DataDesignerConfig`` format.

    Args:
        path: Path or URL to the config file.

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

    # Temporarily add the module's parent directory to sys.path so the user's
    # config script can import sibling modules (e.g. helpers in the same folder).
    # We only insert if the directory isn't already on the path, and track
    # whether we did so we can clean it up in the finally block.
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
        # Remove the parent directory we added to sys.path. We use remove()
        # instead of checking sys.path[0] because exec_module could have
        # caused other entries to be inserted at index 0, pushing ours deeper.
        # remove() finds the first occurrence by value, which is ours since we
        # confirmed parent_dir was absent before inserting it.
        if prepended_path:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass
