# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.script_params import DataDesignerScriptParams
from data_designer.config.utils.io_helpers import VALID_CONFIG_FILE_EXTENSIONS, is_http_url


class ConfigLoadError(Exception):
    """Raised when a configuration source cannot be loaded."""


class WorkflowHelpRequested(Exception):
    """Raised when a Python workflow prints help and exits successfully."""


PYTHON_EXTENSIONS = {".py"}
ALL_SUPPORTED_EXTENSIONS = VALID_CONFIG_FILE_EXTENSIONS | PYTHON_EXTENSIONS

USER_MODULE_FUNC_NAME = "load_config_builder"


def load_config_builder(
    config_source: str,
    script_params: DataDesignerScriptParams | None = None,
) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a file path or URL.

    Auto-detects the file type by extension:
    - .yaml/.yml/.json: Loads as a config file via DataDesignerConfigBuilder.from_config()
      (supports local paths and HTTP(S) URLs)
    - .py: Loads as a Python module and calls its load_config_builder() function

    Args:
        config_source: Path or URL to the configuration file, or path to a Python module.
        script_params: Optional runtime arguments for Python config workflows.

    Returns:
        A DataDesignerConfigBuilder instance.

    Raises:
        ConfigLoadError: If the file cannot be loaded or is invalid.
    """
    if is_http_url(config_source):
        _reject_script_params_for_static_source(config_source, script_params)
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
        _reject_script_params_for_static_source(str(path), script_params)
        return _load_from_config_file(path)

    return _load_from_python_module(path, script_params)


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


def _load_from_python_module(
    path: Path,
    script_params: DataDesignerScriptParams | None = None,
) -> DataDesignerConfigBuilder:
    """Load a DataDesignerConfigBuilder from a Python module.

    The module must define a load_config_builder() function that returns
    a DataDesignerConfigBuilder instance.

    Args:
        path: Path to the Python module.
        script_params: Optional runtime arguments for Python config workflows.

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

        config_builder = call_config_builder_function(func, str(path), script_params)

        if not isinstance(config_builder, DataDesignerConfigBuilder):
            raise ConfigLoadError(
                f"'{USER_MODULE_FUNC_NAME}()' in '{path}' returned "
                f"{type(config_builder).__name__}, expected DataDesignerConfigBuilder"
            )

        return config_builder

    except (ConfigLoadError, WorkflowHelpRequested):
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


def call_config_builder_function(
    func: Callable[..., Any],
    source_name: str,
    script_params: DataDesignerScriptParams | None = None,
) -> DataDesignerConfigBuilder:
    """Call a user-provided config builder function with a supported signature."""
    params = script_params or DataDesignerScriptParams()
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as e:
        raise ConfigLoadError(f"Could not inspect '{USER_MODULE_FUNC_NAME}()' in '{source_name}': {e}") from e

    config_builder: Any
    if len(signature.parameters) == 0:
        if params.argv:
            raise ConfigLoadError(
                f"'{USER_MODULE_FUNC_NAME}()' in '{source_name}' does not accept workflow arguments. "
                "Update it to accept a DataDesignerScriptParams parameter."
            )
        try:
            config_builder = func()
        except SystemExit as e:
            if _is_successful_system_exit(e):
                raise WorkflowHelpRequested from e
            raise ConfigLoadError(f"'{USER_MODULE_FUNC_NAME}()' in '{source_name}' exited with code {e.code}") from e
        except Exception as e:
            raise ConfigLoadError(f"Error calling '{USER_MODULE_FUNC_NAME}()' in '{source_name}': {e}") from e
    else:
        _validate_params_signature(signature, source_name)
        try:
            config_builder = _call_params_aware_function(func, signature, params)
        except SystemExit as e:
            if _is_successful_system_exit(e):
                raise WorkflowHelpRequested from e
            raise ConfigLoadError(
                f"'{USER_MODULE_FUNC_NAME}(params)' in '{source_name}' exited with code {e.code}"
            ) from e
        except Exception as e:
            raise ConfigLoadError(f"Error calling '{USER_MODULE_FUNC_NAME}(params)' in '{source_name}': {e}") from e

    if not isinstance(config_builder, DataDesignerConfigBuilder):
        raise ConfigLoadError(
            f"'{USER_MODULE_FUNC_NAME}()' in '{source_name}' returned "
            f"{type(config_builder).__name__}, expected DataDesignerConfigBuilder"
        )

    return config_builder


def _validate_params_signature(signature: inspect.Signature, source_name: str) -> None:
    parameters = list(signature.parameters.values())
    if len(parameters) != 1:
        raise ConfigLoadError(
            f"Unsupported '{USER_MODULE_FUNC_NAME}()' signature in '{source_name}'. "
            "Expected zero arguments or one DataDesignerScriptParams parameter."
        )

    parameter = parameters[0]
    supported_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if parameter.kind not in supported_kinds:
        raise ConfigLoadError(
            f"Unsupported '{USER_MODULE_FUNC_NAME}()' signature in '{source_name}'. "
            "Expected zero arguments or one DataDesignerScriptParams parameter."
        )

    if parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.name != "params":
        raise ConfigLoadError(
            f"Unsupported '{USER_MODULE_FUNC_NAME}()' signature in '{source_name}'. "
            "Keyword-only workflow parameters must be named 'params'."
        )


def _call_params_aware_function(
    func: Callable[..., Any],
    signature: inspect.Signature,
    params: DataDesignerScriptParams,
) -> Any:
    parameter = next(iter(signature.parameters.values()))
    if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
        return func(params=params)
    return func(params)


def _reject_script_params_for_static_source(
    source_name: str,
    script_params: DataDesignerScriptParams | None,
) -> None:
    params = script_params or DataDesignerScriptParams()
    if params.argv:
        raise ConfigLoadError(
            f"Workflow arguments are only supported for Python config modules, but '{source_name}' is not a "
            "local Python module."
        )


def _is_successful_system_exit(exc: SystemExit) -> bool:
    return exc.code is None or exc.code == 0
