# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from data_designer._version import __version__
except ImportError:
    # Fallback for editable installs without build
    try:
        from importlib.metadata import version

        __version__ = version("data-designer")
    except Exception:
        __version__ = "0.0.0.dev0+unknown"

# Initialize default model settings on import if running locally
# This ensures default model configs and providers are available throughout the library
try:
    from data_designer.config.default_model_settings import resolve_seed_default_model_settings
    from data_designer.config.utils.misc import can_run_data_designer_locally

    if can_run_data_designer_locally():
        resolve_seed_default_model_settings()
except ImportError:
    # If config module can't be imported, skip initialization
    # This handles edge cases during package installation or incomplete environments
    pass

__all__ = ["__version__"]
