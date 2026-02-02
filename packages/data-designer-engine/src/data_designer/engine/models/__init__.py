# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

_ASYNC_ENGINE_ENV_VAR = "DATA_DESIGNER_ASYNC_ENGINE"
_TRUTHY_ENV_VALUES = {"1", "true", "yes"}


def _is_async_engine_enabled() -> bool:
    return os.getenv(_ASYNC_ENGINE_ENV_VAR, "").lower() in _TRUTHY_ENV_VALUES


def _redirect_to_models_v2() -> None:
    models_v2_path = Path(__file__).resolve().parent.parent / "models_v2"
    # Set DATA_DESIGNER_ASYNC_ENGINE before importing this package for it to take effect.
    global __path__
    __path__ = [str(models_v2_path)]
    if __spec__ is not None:
        __spec__.submodule_search_locations = [str(models_v2_path)]


if __name__ == "data_designer.engine.models" and _is_async_engine_enabled():
    _redirect_to_models_v2()
