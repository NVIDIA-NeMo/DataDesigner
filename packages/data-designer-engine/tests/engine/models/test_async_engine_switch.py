# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _clear_models_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "data_designer.engine.models" or module_name.startswith("data_designer.engine.models."):
            sys.modules.pop(module_name, None)
        if module_name == "data_designer.engine.models_v2" or module_name.startswith("data_designer.engine.models_v2."):
            sys.modules.pop(module_name, None)


def _module_path_parts(module: ModuleType) -> tuple[str, ...]:
    module_file = module.__file__
    assert module_file is not None
    return Path(module_file).parts


def test_async_engine_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_DESIGNER_ASYNC_ENGINE", raising=False)
    _clear_models_modules()
    default_facade = importlib.import_module("data_designer.engine.models.facade")
    default_parts = _module_path_parts(default_facade)
    assert "models_v2" not in default_parts
    assert "models" in default_parts

    _clear_models_modules()
    monkeypatch.setenv("DATA_DESIGNER_ASYNC_ENGINE", "1")
    async_facade = importlib.import_module("data_designer.engine.models.facade")
    async_parts = _module_path_parts(async_facade)
    assert "models_v2" in async_parts

    _clear_models_modules()
