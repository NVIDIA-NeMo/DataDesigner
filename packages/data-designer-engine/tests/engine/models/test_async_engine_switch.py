# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODEL_PREFIXES = (
    "data_designer.engine.models",
    "data_designer.engine.models_v2",
)


def _matches_models_namespace(module_name: str) -> bool:
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in _MODEL_PREFIXES)


def _purge_models_modules() -> dict[str, ModuleType]:
    saved: dict[str, ModuleType] = {}
    for module_name in list(sys.modules):
        if _matches_models_namespace(module_name):
            module = sys.modules.pop(module_name, None)
            if isinstance(module, ModuleType):
                saved[module_name] = module
    return saved


def _restore_models_modules(saved: dict[str, ModuleType]) -> None:
    for module_name in list(sys.modules):
        if _matches_models_namespace(module_name):
            sys.modules.pop(module_name, None)
    sys.modules.update(saved)


def _module_path_parts(module: ModuleType) -> tuple[str, ...]:
    module_file = module.__file__
    assert module_file is not None
    return Path(module_file).parts


def test_async_engine_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    saved_modules = _purge_models_modules()
    try:
        monkeypatch.delenv("DATA_DESIGNER_ASYNC_ENGINE", raising=False)
        default_facade = importlib.import_module("data_designer.engine.models.facade")
        default_parts = _module_path_parts(default_facade)
        assert "models_v2" not in default_parts
        assert "models" in default_parts

        _purge_models_modules()
        monkeypatch.setenv("DATA_DESIGNER_ASYNC_ENGINE", "1")
        async_facade = importlib.import_module("data_designer.engine.models.facade")
        async_parts = _module_path_parts(async_facade)
        assert "models_v2" in async_parts
    finally:
        _restore_models_modules(saved_modules)
