# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.analysis.column_profilers import ColumnProfilerType
from data_designer.engine.analysis.column_profilers.judge_score_profiler import (
    JudgeScoreProfiler,
    JudgeScoreProfilerConfig,
)
from data_designer.engine.analysis.column_profilers.registry import (
    ColumnProfilerRegistry,
    create_default_column_profiler_registry,
)
from data_designer.engine.registry.errors import NotFoundInRegistryError


def _register_test_profiler(registry):
    """Helper function to register test profiler."""
    registry.register(ColumnProfilerType.JUDGE_SCORE, JudgeScoreProfiler, JudgeScoreProfilerConfig, False)


def test_registry_creation():
    registry = ColumnProfilerRegistry()
    assert registry is not None


def test_register_profiler():
    registry = ColumnProfilerRegistry()
    _register_test_profiler(registry)

    assert ColumnProfilerType.JUDGE_SCORE in ColumnProfilerRegistry._registry
    assert ColumnProfilerRegistry._registry[ColumnProfilerType.JUDGE_SCORE] == JudgeScoreProfiler


def test_get_profiler_class():
    registry = ColumnProfilerRegistry()
    _register_test_profiler(registry)

    profiler_class = ColumnProfilerRegistry.get_task_type(ColumnProfilerType.JUDGE_SCORE)
    assert profiler_class == JudgeScoreProfiler


def test_get_nonexistent_profiler():
    ColumnProfilerRegistry._registry.clear()
    ColumnProfilerRegistry._reverse_registry.clear()
    ColumnProfilerRegistry._config_registry.clear()
    ColumnProfilerRegistry._reverse_config_registry.clear()

    with pytest.raises(NotFoundInRegistryError):
        ColumnProfilerRegistry.get_task_type(ColumnProfilerType.JUDGE_SCORE)


def test_create_default_registry():
    registry = create_default_column_profiler_registry()

    assert isinstance(registry, ColumnProfilerRegistry)
    assert ColumnProfilerType.JUDGE_SCORE in ColumnProfilerRegistry._registry
    assert ColumnProfilerRegistry._registry[ColumnProfilerType.JUDGE_SCORE] == JudgeScoreProfiler
