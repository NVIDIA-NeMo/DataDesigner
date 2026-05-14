# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.analysis.column_profilers import ColumnProfilerType
from data_designer.config.base import ConfigBase
from data_designer.engine.analysis.column_profilers.base import ColumnProfiler
from data_designer.engine.analysis.column_profilers.judge_score_profiler import (
    JudgeScoreProfiler,
    JudgeScoreProfilerConfig,
)
from data_designer.engine.registry.base import TaskRegistry


class ColumnProfilerRegistry(TaskRegistry[ColumnProfilerType, ColumnProfiler, ConfigBase]): ...


def create_default_column_profiler_registry() -> ColumnProfilerRegistry:
    registry = ColumnProfilerRegistry()
    registry.register(ColumnProfilerType.JUDGE_SCORE, JudgeScoreProfiler, JudgeScoreProfilerConfig, False)
    return registry
