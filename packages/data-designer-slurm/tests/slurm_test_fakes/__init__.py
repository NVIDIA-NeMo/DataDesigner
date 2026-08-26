# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic test doubles for the Slurm integration."""

from __future__ import annotations

from slurm_test_fakes.clock import FakeClock
from slurm_test_fakes.dependencies import FakeDependencyInstaller, FakeDependencyResolver
from slurm_test_fakes.services import (
    FakeBatchScriptRenderer,
    FakeBenchmarkBackend,
    FakeImageResolver,
    FakeRunPlanningBackend,
    FakeScriptError,
)
from slurm_test_fakes.serving import FakeLogicalEndpoint, FakeServingState, FakeVllmBackend
from slurm_test_fakes.slurm import (
    FakeCommandResponse,
    FakeSlurmArray,
    FakeSlurmRunner,
    FakeSlurmTask,
)

__all__ = [
    "FakeBatchScriptRenderer",
    "FakeBenchmarkBackend",
    "FakeClock",
    "FakeCommandResponse",
    "FakeDependencyInstaller",
    "FakeDependencyResolver",
    "FakeImageResolver",
    "FakeLogicalEndpoint",
    "FakeRunPlanningBackend",
    "FakeScriptError",
    "FakeServingState",
    "FakeSlurmArray",
    "FakeSlurmRunner",
    "FakeSlurmTask",
    "FakeVllmBackend",
]
