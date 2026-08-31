# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic fakes for public Slurm service boundaries."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Generic, TypeVar

from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.config import (
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    ImageKind,
    ImageRef,
)
from data_designer.slurm.contracts import Identifier
from data_designer.slurm.planning import ResolvedImage, ResolvedSlurmRunPlan
from data_designer.slurm.services import (
    SlurmBatchScriptRenderer,
    SlurmBenchmarkBackend,
    SlurmImageResolver,
    SlurmRunPlanner,
)

_RequestT = TypeVar("_RequestT")
_ResultT = TypeVar("_ResultT")


class FakeScriptError(BaseException):
    """Signal an invalid deterministic-fake interaction."""


class _ScriptedResponses(Generic[_RequestT, _ResultT]):
    def __init__(self, responses: Iterable[tuple[_RequestT, _ResultT | BaseException]]) -> None:
        self._responses = deque(responses)
        self.calls: list[_RequestT] = []

    def next(self, request: _RequestT, *, operation: str) -> _ResultT:
        self.calls.append(request)
        if not self._responses:
            raise FakeScriptError(f"unexpected {operation}")
        expected, response = self._responses.popleft()
        if request != expected:
            raise FakeScriptError(f"expected {operation} {expected!r}, got {request!r}")
        if isinstance(response, BaseException):
            raise response
        return response

    def assert_complete(self, *, operation: str) -> None:
        if self._responses:
            raise FakeScriptError(f"{len(self._responses)} {operation} responses remain")


class FakeRunPlanningBackend(SlurmRunPlanner):
    """Return or raise exact scripted run-planning outcomes."""

    def __init__(
        self,
        responses: Iterable[tuple[DataDesignerSlurmConfig, ResolvedSlurmRunPlan | BaseException]],
    ) -> None:
        self._script = _ScriptedResponses(responses)
        self.calls = self._script.calls

    def plan(self, config: DataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Return the plan scripted for one exact authored config."""
        return self._script.next(config, operation="run plan")

    def assert_complete(self) -> None:
        """Assert that every scripted plan was consumed."""
        self._script.assert_complete(operation="run plan")


class FakeBatchScriptRenderer(SlurmBatchScriptRenderer):
    """Return or raise exact scripted batch-rendering outcomes."""

    def __init__(
        self,
        responses: Iterable[tuple[tuple[ResolvedSlurmRunPlan, int], str | BaseException]],
    ) -> None:
        self._script = _ScriptedResponses(responses)
        self.calls = self._script.calls

    def __call__(self, plan: ResolvedSlurmRunPlan, *, attempt_ordinal: int) -> str:
        """Return the script for one exact plan and attempt ordinal."""
        return self._script.next((plan, attempt_ordinal), operation="batch render")

    def assert_complete(self) -> None:
        """Assert that every scripted render was consumed."""
        self._script.assert_complete(operation="batch render")


class FakeImageResolver(SlurmImageResolver):
    """Return or raise exact scripted image-resolution outcomes."""

    def __init__(
        self,
        responses: Iterable[tuple[tuple[ImageRef, ImageKind], ResolvedImage | BaseException]],
    ) -> None:
        self._script = _ScriptedResponses(responses)
        self.calls = self._script.calls

    def resolve(self, reference: ImageRef, *, expected_kind: ImageKind) -> ResolvedImage:
        """Return the image scripted for one exact reference and kind."""
        return self._script.next((reference, expected_kind), operation="image resolution")

    def assert_complete(self) -> None:
        """Assert that every scripted image resolution was consumed."""
        self._script.assert_complete(operation="image resolution")


class FakeBenchmarkBackend(SlurmBenchmarkBackend):
    """Return or raise exact scripted benchmark outcomes."""

    def __init__(
        self,
        *,
        run_responses: Iterable[tuple[DataDesignerSlurmBenchmarkConfig, BenchmarkManifest | BaseException]] = (),
        analysis_responses: Iterable[tuple[tuple[Identifier, bool], BenchmarkReport | BaseException]] = (),
    ) -> None:
        self._run_script = _ScriptedResponses(run_responses)
        self._analysis_script = _ScriptedResponses(analysis_responses)
        self.run_calls = self._run_script.calls
        self.analysis_calls = self._analysis_script.calls

    def run(self, config: DataDesignerSlurmBenchmarkConfig) -> BenchmarkManifest:
        """Return the manifest scripted for one exact benchmark config."""
        return self._run_script.next(config, operation="benchmark run")

    def analyze(self, benchmark_id: Identifier, *, refresh_state: bool = False) -> BenchmarkReport:
        """Return the report scripted for one benchmark and refresh action."""
        return self._analysis_script.next((benchmark_id, refresh_state), operation="benchmark analysis")

    def assert_complete(self) -> None:
        """Assert that every scripted benchmark response was consumed."""
        self._run_script.assert_complete(operation="benchmark run")
        self._analysis_script.assert_complete(operation="benchmark analysis")
