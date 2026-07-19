# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
import tracemalloc
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import data_designer.engine.dataset_builders.async_scheduler as async_scheduler_module
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import SkipConfig
from data_designer.config.column_configs import (
    CustomColumnConfig,
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.config.sampler_params import SamplerType
from data_designer.config.scheduling import SchedulingMetadata
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorFullColumn,
    ColumnGeneratorWithModelRegistry,
    FromScratchColumnGenerator,
)
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.context import current_row_group_start_offset
from data_designer.engine.dataset_builders.async_scheduler import AsyncTaskScheduler
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.row_group_plan import CompactRowGroupPlan
from data_designer.engine.dataset_builders.scheduling.completion import CompletionTracker, FrontierDelta
from data_designer.engine.dataset_builders.scheduling.task_admission import TaskAdmissionConfig, TaskAdmissionLease
from data_designer.engine.dataset_builders.scheduling.task_model import Task
from data_designer.engine.dataset_builders.scheduling.task_policies import BoundedBorrowTaskAdmissionPolicyConfig
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.models.errors import (
    RETRYABLE_MODEL_ERRORS,
    ModelAPIConnectionError,
    ModelInternalServerError,
    ModelRateLimitError,
    ModelRequestAdmissionTimeoutError,
    ModelTimeoutError,
)
from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.controller import (
    AdaptiveRequestAdmissionController,
    RequestAdmissionLease,
)
from data_designer.engine.models.request_admission.outcomes import RequestReleaseOutcome
from data_designer.engine.models.request_admission.pressure import RequestPressureSnapshot
from data_designer.engine.models.request_admission.resources import (
    RequestAdmissionItem,
    RequestDomain,
    RequestGroupSpec,
    RequestResourceKey,
)
from data_designer.engine.models.resources import ProviderModelKey
from data_designer.engine.observability import (
    RuntimeCorrelation,
    SchedulerAdmissionEventKind,
    runtime_correlation_provider,
)
from data_designer.engine.progress.reporter import AsyncProgressReporter
from data_designer.engine.progress.tracker import ProgressTracker
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.testing import InMemoryAdmissionEventSink

MODEL_ALIAS = "stub"


# -- Mock generators -----------------------------------------------------------


def _mock_provider() -> MagicMock:
    return MagicMock(spec=ResourceProvider)


def _expr_config(name: str = "test") -> ExpressionColumnConfig:
    return ExpressionColumnConfig(name=name, expr="{{ x }}", dtype="str")


class MockSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
    """Mock from-scratch generator that produces a DataFrame."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        return lazy.pd.DataFrame({self.config.name: list(range(num_records))})


class MockCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Mock cell-by-cell generator."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"processed_{data.get('seed', '?')}"
        return data


class MockRootCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Root cell generator that records the shape it receives."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.call_types: list[str] = []

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        self.call_types.append(type(data).__name__)
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        data[self.config.name] = f"root_{len(self.call_types)}"
        return data


class MockFullColumnGenerator(ColumnGeneratorFullColumn[ExpressionColumnConfig]):
    """Mock full-column generator."""

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        data[self.config.name] = "batch_val"
        return data


class MockStatefulSeed(FromScratchColumnGenerator[ExpressionColumnConfig]):
    """Stateful mock seed generator."""

    @property
    def is_order_dependent(self) -> bool:
        return True

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        return lazy.pd.DataFrame({self.config.name: list(range(num_records))})


class MockFailingSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
    """Seed generator that always fails with a non-retryable error."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        raise ValueError("permanent seed failure")

    async def agenerate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        raise ValueError("permanent seed failure")


class MockFailingGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Generator that fails with a configurable error.

    By default fails permanently. Set ``transient_failures`` to make the first
    N calls fail with a retryable 503 error before succeeding.
    """

    def __init__(self, *args: Any, transient_failures: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._transient_failures = transient_failures
        self._calls = 0

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        self._calls += 1
        if self._transient_failures > 0 and self._calls <= self._transient_failures:
            raise ModelInternalServerError("503 Service Unavailable")
        if self._transient_failures == 0:
            raise ValueError("permanent failure")
        data[self.config.name] = f"recovered_{data.get('seed', '?')}"
        return data


class MockBuggyGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Generator that raises a bare built-in exception from generator code."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, _data: dict) -> dict:
        raise KeyError("missing internal key")


class MockBuggyFromScratchGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
    """From-scratch generator that raises a bare built-in exception from generator code."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, _num_records: int) -> lazy.pd.DataFrame:
        raise AssertionError("invalid seed source")


class MockMalformedFromScratchGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
    """From-scratch generator that returns a non-DataFrame object."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> Any:
        return [{"seed": index} for index in range(num_records)]


class MockBuggyFullColumnGenerator(ColumnGeneratorFullColumn[ExpressionColumnConfig]):
    """Full-column generator that raises a bare built-in exception from generator code."""

    def generate(self, _data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        raise TypeError("bad batch shape")


class MockMalformedFullColumnGenerator(ColumnGeneratorFullColumn[ExpressionColumnConfig]):
    """Full-column generator that returns a non-DataFrame object."""

    def generate(self, data: lazy.pd.DataFrame) -> Any:
        return [{"seed": value, self.config.name: "bad"} for value in data.get("seed", [])]


class MockRateLimitGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Generator that fails with rate-limit errors before succeeding.

    The first ``rate_limit_failures`` calls raise ``ModelRateLimitError``,
    then all subsequent calls succeed.
    """

    def __init__(self, *args: Any, rate_limit_failures: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_failures = rate_limit_failures
        self._calls = 0

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        self._calls += 1
        if self._calls <= self._rate_limit_failures:
            raise ModelRateLimitError("429 Too Many Requests")
        data[self.config.name] = f"ok_{data.get('seed', '?')}"
        return data


class MockSelectiveFailGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Cell generator with deterministic per-seed behavior.

    - Seeds in ``fail_on_seeds``: raise a non-retryable ``ValueError`` immediately.
    - Seeds in ``slow_seeds``: block on ``asyncio.sleep`` so they remain
      in-flight when the early-shutdown gate fires.
    - All others: succeed.

    Cell-by-cell only — exercised through ``agenerate`` from the async scheduler.
    """

    def __init__(
        self,
        *args: Any,
        fail_on_seeds: set[int] = frozenset(),
        slow_seeds: set[int] = frozenset(),
        slow_timeout_s: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._fail = set(fail_on_seeds)
        self._slow = set(slow_seeds)
        self._slow_timeout_s = slow_timeout_s

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    async def agenerate(self, data: dict) -> dict:
        seed = data.get("seed")
        if seed in self._fail:
            raise ValueError(f"non-retryable on seed={seed}")
        if seed in self._slow:
            await asyncio.sleep(self._slow_timeout_s)
        data[self.config.name] = f"ok_{seed}"
        return data

    def generate(self, data: dict) -> dict:
        # Sync path: kept minimal because this mock is exercised exclusively
        # through ``agenerate`` from the async scheduler. ``slow_seeds`` is
        # intentionally not honored here — callers needing sync slow behavior
        # should use a different fixture.
        seed = data.get("seed")
        if seed in self._fail:
            raise ValueError(f"non-retryable on seed={seed}")
        data[self.config.name] = f"ok_{seed}"
        return data


class MockRetryableErrorGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Generator that raises a parametrizable retryable error then succeeds.

    Declares model scheduling metadata because it mimics model-call behavior;
    the scheduler's degraded-provider WARN window counts model-stage tasks.
    """

    def __init__(
        self,
        *args: Any,
        error_factory: Callable[[], Exception],
        retryable_failures: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._error_factory = error_factory
        self._retryable_failures = retryable_failures
        self._calls = 0

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.custom_model("test", self.config.name, "v1")

    def generate(self, data: dict) -> dict:
        self._calls += 1
        if self._calls <= self._retryable_failures:
            raise self._error_factory()
        data[self.config.name] = f"ok_{data.get('seed', '?')}"
        return data


class _BrokenSchedulerSink:
    def emit_scheduler_event(self, _event: object) -> None:
        raise RuntimeError("sink boom")


class _FilteringSchedulerSink(InMemoryAdmissionEventSink):
    def __init__(self, accepted: set[SchedulerAdmissionEventKind]) -> None:
        super().__init__()
        self._accepted = accepted

    def accepts_scheduler_event(self, event_kind: SchedulerAdmissionEventKind) -> bool:
        return event_kind in self._accepted


# -- Helper to build graph + scheduler ----------------------------------------


def _build_simple_pipeline(
    num_records: int = 3,
    buffer_size: int = 3,
    trace: bool = False,
    generators: dict[str, ColumnGenerator] | None = None,
    configs: list[SamplerColumnConfig | LLMTextColumnConfig | ExpressionColumnConfig] | None = None,
    strategies: dict[str, GenerationStrategy] | None = None,
    scheduler_event_sink: Any | None = None,
    max_concurrent_row_groups: int = 3,
    adaptive_row_group_admission: bool = False,
    adaptive_row_group_initial_target: int = 1,
) -> tuple[AsyncTaskScheduler, CompletionTracker]:
    """Build a simple seed → cell pipeline for testing."""
    if configs is None:
        configs = [
            SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
            LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ]
    if strategies is None:
        strategies = {
            "seed": GenerationStrategy.FULL_COLUMN,
            "cell_out": GenerationStrategy.CELL_BY_CELL,
        }
    if generators is None:
        provider = _mock_provider()
        generators = {
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
        }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, num_records)] if num_records <= buffer_size else []
    if not row_groups:
        remaining = num_records
        rg_id = 0
        while remaining > 0:
            size = min(buffer_size, remaining)
            row_groups.append((rg_id, size))
            remaining -= size
            rg_id += 1

    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_concurrent_row_groups=max_concurrent_row_groups,
        trace=trace,
        scheduler_event_sink=scheduler_event_sink,
        adaptive_row_group_admission=adaptive_row_group_admission,
        adaptive_row_group_initial_target=adaptive_row_group_initial_target,
    )
    return scheduler, tracker


def _make_storage() -> MagicMock:
    """Standard mock storage for buffer-manager-backed scheduler tests."""
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    return storage


def test_scheduler_preparation_memory_stays_bounded_for_million_row_groups() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }
    graph = ExecutionGraph.create(configs, strategies)

    tracemalloc.start()
    try:
        row_groups = CompactRowGroupPlan.fresh(num_records=2_000_000, buffer_size=2)
        tracker = CompletionTracker.with_graph(graph, row_groups)
        scheduler = AsyncTaskScheduler(
            generators=generators,
            graph=graph,
            tracker=tracker,
            row_groups=row_groups,
            num_records=2_000_000,
            buffer_size=2,
        )
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(row_groups) == 1_000_000
    assert row_groups.scheduled_total_rows == 2_000_000
    assert scheduler._scheduled_records == 2_000_000
    assert peak_bytes < 5 * 1024 * 1024


def _seed_plus_cell_setup(
    cell_generator: ColumnGenerator,
    num_records: int,
) -> tuple[
    dict[str, ColumnGenerator],
    ExecutionGraph,
    list[tuple[int, int]],
    CompletionTracker,
    RowGroupBufferManager,
    MagicMock,
]:
    """Build the shared seed → LLM cell pipeline scaffolding (no scheduler yet).

    Used by early-shutdown / WARN tests that need a real ``buffer_manager``
    *before* constructing the scheduler (e.g. to wire a checkpoint callback
    that closes over it).
    """
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN, "cell_out": GenerationStrategy.CELL_BY_CELL}
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": cell_generator,
    }
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, num_records)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    storage = _make_storage()
    buffer_manager = RowGroupBufferManager(storage)
    return generators, graph, row_groups, tracker, buffer_manager, storage


# -- Tests --------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_dispatches_seeds_first() -> None:
    """Seeds (no upstream) are dispatched before downstream columns."""
    scheduler, tracker = _build_simple_pipeline(num_records=2, trace=True)
    await scheduler.run()

    # All tasks should be complete
    assert tracker.is_row_group_complete(0, 2, ["seed", "cell_out"])

    # Verify dispatch order: seeds before cells
    seed_traces = [t for t in scheduler.traces if t.column == "seed"]
    cell_traces = [t for t in scheduler.traces if t.column == "cell_out"]
    assert len(seed_traces) == 1  # one batch task
    assert len(cell_traces) == 2  # two cell tasks
    assert seed_traces[0].dispatched_at < cell_traces[0].dispatched_at


@pytest.mark.asyncio(loop_scope="session")
async def test_request_cancel_wakes_scheduler_and_drains_workers() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingCellGenerator(MockCellGenerator):
        async def agenerate(self, data: dict) -> dict:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return data

    provider = _mock_provider()
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": BlockingCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }
    scheduler, _tracker = _build_simple_pipeline(num_records=1, generators=generators)
    run_task = asyncio.create_task(scheduler.run())

    await asyncio.wait_for(started.wait(), timeout=2)
    await asyncio.to_thread(scheduler.request_cancel)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=2)
    assert cancelled.is_set()
    assert scheduler.active_worker_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_dispatches_root_cell_by_cell_columns_per_row() -> None:
    provider = _mock_provider()
    generator = MockRootCellGenerator(config=_expr_config("root_cell"), resource_provider=provider)
    configs = [SamplerColumnConfig(name="root_cell", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})]
    strategies = {"root_cell": GenerationStrategy.CELL_BY_CELL}
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators={"root_cell": generator},
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        trace=True,
    )

    await scheduler.run()

    assert generator.call_types == ["dict", "dict", "dict"]
    assert [trace.task_type for trace in scheduler.traces] == ["cell", "cell", "cell"]
    assert not any(tracker.is_dropped(0, row_index) for row_index in range(3))
    assert tracker.is_row_group_complete(0, 3, ["root_cell"])


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_with_buffer_manager() -> None:
    """Scheduler writes results to buffer manager and checkpoints."""
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    buffer_mgr = RowGroupBufferManager(storage)
    provider = _mock_provider()

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    checkpointed: list[int] = []

    def finalize(rg_id: int) -> None:
        buffer_mgr.checkpoint_row_group(rg_id)
        checkpointed.append(rg_id)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_finalize_row_group=finalize,
    )
    await scheduler.run()

    assert 0 in checkpointed
    assert buffer_mgr.actual_num_records == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_multiple_row_groups() -> None:
    """Scheduler handles multiple row groups."""
    scheduler, tracker = _build_simple_pipeline(num_records=5, buffer_size=2, trace=True)
    await scheduler.run()

    # 3 row groups: (0, 2), (1, 2), (2, 1)
    assert tracker.is_row_group_complete(0, 2, ["seed", "cell_out"])
    assert tracker.is_row_group_complete(1, 2, ["seed", "cell_out"])
    assert tracker.is_row_group_complete(2, 1, ["seed", "cell_out"])


class _OffsetSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
    """Synthetic seed generator that emits ``[offset, offset + n)`` for a row group."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        offset = current_row_group_start_offset.get()
        assert offset is not None
        return lazy.pd.DataFrame({"seed": list(range(offset, offset + num_records))})


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_sets_row_group_start_offsets_for_generators() -> None:
    """Ordered generators can seek by planned row-group offset during async resume."""
    provider = _mock_provider()
    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": _OffsetSeedGenerator(config=_expr_config("seed"), resource_provider=provider)}
    row_groups = CompactRowGroupPlan.resume(
        original_target=4,
        num_records=4,
        buffer_size=1,
        completed_ids={0, 2},
    )

    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    storage = _make_storage()
    buffer_manager = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
    )
    await scheduler.run()

    assert buffer_manager.get_row(1, 0)["seed"] == 1
    assert buffer_manager.get_row(3, 0)["seed"] == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_auto_computes_row_group_start_offsets_for_fresh_runs() -> None:
    """Fresh async runs (no caller-supplied offsets) auto-derive offsets from row-group sizes.

    This locks in the per-row-group seek behavior for ordered generators on fresh
    runs. Previously the scheduler relied on a single shared seed reader whose
    state advanced under a stateful lock; now each row group seeks to its own
    planned offset, which is parallel-safe and order-independent.
    """
    provider = _mock_provider()
    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": _OffsetSeedGenerator(config=_expr_config("seed"), resource_provider=provider)}
    row_groups = [(0, 2), (1, 2), (2, 1)]  # non-aligned last group exercises offset accumulation

    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    storage = _make_storage()
    buffer_manager = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        # row_group_start_offsets intentionally omitted — scheduler should derive
        # {0: 0, 1: 2, 2: 4} from the row-group sizes.
    )
    await scheduler.run()

    assert buffer_manager.get_row(0, 0)["seed"] == 0
    assert buffer_manager.get_row(0, 1)["seed"] == 1
    assert buffer_manager.get_row(1, 0)["seed"] == 2
    assert buffer_manager.get_row(1, 1)["seed"] == 3
    assert buffer_manager.get_row(2, 0)["seed"] == 4


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_non_retryable_failure_drops_row() -> None:
    """Non-retryable failure drops the row."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="fail_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "fail_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "fail_col": MockFailingGenerator(config=_expr_config("fail_col"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
    )
    await scheduler.run()

    # All rows should be dropped since all fail non-retryably
    assert tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    # Row group is "complete" because all non-dropped rows have all columns
    # (there are no non-dropped rows)
    assert tracker.is_row_group_complete(0, 2, ["seed", "fail_col"])


def test_scheduler_internal_bug_classifier_preserves_scheduler_builtin_failures() -> None:
    scheduler, tracker = _build_simple_pipeline(num_records=1)
    assert scheduler._is_internal_bug(KeyError("missing internal key"))
    assert not scheduler._is_internal_bug(DatasetGenerationError("generator failure"))
    assert not tracker.is_dropped(0, 0)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_generator_builtin_exception_drops_cell_without_fatal_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _mock_provider()
    scheduler, tracker = _build_simple_pipeline(
        num_records=1,
        configs=[
            SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
            LLMTextColumnConfig(name="buggy_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ],
        strategies={
            "seed": GenerationStrategy.FULL_COLUMN,
            "buggy_col": GenerationStrategy.CELL_BY_CELL,
        },
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "buggy_col": MockBuggyGenerator(config=_expr_config("buggy_col"), resource_provider=provider),
        },
    )

    with caplog.at_level(logging.WARNING, logger="data_designer.engine.dataset_builders.async_scheduler"):
        await scheduler.run()

    assert tracker.is_dropped(0, 0)
    assert isinstance(scheduler.first_non_retryable_error, DatasetGenerationError)
    assert isinstance(scheduler.first_non_retryable_error.__cause__, KeyError)
    assert "Unexpected fatal Non-retryable failure" not in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_generator_builtin_exception_drops_from_scratch_group_without_fatal_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _mock_provider()
    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": MockBuggyFromScratchGenerator(config=_expr_config("seed"), resource_provider=provider)}
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, [(0, 2)])
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=[(0, 2)],
    )

    with caplog.at_level(logging.WARNING, logger="data_designer.engine.dataset_builders.async_scheduler"):
        await scheduler.run()

    assert tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    assert isinstance(scheduler.first_non_retryable_error, DatasetGenerationError)
    assert isinstance(scheduler.first_non_retryable_error.__cause__, AssertionError)
    assert "Unexpected fatal Non-retryable failure" not in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_generator_builtin_exception_drops_batch_group_without_fatal_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="buggy_batch", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "buggy_batch": GenerationStrategy.FULL_COLUMN,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "buggy_batch": MockBuggyFullColumnGenerator(
            config=_expr_config("buggy_batch"),
            resource_provider=provider,
        ),
    }
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=RowGroupBufferManager(_make_storage()),
    )

    with caplog.at_level(logging.WARNING, logger="data_designer.engine.dataset_builders.async_scheduler"):
        await scheduler.run()

    assert tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    assert isinstance(scheduler.first_non_retryable_error, DatasetGenerationError)
    assert isinstance(scheduler.first_non_retryable_error.__cause__, TypeError)
    assert "Unexpected fatal Non-retryable failure" not in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_generator_malformed_from_scratch_return_drops_group_without_fatal_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _mock_provider()
    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": MockMalformedFromScratchGenerator(config=_expr_config("seed"), resource_provider=provider)}
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=RowGroupBufferManager(_make_storage()),
    )

    with caplog.at_level(logging.WARNING, logger="data_designer.engine.dataset_builders.async_scheduler"):
        await scheduler.run()

    assert tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    assert isinstance(scheduler.first_non_retryable_error, DatasetGenerationError)
    assert "must return a DataFrame, got list" in str(scheduler.first_non_retryable_error)
    assert "Unexpected fatal Non-retryable failure" not in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_generator_malformed_batch_return_drops_group_without_fatal_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="malformed_batch", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "malformed_batch": GenerationStrategy.FULL_COLUMN,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "malformed_batch": MockMalformedFullColumnGenerator(
            config=_expr_config("malformed_batch"),
            resource_provider=provider,
        ),
    }
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=RowGroupBufferManager(_make_storage()),
    )

    with caplog.at_level(logging.WARNING, logger="data_designer.engine.dataset_builders.async_scheduler"):
        await scheduler.run()

    assert tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    assert isinstance(scheduler.first_non_retryable_error, DatasetGenerationError)
    assert "must return a DataFrame, got list" in str(scheduler.first_non_retryable_error)
    assert "Unexpected fatal Non-retryable failure" not in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_custom_generator_key_error_drops_row_without_fatal_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @custom_column_generator()
    def failing_custom(row: dict) -> dict:
        raise KeyError("missing user field")

    provider = _mock_provider()
    custom_config = CustomColumnConfig(name="custom_col", generator_function=failing_custom)
    scheduler, tracker = _build_simple_pipeline(
        num_records=1,
        configs=[
            SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
            custom_config,
        ],
        strategies={
            "seed": GenerationStrategy.FULL_COLUMN,
            "custom_col": GenerationStrategy.CELL_BY_CELL,
        },
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "custom_col": CustomColumnGenerator(config=custom_config, resource_provider=provider),
        },
    )

    with caplog.at_level(logging.WARNING):
        await scheduler.run()

    assert tracker.is_dropped(0, 0)
    assert "This record will be skipped" in caplog.text
    assert "Unexpected fatal Non-retryable failure" not in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_logs_sink_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="data_designer.engine.dataset_builders.async_scheduler")
    scheduler, tracker = _build_simple_pipeline(num_records=1, scheduler_event_sink=_BrokenSchedulerSink())

    await scheduler.run()

    assert tracker.is_row_group_complete(0, 1, ["seed", "cell_out"])
    assert "Scheduler admission event sink raised; dropping event." in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_stateful_generator_serializes() -> None:
    """Stateful generators serialize across row groups."""
    provider = _mock_provider()
    gen = MockStatefulSeed(config=_expr_config("seed"), resource_provider=provider)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2), (1, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        trace=True,
    )
    await scheduler.run()

    # Both row groups should complete
    assert tracker.is_row_group_complete(0, 2, ["seed"])
    assert tracker.is_row_group_complete(1, 2, ["seed"])

    # Stateful: verify both row groups completed (the lock ensures serial
    # execution, but sub-microsecond mock generators make timestamp-based
    # ordering assertions flaky)
    assert len(scheduler.traces) == 2
    rg_ids = [t.row_group for t in scheduler.traces]
    assert set(rg_ids) == {0, 1}


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_bounded_in_flight_tasks() -> None:
    """In-flight task count respects max_in_flight_tasks."""
    provider = _mock_provider()

    # Use a pipeline with many cells and low submission limit
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 5)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=2,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, 5, ["seed", "cell_out"])


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_trace_disabled_by_default() -> None:
    """Traces are empty when trace=False (default)."""
    scheduler, _ = _build_simple_pipeline(num_records=2)
    await scheduler.run()

    assert len(scheduler.traces) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_trace_enabled() -> None:
    """Traces are populated when trace=True."""
    scheduler, _ = _build_simple_pipeline(num_records=2, trace=True)
    await scheduler.run()

    assert len(scheduler.traces) > 0
    for t in scheduler.traces:
        assert t.dispatched_at > 0
        assert t.completed_at >= t.dispatched_at
        assert t.status in ("ok", "error")


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_three_column_pipeline() -> None:
    """Test a three-column pipeline: seed → cell → full_column."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="full_out", expr="{{ cell_out }}"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
        "full_out": GenerationStrategy.FULL_COLUMN,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
        "full_out": MockFullColumnGenerator(config=_expr_config("full_out"), resource_provider=provider),
    }

    scheduler, tracker = _build_simple_pipeline(
        num_records=3,
        generators=generators,
        configs=configs,
        strategies=strategies,
        trace=True,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, 3, ["seed", "cell_out", "full_out"])


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_retryable_failure_recovers_in_salvage() -> None:
    """Transient (retryable) failures are retried in salvage rounds and succeed."""
    provider = _mock_provider()
    # Fail the first 2 calls with 503, then succeed
    fail_gen = MockFailingGenerator(config=_expr_config("fail_col"), resource_provider=provider, transient_failures=2)
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="fail_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "fail_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "fail_col": fail_gen,
    }
    scheduler, tracker = _build_simple_pipeline(
        num_records=2, generators=generators, configs=configs, strategies=strategies
    )
    await scheduler.run()

    # Rows should NOT be dropped - salvage recovered them
    assert not tracker.is_dropped(0, 0)
    assert not tracker.is_dropped(0, 1)
    assert tracker.is_row_group_complete(0, 2, ["seed", "fail_col"])


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_eager_row_drop_skips_downstream_of_failed_column() -> None:
    """When fail_col drops a row, a downstream column never processes it."""
    provider = _mock_provider()

    # Pipeline: seed -> fail_col (cell, permanent failure) -> downstream (cell)
    # downstream depends on fail_col, so its tasks only enter the frontier
    # after fail_col completes for each row. Since fail_col always fails,
    # the row is dropped before downstream is ever enqueued.
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="fail_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="downstream", prompt="{{ fail_col }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "fail_col": GenerationStrategy.CELL_BY_CELL,
        "downstream": GenerationStrategy.CELL_BY_CELL,
    }
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "fail_col": MockFailingGenerator(config=_expr_config("fail_col"), resource_provider=provider),
        "downstream": MockCellGenerator(config=_expr_config("downstream"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        trace=True,
        num_records=2,
        buffer_size=2,
    )
    await scheduler.run()

    # All rows dropped by fail_col
    assert tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    # downstream was never dispatched for the dropped rows
    downstream_traces = [t for t in scheduler.traces if t.column == "downstream"]
    assert len(downstream_traces) == 0
    # Row group is still "complete" (no non-dropped rows remain)
    assert tracker.is_row_group_complete(0, 2, ["seed", "fail_col", "downstream"])
    assert scheduler._reporter is not None
    assert scheduler._reporter._trackers["fail_col"].failed == 2
    assert scheduler._reporter._trackers["downstream"].skipped == 2
    assert scheduler._reporter._trackers["downstream"].completed == 2


def test_resume_progress_reporter_starts_from_completed_records(caplog: pytest.LogCaptureFixture) -> None:
    """Resume progress should include persisted records while logging only remaining scheduled work."""
    trackers = {
        "cell_a": ProgressTracker(total_records=1000, label="column 'cell_a'", quiet=True, initial_completed=252),
        "cell_b": ProgressTracker(total_records=1000, label="column 'cell_b'", quiet=True, initial_completed=252),
    }
    completed, total, _success, _failed, _skipped, _pct, rate, _emoji = trackers["cell_a"].get_snapshot(elapsed=1.0)
    assert completed == 252
    assert total == 1000
    assert rate == 0.0

    trackers["cell_a"].record_success()
    completed, _total, _success, _failed, _skipped, _pct, rate, _emoji = trackers["cell_a"].get_snapshot(elapsed=1.0)
    assert completed == 253
    assert rate == 1.0

    reporter = AsyncProgressReporter(trackers)
    with caplog.at_level(logging.INFO):
        reporter.log_start(num_row_groups=2, scheduled_records=128)

    assert "256 tasks across 2 row group(s)" in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_non_retryable_seed_failure_no_keyerror_on_downstream() -> None:
    """Non-retryable seed failure does not cause KeyError on vacuously-ready downstream.

    Pipeline: seed (full_column) -> cell_out (cell_by_cell) -> full_out (full_column).
    When seed fails non-retryably, all rows are dropped. cell_out's cell tasks
    become vacuously complete (all rows dropped), which makes full_out ready.
    full_out must not crash with a KeyError when its row group buffer has been
    checkpointed.
    """
    provider = _mock_provider()
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="full_out", expr="{{ cell_out }}"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
        "full_out": GenerationStrategy.FULL_COLUMN,
    }
    generators: dict[str, ColumnGenerator] = {
        "seed": MockFailingSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
        "full_out": MockFullColumnGenerator(config=_expr_config("full_out"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    buffer_mgr = RowGroupBufferManager(storage)

    finalized: list[int] = []

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_finalize_row_group=lambda rg: finalized.append(rg),
        trace=True,
        num_records=3,
        buffer_size=3,
    )
    await scheduler.run()

    # All rows dropped due to seed failure
    for ri in range(3):
        assert tracker.is_dropped(0, ri)

    # Row group is NOT finalized when all rows are dropped (freed instead)
    assert 0 not in finalized

    # full_out was either never dispatched or silently skipped (no KeyError)
    full_out_errors = [t for t in scheduler.traces if t.column == "full_out" and t.status == "error"]
    assert len(full_out_errors) == 0
    assert scheduler._reporter is not None
    assert scheduler._reporter._trackers["cell_out"].skipped == 3
    assert scheduler._reporter._trackers["cell_out"].completed == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_pre_batch_failure_raises() -> None:
    """Pre-batch processor failure propagates as DatasetGenerationError."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    def fail_pre_batch(row_group: int, row_group_size: int) -> None:
        raise ValueError(f"pre-batch failed for {row_group}/{row_group_size}")

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        on_seeds_complete=fail_pre_batch,
        num_records=3,
        buffer_size=3,
    )
    with pytest.raises(DatasetGenerationError, match="Pre-batch processor failed"):
        await scheduler.run()


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_error_rate_shutdown(caplog: pytest.LogCaptureFixture) -> None:
    """Early shutdown triggers when error rate exceeds threshold."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="fail_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "fail_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "fail_col": MockFailingGenerator(config=_expr_config("fail_col"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 10)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    buffer_mgr = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        shutdown_error_rate=0.5,
        shutdown_error_window=2,
    )
    with caplog.at_level("ERROR", logger="data_designer.engine.dataset_builders.async_scheduler"):
        await scheduler.run()

    # Early shutdown: not all rows should be checkpointed (some row groups incomplete)
    assert scheduler.early_shutdown
    assert buffer_mgr.actual_num_records < 10
    assert not any("unfinished row group" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio(loop_scope="session")
async def test_partial_row_group_salvaged_after_early_shutdown() -> None:
    """Mid-run shutdown drops incomplete rows and checkpoints survivors."""
    # 3 succeed (0,1,2), 3 fail non-retryable (5,6,7), 4 stay in-flight (3,4,8,9)
    # until cancellation. Window=4, rate=0.5 → gate trips after ~3-5 outcomes.
    cell = MockSelectiveFailGenerator(
        config=_expr_config("cell_out"),
        resource_provider=_mock_provider(),
        fail_on_seeds={5, 6, 7},
        slow_seeds={3, 4, 8, 9},
    )
    generators, graph, row_groups, tracker, buffer_mgr, _storage = _seed_plus_cell_setup(cell, num_records=10)
    finalized: list[int] = []

    def on_finalize(rg_id: int) -> None:
        buffer_mgr.checkpoint_row_group(rg_id)
        finalized.append(rg_id)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_finalize_row_group=on_finalize,
        shutdown_error_rate=0.5,
        shutdown_error_window=4,
    )
    await scheduler.run()

    assert scheduler.early_shutdown
    # Survivor count depends on event-loop dispatch ordering between fast/fail/slow
    # seeds, so the assertion is bounded rather than exact: 3 fail → at least 3
    # dropped, so survivors ≤ 7; at least 1 success is needed for the gate to
    # start counting. The point of the test is "salvage works", not exact counts.
    assert 0 in finalized
    assert scheduler.partial_row_groups == (0,)
    assert 1 <= buffer_mgr.actual_num_records <= 7


@pytest.mark.asyncio(loop_scope="session")
async def test_zero_survivor_shutdown_does_not_raise() -> None:
    """If every row is dropped at shutdown, the row group is freed without writing parquet.

    Also covers the healthy-run baseline: ``partial_row_groups`` stays empty
    when no rows survived (all dropped, none salvaged).
    """
    cell = MockSelectiveFailGenerator(
        config=_expr_config("cell_out"),
        resource_provider=_mock_provider(),
        fail_on_seeds=set(range(5)),
    )
    generators, graph, row_groups, tracker, buffer_mgr, storage = _seed_plus_cell_setup(cell, num_records=5)
    finalized: list[int] = []
    sink = _FilteringSchedulerSink({"scheduler_job_completed"})

    def on_finalize(rg_id: int) -> None:
        buffer_mgr.checkpoint_row_group(rg_id)
        finalized.append(rg_id)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_finalize_row_group=on_finalize,
        shutdown_error_rate=0.5,
        shutdown_error_window=2,
        scheduler_event_sink=sink,
    )
    # Must not raise (no FileNotFoundError, no DataDesignerGenerationError).
    await scheduler.run()

    assert scheduler.early_shutdown
    assert buffer_mgr.actual_num_records == 0
    # All rows dropped → checkpoint path frees buffer without writing; on_finalize
    # is *not* called because every row was dropped before survivors could exist.
    assert finalized == []
    assert scheduler.partial_row_groups == ()
    storage.write_batch_to_parquet_file.assert_not_called()
    [completed] = sink.scheduler_events
    assert completed.reason_or_result == "early_shutdown"
    assert completed.diagnostics["outcome"] == "early_shutdown"
    assert completed.diagnostics["reason"] == "completed"


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_early_shutdown_disabled() -> None:
    """shutdown_error_rate=1.0 prevents shutdown even at 100% error rate."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="fail_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "fail_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "fail_col": MockFailingGenerator(config=_expr_config("fail_col"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 5)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    buffer_mgr = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        shutdown_error_rate=1.0,
    )
    await scheduler.run()

    # All rows dropped (all fail) but no early shutdown - all row groups processed
    assert all(tracker.is_dropped(0, ri) for ri in range(5))
    assert tracker.is_row_group_complete(0, 5, ["seed", "fail_col"])


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_sliding_window_error_rate_recovers() -> None:
    """Transient errors diluted by successes do not trigger early shutdown."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "col": GenerationStrategy.CELL_BY_CELL,
    }
    # First 2 calls fail (retryable 503), rest succeed.
    # With window=10 and 10 cell tasks, at most 2/10 = 20% error rate
    # when the window first fills - well below the 0.4 threshold.
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "col": MockFailingGenerator(config=_expr_config("col"), resource_provider=provider, transient_failures=2),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 10)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    buffer_mgr = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        shutdown_error_rate=0.4,
        shutdown_error_window=10,
    )
    await scheduler.run()

    # No early shutdown - transient errors recovered in salvage
    assert not scheduler._early_shutdown
    assert tracker.is_row_group_complete(0, 10, ["seed", "col"])


@pytest.mark.asyncio(loop_scope="session")
async def test_rate_limit_errors_do_not_trigger_early_shutdown() -> None:
    """Rate-limit (429) errors are expected AIMD behavior and must not count toward early shutdown."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "col": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "col": MockRateLimitGenerator(config=_expr_config("col"), resource_provider=provider, rate_limit_failures=8),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 10)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    buffer_mgr = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        shutdown_error_rate=0.5,
        shutdown_error_window=10,
    )
    await scheduler.run()

    assert not scheduler._early_shutdown
    assert tracker.is_row_group_complete(0, 10, ["seed", "col"])


@pytest.mark.asyncio(loop_scope="session")
async def test_preserved_429_retries_after_unrelated_early_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Early shutdown must not turn rate-limited deferred work into dropped rows."""
    monkeypatch.setattr(async_scheduler_module, "RETRYABLE_RESALVAGE_BACKOFF_S", 0)
    cell = MockRateLimitThenNonRetryableGenerator(
        config=_expr_config("cell_out"),
        resource_provider=_mock_provider(),
        rate_limit_failures=2,
    )
    generators, graph, row_groups, tracker, buffer_mgr, _storage = _seed_plus_cell_setup(cell, num_records=3)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_finalize_row_group=lambda rg_id: buffer_mgr.checkpoint_row_group(rg_id),
        shutdown_error_rate=0.5,
        shutdown_error_window=1,
        salvage_max_rounds=1,
    )

    await scheduler.run()

    assert scheduler.early_shutdown
    assert not tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    assert not tracker.is_dropped(0, 2)
    assert tracker.is_row_group_complete(0, 3, ["seed", "cell_out"])
    assert buffer_mgr.actual_num_records == 2


@pytest.mark.parametrize("exc_cls", RETRYABLE_MODEL_ERRORS, ids=lambda c: c.__name__)
@pytest.mark.asyncio(loop_scope="session")
async def test_retryable_errors_do_not_trigger_early_shutdown(
    exc_cls: type[Exception],
) -> None:
    """All retryable errors (rate-limit, timeout, 5xx, connection) bypass the early-shutdown gate.

    Regression test for #575: clustered ``ModelTimeoutError`` during provider degradation
    used to trip the gate even though salvage could recover the rows.
    """
    cell = MockRetryableErrorGenerator(
        config=_expr_config("cell_out"),
        resource_provider=_mock_provider(),
        error_factory=lambda: exc_cls("boom"),
        retryable_failures=8,
    )
    generators, graph, row_groups, tracker, buffer_mgr, _storage = _seed_plus_cell_setup(cell, num_records=10)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        shutdown_error_rate=0.5,
        shutdown_error_window=10,
    )
    await scheduler.run()

    assert not scheduler._early_shutdown
    assert scheduler._recent_outcomes.count(False) == 0
    assert tracker.is_row_group_complete(0, 10, ["seed", "cell_out"])


def _count_degraded_msgs(caplog: pytest.LogCaptureFixture) -> int:
    return sum(1 for r in caplog.records if "degraded performance" in r.getMessage())


@pytest.mark.parametrize(
    "retryable_failures,num_records,window,interval_s,expected_count",
    [
        # Above-threshold + no log interval: at least one WARN should fire.
        pytest.param(6, 10, 8, 0.0, "at_least_one", id="fires_above_threshold"),
        # Above-threshold + 1h log interval: only one WARN despite sustained degradation.
        pytest.param(8, 12, 4, 3600.0, 1, id="rate_limited_to_one"),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_degraded_provider_warn_emission(
    caplog: pytest.LogCaptureFixture,
    retryable_failures: int,
    num_records: int,
    window: int,
    interval_s: float,
    expected_count: int | str,
) -> None:
    cell = MockRetryableErrorGenerator(
        config=_expr_config("cell_out"),
        resource_provider=_mock_provider(),
        error_factory=lambda: ModelTimeoutError("read timeout"),
        retryable_failures=retryable_failures,
    )
    generators, graph, row_groups, tracker, buffer_mgr, _storage = _seed_plus_cell_setup(cell, num_records=num_records)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        degraded_warn_rate=0.5,
        degraded_warn_window=window,
        degraded_warn_interval_s=interval_s,
    )
    with caplog.at_level("WARNING"):
        await scheduler.run()

    n = _count_degraded_msgs(caplog)
    if expected_count == "at_least_one":
        assert n >= 1
    else:
        assert n == expected_count


@pytest.mark.asyncio(loop_scope="session")
async def test_degraded_provider_warn_silent_under_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """Healthy runs (no errors) never emit the degraded-provider WARN."""
    scheduler, _tracker = _build_simple_pipeline(num_records=5)
    with caplog.at_level("WARNING"):
        await scheduler.run()
    assert _count_degraded_msgs(caplog) == 0


@pytest.mark.parametrize(
    ("exc_cls", "expected_kind"),
    [
        (ModelRateLimitError, "rate_limit"),
        (ModelRequestAdmissionTimeoutError, "request_admission_timeout"),
        (ModelTimeoutError, "timeout"),
        (ModelInternalServerError, "internal_server"),
        (ModelAPIConnectionError, "connection"),
    ],
)
def test_retryable_outcome_metrics_classify_errors(
    exc_cls: type[Exception],
    expected_kind: str,
) -> None:
    scheduler, _tracker = _build_simple_pipeline(num_records=1)

    scheduler._record_retryable_outcome(
        retryable=True,
        exc=exc_cls("sensitive provider text"),
    )
    scheduler._record_retryable_outcome(retryable=False)

    metrics = scheduler.retryable_outcome_metrics
    assert metrics["cumulative_counts"] == {expected_kind: 1, "success": 1}
    assert metrics["rolling_counts"] == {expected_kind: 1, "success": 1}
    assert metrics["retryable_details"] == {expected_kind: 1}
    assert "sensitive provider text" not in str(metrics)


def test_retryable_outcome_metrics_distinguish_non_retryable_failures() -> None:
    scheduler, _tracker = _build_simple_pipeline(num_records=1)

    scheduler._record_retryable_outcome(
        retryable=False,
        exc=RuntimeError("sensitive non-retryable text"),
    )
    scheduler._record_retryable_outcome(retryable=False)

    metrics = scheduler.retryable_outcome_metrics
    assert metrics["cumulative_counts"] == {"non_retryable_failure": 1, "success": 1}
    assert metrics["rolling_counts"] == {"non_retryable_failure": 1, "success": 1}
    assert metrics["retryable_details"] == {}
    assert "sensitive non-retryable text" not in str(metrics)

    diagnostics = scheduler._scheduler_health_diagnostics(reason="test")
    assert diagnostics["deferred_tasks"] == 0
    retryable_outcomes = diagnostics["retryable_outcomes"]
    assert isinstance(retryable_outcomes, dict)
    assert "deferred_tasks" not in retryable_outcomes


@pytest.mark.asyncio(loop_scope="session")
async def test_degraded_provider_warn_only_counts_llm_tasks() -> None:
    """The WARN window must ignore non-LLM task outcomes (samplers, expressions, etc).

    Without this, a healthy non-model column mix dilutes the retryable rate and
    the WARN never fires under genuine provider stress.
    """
    # Sampler-only graph: no LLM tasks → window must stay empty regardless of
    # how many task outcomes feed in.
    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=_mock_provider())}
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 5)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    buffer_mgr = RowGroupBufferManager(_make_storage())
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        degraded_warn_rate=0.5,
        degraded_warn_window=2,
        degraded_warn_interval_s=0.0,
    )
    await scheduler.run()
    assert len(scheduler._recent_retryable) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_on_before_checkpoint_callback() -> None:
    """on_before_checkpoint is called before each row group is checkpointed."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider)}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3), (1, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    buffer_mgr = RowGroupBufferManager(storage)
    callback_log: list[tuple[int, int]] = []

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_before_checkpoint=lambda rg, sz: callback_log.append((rg, sz)),
    )
    await scheduler.run()

    assert sorted(callback_log) == [(0, 3), (1, 2)]


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_on_finalize_row_group_callback_fires() -> None:
    """on_finalize_row_group is called for each completed row group."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider)}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    buffer_mgr = RowGroupBufferManager(storage)
    finalized: list[int] = []

    def finalize_row_group(rg_id: int) -> None:
        buffer_mgr.checkpoint_row_group(rg_id)
        finalized.append(rg_id)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_finalize_row_group=finalize_row_group,
    )
    await scheduler.run()

    assert finalized == [0]
    assert storage.write_batch_to_parquet_file.call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_on_finalize_skips_empty_row_group() -> None:
    """on_finalize_row_group is not called when all rows are dropped."""
    provider = _mock_provider()
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {
        "seed": MockFailingSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    buffer_mgr = RowGroupBufferManager(storage)
    callback = MagicMock()

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_finalize_row_group=callback,
    )
    await scheduler.run()

    callback.assert_not_called()
    storage.write_batch_to_parquet_file.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_pre_batch_failure_propagates_across_row_groups() -> None:
    """Pre-batch processor failure propagates even when other row groups exist."""
    provider = _mock_provider()
    seed_gen = MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider)
    cell_gen = MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {"seed": seed_gen, "cell_out": cell_gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3), (1, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    buffer_mgr = RowGroupBufferManager(storage)

    def failing_pre_batch(rg_id: int, rg_size: int) -> None:
        if rg_id == 0:
            raise RuntimeError("pre-batch processor failed")

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_seeds_complete=failing_pre_batch,
    )
    with pytest.raises(DatasetGenerationError, match="Pre-batch processor failed"):
        await scheduler.run()


class _SlowSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
    """Seed generator whose async cost scales with row count.

    Both RGs' seed tasks run concurrently. The task with fewer rows sleeps for
    less real time, causing its downstream to be dispatched and completed first.
    """

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        return lazy.pd.DataFrame({self.config.name: list(range(num_records))})

    async def agenerate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        await asyncio.sleep(num_records * 0.02)
        return self.generate_from_scratch(num_records)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_out_of_order_row_group_completion() -> None:
    """Row groups may complete out of order; both are checkpointed correctly."""
    provider = _mock_provider()
    # Slow seed generator: RG 0 (5 rows) sleeps 100ms, RG 1 (1 row) sleeps 20ms.
    # RG 1 finishes seeds first, its downstream is dispatched and completes before RG 0.
    slow_seed = _SlowSeedGenerator(config=_expr_config("seed"), resource_provider=provider)
    cell_gen = MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {"seed": slow_seed, "cell_out": cell_gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 5), (1, 1)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    buffer_mgr = RowGroupBufferManager(storage)

    checkpoint_order: list[int] = []

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        max_concurrent_row_groups=2,
        on_finalize_row_group=lambda rg_id: checkpoint_order.append(rg_id),
    )
    await scheduler.run()

    # Both row groups completed
    assert tracker.is_row_group_complete(0, 5, ["seed", "cell_out"])
    assert tracker.is_row_group_complete(1, 1, ["seed", "cell_out"])
    # Both were checkpointed
    assert set(checkpoint_order) == {0, 1}
    # RG 1 (fewer rows, fewer seed yields) checkpoints before RG 0
    assert checkpoint_order.index(1) < checkpoint_order.index(0)


# -- Task-admission / model-stage tests ---------------------------------------


class MockLLMBoundCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Mock cell-by-cell generator that reports model-stage scheduling metadata."""

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.custom_model("test", self.config.name, "v1")

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"llm_{data.get('seed', '?')}"
        return data


class MockConfiguredModelCellGenerator(ColumnGeneratorWithModelRegistry[LLMTextColumnConfig]):
    """Mock cell generator with model-registry helpers."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"model_{data.get('seed', '?')}"
        return data

    def get_model_config(self, model_alias: str) -> ModelConfig:
        return self.resource_provider.model_registry.get_model_config(model_alias=model_alias)

    def get_model_provider_name(self, model_alias: str) -> str:
        provider = self.resource_provider.model_registry.get_model_provider(model_alias=model_alias)
        return str(provider.name)


class MockLLMBoundRateLimitGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """LLM-bound generator that raises ModelRateLimitError for the first N calls, then succeeds."""

    def __init__(self, *args: Any, rate_limit_failures: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_failures = rate_limit_failures
        self._calls = 0

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.custom_model("test", self.config.name, "v1")

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        self._calls += 1
        if self._calls <= self._rate_limit_failures:
            raise ModelRateLimitError("429 Too Many Requests")
        data[self.config.name] = f"llm_ok_{data.get('seed', '?')}"
        return data


class MockMixedRetryableGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Generator that preserves one rate-limited row while another retryable row exhausts."""

    def __init__(self, *args: Any, rate_limit_failures: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_failures = rate_limit_failures
        self._rate_limit_calls = 0
        self._timeout_calls = 0

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.custom_model("test", self.config.name, "v1")

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        seed = data.get("seed")
        if seed == 0:
            self._rate_limit_calls += 1
            if self._rate_limit_calls <= self._rate_limit_failures:
                raise ModelRateLimitError("429 Too Many Requests")
        elif seed == 1:
            self._timeout_calls += 1
            raise ModelTimeoutError("timed out")
        data[self.config.name] = f"mixed_ok_{seed}"
        return data


class MockRateLimitThenNonRetryableGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Generator that combines preserved 429 work with an early-shutdown failure."""

    def __init__(self, *args: Any, rate_limit_failures: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_failures = rate_limit_failures
        self._rate_limit_calls = 0
        self._first_rate_limit_recorded = asyncio.Event()

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.custom_model("test", self.config.name, "v1")

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    async def agenerate(self, data: dict) -> dict:
        seed = data.get("seed")
        if seed == 0:
            self._rate_limit_calls += 1
            if self._rate_limit_calls <= self._rate_limit_failures:
                self._first_rate_limit_recorded.set()
                raise ModelRateLimitError("429 Too Many Requests")
        elif seed == 1:
            await self._first_rate_limit_recorded.wait()
            raise ValueError("non-retryable failure")
        data[self.config.name] = f"shutdown_ok_{seed}"
        return data


class MockModelRateLimitGenerator(MockLLMBoundRateLimitGenerator):
    """Rate-limit fixture with request-admission resource metadata."""

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.model("provider", "model", RequestDomain.CHAT.value, weight=1)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_llm_bound_one_way_handoff() -> None:
    """LLM-bound tasks release submission slot and hold LLM-wait slot during execution."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="llm_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "llm_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "llm_col": MockLLMBoundCellGenerator(config=_expr_config("llm_col"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    max_in_flight = 2
    max_llm_wait = 2
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=max_in_flight,
        max_model_task_admission=max_llm_wait,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, 3, ["seed", "llm_col"])

    snapshot = scheduler.task_admission_snapshot()
    assert snapshot.resources_available["submission"] == max_in_flight
    assert snapshot.resources_available["llm_wait"] == max_llm_wait


def test_scheduler_default_task_admission_uses_bounded_borrow_policy() -> None:
    provider = _mock_provider()
    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    generators = {"seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider)}
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 1)]

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
    )

    assert isinstance(scheduler.task_admission_config.bounded_borrow, BoundedBorrowTaskAdmissionPolicyConfig)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_non_llm_holds_submission_slot() -> None:
    """Non-LLM generators hold the submission slot for the entire execution (no handoff)."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    max_llm_wait = 2
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=2,
        max_model_task_admission=max_llm_wait,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, 3, ["seed", "cell_out"])

    snapshot = scheduler.task_admission_snapshot()
    assert snapshot.resources_available["llm_wait"] == max_llm_wait


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_deadlock_regression() -> None:
    """max_in_flight_tasks=1, max_model_task_admission=1, two ready LLM tasks completes without deadlock."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="llm_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "llm_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "llm_col": MockLLMBoundCellGenerator(config=_expr_config("llm_col"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=1,
        max_model_task_admission=1,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)
    assert tracker.is_row_group_complete(0, 2, ["seed", "llm_col"])


@pytest.mark.asyncio(loop_scope="session")
async def test_drain_frontier_raises_when_ready_but_no_capacity_or_inflight() -> None:
    """A broken admission state fails fast instead of spinning in the drain loop.

    This intentionally calls private frontier helpers: the state is an invariant
    violation that public ``run()`` should never construct, but the fail-fast
    guard prevents infinite waits if future scheduler changes create it.
    """
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 1)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    seed_delta = tracker.mark_row_range_complete("seed", 0, 1)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        task_admission_config=TaskAdmissionConfig(submission_capacity=1),
    )
    scheduler._rg_states[0] = MagicMock(size=1)
    blocker = scheduler._schedulable_task(Task(column="cell_out", row_group=0, row_index=99, task_type="cell"))
    lease = scheduler._task_admission.try_acquire(blocker, scheduler._fair_queue.view())
    assert isinstance(lease, TaskAdmissionLease)
    scheduler._apply_frontier_delta(seed_delta)

    with pytest.raises(RuntimeError, match="Ready frontier is admission-blocked"):
        await scheduler._drain_frontier(("seed",), False)


@pytest.mark.asyncio(loop_scope="session")
async def test_drain_frontier_stops_when_cancel_requested() -> None:
    """Cancellation wakes and exits the salvage dispatch loop."""
    scheduler, _tracker = _build_simple_pipeline(num_records=1)
    scheduler._in_flight.add(Task(column="cell_out", row_group=0, row_index=0, task_type="cell"))
    scheduler._run_loop = asyncio.get_running_loop()
    drain_task = asyncio.create_task(scheduler._drain_frontier(("seed",), False))

    await asyncio.sleep(0)
    await asyncio.to_thread(scheduler.request_cancel)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(drain_task, timeout=2)


def test_dispatch_selected_task_rolls_back_scheduler_state_when_worker_spawn_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _mock_provider()
    config = ExpressionColumnConfig(name="cell_out", expr="'x'", dtype="str")
    graph = ExecutionGraph.create([config], {"cell_out": GenerationStrategy.CELL_BY_CELL})
    scheduler = AsyncTaskScheduler(
        generators={"cell_out": MockCellGenerator(config=config, resource_provider=provider)},
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, [(0, 1)]),
        row_groups=[(0, 1)],
        scheduler_event_sink=(sink := InMemoryAdmissionEventSink()),
    )
    task = Task(column="cell_out", row_group=0, row_index=0, task_type="cell")
    item = scheduler._schedulable_task(task)
    lease = scheduler._task_admission.try_acquire(item, scheduler._fair_queue.view())
    assert isinstance(lease, TaskAdmissionLease)
    scheduler._rg_states[0] = SimpleNamespace(size=1, in_flight_count=0)

    def fail_spawn(coro: Any) -> None:
        coro.close()
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(scheduler, "_spawn_worker", fail_spawn)

    with pytest.raises(RuntimeError, match="spawn failed"):
        scheduler._dispatch_selected_task(item, lease)

    assert task not in scheduler._dispatched
    assert task not in scheduler._in_flight
    assert scheduler._rg_states[0].in_flight_count == 0
    assert scheduler.task_admission_snapshot().leased_resources == {}
    assert scheduler.task_admission_snapshot().running_counts_by_group == {}
    assert any(event.event_kind == "worker_spawn_failed" for event in sink.scheduler_events)


@pytest.mark.asyncio(loop_scope="session")
async def test_main_dispatch_loop_yields_when_pre_batch_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _mock_provider()
    seed_config = ExpressionColumnConfig(name="seed", expr="'seed'", dtype="str")
    graph = ExecutionGraph.create([seed_config], {"seed": GenerationStrategy.FULL_COLUMN})
    scheduler = AsyncTaskScheduler(
        generators={"seed": MockSeedGenerator(config=seed_config, resource_provider=provider)},
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, [(0, 1)]),
        row_groups=[(0, 1)],
    )
    scheduler._all_rgs_admitted = True
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=False)
    monkeypatch.setattr(scheduler, "_run_seeds_complete_check", lambda seed_cols: None)
    monkeypatch.setattr(
        scheduler, "_dispatch_queued_tasks", lambda: SimpleNamespace(dispatched=False, admission_blocked=False)
    )
    monkeypatch.setattr(scheduler, "_checkpoint_completed_row_groups", lambda all_columns: None)
    monkeypatch.setattr(scheduler, "_maybe_update_adaptive_row_group_target", lambda: None)
    yielded_delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        yielded_delays.append(delay)
        scheduler._rg_states[0].pre_batch_done = True

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    await scheduler._main_dispatch_loop(("seed",), True, ["seed"])

    assert yielded_delays == [0]


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_pre_batch_drop_removes_pending_ready_task() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, [(0, 3)])

    def drop_middle_row(row_group: int, row_group_size: int) -> FrontierDelta:
        del row_group_size
        return tracker.drop_row(row_group, 1)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=[(0, 3)],
        on_seeds_complete=drop_middle_row,
        trace=True,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    cell_traces = [trace for trace in scheduler.traces if trace.column == "cell_out"]
    assert {trace.row_index for trace in cell_traces} == {0, 2}
    assert tracker.is_dropped(0, 1)
    assert tracker.is_row_group_complete(0, 3, ["seed", "cell_out"])


def test_apply_frontier_delta_enqueues_ready_tasks_in_one_queue_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _mock_provider()
    configs = [
        LLMTextColumnConfig(name="root", prompt="root", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "root": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "root": MockCellGenerator(config=_expr_config("root"), resource_provider=provider),
    }
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 5)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=5, pre_batch_done=True)

    enqueue_sizes: list[int] = []
    original_enqueue = scheduler._fair_queue.enqueue

    def spy_enqueue(items: Any) -> tuple[str, ...]:
        materialized = tuple(items)
        enqueue_sizes.append(len(materialized))
        return original_enqueue(materialized)

    monkeypatch.setattr(scheduler._fair_queue, "enqueue", spy_enqueue)

    scheduler._apply_frontier_delta(tracker.add_root_tasks(0, 5))

    assert enqueue_sizes == [5]
    assert tracker.ready_frontier() == ()
    assert scheduler._fair_queue.view().queued_total == 5


def test_pre_batch_flush_batches_pending_ready_and_skips_dropped_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    sink = InMemoryAdmissionEventSink()
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        on_seeds_complete=lambda row_group, row_group_size: None,
        scheduler_event_sink=sink,
    )
    state = SimpleNamespace(size=3, pre_batch_done=False)
    scheduler._rg_states[0] = state

    enqueue_sizes: list[int] = []
    original_enqueue = scheduler._fair_queue.enqueue

    def spy_enqueue(items: Any) -> tuple[str, ...]:
        materialized = tuple(items)
        enqueue_sizes.append(len(materialized))
        return original_enqueue(materialized)

    monkeypatch.setattr(scheduler._fair_queue, "enqueue", spy_enqueue)

    scheduler._apply_frontier_delta(tracker.mark_row_range_complete("seed", 0, 3))
    scheduler._apply_frontier_delta(tracker.drop_row(0, 1))
    state.pre_batch_done = True
    scheduler._flush_pre_batch_ready(0)

    assert enqueue_sizes == [2]
    assert scheduler._fair_queue.view().queued_total == 2
    assert {item.payload.row_index for item in scheduler._fair_queue._queued.values()} == {0, 2}
    assert tracker.is_dropped(0, 1)
    assert sum(event.event_kind == "ready_enqueued" for event in sink.scheduler_events) == 2
    assert sum(event.event_kind == "dependency_ready" for event in sink.scheduler_events) == 5


def test_custom_generator_with_model_aliases_reports_custom_model_metadata() -> None:
    """CustomColumnGenerator with model_aliases reports custom-model metadata."""

    @custom_column_generator(model_aliases=["my_model"])
    def gen_with_models(row: dict, generator_params: None, models: dict) -> dict:
        row["custom_llm"] = "val"
        return row

    @custom_column_generator()
    def gen_no_models(row: dict) -> dict:
        row["custom_plain"] = "val"
        return row

    provider = _mock_provider()
    llm_config = CustomColumnConfig(name="custom_llm", generator_function=gen_with_models)
    plain_config = CustomColumnConfig(name="custom_plain", generator_function=gen_no_models)

    llm_gen = CustomColumnGenerator(config=llm_config, resource_provider=provider)
    plain_gen = CustomColumnGenerator(config=plain_config, resource_provider=provider)

    assert llm_gen.get_scheduling_metadata().kind == "custom_model"
    assert plain_gen.get_scheduling_metadata().kind == "local"


def _provider_with_model_configs(configs: dict[str, ModelConfig]) -> MagicMock:
    provider = MagicMock()
    provider.model_registry = MagicMock()
    provider.model_registry.get_model_config.side_effect = lambda model_alias: configs[model_alias]
    provider.model_registry.get_model_provider.return_value = SimpleNamespace(name="mock-provider")
    return provider


def test_scheduler_model_task_group_spec_uses_model_resource_and_flow() -> None:
    """Direct spec coverage keeps model identity and flow composition deterministic."""
    model_config = ModelConfig(
        alias=MODEL_ALIAS,
        model="model-text",
        inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=3),
        provider="mock-provider",
    )
    provider = _provider_with_model_configs({MODEL_ALIAS: model_config})
    column_config = LLMTextColumnConfig(name="answer", prompt="hello", model_alias=MODEL_ALIAS)
    generator = MockConfiguredModelCellGenerator(config=column_config, resource_provider=provider)
    graph = ExecutionGraph.create([column_config], {"answer": GenerationStrategy.CELL_BY_CELL})
    tracker = CompletionTracker.with_graph(graph, [(0, 1)])
    scheduler = AsyncTaskScheduler(
        generators={"answer": generator},
        graph=graph,
        tracker=tracker,
        row_groups=[(0, 1)],
        max_model_task_admission=5,
    )

    spec = scheduler._schedulable_task(Task(column="answer", row_group=0, row_index=0, task_type="cell")).group

    assert spec.key.kind == "model"
    assert spec.key.identity[:3] == ("model", "mock-provider", "model-text")
    assert spec.key.identity[-1] == "answer"
    assert spec.weight == 3.0
    assert spec.admitted_limit == 5


def test_scheduler_task_group_spec_is_cached_per_generator() -> None:
    """The per-generator spec cache has no stable public signal, so isolate it directly."""
    model_config = ModelConfig(
        alias=MODEL_ALIAS,
        model="model-text",
        inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=3),
        provider="mock-provider",
    )
    provider = _provider_with_model_configs({MODEL_ALIAS: model_config})
    column_config = LLMTextColumnConfig(name="answer", prompt="hello", model_alias=MODEL_ALIAS)
    generator = MockConfiguredModelCellGenerator(config=column_config, resource_provider=provider)
    graph = ExecutionGraph.create([column_config], {"answer": GenerationStrategy.CELL_BY_CELL})
    tracker = CompletionTracker.with_graph(graph, [(0, 2)])
    scheduler = AsyncTaskScheduler(
        generators={"answer": generator},
        graph=graph,
        tracker=tracker,
        row_groups=[(0, 2)],
        max_model_task_admission=5,
    )

    spec_a = scheduler._schedulable_task(Task(column="answer", row_group=0, row_index=0, task_type="cell")).group
    spec_b = scheduler._schedulable_task(Task(column="answer", row_group=0, row_index=1, task_type="cell")).group

    assert spec_a == spec_b
    assert provider.model_registry.get_model_config.call_count == 1
    assert provider.model_registry.get_model_provider.call_count == 1


def test_scheduler_task_group_spec_raises_on_model_resolution_failure() -> None:
    """Model metadata resolution failures are fatal without an explicit fallback."""
    provider = MagicMock()
    provider.model_registry = MagicMock()
    provider.model_registry.get_model_config.side_effect = RuntimeError("registry unavailable")
    provider.model_registry.get_model_provider.return_value = SimpleNamespace(name="mock-provider")
    column_config = LLMTextColumnConfig(name="answer", prompt="hello", model_alias=MODEL_ALIAS)
    generator = MockConfiguredModelCellGenerator(config=column_config, resource_provider=provider)
    graph = ExecutionGraph.create([column_config], {"answer": GenerationStrategy.CELL_BY_CELL})
    tracker = CompletionTracker.with_graph(graph, [(0, 2)])

    with pytest.raises(Exception):
        AsyncTaskScheduler(
            generators={"answer": generator},
            graph=graph,
            tracker=tracker,
            row_groups=[(0, 2)],
            max_model_task_admission=5,
        )


def test_scheduler_custom_model_task_group_spec_uses_alias_set_weight() -> None:
    """Direct spec coverage verifies custom-model alias aggregation before fair admission."""

    @custom_column_generator(model_aliases=["draft", "judge"])
    def gen_with_models(row: dict, generator_params: None, models: dict) -> dict:
        row["custom_llm"] = "val"
        return row

    provider = _provider_with_model_configs(
        {
            "draft": ModelConfig(
                alias="draft",
                model="model-draft",
                inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=2),
                provider="mock-provider",
            ),
            "judge": ModelConfig(
                alias="judge",
                model="model-judge",
                inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=3),
                provider="mock-provider",
            ),
        }
    )
    config = CustomColumnConfig(name="custom_llm", generator_function=gen_with_models)
    generator = CustomColumnGenerator(config=config, resource_provider=provider)
    graph = ExecutionGraph.create([config], {"custom_llm": GenerationStrategy.CELL_BY_CELL})
    tracker = CompletionTracker.with_graph(graph, [(0, 1)])
    scheduler = AsyncTaskScheduler(
        generators={"custom_llm": generator},
        graph=graph,
        tracker=tracker,
        row_groups=[(0, 1)],
        max_model_task_admission=10,
    )

    spec = scheduler._schedulable_task(Task(column="custom_llm", row_group=0, row_index=0, task_type="cell")).group

    assert spec.key.kind == "custom_model"
    assert spec.key.identity[:3] == ("custom_model", "custom_column", "draft-judge")
    assert spec.weight == 2.0
    assert spec.admitted_limit == 4


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_llm_bound_429_retried_in_salvage() -> None:
    """A 429'd LLM-bound task is deferred, retried in salvage (handoff runs twice), and completes."""
    provider = _mock_provider()
    num_records = 3
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="llm_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "llm_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "llm_col": MockLLMBoundRateLimitGenerator(
            config=_expr_config("llm_col"),
            resource_provider=provider,
            rate_limit_failures=num_records,
        ),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, num_records)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    buffer_mgr = RowGroupBufferManager(storage)

    max_in_flight = 4
    max_llm_wait = 2
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        max_in_flight_tasks=max_in_flight,
        max_model_task_admission=max_llm_wait,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, num_records, ["seed", "llm_col"])

    snapshot = scheduler.task_admission_snapshot()
    assert snapshot.resources_available["submission"] == max_in_flight
    assert snapshot.resources_available["llm_wait"] == max_llm_wait


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_429_beyond_salvage_cap_is_delayed_not_dropped() -> None:
    """429s may outlast the salvage cap; those rows must wait and retry instead of being dropped."""
    provider = _mock_provider()
    num_records = 3
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="llm_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "llm_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "llm_col": MockLLMBoundRateLimitGenerator(
            config=_expr_config("llm_col"),
            resource_provider=provider,
            rate_limit_failures=num_records * 3,
        ),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, num_records)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    buffer_mgr = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        salvage_max_rounds=2,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, num_records, ["seed", "llm_col"])
    assert not any(tracker.is_dropped(0, row_index) for row_index in range(num_records))
    assert scheduler._deferred == []
    assert scheduler._deferred_errors == {}
    assert scheduler._preserved_retryable_counts == {}
    assert scheduler._preserved_retryable_log_state == {}


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_request_admission_timeout_beyond_salvage_cap_is_delayed_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local request-admission timeouts may outlast salvage rounds without becoming row drops."""
    provider = _mock_provider()
    monkeypatch.setattr(async_scheduler_module, "RETRYABLE_RESALVAGE_BACKOFF_S", 0)
    config = ExpressionColumnConfig(name="llm_col", expr="'x'", dtype="str")
    generator = MockRetryableErrorGenerator(
        config=config,
        resource_provider=provider,
        error_factory=lambda: ModelRequestAdmissionTimeoutError("request admission queue timeout"),
        retryable_failures=3,
    )
    graph = ExecutionGraph.create([config], {"llm_col": GenerationStrategy.CELL_BY_CELL})
    row_groups = [(0, 1)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators={"llm_col": generator},
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        salvage_max_rounds=1,
    )
    await asyncio.wait_for(scheduler.run(), timeout=5.0)

    assert tracker.is_row_group_complete(0, 1, ["llm_col"])
    assert not tracker.is_dropped(0, 0)
    assert generator._calls == 4
    assert scheduler._deferred == []
    assert scheduler._deferred_errors == {}


@pytest.mark.parametrize("retryable_kind", ["rate_limit", "request_admission_timeout"])
@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_paces_sustained_preserved_retryable_resalvage(
    retryable_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserved retryable loops should wait between salvage cycles instead of spinning CPU-bound."""
    provider = _mock_provider()
    retrying_generator = (
        MockLLMBoundRateLimitGenerator(
            config=_expr_config("cell_out"),
            resource_provider=provider,
            rate_limit_failures=10_000,
        )
        if retryable_kind == "rate_limit"
        else MockRetryableErrorGenerator(
            config=_expr_config("cell_out"),
            resource_provider=provider,
            error_factory=lambda: ModelRequestAdmissionTimeoutError("request admission queue timeout"),
            retryable_failures=10_000,
        )
    )
    generators, graph, row_groups, tracker, buffer_mgr, _storage = _seed_plus_cell_setup(
        retrying_generator,
        num_records=1,
    )
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        salvage_max_rounds=1,
    )

    first_wait_started = asyncio.Event()
    release_first_wait = asyncio.Event()
    second_wait_started = asyncio.Event()
    block_second_wait = asyncio.Event()
    wait_calls = 0

    async def controlled_resalvage_wait() -> None:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            first_wait_started.set()
            await release_first_wait.wait()
            return
        second_wait_started.set()
        await block_second_wait.wait()

    monkeypatch.setattr(scheduler, "_wait_before_retryable_resalvage", controlled_resalvage_wait)

    with caplog.at_level(logging.INFO, logger=async_scheduler_module.__name__):
        run_task = asyncio.create_task(scheduler.run())
        try:
            await asyncio.wait_for(first_wait_started.wait(), timeout=5.0)
            calls_after_preserve = retrying_generator._calls
            for _ in range(5):
                await asyncio.sleep(0)

            assert retrying_generator._calls == calls_after_preserve
            assert not run_task.done()

            release_first_wait.set()
            await asyncio.wait_for(second_wait_started.wait(), timeout=5.0)

            assert wait_calls == 2
            assert retrying_generator._calls > calls_after_preserve
        finally:
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

    messages = [record.getMessage() for record in caplog.records]
    salvage_logs = [message for message in messages if "Salvaging" in message]
    preserving_logs = [message for message in messages if "Preserving" in message]

    assert len(salvage_logs) == 1
    assert len(preserving_logs) == 1
    assert "(1/1)" in preserving_logs[0]
    assert "Row group 0" not in "\n".join(messages)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_drops_non_preserved_retryable_errors_when_salvage_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausted retryable errors are dropped unless they are explicitly preserved."""
    provider = _mock_provider()
    monkeypatch.setattr(async_scheduler_module, "RETRYABLE_RESALVAGE_BACKOFF_S", 0)
    mixed = MockMixedRetryableGenerator(
        config=_expr_config("cell_out"),
        resource_provider=provider,
        rate_limit_failures=3,
    )
    generators, graph, row_groups, tracker, buffer_mgr, _storage = _seed_plus_cell_setup(
        mixed,
        num_records=2,
    )
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        salvage_max_rounds=1,
    )

    await scheduler.run()

    assert tracker.is_row_group_complete(0, 2, ["seed", "cell_out"])
    assert not tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 1)
    assert buffer_mgr.get_row(0, 0)["cell_out"] == "mixed_ok_0"
    assert scheduler._deferred == []
    assert scheduler._deferred_errors == {}
    assert scheduler._preserved_retryable_counts == {}
    assert scheduler._preserved_retryable_log_state == {}


def test_scheduler_rejects_zero_salvage_rounds() -> None:
    """At least one salvage round is required so preserved 429 work can retry."""
    config = ExpressionColumnConfig(name="llm_col", expr="'x'", dtype="str")
    graph = ExecutionGraph.create([config], {"llm_col": GenerationStrategy.CELL_BY_CELL})

    with pytest.raises(ValueError, match="salvage_max_rounds must be at least 1"):
        AsyncTaskScheduler(
            generators={"llm_col": MockCellGenerator(config=config, resource_provider=_mock_provider())},
            graph=graph,
            tracker=CompletionTracker.with_graph(graph, [(0, 1)]),
            row_groups=[(0, 1)],
            salvage_max_rounds=0,
        )


@pytest.mark.parametrize(
    "error",
    [
        ModelRateLimitError("429 Too Many Requests"),
        ModelRequestAdmissionTimeoutError("request admission queue timeout"),
    ],
)
def test_retryable_resalvage_delay_uses_request_cooldown(error: Exception) -> None:
    """Scheduler-level pacing should respect request-admission cooldown when available."""
    provider = _mock_provider()
    config = ExpressionColumnConfig(name="llm_col", expr="'x'", dtype="str")
    graph = ExecutionGraph.create([config], {"llm_col": GenerationStrategy.CELL_BY_CELL})
    task = Task(column="llm_col", row_group=0, row_index=0, task_type="cell")
    resource = RequestResourceKey("provider", "model", RequestDomain.CHAT)

    class PressureProvider:
        @property
        def config(self) -> None:
            return None

        def snapshot(self, request_resource: RequestResourceKey) -> RequestPressureSnapshot | None:
            if request_resource != resource:
                return None
            return RequestPressureSnapshot(
                captured_at=0.0,
                sequence=1,
                resource=resource,
                effective_max=1,
                current_limit=1,
                in_flight_count=0,
                active_lease_count=0,
                waiters=0,
                blocked_until_monotonic=100.0,
                cooldown_remaining_seconds=0.25,
                rate_limit_ceiling=1,
                consecutive_rate_limits=1,
                last_outcome="rate_limited",
                leak_diagnostics={},
            )

        def snapshots(self) -> dict[RequestResourceKey, RequestPressureSnapshot]:
            snapshot = self.snapshot(resource)
            return {resource: snapshot} if snapshot is not None else {}

        def global_snapshot(self, provider_name: str, model: str) -> None:
            return None

        def global_snapshots(self) -> dict[ProviderModelKey, object]:
            return {}

    scheduler = AsyncTaskScheduler(
        generators={"llm_col": MockModelRateLimitGenerator(config=config, resource_provider=provider)},
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, [(0, 1)]),
        row_groups=[(0, 1)],
        request_pressure_provider=PressureProvider(),
    )
    scheduler._deferred = [task]
    scheduler._deferred_errors[task] = error

    assert scheduler._retryable_resalvage_delay_seconds() == pytest.approx(0.25)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_cancellation_releases_task_admission_leases() -> None:
    """Cancelling the scheduler while model-stage tasks are in-flight releases task leases."""
    provider = _mock_provider()

    blocked = asyncio.Event()
    proceed = asyncio.Event()

    class BlockingLLMGenerator(ColumnGenerator[ExpressionColumnConfig]):
        def get_scheduling_metadata(self) -> SchedulingMetadata:
            return SchedulingMetadata.custom_model("test", self.config.name, "v1")

        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.CELL_BY_CELL

        def generate(self, data: dict) -> dict:
            data[self.config.name] = "val"
            return data

        async def agenerate(self, data: dict) -> dict:
            blocked.set()
            await proceed.wait()
            return self.generate(data)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="llm_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "llm_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "llm_col": BlockingLLMGenerator(config=_expr_config("llm_col"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    max_in_flight = 4
    max_llm_wait = 2
    sink = InMemoryAdmissionEventSink()
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=max_in_flight,
        max_model_task_admission=max_llm_wait,
        scheduler_event_sink=sink,
    )

    run_task = asyncio.create_task(scheduler.run())

    await asyncio.wait_for(blocked.wait(), timeout=5.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    snapshot = scheduler.task_admission_snapshot()
    assert snapshot.resources_available["submission"] == max_in_flight
    assert snapshot.resources_available["llm_wait"] == max_llm_wait
    assert "cancelled" in [event.event_kind for event in sink.scheduler_events]
    assert all(event.snapshot is not None for event in sink.scheduler_events)
    task_events = [event for event in sink.scheduler_events if event.task_id is not None]
    assert all("resource_request" in event.diagnostics for event in task_events)
    assert any("llm_wait" in event.diagnostics["resource_request"] for event in task_events)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_rg_semaphore_deadlock_with_transient_failures() -> None:
    """Row groups stalled by transient failures don't block admission of new row groups.

    Regression test: with max_concurrent_row_groups=1 and 2 row groups, if all
    tasks in RG0 fail transiently, row-group capacity must still be released so RG1
    can be admitted.  The scheduler salvages RG0 inline and continues.
    """
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "col": GenerationStrategy.CELL_BY_CELL,
    }
    # Fail the first 2 calls (all of RG0), then succeed for everything after.
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "col": MockFailingGenerator(config=_expr_config("col"), resource_provider=provider, transient_failures=2),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2), (1, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_concurrent_row_groups=1,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    assert tracker.is_row_group_complete(0, 2, ["seed", "col"])
    assert tracker.is_row_group_complete(1, 2, ["seed", "col"])


def test_side_effect_columns_separated_from_completion_tracking() -> None:
    """Side-effect columns must appear in _gen_instance_to_columns_including_side_effects
    (buffer writes) but NOT in _gen_instance_to_columns (completion tracking), because
    they are not registered in the execution graph and would cause KeyError in
    CompletionTracker.
    """
    graph = ExecutionGraph()
    graph.add_column("seed", GenerationStrategy.FULL_COLUMN)
    graph.add_column("primary", GenerationStrategy.CELL_BY_CELL)
    graph.add_edge(upstream="seed", downstream="primary")

    row_groups = [(0, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    provider = _mock_provider()
    seed_gen = MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider)
    cell_gen = MockCellGenerator(config=_expr_config("primary"), resource_provider=provider)
    # Replace the config with a mock that reports side-effect columns.
    mock_config = MagicMock()
    mock_config.side_effect_columns = ["side_a", "side_b"]
    object.__setattr__(cell_gen, "_config", mock_config)

    generators: dict[str, ColumnGenerator] = {"seed": seed_gen, "primary": cell_gen}

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
    )

    cell_id = id(cell_gen)

    # Completion tracking dict: only real columns
    assert "side_a" not in scheduler._gen_instance_to_columns.get(cell_id, [])
    assert "side_b" not in scheduler._gen_instance_to_columns.get(cell_id, [])
    assert "primary" in scheduler._gen_instance_to_columns.get(cell_id, [])

    # Buffer write dict: includes side-effect columns
    write_cols = scheduler._gen_instance_to_columns_including_side_effects.get(cell_id, [])
    assert "primary" in write_cols
    assert "side_a" in write_cols
    assert "side_b" in write_cols


# -- Pipeline parallelism (stale dispatch fix, issue #504) ---------------------


class SlowCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Cell-by-cell generator with configurable async delay."""

    def __init__(self, *args: Any, delay: float = 0.05, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._delay = delay

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"gen_{data.get('seed', '?')}"
        return data

    async def agenerate(self, data: dict) -> dict:
        await asyncio.sleep(self._delay)
        return self.generate(data)


class SlowLLMBoundCellGenerator(SlowCellGenerator):
    """Slow cell generator that participates in model-stage scheduling."""

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.custom_model("test", self.config.name, "v1")


class SlowModelBoundCellGenerator(SlowCellGenerator):
    """Slow cell generator with concrete request-pressure identity."""

    def __init__(
        self,
        *args: Any,
        provider_name: str = "provider",
        model_id: str = "model",
        generation_kind: str = "chat",
        request_weight: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._provider_name = provider_name
        self._model_id = model_id
        self._generation_kind = generation_kind
        self._request_weight = request_weight

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.model(
            self._provider_name,
            self._model_id,
            self._generation_kind,
            weight=self._request_weight,
        )


class GatedRequestAdmissionCellGenerator(SlowModelBoundCellGenerator):
    """Model-bound generator that holds initial request-admission leases."""

    def __init__(
        self,
        *args: Any,
        request_admission: AdaptiveRequestAdmissionController,
        hold_until_active: int,
        provider_name: str = "provider",
        model_id: str = "model",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, provider_name=provider_name, model_id=model_id, **kwargs)
        self._request_admission = request_admission
        self._hold_until_active = hold_until_active
        self._resource = RequestResourceKey(provider_name, model_id, RequestDomain.CHAT)
        self._started_count = 0
        self._active_leases = 0
        self.initial_leases_acquired: asyncio.Event = asyncio.Event()
        self.release_held_leases: asyncio.Event = asyncio.Event()

    async def agenerate(self, data: dict) -> dict:
        item = RequestAdmissionItem(self._resource, RequestGroupSpec(self._resource))
        lease = await self._request_admission.acquire_async(item)
        self._started_count += 1
        started_index = self._started_count
        self._active_leases += 1
        if self._active_leases >= self._hold_until_active:
            self.initial_leases_acquired.set()
        try:
            if started_index <= self._hold_until_active:
                await self.release_held_leases.wait()
            data[self.config.name] = f"gated_{started_index}"
            return data
        finally:
            self._active_leases = max(0, self._active_leases - 1)
            self._request_admission.release(lease, RequestReleaseOutcome(kind="success"))


class _StaticRequestPressureProvider:
    def __init__(
        self,
        snapshots: dict[RequestResourceKey, RequestPressureSnapshot],
    ) -> None:
        self._snapshots = snapshots

    @property
    def config(self) -> RequestAdmissionConfig | None:
        return None

    def snapshot(self, resource: RequestResourceKey) -> RequestPressureSnapshot | None:
        return self._snapshots.get(resource)

    def snapshots(self) -> dict[RequestResourceKey, RequestPressureSnapshot]:
        return dict(self._snapshots)

    def global_snapshot(self, _provider: str, _model: str) -> None:
        return None

    def global_snapshots(self) -> dict[ProviderModelKey, object]:
        return {}


def _pressure_snapshot(
    resource: RequestResourceKey,
    *,
    current_limit: int = 1,
    in_flight: int = 0,
    waiters: int = 0,
    cooldown: float = 0.0,
) -> RequestPressureSnapshot:
    return RequestPressureSnapshot(
        captured_at=time.monotonic(),
        sequence=1,
        resource=resource,
        effective_max=max(1, current_limit),
        current_limit=current_limit,
        in_flight_count=in_flight,
        active_lease_count=in_flight,
        waiters=waiters,
        blocked_until_monotonic=time.monotonic() + cooldown if cooldown > 0.0 else None,
        cooldown_remaining_seconds=cooldown,
        rate_limit_ceiling=max(1, current_limit),
        consecutive_rate_limits=0,
        last_outcome=None,
        leak_diagnostics={},
    )


def _build_queued_model_pressure_scheduler(
    *,
    column: str = "pressured",
    provider_name: str = "provider-a",
    model_id: str = "model-a",
    queued_rows: int = 5,
    request_pressure_provider: Any,
    scheduler_event_sink: InMemoryAdmissionEventSink | None = None,
) -> AsyncTaskScheduler:
    provider = _mock_provider()
    config = LLMTextColumnConfig(name=column, prompt="A", model_alias=MODEL_ALIAS)
    graph = ExecutionGraph.create([config], {column: GenerationStrategy.CELL_BY_CELL})
    row_groups = [(0, queued_rows)]
    generator = SlowModelBoundCellGenerator(
        config=_expr_config(column),
        resource_provider=provider,
        provider_name=provider_name,
        model_id=model_id,
        delay=30.0,
    )
    scheduler = AsyncTaskScheduler(
        generators={column: generator},
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        request_pressure_provider=request_pressure_provider,
        request_pressure_advisory=True,
        scheduler_event_sink=scheduler_event_sink,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=queued_rows, pre_batch_done=True, in_flight_count=0)
    tasks = tuple(
        scheduler._schedulable_task(Task(column=column, row_group=0, row_index=row_index, task_type="cell"))
        for row_index in range(queued_rows)
    )
    scheduler._fair_queue.enqueue(tasks)
    return scheduler


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_fair_admission_across_ready_columns() -> None:
    """A large ready frontier is admitted across columns instead of one column at a time."""
    provider = _mock_provider()
    gen_names = ["gen_a", "gen_b", "gen_c"]
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        *[LLMTextColumnConfig(name=g, prompt="{{ topic }}", model_alias=MODEL_ALIAS) for g in gen_names],
    ]
    strategies: dict[str, GenerationStrategy] = {"topic": GenerationStrategy.FULL_COLUMN}
    strategies.update({c: GenerationStrategy.CELL_BY_CELL for c in gen_names})
    generators: dict[str, ColumnGenerator] = {
        "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
        **{
            name: SlowCellGenerator(config=_expr_config(name), resource_provider=provider, delay=0.05)
            for name in gen_names
        },
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 12)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=4,
        trace=True,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    first_window = [
        trace.column
        for trace in sorted((t for t in scheduler.traces if t.column in gen_names), key=lambda t: t.dispatched_at)[:4]
    ]

    assert set(first_window[:3]) == set(gen_names)
    assert max(first_window.count(column) for column in gen_names) <= 2
    assert tracker.is_row_group_complete(0, 12, ["topic", *gen_names])


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_fair_admission_across_ready_columns_and_row_groups() -> None:
    """Fair admission stays column-balanced when multiple row groups are ready."""
    provider = _mock_provider()
    gen_names = ["gen_a", "gen_b", "gen_c"]

    class BarrierSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
        def __init__(self, *args: Any, expected_calls: int, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._expected_calls = expected_calls
            self._started = 0
            self._lock = asyncio.Lock()
            self._release = asyncio.Event()

        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.FULL_COLUMN

        def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
            return data

        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            return lazy.pd.DataFrame({self.config.name: ["A"] * num_records})

        async def agenerate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            async with self._lock:
                self._started += 1
                if self._started == self._expected_calls:
                    self._release.set()
            await self._release.wait()
            return self.generate_from_scratch(num_records)

    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        *[LLMTextColumnConfig(name=g, prompt="{{ topic }}", model_alias=MODEL_ALIAS) for g in gen_names],
    ]
    strategies: dict[str, GenerationStrategy] = {"topic": GenerationStrategy.FULL_COLUMN}
    strategies.update({c: GenerationStrategy.CELL_BY_CELL for c in gen_names})
    row_groups = [(0, 3), (1, 3)]
    generators: dict[str, ColumnGenerator] = {
        "topic": BarrierSeedGenerator(
            config=_expr_config("topic"),
            resource_provider=provider,
            expected_calls=len(row_groups),
        ),
        **{
            name: SlowCellGenerator(config=_expr_config(name), resource_provider=provider, delay=0.05)
            for name in gen_names
        },
    }

    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=8,
        max_concurrent_row_groups=2,
        trace=True,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    cell_traces = sorted(
        (t for t in scheduler.traces if t.column in gen_names),
        key=lambda t: t.dispatched_at,
    )
    first_six = cell_traces[:6]
    first_twelve = cell_traces[:12]

    assert len(cell_traces) == 18
    assert all({t.column for t in first_six[i : i + 3]} == set(gen_names) for i in range(0, 6, 3))
    assert all(sum(1 for t in first_twelve if t.column == column) == 4 for column in gen_names)
    assert {t.row_group for t in first_twelve} == {0, 1}
    assert all(tracker.is_row_group_complete(rg_id, rg_size, ["topic", *gen_names]) for rg_id, rg_size in row_groups)


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_fair_llm_group_cap_preserves_peer_admission() -> None:
    """One LLM-bound column cannot consume the whole initial LLM admission window."""
    provider = _mock_provider()
    gen_names = ["hot", "peer"]
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        *[LLMTextColumnConfig(name=g, prompt="{{ topic }}", model_alias=MODEL_ALIAS) for g in gen_names],
    ]
    strategies: dict[str, GenerationStrategy] = {"topic": GenerationStrategy.FULL_COLUMN}
    strategies.update({c: GenerationStrategy.CELL_BY_CELL for c in gen_names})
    generators: dict[str, ColumnGenerator] = {
        "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
        **{
            name: SlowLLMBoundCellGenerator(config=_expr_config(name), resource_provider=provider, delay=0.05)
            for name in gen_names
        },
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 8)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=4,
        max_model_task_admission=4,
        trace=True,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    first_window = [
        trace.column
        for trace in sorted((t for t in scheduler.traces if t.column in gen_names), key=lambda t: t.dispatched_at)[:4]
    ]

    assert first_window.count("hot") == 2
    assert first_window.count("peer") == 2
    assert tracker.is_row_group_complete(0, 8, ["topic", *gen_names])
    snapshot = scheduler.task_admission_snapshot()
    assert snapshot.resources_available["submission"] == 4
    assert snapshot.resources_available["llm_wait"] == 4


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_downstream_interleaves_with_upstream() -> None:
    """Downstream judge tasks begin before all upstream gen tasks complete (issue #504).

    Mirrors the reported pipeline topology:

        topic (sampler, instant)
        ├── gen_a (slow, 50ms) → judge_a (instant)
        ├── gen_b (slow, 50ms) → judge_b (instant)
        └── gen_c (slow, 50ms) → judge_c (instant)

    With small task admission capacity (4) and 10 records, the 30 gen tasks
    saturate admission. The dispatch loop must re-query the frontier when capacity
    is full so that judge tasks from completed gen rows are picked up
    before all gen tasks finish.
    """
    provider = _mock_provider()
    gen_names = ["gen_a", "gen_b", "gen_c"]
    judge_names = ["judge_a", "judge_b", "judge_c"]

    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        *[LLMTextColumnConfig(name=g, prompt="{{ topic }}", model_alias=MODEL_ALIAS) for g in gen_names],
        *[
            LLMTextColumnConfig(name=j, prompt=f"{{{{ {g} }}}}", model_alias=MODEL_ALIAS)
            for j, g in zip(judge_names, gen_names)
        ],
    ]
    all_col_names = ["topic", *gen_names, *judge_names]
    strategies: dict[str, GenerationStrategy] = {"topic": GenerationStrategy.FULL_COLUMN}
    strategies.update({c: GenerationStrategy.CELL_BY_CELL for c in gen_names + judge_names})

    generators: dict[str, ColumnGenerator] = {
        "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
    }
    for g in gen_names:
        generators[g] = SlowCellGenerator(config=_expr_config(g), resource_provider=provider, delay=0.05)
    for j in judge_names:
        generators[j] = MockCellGenerator(config=_expr_config(j), resource_provider=provider)

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 10)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    buffer_manager = RowGroupBufferManager(graph.columns)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        max_in_flight_tasks=4,
        trace=True,
    )
    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    assert tracker.is_row_group_complete(0, 10, all_col_names)

    gen_traces = [t for t in scheduler.traces if t.column in gen_names]
    judge_traces = [t for t in scheduler.traces if t.column in judge_names]
    assert len(gen_traces) == 30  # 3 cols x 10 rows
    assert len(judge_traces) == 30

    last_gen_dispatched = max(t.dispatched_at for t in gen_traces)
    first_judge_dispatched = min(t.dispatched_at for t in judge_traces)

    assert first_judge_dispatched < last_gen_dispatched, (
        "Judge tasks should begin before all gen tasks are dispatched. "
        f"First judge dispatched at {first_judge_dispatched:.4f}, "
        f"last gen dispatched at {last_gen_dispatched:.4f}."
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_capacity_plan_observes_buffer_backpressure() -> None:
    provider = _mock_provider()
    gen_names = ["gen_a", "gen_b", "gen_c"]
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        *[LLMTextColumnConfig(name=g, prompt="{{ topic }}", model_alias=MODEL_ALIAS) for g in gen_names],
    ]
    strategies: dict[str, GenerationStrategy] = {"topic": GenerationStrategy.FULL_COLUMN}
    strategies.update({column: GenerationStrategy.CELL_BY_CELL for column in gen_names})
    generators: dict[str, ColumnGenerator] = {
        "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
        **{
            name: SlowCellGenerator(config=_expr_config(name), resource_provider=provider, delay=0.02)
            for name in gen_names
        },
    }
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3), (1, 3), (2, 3), (3, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_concurrent_row_groups=2,
        max_in_flight_tasks=2,
        trace=True,
        num_records=12,
        buffer_size=3,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    plan = scheduler.capacity_plan()
    for row_group_index, row_count in row_groups:
        assert tracker.is_row_group_complete(row_group_index, row_count, ["topic", *gen_names])
    assert plan.configured.row_group_admission.observed_in_flight == 0
    assert plan.observed_maxima.row_groups_in_flight == 2
    assert plan.observed_maxima.queued_tasks_by_group
    assert max(plan.observed_maxima.task_leases_by_resource.values()) <= 2


def test_scheduler_capacity_plan_reports_request_admission_state() -> None:
    resource = RequestResourceKey("provider", "model", RequestDomain.CHAT)
    request_admission = AdaptiveRequestAdmissionController(
        RequestAdmissionConfig(initial_limits={resource: 2}, max_limit_clamps={resource: 3})
    )
    request_admission.register(
        provider_name="provider",
        model_id="model",
        alias="primary",
        max_parallel_requests=4,
    )
    lease = request_admission.try_acquire(RequestAdmissionItem(resource, RequestGroupSpec(resource)))
    assert isinstance(lease, RequestAdmissionLease)

    scheduler, _tracker = _build_simple_pipeline()
    scheduler._request_pressure_provider = request_admission
    scheduler._record_observed_task_state()
    plan = scheduler.capacity_plan()

    assert plan.configured.request_resources.value == (resource,)
    assert plan.configured.request_domain_initial_limits.value[resource] == 2
    assert plan.configured.request_admission_config.value is not None
    assert plan.configured.provider_model_static_caps.value[ProviderModelKey("provider", "model")].cap == 4
    assert plan.runtime_snapshot.request_domain_current_limits[resource] == 2
    assert plan.runtime_snapshot.request_domain_effective_max[resource] == 3
    assert plan.runtime_snapshot.provider_model_aggregate_in_flight[ProviderModelKey("provider", "model")] == 1
    assert plan.observed_maxima.request_in_flight_by_resource[resource] == 1
    assert plan.observed_maxima.provider_model_aggregate_in_flight[ProviderModelKey("provider", "model")] == 1
    request_admission.release(lease, RequestReleaseOutcome(kind="success"))


def test_scheduler_capacity_plan_reports_default_request_initial_limit_after_aimd_drop() -> None:
    resource = RequestResourceKey("provider", "model", RequestDomain.CHAT)
    request_admission = AdaptiveRequestAdmissionController()
    request_admission.register(
        provider_name="provider",
        model_id="model",
        alias="primary",
        max_parallel_requests=4,
    )
    lease = request_admission.try_acquire(RequestAdmissionItem(resource, RequestGroupSpec(resource)))
    assert isinstance(lease, RequestAdmissionLease)
    request_admission.release(lease, RequestReleaseOutcome(kind="rate_limited"))

    scheduler, _tracker = _build_simple_pipeline()
    scheduler._request_pressure_provider = request_admission
    plan = scheduler.capacity_plan()

    assert plan.configured.request_domain_initial_limits.value[resource] == 4
    assert plan.runtime_snapshot.request_domain_effective_max[resource] == 4
    assert plan.runtime_snapshot.request_domain_current_limits[resource] == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_emits_job_health_and_row_group_telemetry() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="model_col", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "topic": GenerationStrategy.FULL_COLUMN,
        "model_col": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 2)]
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    sink = InMemoryAdmissionEventSink()
    scheduler = AsyncTaskScheduler(
        generators={
            "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
            "model_col": SlowLLMBoundCellGenerator(
                config=_expr_config("model_col"),
                resource_provider=provider,
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_concurrent_row_groups=1,
        max_in_flight_tasks=2,
        max_model_task_admission=1,
        scheduler_event_sink=sink,
        num_records=2,
        buffer_size=2,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    kinds = [event.event_kind for event in sink.scheduler_events]
    assert "scheduler_job_started" in kinds
    assert "scheduler_health_snapshot" in kinds
    assert "row_group_checkpointed" in kinds
    assert "scheduler_job_completed" in kinds

    started = next(event for event in sink.scheduler_events if event.event_kind == "scheduler_job_started")
    assert started.diagnostics["num_records"] == 2
    assert started.diagnostics["buffer_size"] == 2
    assert started.diagnostics["row_group_count"] == 1
    assert started.diagnostics["graph_depth"] == 2
    column_scheduling = started.diagnostics["column_scheduling"]
    assert isinstance(column_scheduling, list)
    model_column = next(item for item in column_scheduling if item["column"] == "model_col")
    assert model_column["group_kind"] == "custom_model"
    assert model_column["resource_request"] == {"submission": 1, "llm_wait": 1}

    health = next(event for event in sink.scheduler_events if event.event_kind == "scheduler_health_snapshot")
    assert "queued_total" in health.diagnostics
    assert "leased_resources" in health.diagnostics
    assert "request_pressure" in health.diagnostics

    checkpointed = next(event for event in sink.scheduler_events if event.event_kind == "row_group_checkpointed")
    assert checkpointed.diagnostics["row_group"] == 0
    assert checkpointed.diagnostics["row_group_size"] == 2
    assert checkpointed.diagnostics["surviving_rows"] == 2

    completed = [event for event in sink.scheduler_events if event.event_kind == "scheduler_job_completed"]
    assert len(completed) == 1
    assert completed[0].reason_or_result == "success"
    assert completed[0].diagnostics["outcome"] == "success"
    assert completed[0].diagnostics["reason"] == "completed"
    assert "queued_total" in completed[0].diagnostics
    assert "request_pressure" in completed[0].diagnostics
    assert "error_type" not in completed[0].diagnostics


def test_scheduler_filter_skips_snapshot_and_health_diagnostics() -> None:
    sink = _FilteringSchedulerSink({"row_group_checkpointed"})
    scheduler, _tracker = _build_simple_pipeline(scheduler_event_sink=sink)
    scheduler.task_admission_snapshot = MagicMock(wraps=scheduler.task_admission_snapshot)
    scheduler._scheduler_health_diagnostics = MagicMock(wraps=scheduler._scheduler_health_diagnostics)

    scheduler._emit_scheduler_event("selected")
    scheduler._emit_scheduler_health_snapshot("test")

    scheduler.task_admission_snapshot.assert_not_called()
    scheduler._scheduler_health_diagnostics.assert_not_called()
    assert sink.scheduler_events == []

    scheduler._emit_scheduler_event("row_group_checkpointed")
    scheduler.task_admission_snapshot.assert_called_once_with()
    assert [event.event_kind for event in sink.scheduler_events] == ["row_group_checkpointed"]


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_inherits_ambient_run_correlation_for_taskless_events() -> None:
    sink = _FilteringSchedulerSink({"scheduler_job_completed"})
    ambient = RuntimeCorrelation(
        run_id="ambient-run",
        row_group=None,
        task_column=None,
        task_type=None,
        scheduling_group_kind=None,
        scheduling_group_identity_hash=None,
        task_execution_id=None,
    )
    token = runtime_correlation_provider.set(ambient)
    try:
        scheduler, _tracker = _build_simple_pipeline(num_records=1, scheduler_event_sink=sink)
    finally:
        runtime_correlation_provider.reset(token)

    await scheduler.run()

    [completed] = sink.scheduler_events
    assert completed.task_id is None
    assert completed.captured_correlation["run_id"] == "ambient-run"
    assert completed.captured_correlation["row_group"] is None


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("raised", "expected_outcome", "expected_error_type"),
    [
        (RuntimeError("scheduler failed"), "error", "RuntimeError"),
        (asyncio.CancelledError(), "cancelled", None),
    ],
)
async def test_scheduler_emits_one_terminal_event_without_replacing_failure(
    monkeypatch: pytest.MonkeyPatch,
    raised: BaseException,
    expected_outcome: str,
    expected_error_type: str | None,
) -> None:
    sink = _FilteringSchedulerSink({"scheduler_job_completed"})
    scheduler, _tracker = _build_simple_pipeline(num_records=1, scheduler_event_sink=sink)

    async def fail(*_args: object) -> None:
        raise raised

    monkeypatch.setattr(scheduler, "_main_dispatch_loop", fail)

    with pytest.raises(type(raised)) as exc_info:
        await scheduler.run()

    assert exc_info.value is raised
    [completed] = sink.scheduler_events
    assert completed.reason_or_result == expected_outcome
    assert completed.diagnostics["outcome"] == expected_outcome
    assert completed.diagnostics.get("error_type") == expected_error_type


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_adaptive_row_group_admission_expands_target_for_horizon_idle() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="model_col", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "topic": GenerationStrategy.FULL_COLUMN,
        "model_col": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1), (1, 1), (2, 1), (3, 1)]
    generators: dict[str, ColumnGenerator] = {
        "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
        "model_col": SlowLLMBoundCellGenerator(
            config=_expr_config("model_col"),
            resource_provider=provider,
            delay=0.04,
        ),
    }
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    sink = InMemoryAdmissionEventSink()
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_concurrent_row_groups=4,
        max_in_flight_tasks=4,
        max_model_task_admission=4,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        scheduler_event_sink=sink,
        trace=True,
        num_records=4,
        buffer_size=1,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    plan = scheduler.capacity_plan()
    assert tracker.is_row_group_complete(0, 1, ["topic", "model_col"])
    assert plan.configured.row_group_admission.mode == "adaptive"
    assert plan.configured.row_group_admission.observed_max_target is not None
    assert plan.configured.row_group_admission.observed_max_target > 1
    assert plan.observed_maxima.row_groups_in_flight > 1
    assert any(event.event_kind == "row_group_admission_target_changed" for event in sink.scheduler_events)


def _build_adaptive_model_resource_scheduler(
    *,
    columns: tuple[str, ...] = ("cooling", "healthy"),
    row_groups: list[tuple[int, int]] | None = None,
) -> tuple[AsyncTaskScheduler, CompletionTracker]:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        *[LLMTextColumnConfig(name=column, prompt="{{ seed }}", model_alias=MODEL_ALIAS) for column in columns],
    ]
    strategies: dict[str, GenerationStrategy] = {"seed": GenerationStrategy.FULL_COLUMN}
    strategies.update({column: GenerationStrategy.CELL_BY_CELL for column in columns})
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
    }
    for column in columns:
        generators[column] = SlowModelBoundCellGenerator(
            config=_expr_config(column),
            resource_provider=provider,
            provider_name="provider",
            model_id=column,
            delay=0.0,
        )
    graph = ExecutionGraph.create(configs, strategies)
    row_groups = row_groups or [(0, 1), (1, 1)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_concurrent_row_groups=max(2, len(row_groups)),
        max_in_flight_tasks=16,
        max_model_task_admission=16,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        num_records=sum(size for _row_group, size in row_groups),
        buffer_size=1,
    )
    return scheduler, tracker


def _set_next_row_group_pending(scheduler: AsyncTaskScheduler, row_group: int = 1, row_group_size: int = 1) -> None:
    scheduler._row_group_admission_pending = (row_group, row_group_size)


def test_scheduler_adaptive_row_group_ignores_unrelated_deferred_retry_resource() -> None:
    scheduler, _tracker = _build_adaptive_model_resource_scheduler()
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler)
    deferred = Task(column="cooling", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    diagnostics = scheduler._row_group_admission_diagnostics(reason="probe")["deferred_admission"]

    assert scheduler._adaptive_row_group_block_reason() is None
    assert diagnostics["blocks_next_row_group"] is False
    assert diagnostics["scope"] == "localized"
    assert diagnostics["columns"] == {"cooling": 1}
    assert diagnostics["request_resources"] == {"provider/cooling/chat": 1}
    assert "healthy" in diagnostics["independent_candidate_columns"]


def test_scheduler_adaptive_row_group_blocks_same_deferred_retry_resource() -> None:
    scheduler, _tracker = _build_adaptive_model_resource_scheduler(columns=("cooling",))
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler)
    deferred = Task(column="cooling", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    diagnostics = scheduler._row_group_admission_diagnostics(reason="probe")["deferred_admission"]

    assert scheduler._adaptive_row_group_block_reason() == "deferred_tasks"
    assert diagnostics["blocks_next_row_group"] is True
    assert set(diagnostics["candidate_columns"]) == {"seed", "cooling"}
    assert diagnostics["independent_candidate_columns"] == ()


def test_scheduler_adaptive_row_group_blocks_downstream_candidate_behind_deferred_resource() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cooling_a", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="cooling_b", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="healthy", prompt="{{ cooling_b }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cooling_a": GenerationStrategy.CELL_BY_CELL,
        "cooling_b": GenerationStrategy.CELL_BY_CELL,
        "healthy": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1), (1, 1)]
    graph = ExecutionGraph.create(configs, strategies)
    scheduler = AsyncTaskScheduler(
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "cooling_a": SlowModelBoundCellGenerator(
                config=_expr_config("cooling_a"),
                resource_provider=provider,
                provider_name="provider",
                model_id="cooling",
                delay=0.0,
            ),
            "cooling_b": SlowModelBoundCellGenerator(
                config=_expr_config("cooling_b"),
                resource_provider=provider,
                provider_name="provider",
                model_id="cooling",
                delay=0.0,
            ),
            "healthy": SlowModelBoundCellGenerator(
                config=_expr_config("healthy"),
                resource_provider=provider,
                provider_name="provider",
                model_id="healthy",
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        max_concurrent_row_groups=2,
        max_in_flight_tasks=16,
        max_model_task_admission=16,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        num_records=2,
        buffer_size=1,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler)
    deferred = Task(column="cooling_a", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    diagnostics = scheduler._row_group_admission_diagnostics(reason="probe")["deferred_admission"]

    assert scheduler._adaptive_row_group_block_reason() == "deferred_tasks"
    assert diagnostics["blocks_next_row_group"] is True
    assert set(diagnostics["candidate_columns"]) == {"seed", "cooling_a", "cooling_b", "healthy"}
    assert diagnostics["independent_candidate_columns"] == ()


def test_scheduler_adaptive_row_group_blocks_multi_output_sibling_dependency() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cooling_a", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="cooling_b", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="healthy", prompt="{{ cooling_b }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cooling_a": GenerationStrategy.CELL_BY_CELL,
        "cooling_b": GenerationStrategy.CELL_BY_CELL,
        "healthy": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1), (1, 1)]
    graph = ExecutionGraph.create(configs, strategies)
    cooling = SlowModelBoundCellGenerator(
        config=_expr_config("cooling_a"),
        resource_provider=provider,
        provider_name="provider",
        model_id="cooling",
        delay=0.0,
    )
    scheduler = AsyncTaskScheduler(
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "cooling_a": cooling,
            "cooling_b": cooling,
            "healthy": SlowModelBoundCellGenerator(
                config=_expr_config("healthy"),
                resource_provider=provider,
                provider_name="provider",
                model_id="healthy",
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        max_concurrent_row_groups=2,
        max_in_flight_tasks=16,
        max_model_task_admission=16,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        num_records=2,
        buffer_size=1,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler)
    deferred = Task(column="cooling_a", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    diagnostics = scheduler._row_group_admission_diagnostics(reason="probe")["deferred_admission"]

    assert scheduler._adaptive_row_group_block_reason() == "deferred_tasks"
    assert diagnostics["blocks_next_row_group"] is True
    assert "healthy" not in diagnostics["independent_candidate_columns"]


def test_scheduler_adaptive_row_group_blocks_shared_scheduler_resource_across_domains() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="chat_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="embedding_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "chat_col": GenerationStrategy.CELL_BY_CELL,
        "embedding_col": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1), (1, 1)]
    graph = ExecutionGraph.create(configs, strategies)
    scheduler = AsyncTaskScheduler(
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "chat_col": SlowModelBoundCellGenerator(
                config=_expr_config("chat_col"),
                resource_provider=provider,
                provider_name="provider",
                model_id="model",
                generation_kind=RequestDomain.CHAT.value,
                delay=0.0,
            ),
            "embedding_col": SlowModelBoundCellGenerator(
                config=_expr_config("embedding_col"),
                resource_provider=provider,
                provider_name="provider",
                model_id="model",
                generation_kind=RequestDomain.EMBEDDING.value,
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        max_concurrent_row_groups=2,
        max_in_flight_tasks=16,
        max_model_task_admission=16,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        num_records=2,
        buffer_size=1,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler)
    deferred = Task(column="chat_col", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    diagnostics = scheduler._row_group_admission_diagnostics(reason="probe")["deferred_admission"]

    assert scheduler._adaptive_row_group_block_reason() == "deferred_tasks"
    assert diagnostics["blocks_next_row_group"] is True
    assert diagnostics["request_resources"] == {"provider/model/chat": 1}
    assert diagnostics["scheduler_resources"] == {"request:provider/model": 1}
    assert diagnostics["independent_candidate_columns"] == ()


def test_scheduler_adaptive_row_group_counts_independent_local_branch() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cooling", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="local_branch", expr="'local'", dtype="str"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cooling": GenerationStrategy.CELL_BY_CELL,
        "local_branch": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1), (1, 1)]
    graph = ExecutionGraph.create(configs, strategies)
    scheduler = AsyncTaskScheduler(
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "cooling": SlowModelBoundCellGenerator(
                config=_expr_config("cooling"),
                resource_provider=provider,
                provider_name="provider",
                model_id="cooling",
                delay=0.0,
            ),
            "local_branch": MockCellGenerator(config=_expr_config("local_branch"), resource_provider=provider),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        max_concurrent_row_groups=2,
        max_in_flight_tasks=16,
        max_model_task_admission=16,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        num_records=2,
        buffer_size=1,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler)
    deferred = Task(column="cooling", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    diagnostics = scheduler._row_group_admission_diagnostics(reason="probe")["deferred_admission"]

    assert scheduler._adaptive_row_group_block_reason() is None
    assert diagnostics["blocks_next_row_group"] is False
    assert "local_branch" in diagnostics["independent_candidate_columns"]
    assert "seed" not in diagnostics["independent_candidate_columns"]


def test_scheduler_adaptive_row_group_localizes_custom_model_deferred_retry() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cooling", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="healthy", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cooling": GenerationStrategy.CELL_BY_CELL,
        "healthy": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1), (1, 1)]
    graph = ExecutionGraph.create(configs, strategies)
    scheduler = AsyncTaskScheduler(
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "cooling": SlowLLMBoundCellGenerator(
                config=_expr_config("cooling"),
                resource_provider=provider,
                delay=0.0,
            ),
            "healthy": SlowLLMBoundCellGenerator(
                config=_expr_config("healthy"),
                resource_provider=provider,
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        max_concurrent_row_groups=2,
        max_in_flight_tasks=16,
        max_model_task_admission=16,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        num_records=2,
        buffer_size=1,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler)
    deferred = Task(column="cooling", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    diagnostics = scheduler._row_group_admission_diagnostics(reason="probe")["deferred_admission"]

    assert scheduler._adaptive_row_group_block_reason() is None
    assert diagnostics["blocks_next_row_group"] is False
    assert diagnostics["request_resources"] == {}
    assert "healthy" in diagnostics["independent_candidate_columns"]


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_adaptive_row_group_admission_keeps_healthy_resource_exposed_during_deferred_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _mock_provider()
    monkeypatch.setattr(async_scheduler_module, "RETRYABLE_RESALVAGE_BACKOFF_S", 0.001)
    healthy_threshold = asyncio.Event()

    class AlwaysCoolingModelGenerator(SlowModelBoundCellGenerator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0

        async def agenerate(self, data: dict) -> dict:
            self.calls += 1
            raise ModelRateLimitError("429 Too Many Requests")

    class CountingHealthyModelGenerator(SlowModelBoundCellGenerator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0

        async def agenerate(self, data: dict) -> dict:
            self.calls += 1
            if self.calls >= 16:
                healthy_threshold.set()
            return await super().agenerate(data)

    cooling = AlwaysCoolingModelGenerator(
        config=_expr_config("cooling"),
        resource_provider=provider,
        provider_name="provider",
        model_id="cooling",
        request_weight=1,
        delay=0.0,
    )
    healthy = CountingHealthyModelGenerator(
        config=_expr_config("healthy"),
        resource_provider=provider,
        provider_name="provider",
        model_id="healthy",
        request_weight=8,
        delay=0.0,
    )
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cooling", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="healthy", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cooling": GenerationStrategy.CELL_BY_CELL,
        "healthy": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(row_group, 1) for row_group in range(64)]
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators={
            "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
            "cooling": cooling,
            "healthy": healthy,
        },
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_concurrent_row_groups=32,
        max_in_flight_tasks=32,
        max_model_task_admission=32,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
        salvage_max_rounds=1,
        num_records=64,
        buffer_size=1,
    )

    run_task = asyncio.create_task(scheduler.run())
    threshold_task = asyncio.create_task(healthy_threshold.wait())
    pending: set[asyncio.Task] = set()
    try:
        done, pending = await asyncio.wait(
            {run_task, threshold_task},
            timeout=5.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if run_task in done:
            await run_task

        assert threshold_task in done
        assert healthy.calls >= 16
        assert cooling.calls >= 1
        assert scheduler._observed_max_row_group_admission_target > 1
        assert scheduler._observed_max_row_groups_in_flight > 1
    finally:
        for pending_task in pending:
            pending_task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass


def test_scheduler_adaptive_row_group_row_guard_blocks_extra_large_groups() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="model_col", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "topic": GenerationStrategy.FULL_COLUMN,
        "model_col": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 5_000), (1, 5_000)]
    graph = ExecutionGraph.create(configs, strategies)
    scheduler = AsyncTaskScheduler(
        generators={
            "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
            "model_col": SlowLLMBoundCellGenerator(
                config=_expr_config("model_col"),
                resource_provider=provider,
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        max_concurrent_row_groups=4,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=4,
        num_records=10_000,
        buffer_size=1,
    )

    scheduler._rg_states[0] = SimpleNamespace(size=5_000)

    assert scheduler._adaptive_max_admitted_rows == 8_192
    assert not scheduler._row_group_row_guard_allows(5_000)
    assert scheduler._row_group_row_guard_allows(1_000)
    scheduler._rg_states.clear()
    assert scheduler._row_group_row_guard_allows(9_000)


def _stub_row_group_admission_resource_views(
    scheduler: AsyncTaskScheduler,
    *,
    queued_total: int,
    queued_llm: int,
    llm_limit: int,
    llm_available: int,
    llm_leased: int,
) -> None:
    scheduler._fair_queue = SimpleNamespace(
        view=lambda: SimpleNamespace(
            queued_total=queued_total,
            queued_by_group={},
            queued_peer_demand_by_resource={"llm_wait": queued_llm} if queued_llm else {},
        )
    )
    scheduler._task_admission = SimpleNamespace(
        view=lambda: SimpleNamespace(
            resource_limits={"llm_wait": llm_limit},
            resources_available={"llm_wait": llm_available},
            leased_resources={"llm_wait": llm_leased} if llm_leased else {},
        )
    )


def test_scheduler_adaptive_row_group_row_guard_precedes_unrelated_deferred_retry() -> None:
    row_groups = [(0, 5_000), (1, 5_000)]
    scheduler, _tracker = _build_adaptive_model_resource_scheduler(row_groups=row_groups)
    scheduler._rg_states[0] = SimpleNamespace(size=5_000, seeds_dispatched=True, pre_batch_done=True, in_flight_count=0)
    _set_next_row_group_pending(scheduler, row_group=1, row_group_size=5_000)
    deferred = Task(column="cooling", row_group=0, row_index=0, task_type="cell")
    scheduler._deferred = [deferred]
    scheduler._deferred_errors[deferred] = ModelRateLimitError("429 Too Many Requests")

    assert scheduler._adaptive_max_admitted_rows == 8_192
    assert scheduler._adaptive_row_group_block_reason() == "max_admitted_rows"


def test_scheduler_adaptive_row_group_block_reason_prefers_llm_saturation() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="model_col", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "topic": GenerationStrategy.FULL_COLUMN,
        "model_col": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1), (1, 1)]
    graph = ExecutionGraph.create(configs, strategies)
    scheduler = AsyncTaskScheduler(
        generators={
            "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
            "model_col": SlowLLMBoundCellGenerator(
                config=_expr_config("model_col"),
                resource_provider=provider,
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        max_concurrent_row_groups=2,
        adaptive_row_group_admission=True,
        num_records=2,
        buffer_size=1,
    )
    _stub_row_group_admission_resource_views(
        scheduler,
        queued_total=1,
        queued_llm=0,
        llm_limit=1,
        llm_available=0,
        llm_leased=1,
    )
    _set_next_row_group_pending(scheduler)

    assert scheduler._adaptive_row_group_block_reason() == "llm_wait_saturated"


def test_scheduler_adaptive_row_group_block_reason_allows_zero_llm_lease_bootstrap() -> None:
    scheduler, _tracker = _build_simple_pipeline(num_records=2, buffer_size=1)
    _stub_row_group_admission_resource_views(
        scheduler,
        queued_total=3,
        queued_llm=3,
        llm_limit=3,
        llm_available=3,
        llm_leased=0,
    )
    _set_next_row_group_pending(scheduler)

    assert scheduler._adaptive_row_group_block_reason() is None


def test_scheduler_adaptive_row_group_block_reason_blocks_queued_llm_demand_after_bootstrap() -> None:
    scheduler, _tracker = _build_simple_pipeline(num_records=2, buffer_size=1)
    _stub_row_group_admission_resource_views(
        scheduler,
        queued_total=3,
        queued_llm=3,
        llm_limit=4,
        llm_available=3,
        llm_leased=1,
    )
    _set_next_row_group_pending(scheduler)

    assert scheduler._adaptive_row_group_block_reason() == "queued_llm_demand"


def test_scheduler_row_group_admission_diagnostics_include_resource_state_for_events() -> None:
    class ExplodingRequestPressureProvider:
        def snapshots(self) -> None:
            raise AssertionError("request pressure diagnostics should not be captured")

        def global_snapshots(self) -> None:
            raise AssertionError("request pressure diagnostics should not be captured")

    scheduler, _tracker = _build_simple_pipeline(
        num_records=2,
        buffer_size=1,
    )
    scheduler._request_pressure_provider = ExplodingRequestPressureProvider()
    _stub_row_group_admission_resource_views(
        scheduler,
        queued_total=3,
        queued_llm=3,
        llm_limit=4,
        llm_available=3,
        llm_leased=1,
    )
    scheduler._in_flight.add(Task(column="cell_out", row_group=0, row_index=0, task_type="cell"))

    diagnostics = scheduler._row_group_admission_diagnostics(reason="queued_llm_demand")

    assert diagnostics["queued_demand_by_resource"] == {"llm_wait": 3}
    assert diagnostics["resource_limits"] == {"llm_wait": 4}
    assert diagnostics["leased_resources"] == {"llm_wait": 1}
    assert diagnostics["resources_available"] == {"llm_wait": 3}
    assert diagnostics["in_flight_tasks"] == 1
    assert "request_pressure" not in diagnostics


def test_scheduler_adaptive_row_group_target_grows_for_zero_llm_lease_bootstrap() -> None:
    scheduler, _tracker = _build_simple_pipeline(
        num_records=2,
        buffer_size=1,
        scheduler_event_sink=(sink := InMemoryAdmissionEventSink()),
        max_concurrent_row_groups=2,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1)
    _stub_row_group_admission_resource_views(
        scheduler,
        queued_total=3,
        queued_llm=3,
        llm_limit=3,
        llm_available=3,
        llm_leased=0,
    )
    _set_next_row_group_pending(scheduler)

    scheduler._maybe_update_adaptive_row_group_target()
    assert scheduler._row_group_admission_target == 1

    scheduler._maybe_update_adaptive_row_group_target()

    assert scheduler._row_group_admission_target == 2
    assert any(event.event_kind == "row_group_admission_target_changed" for event in sink.scheduler_events)


def test_scheduler_adaptive_row_group_target_stays_blocked_after_llm_lease_bootstrap() -> None:
    scheduler, _tracker = _build_simple_pipeline(
        num_records=2,
        buffer_size=1,
        scheduler_event_sink=(sink := InMemoryAdmissionEventSink()),
        max_concurrent_row_groups=2,
        adaptive_row_group_admission=True,
        adaptive_row_group_initial_target=1,
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1)
    _stub_row_group_admission_resource_views(
        scheduler,
        queued_total=3,
        queued_llm=3,
        llm_limit=4,
        llm_available=3,
        llm_leased=1,
    )
    _set_next_row_group_pending(scheduler)

    scheduler._maybe_update_adaptive_row_group_target()
    scheduler._maybe_update_adaptive_row_group_target()

    assert scheduler._row_group_admission_target == 1
    assert not any(event.event_kind == "row_group_admission_target_changed" for event in sink.scheduler_events)
    assert scheduler._row_group_admission_blocked_reasons["queued_llm_demand"] == 2


def test_scheduler_adaptive_row_group_queue_guard_uses_in_flight_task_cap() -> None:
    scheduler, _tracker = _build_simple_pipeline(num_records=2, buffer_size=1)
    scheduler._max_in_flight_tasks = 2
    scheduler._fair_queue = SimpleNamespace(
        view=lambda: SimpleNamespace(queued_total=8, queued_peer_demand_by_resource={})
    )
    _set_next_row_group_pending(scheduler)

    assert scheduler._adaptive_row_group_block_reason() == "queued_task_guardrail"


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_raises_when_ready_frontier_blocked_without_in_flight() -> None:
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="model_col", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "topic": GenerationStrategy.FULL_COLUMN,
        "model_col": GenerationStrategy.CELL_BY_CELL,
    }
    row_groups = [(0, 1)]
    graph = ExecutionGraph.create(configs, strategies)
    scheduler = AsyncTaskScheduler(
        generators={
            "topic": MockSeedGenerator(config=_expr_config("topic"), resource_provider=provider),
            "model_col": SlowLLMBoundCellGenerator(
                config=_expr_config("model_col"),
                resource_provider=provider,
                delay=0.0,
            ),
        },
        graph=graph,
        tracker=CompletionTracker.with_graph(graph, row_groups),
        row_groups=row_groups,
        task_admission_config=TaskAdmissionConfig(
            submission_capacity=1,
            resource_limits={"submission": 1, "local": 1},
        ),
    )

    with pytest.raises(RuntimeError, match="Ready frontier is admission-blocked"):
        await asyncio.wait_for(scheduler.run(), timeout=2.0)


def test_scheduler_request_pressure_advisory_prefers_pressure_open_peer() -> None:
    provider = _mock_provider()
    configs = [
        LLMTextColumnConfig(name="pressured", prompt="A", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="open", prompt="B", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "pressured": GenerationStrategy.CELL_BY_CELL,
        "open": GenerationStrategy.CELL_BY_CELL,
    }
    generators: dict[str, ColumnGenerator] = {
        "pressured": SlowModelBoundCellGenerator(
            config=_expr_config("pressured"),
            resource_provider=provider,
            provider_name="provider-a",
            model_id="model-a",
        ),
        "open": SlowModelBoundCellGenerator(
            config=_expr_config("open"),
            resource_provider=provider,
            provider_name="provider-b",
            model_id="model-b",
        ),
    }
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, [(0, 1)])
    pressured_key = RequestResourceKey("provider-a", "model-a", RequestDomain.CHAT)
    open_key = RequestResourceKey("provider-b", "model-b", RequestDomain.CHAT)
    pressure = _StaticRequestPressureProvider(
        {
            pressured_key: _pressure_snapshot(pressured_key, current_limit=1, in_flight=1, waiters=1),
            open_key: _pressure_snapshot(open_key, current_limit=1, in_flight=0, waiters=0),
        }
    )
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=[(0, 1)],
        request_pressure_provider=pressure,
        request_pressure_advisory=True,
        scheduler_event_sink=(sink := InMemoryAdmissionEventSink()),
    )
    scheduler._rg_states[0] = SimpleNamespace(size=1, pre_batch_done=True, in_flight_count=0)
    pressured = scheduler._schedulable_task(Task(column="pressured", row_group=0, row_index=0, task_type="cell"))
    open_task = scheduler._schedulable_task(Task(column="open", row_group=0, row_index=0, task_type="cell"))
    scheduler._fair_queue.enqueue((pressured, open_task))

    selection = scheduler._fair_queue.select_next(scheduler._is_dispatch_eligible)

    assert selection is not None
    assert selection.item.payload.column == "open"
    skip = next(event for event in sink.scheduler_events if event.event_kind == "request_pressure_advisory_skipped")
    assert skip.diagnostics["request_resource"] == "provider-a/model-a/chat"
    assert skip.diagnostics["pressure_reason"] == "waiters"
    assert skip.diagnostics["open_peer_column"] == "open"
    assert skip.diagnostics["open_peer_request_resource"] == "provider-b/model-b/chat"


def test_scheduler_request_pressure_advisory_preserves_liveness_when_all_candidates_pressured() -> None:
    pressured_key = RequestResourceKey("provider-a", "model-a", RequestDomain.CHAT)
    pressure = _StaticRequestPressureProvider(
        {pressured_key: _pressure_snapshot(pressured_key, current_limit=1, in_flight=1, waiters=1)}
    )
    scheduler = _build_queued_model_pressure_scheduler(
        queued_rows=1,
        request_pressure_provider=pressure,
    )

    selection = scheduler._fair_queue.select_next(scheduler._is_dispatch_eligible)

    assert selection is not None
    assert selection.item.payload.column == "pressured"


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_request_resource_admission_avoids_creating_waiters() -> None:
    provider = _mock_provider()
    sink = InMemoryAdmissionEventSink()
    resource = RequestResourceKey("provider-a", "model-a", RequestDomain.CHAT)
    request_admission = AdaptiveRequestAdmissionController(
        RequestAdmissionConfig(
            initial_limits={resource: 4},
            default_queue_wait_timeout_seconds=0.02,
        ),
        event_sink=sink,
    )
    request_admission.register(
        provider_name="provider-a",
        model_id="model-a",
        alias="primary",
        max_parallel_requests=4,
    )
    config = LLMTextColumnConfig(name="pressured", prompt="A", model_alias=MODEL_ALIAS)
    generator = GatedRequestAdmissionCellGenerator(
        config=_expr_config("pressured"),
        resource_provider=provider,
        request_admission=request_admission,
        hold_until_active=3,
        provider_name="provider-a",
        model_id="model-a",
        request_weight=4,
    )
    graph = ExecutionGraph.create([config], {"pressured": GenerationStrategy.CELL_BY_CELL})
    row_groups = [(0, 6)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    scheduler = AsyncTaskScheduler(
        generators={"pressured": generator},
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_in_flight_tasks=16,
        max_model_task_admission=16,
        request_pressure_provider=request_admission,
        request_pressure_advisory=True,
        scheduler_event_sink=sink,
    )

    run_task = asyncio.create_task(scheduler.run())
    try:
        await asyncio.wait_for(generator.initial_leases_acquired.wait(), timeout=5.0)
        for _ in range(5):
            await asyncio.sleep(0)

        waiters = sum(snapshot.waiters for snapshot in request_admission.pressure.snapshots().values())
        assert waiters == 0
        assert len(scheduler._in_flight) == 3

        generator.release_held_leases.set()
        await asyncio.wait_for(run_task, timeout=5.0)
    finally:
        if not run_task.done():
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

    assert tracker.is_row_group_complete(0, 6, ["pressured"])
    assert not any(tracker.is_dropped(0, row_index) for row_index in range(6))
    assert not any(event.event_kind == "request_wait_timeout" for event in sink.request_events)


# -- Skip / conditional generation tests (async engine) -----------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_skip_cell_by_cell_with_propagation() -> None:
    """Cell-by-cell column skips rows via expression gate, downstream propagates.

    Pipeline: seed(sampler) -> review(cell, skip.when seed<2) -> complaint(cell, propagate_skip)
    Rows with seed < 2 should be skipped for review and propagated to complaint.
    """
    provider = _mock_provider()
    num_records = 4

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(
            name="review",
            prompt="{{ seed }}",
            model_alias=MODEL_ALIAS,
            skip=SkipConfig(when="{{ seed < 2 }}"),
        ),
        LLMTextColumnConfig(
            name="complaint",
            prompt="{{ review }}",
            model_alias=MODEL_ALIAS,
            propagate_skip=True,
        ),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "review": GenerationStrategy.CELL_BY_CELL,
        "complaint": GenerationStrategy.CELL_BY_CELL,
    }

    class IntSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.FULL_COLUMN

        def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
            return data

        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            return lazy.pd.DataFrame({"seed": list(range(num_records))})

    generators: dict[str, ColumnGenerator] = {
        "seed": IntSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "review": MockCellGenerator(config=_expr_config("review"), resource_provider=provider),
        "complaint": MockCellGenerator(config=_expr_config("complaint"), resource_provider=provider),
    }

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    buffer_mgr = RowGroupBufferManager(storage)

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, num_records)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        trace=True,
        num_records=num_records,
        buffer_size=num_records,
    )
    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    assert tracker.is_row_group_complete(0, num_records, ["seed", "review", "complaint"])

    for ri in range(num_records):
        row = buffer_mgr.get_row(0, ri)
        seed_val = row["seed"]
        if seed_val < 2:
            assert row.get("review") is None, f"row {ri}: review should be skipped (seed={seed_val})"
            assert row.get("complaint") is None, f"row {ri}: complaint should propagate skip (seed={seed_val})"
        else:
            assert row.get("review") is not None, f"row {ri}: review should be generated (seed={seed_val})"
            assert row.get("complaint") is not None, f"row {ri}: complaint should be generated (seed={seed_val})"


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_skip_propagates_through_side_effect_dependency() -> None:
    """A downstream dependency on a skipped side-effect should auto-skip.

    Pipeline: seed(sampler) -> review(cell, skip.when seed<2, produces
    review__trace) -> complaint(cell, depends on review__trace,
    propagate_skip=True).
    """
    provider = _mock_provider()
    num_records = 4

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(
            name="review",
            prompt="{{ seed }}",
            model_alias=MODEL_ALIAS,
            with_trace="last_message",
            skip=SkipConfig(when="{{ seed < 2 }}"),
        ),
        LLMTextColumnConfig(
            name="complaint",
            prompt="{{ review__trace }}",
            model_alias=MODEL_ALIAS,
            propagate_skip=True,
        ),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "review": GenerationStrategy.CELL_BY_CELL,
        "complaint": GenerationStrategy.CELL_BY_CELL,
    }

    class IntSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.FULL_COLUMN

        def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
            return data

        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            return lazy.pd.DataFrame({"seed": list(range(num_records))})

    generators: dict[str, ColumnGenerator] = {
        "seed": IntSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "review": MockCellGenerator(config=_expr_config("review"), resource_provider=provider),
        "complaint": MockCellGenerator(config=_expr_config("complaint"), resource_provider=provider),
    }

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    buffer_mgr = RowGroupBufferManager(storage)

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, num_records)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        trace=True,
        num_records=num_records,
        buffer_size=num_records,
    )
    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    assert tracker.is_row_group_complete(0, num_records, ["seed", "review", "complaint"])

    for ri in range(num_records):
        row = buffer_mgr.get_row(0, ri)
        seed_val = row["seed"]
        if seed_val < 2:
            assert row.get("review") is None, f"row {ri}: review should be skipped (seed={seed_val})"
            assert row.get("review__trace") is None, f"row {ri}: review__trace should be cleared on skip"
            assert row.get("complaint") is None, f"row {ri}: complaint should propagate skip (seed={seed_val})"
        else:
            assert row.get("complaint") is not None, f"row {ri}: complaint should be generated (seed={seed_val})"


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_skip_full_column_batch() -> None:
    """Full-column (batch) generator skips rows via expression gate.

    Pipeline: seed(sampler) -> review(full_column, skip.when seed<2)
    Only active (non-skipped) rows should be passed to the generator.
    """
    provider = _mock_provider()
    num_records = 4

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(
            name="review",
            prompt="{{ seed }}",
            model_alias=MODEL_ALIAS,
            skip=SkipConfig(when="{{ seed < 2 }}"),
        ),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "review": GenerationStrategy.FULL_COLUMN,
    }

    class IntSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.FULL_COLUMN

        def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
            return data

        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            return lazy.pd.DataFrame({"seed": list(range(num_records))})

    generators: dict[str, ColumnGenerator] = {
        "seed": IntSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "review": MockFullColumnGenerator(config=_expr_config("review"), resource_provider=provider),
    }

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    buffer_mgr = RowGroupBufferManager(storage)

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, num_records)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        trace=True,
        num_records=num_records,
        buffer_size=num_records,
    )
    await asyncio.wait_for(scheduler.run(), timeout=10.0)

    assert tracker.is_row_group_complete(0, num_records, ["seed", "review"])

    for ri in range(num_records):
        row = buffer_mgr.get_row(0, ri)
        seed_val = row["seed"]
        if seed_val < 2:
            assert row.get("review") is None, f"row {ri}: review should be skipped (seed={seed_val})"
        else:
            assert row["review"] == "batch_val", f"row {ri}: review should be generated (seed={seed_val})"


# -- Post-batch (on_before_checkpoint) failure propagation --------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_post_batch_failure_raises() -> None:
    """Post-batch processor failure propagates as DatasetGenerationError."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "cell_out": MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    buffer_mgr = RowGroupBufferManager(storage)

    def fail_post_batch(rg_id: int, rg_size: int) -> None:
        raise RuntimeError("post-batch processor exploded")

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_before_checkpoint=fail_post_batch,
    )
    with pytest.raises(DatasetGenerationError, match="Post-batch processor failed"):
        await scheduler.run()


# -- Early shutdown drains workers -------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_early_shutdown_drains_workers() -> None:
    """Workers are cancelled after early shutdown, not left dangling."""
    provider = _mock_provider()
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="fail_col", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "fail_col": GenerationStrategy.CELL_BY_CELL,
    }
    generators = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "fail_col": MockFailingGenerator(config=_expr_config("fail_col"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 5)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        shutdown_error_rate=0.5,
        shutdown_error_window=5,
        num_records=5,
        buffer_size=5,
    )
    await scheduler.run()

    # After run() returns, no worker tasks should remain.
    assert scheduler.active_worker_count == 0
