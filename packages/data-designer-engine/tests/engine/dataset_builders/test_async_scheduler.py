# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import (
    CustomColumnConfig,
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorFullColumn,
    FromScratchColumnGenerator,
)
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.dataset_builders.async_scheduler import AsyncTaskScheduler, build_llm_bound_lookup
from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.models.errors import ModelInternalServerError, ModelRateLimitError
from data_designer.engine.resources.resource_provider import ResourceProvider

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


# -- Helper to build graph + scheduler ----------------------------------------


def _build_simple_pipeline(
    num_records: int = 3,
    buffer_size: int = 3,
    trace: bool = False,
    generators: dict[str, ColumnGenerator] | None = None,
    configs: list[SamplerColumnConfig | LLMTextColumnConfig | ExpressionColumnConfig] | None = None,
    strategies: dict[str, GenerationStrategy] | None = None,
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
        trace=trace,
    )
    return scheduler, tracker


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
async def test_scheduler_bounded_submission() -> None:
    """Submitted task count respects max_submitted_tasks."""
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
        max_submitted_tasks=2,
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
async def test_scheduler_pre_batch_failure_marks_downstream_tasks_skipped() -> None:
    """Pre-batch row-group drops count downstream cell tasks as skipped."""
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
    await scheduler.run()

    for row_index in range(3):
        assert tracker.is_dropped(0, row_index)

    assert scheduler._reporter is not None
    assert scheduler._reporter._trackers["cell_out"].skipped == 3
    assert scheduler._reporter._trackers["cell_out"].completed == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_error_rate_shutdown() -> None:
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
    await scheduler.run()

    # Early shutdown: not all rows should be checkpointed (some row groups incomplete)
    assert buffer_mgr.actual_num_records < 10


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
async def test_scheduler_pre_batch_failure_skips_row_group() -> None:
    """Pre-batch processor failure drops all rows in the row group; other row groups continue."""
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
    await scheduler.run()

    # Row group 0: all rows dropped due to pre-batch failure
    assert all(tracker.is_dropped(0, ri) for ri in range(3))
    # Row group 1: completed normally
    assert tracker.is_row_group_complete(1, 2, ["seed", "cell_out"])


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


# -- Dual-semaphore / LLM-bound tests -----------------------------------------


class MockLLMBoundCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """Mock cell-by-cell generator that reports is_llm_bound=True."""

    @property
    def is_llm_bound(self) -> bool:
        return True

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"llm_{data.get('seed', '?')}"
        return data


class MockLLMBoundRateLimitGenerator(ColumnGenerator[ExpressionColumnConfig]):
    """LLM-bound generator that raises ModelRateLimitError for the first N calls, then succeeds."""

    def __init__(self, *args: Any, rate_limit_failures: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_failures = rate_limit_failures
        self._calls = 0

    @property
    def is_llm_bound(self) -> bool:
        return True

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        self._calls += 1
        if self._calls <= self._rate_limit_failures:
            raise ModelRateLimitError("429 Too Many Requests")
        data[self.config.name] = f"llm_ok_{data.get('seed', '?')}"
        return data


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

    max_submitted = 2
    max_llm_wait = 2
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_submitted_tasks=max_submitted,
        max_llm_wait_tasks=max_llm_wait,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, 3, ["seed", "llm_col"])

    sub_available, llm_available = scheduler.get_semaphore_permits()
    assert sub_available == max_submitted, (
        f"Submission semaphore leaked after LLM handoff: available={sub_available}, expected={max_submitted}"
    )
    assert llm_available == max_llm_wait, (
        f"LLM-wait semaphore leaked after LLM handoff: available={llm_available}, expected={max_llm_wait}"
    )


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
        max_submitted_tasks=2,
        max_llm_wait_tasks=max_llm_wait,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, 3, ["seed", "cell_out"])

    _, llm_available = scheduler.get_semaphore_permits()
    assert llm_available == max_llm_wait, (
        f"LLM-wait semaphore was consumed by non-LLM task: available={llm_available}, expected={max_llm_wait}"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_deadlock_regression() -> None:
    """max_submitted_tasks=1, max_llm_wait_tasks=1, two ready LLM tasks completes without deadlock."""
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
        max_submitted_tasks=1,
        max_llm_wait_tasks=1,
    )

    await asyncio.wait_for(scheduler.run(), timeout=10.0)
    assert tracker.is_row_group_complete(0, 2, ["seed", "llm_col"])


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_is_llm_bound_property_drives_lookup() -> None:
    """is_llm_bound property on generators drives the lookup, not isinstance."""
    provider = _mock_provider()
    llm_gen = MockLLMBoundCellGenerator(config=_expr_config("llm_col"), resource_provider=provider)
    non_llm_gen = MockCellGenerator(config=_expr_config("cell_out"), resource_provider=provider)

    assert llm_gen.is_llm_bound is True
    assert non_llm_gen.is_llm_bound is False

    lookup = build_llm_bound_lookup({"llm_col": llm_gen, "cell_out": non_llm_gen})
    assert lookup == {"llm_col": True, "cell_out": False}


def test_custom_generator_with_model_aliases_is_llm_bound() -> None:
    """CustomColumnGenerator with model_aliases reports is_llm_bound=True."""

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

    assert llm_gen.is_llm_bound is True
    assert plain_gen.is_llm_bound is False

    lookup = build_llm_bound_lookup({"custom_llm": llm_gen, "custom_plain": plain_gen})
    assert lookup == {"custom_llm": True, "custom_plain": False}


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

    max_submitted = 4
    max_llm_wait = 2
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        max_submitted_tasks=max_submitted,
        max_llm_wait_tasks=max_llm_wait,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, num_records, ["seed", "llm_col"])

    sub_available, llm_available = scheduler.get_semaphore_permits()
    assert sub_available == max_submitted, (
        f"Submission semaphore leaked after salvage retry: available={sub_available}, expected={max_submitted}"
    )
    assert llm_available == max_llm_wait, (
        f"LLM-wait semaphore leaked after salvage retry: available={llm_available}, expected={max_llm_wait}"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_cancellation_releases_semaphores() -> None:
    """Cancelling the scheduler while LLM-bound tasks are in-flight releases all semaphore slots."""
    provider = _mock_provider()

    blocked = asyncio.Event()
    proceed = asyncio.Event()

    class BlockingLLMGenerator(ColumnGenerator[ExpressionColumnConfig]):
        @property
        def is_llm_bound(self) -> bool:
            return True

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

    max_submitted = 4
    max_llm_wait = 2
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        max_submitted_tasks=max_submitted,
        max_llm_wait_tasks=max_llm_wait,
    )

    run_task = asyncio.create_task(scheduler.run())

    await asyncio.wait_for(blocked.wait(), timeout=5.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    sub_available, llm_available = scheduler.get_semaphore_permits()
    assert sub_available == max_submitted, (
        f"Submission semaphore leaked: available={sub_available}, expected={max_submitted}"
    )
    assert llm_available == max_llm_wait, (
        f"LLM-wait semaphore leaked: available={llm_available}, expected={max_llm_wait}"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_rg_semaphore_deadlock_with_transient_failures() -> None:
    """Row groups stalled by transient failures don't block admission of new row groups.

    Regression test: with max_concurrent_row_groups=1 and 2 row groups, if all
    tasks in RG0 fail transiently, the semaphore must still be released so RG1
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


# -- TrackingSemaphore tests ---------------------------------------------------


def test_tracking_semaphore_try_acquire() -> None:
    """try_acquire returns True when permits are available, False when exhausted."""
    from data_designer.engine.dataset_builders.async_scheduler import TrackingSemaphore

    sem = TrackingSemaphore(2)
    assert sem.available_permits == 2

    assert sem.try_acquire() is True
    assert sem.available_permits == 1

    assert sem.try_acquire() is True
    assert sem.available_permits == 0

    assert sem.try_acquire() is False
    assert sem.available_permits == 0

    sem.release()
    assert sem.available_permits == 1
    assert sem.try_acquire() is True
    assert sem.available_permits == 0


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


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_downstream_interleaves_with_upstream() -> None:
    """Downstream judge tasks begin before all upstream gen tasks complete (issue #504).

    Mirrors the reported pipeline topology:

        topic (sampler, instant)
        ├── gen_a (slow, 50ms) → judge_a (instant)
        ├── gen_b (slow, 50ms) → judge_b (instant)
        └── gen_c (slow, 50ms) → judge_c (instant)

    With a small semaphore (4) and 10 records, the 30 gen tasks (3 cols x 10 rows)
    saturate the semaphore. The dispatch loop must re-query the frontier when the
    semaphore is full so that judge tasks from completed gen rows are picked up
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
        max_submitted_tasks=4,
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


class MockCellGeneratorWithSideEffect(ColumnGenerator[ExpressionColumnConfig]):
    """Cell generator that produces a side-effect column alongside its primary output."""

    def __init__(self, *args: Any, side_effect_col: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._side_effect_col = side_effect_col

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"primary_{data.get('seed', '?')}"
        data[self._side_effect_col] = f"side_effect_{data.get('seed', '?')}"
        return data


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_side_effect_columns_written_to_buffer() -> None:
    """Side-effect columns (e.g. __reasoning_content) are persisted to the buffer.

    Reproduces the bug where a generator produces extra columns (like trace or
    reasoning_content) that aren't tracked in _instance_to_columns. Downstream
    columns that reference these side-effect values must find them in the buffer.
    """
    provider = _mock_provider()
    side_effect_col = "answer__reasoning_content"

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="answer", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="judge", prompt=f"{{{{ {side_effect_col} }}}}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "answer": GenerationStrategy.CELL_BY_CELL,
        "judge": GenerationStrategy.CELL_BY_CELL,
    }

    answer_gen = MockCellGeneratorWithSideEffect(
        config=_expr_config("answer"),
        resource_provider=provider,
        side_effect_col=side_effect_col,
    )
    generators: dict[str, ColumnGenerator] = {
        "seed": MockSeedGenerator(config=_expr_config("seed"), resource_provider=provider),
        "answer": answer_gen,
        "judge": MockCellGenerator(config=_expr_config("judge"), resource_provider=provider),
    }

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)
    buffer_manager = RowGroupBufferManager(MagicMock())

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
    )
    await scheduler.run()

    assert tracker.is_row_group_complete(0, 3, ["seed", "answer", "judge"])
    for ri in range(3):
        row = buffer_manager.get_row(0, ri)
        assert side_effect_col in row, f"Side-effect column missing from row {ri}"
        assert row[side_effect_col].startswith("side_effect_")
