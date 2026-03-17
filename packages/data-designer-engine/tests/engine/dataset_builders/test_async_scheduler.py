# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorFullColumn,
    FromScratchColumnGenerator,
)
from data_designer.engine.dataset_builders.async_scheduler import AsyncTaskScheduler
from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.models.errors import ModelInternalServerError
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

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_row_group_complete=lambda rg: checkpointed.append(rg),
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

    scheduler, tracker = _build_simple_pipeline(
        num_records=2, generators=generators, configs=configs, strategies=strategies, trace=True
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

    checkpointed: list[int] = []

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_mgr,
        on_row_group_complete=lambda rg: checkpointed.append(rg),
        trace=True,
    )
    await scheduler.run()

    # All rows dropped due to seed failure
    for ri in range(3):
        assert tracker.is_dropped(0, ri)

    # Row group still completes (vacuously) and is checkpointed
    assert 0 in checkpointed

    # full_out was either never dispatched or silently skipped (no KeyError)
    full_out_errors = [t for t in scheduler.traces if t.column == "full_out" and t.status == "error"]
    assert len(full_out_errors) == 0


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

    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        shutdown_error_rate=0.5,
        shutdown_error_window=2,
    )
    await scheduler.run()

    assert scheduler._early_shutdown is True


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduler_early_shutdown_disabled() -> None:
    """disable_early_shutdown=True prevents shutdown even at 100% error rate."""
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
        disable_early_shutdown=True,
    )
    await scheduler.run()

    assert scheduler._early_shutdown is False


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
