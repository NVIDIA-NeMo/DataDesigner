# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import tracemalloc
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import data_designer.engine.dataset_builders.dataset_builder as builder_mod
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import SkipConfig
from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorFullColumn,
    FromScratchColumnGenerator,
)
from data_designer.engine.column_generators.generators.expression import ExpressionColumnGenerator
from data_designer.engine.context import current_row_group
from data_designer.engine.dataset_builders.async_scheduler import AsyncTaskScheduler
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.row_group_plan import CompactRowGroupPlan
from data_designer.engine.dataset_builders.scheduling.completion import CompletionTracker, FrontierDelta
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.resources.resource_provider import ResourceProvider

MODEL_ALIAS = "stub"


# -- Mock generators for integration tests ------------------------------------


def _expr_config(name: str = "test") -> ExpressionColumnConfig:
    return ExpressionColumnConfig(name=name, expr="{{ x }}", dtype="str")


def _mock_provider() -> MagicMock:
    return MagicMock(spec=ResourceProvider)


class MockSeed(FromScratchColumnGenerator[ExpressionColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        return lazy.pd.DataFrame({"seed": list(range(num_records))})


class MockCell(ColumnGenerator[ExpressionColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data["cell_out"] = f"val_{data.get('seed', '?')}"
        return data


class MockFullCol(ColumnGeneratorFullColumn[ExpressionColumnConfig]):
    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        data["expr_out"] = "computed"
        return data


# -- _build_async integration test with mock generators -----------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_build_async_end_to_end() -> None:
    """Test _build_async with mock generators produces correct results in buffer."""

    provider = _mock_provider()
    seed_gen = MockSeed(config=_expr_config("seed"), resource_provider=provider)
    cell_gen = MockCell(config=_expr_config("cell_out"), resource_provider=provider)
    expr_gen = MockFullCol(config=_expr_config("expr_out"), resource_provider=provider)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="cell_out", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="expr_out", expr="{{ cell_out }}"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "cell_out": GenerationStrategy.CELL_BY_CELL,
        "expr_out": GenerationStrategy.FULL_COLUMN,
    }
    gen_map = {
        "seed": seed_gen,
        "cell_out": cell_gen,
        "expr_out": expr_gen,
    }

    num_records = 4
    buffer_size = 2
    graph = ExecutionGraph.create(configs, strategies)

    row_groups: list[tuple[int, int]] = []
    remaining = num_records
    rg_id = 0
    while remaining > 0:
        size = min(buffer_size, remaining)
        row_groups.append((rg_id, size))
        remaining -= size
        rg_id += 1

    tracker = CompletionTracker.with_graph(graph, row_groups)
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    buffer_manager = RowGroupBufferManager(storage)

    finalized: list[int] = []

    def finalize_row_group(rg_id: int) -> None:
        buffer_manager.checkpoint_row_group(rg_id)
        finalized.append(rg_id)

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        on_finalize_row_group=finalize_row_group,
    )
    await scheduler.run()

    # Both row groups should be finalized
    assert sorted(finalized) == [0, 1]
    assert buffer_manager.actual_num_records == 4

    # All columns should be complete
    all_cols = ["seed", "cell_out", "expr_out"]
    assert tracker.is_row_group_complete(0, 2, all_cols)
    assert tracker.is_row_group_complete(1, 2, all_cols)


@pytest.mark.asyncio(loop_scope="session")
async def test_expression_row_drops_shrink_async_row_group(caplog: pytest.LogCaptureFixture) -> None:
    """Expression row drops are applied to the exact row indexes in the async scheduler."""

    class SeedWithEmpty(MockSeed):
        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            values = ["keep", "", "also"]
            return lazy.pd.DataFrame({"seed": values[:num_records]})

    provider = _mock_provider()
    provider.run_config = RunConfig()
    seed_gen = SeedWithEmpty(config=_expr_config("seed"), resource_provider=provider)
    expr_config = ExpressionColumnConfig(name="copy", expr="{{ seed }}")
    expr_gen = ExpressionColumnGenerator(config=expr_config, resource_provider=provider)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        expr_config,
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN, "copy": GenerationStrategy.FULL_COLUMN}
    gen_map = {"seed": seed_gen, "copy": expr_gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    written_batches: list[lazy.pd.DataFrame] = []
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    def capture_batch(**kwargs: object) -> None:
        written_batches.append(kwargs["dataframe"].copy())

    storage.write_batch_to_parquet_file.side_effect = capture_batch
    buffer_manager = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        on_finalize_row_group=lambda rg_id: buffer_manager.checkpoint_row_group(rg_id),
    )
    with caplog.at_level("WARNING"):
        await scheduler.run()

    assert buffer_manager.actual_num_records == 2
    assert tracker.is_dropped(0, 1)
    assert tracker.is_row_group_complete(0, 3, ["seed", "copy"])
    assert len(written_batches) == 1
    assert written_batches[0]["seed"].tolist() == ["keep", "also"]
    assert written_batches[0]["copy"].tolist() == ["keep", "also"]
    assert "Expression column 'copy' dropped 1/3 rows after render: EmptyRenderedExpression=1." in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_expression_all_dropped_async_row_group_fails_loudly() -> None:
    """All-dropped expression batches abort async generation instead of salvaging the row group."""

    class SeedWithAllDroppedFirstGroup(MockSeed):
        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            rg = current_row_group.get()
            values = ["", ""] if rg is not None and rg[0] == 0 else ["keep", "also"]
            return lazy.pd.DataFrame({"seed": values[:num_records]})

    provider = _mock_provider()
    provider.run_config = RunConfig()
    seed_gen = SeedWithAllDroppedFirstGroup(config=_expr_config("seed"), resource_provider=provider)
    expr_config = ExpressionColumnConfig(name="copy", expr="{{ seed }}")
    expr_gen = ExpressionColumnGenerator(config=expr_config, resource_provider=provider)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        expr_config,
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN, "copy": GenerationStrategy.FULL_COLUMN}
    gen_map = {"seed": seed_gen, "copy": expr_gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2), (1, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    written_batches: list[lazy.pd.DataFrame] = []
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    storage.write_batch_to_parquet_file.side_effect = lambda **kwargs: written_batches.append(
        kwargs["dataframe"].copy()
    )
    buffer_manager = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        max_concurrent_row_groups=1,
        on_finalize_row_group=lambda rg_id: buffer_manager.checkpoint_row_group(rg_id),
    )

    with pytest.raises(DatasetGenerationError, match="Expression column 'copy' produced no valid rows."):
        await scheduler.run()

    assert written_batches == []
    assert buffer_manager.actual_num_records == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_root_expression_all_dropped_async_fails_loudly() -> None:
    """Root expression columns (scheduled as ``from_scratch``) raise fatal when all rows drop.

    Regression for PR #757: before, the ``_task_supports_row_drops`` check only
    fired on ``batch`` tasks, so a root expression like ``{{ "" }}`` (no upstream
    refs, scheduled as ``from_scratch``) had its ``UserTemplateError`` wrapped as
    a generic ``DatasetGenerationError`` and the whole row group was silently
    dropped instead of aborting the run.
    """

    provider = _mock_provider()
    provider.run_config = RunConfig()
    expr_config = ExpressionColumnConfig(name="empty_root", expr='{{ "" }}')
    expr_gen = ExpressionColumnGenerator(config=expr_config, resource_provider=provider)

    configs = [expr_config]
    strategies = {"empty_root": GenerationStrategy.FULL_COLUMN}
    gen_map: dict[str, ColumnGenerator] = {"empty_root": expr_gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 2), (1, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    written_batches: list[lazy.pd.DataFrame] = []
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    storage.write_batch_to_parquet_file.side_effect = lambda **kwargs: written_batches.append(
        kwargs["dataframe"].copy()
    )
    buffer_manager = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        max_concurrent_row_groups=1,
        on_finalize_row_group=lambda rg_id: buffer_manager.checkpoint_row_group(rg_id),
    )

    with pytest.raises(DatasetGenerationError, match="Expression column 'empty_root' produced no valid rows."):
        await scheduler.run()

    assert written_batches == []
    assert buffer_manager.actual_num_records == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_expression_row_drops_with_skip_async_row_group(caplog: pytest.LogCaptureFixture) -> None:
    """Skip metadata + expression row drops both round-trip correctly through async batches.

    Mirrors the sync ``test_expression_column_row_drops_shrink_sync_skip_aware_batch``
    coverage: when a row is pre-skipped (via ``SkipConfig``) and another row is
    dropped by the expression generator (empty render), the ``active_row_indices``
    mapping in ``_run_batch`` must correctly write the skip value for the
    skipped row, drop the empty row, and preserve the kept row.
    """

    class SeedWithSkipAndEmpty(MockSeed):
        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            values = ["skip-me", "", "keep"]
            return lazy.pd.DataFrame({"seed": values[:num_records]})

    provider = _mock_provider()
    provider.run_config = RunConfig()
    seed_gen = SeedWithSkipAndEmpty(config=_expr_config("seed"), resource_provider=provider)
    expr_config = ExpressionColumnConfig(
        name="copy",
        expr="{{ seed }}",
        skip=SkipConfig(when="{{ seed == 'skip-me' }}", value="SKIPPED"),
    )
    expr_gen = ExpressionColumnGenerator(config=expr_config, resource_provider=provider)

    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        expr_config,
    ]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN, "copy": GenerationStrategy.FULL_COLUMN}
    gen_map = {"seed": seed_gen, "copy": expr_gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    written_batches: list[lazy.pd.DataFrame] = []
    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"
    storage.write_batch_to_parquet_file.side_effect = lambda **kwargs: written_batches.append(
        kwargs["dataframe"].copy()
    )
    buffer_manager = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        on_finalize_row_group=lambda rg_id: buffer_manager.checkpoint_row_group(rg_id),
    )
    with caplog.at_level("WARNING"):
        await scheduler.run()

    # row 0: skipped → copy="SKIPPED", row 1: empty render → dropped, row 2: kept
    assert tracker.is_dropped(0, 1)
    assert buffer_manager.actual_num_records == 2
    assert len(written_batches) == 1
    assert written_batches[0]["seed"].tolist() == ["skip-me", "keep"]
    assert written_batches[0]["copy"].tolist() == ["SKIPPED", "keep"]
    # Log shows ``1/2`` because the skipped row is filtered out before the
    # expression generator runs — only the 2 active rows reach ``generate``.
    assert "Expression column 'copy' dropped 1/2 rows after render: EmptyRenderedExpression=1." in caplog.text


def test_prepare_async_run_enables_request_pressure_advisory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class _SpyScheduler:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(builder_mod, "AsyncTaskScheduler", _SpyScheduler)
    request_admission = object()
    model_registry = MagicMock()
    model_registry.get_aggregate_max_parallel_requests.side_effect = AssertionError(
        "model task admission should follow max_in_flight_tasks directly"
    )
    model_registry.request_admission = request_admission
    provider = SimpleNamespace(
        model_registry=model_registry,
        run_config=SimpleNamespace(
            max_concurrent_row_groups=8,
            max_in_flight_tasks=64,
            progress_interval=5.0,
            display_tui=False,
        ),
    )
    processor_runner = MagicMock()
    processor_runner.has_processors_for.return_value = False
    config = SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})
    builder = SimpleNamespace(
        _column_configs=[config],
        _processor_runner=processor_runner,
        artifact_storage=MagicMock(),
        _resource_provider=provider,
    )
    generator = MockSeed(config=_expr_config("seed"), resource_provider=provider)

    DatasetBuilder._prepare_async_run(builder, [generator], num_records=1, buffer_size=1)

    assert captured_kwargs["request_pressure_provider"] is request_admission
    assert captured_kwargs["request_pressure_advisory"] is True
    assert captured_kwargs["max_concurrent_row_groups"] == 8
    assert captured_kwargs["max_in_flight_tasks"] == 64
    assert captured_kwargs["max_model_task_admission"] == 64


def test_prepare_async_run_uses_compact_plan_for_large_fresh_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class _SpyScheduler:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(builder_mod, "AsyncTaskScheduler", _SpyScheduler)
    model_registry = MagicMock()
    model_registry.request_admission = None
    provider = SimpleNamespace(
        model_registry=model_registry,
        run_config=SimpleNamespace(
            max_concurrent_row_groups=3,
            max_in_flight_tasks=64,
            progress_interval=5.0,
            display_tui=False,
        ),
    )
    processor_runner = MagicMock()
    processor_runner.has_processors_for.return_value = False
    config = SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})
    builder = SimpleNamespace(
        _column_configs=[config],
        _processor_runner=processor_runner,
        artifact_storage=MagicMock(),
        _resource_provider=provider,
    )
    generator = MockSeed(config=_expr_config("seed"), resource_provider=provider)

    tracemalloc.start()
    try:
        DatasetBuilder._prepare_async_run(builder, [generator], num_records=2_000_000, buffer_size=2)
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    row_groups = captured_kwargs["row_groups"]
    assert isinstance(row_groups, CompactRowGroupPlan)
    assert len(row_groups) == 1_000_000
    assert peak_bytes < 5 * 1024 * 1024


# -- Test execution graph integration with real column configs -----------------


def test_execution_graph_from_real_configs() -> None:
    """Build execution graph from real column config objects."""
    configs = [
        SamplerColumnConfig(name="id", sampler_type=SamplerType.UUID, params={}),
        LLMTextColumnConfig(name="question", prompt="{{ id }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="answer", prompt="{{ question }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="combined", expr="{{ question }} {{ answer }}"),
    ]
    strategies = {
        "id": GenerationStrategy.FULL_COLUMN,
        "question": GenerationStrategy.CELL_BY_CELL,
        "answer": GenerationStrategy.CELL_BY_CELL,
        "combined": GenerationStrategy.FULL_COLUMN,
    }

    graph = ExecutionGraph.create(configs, strategies)

    order = graph.get_topological_order()
    idx = {col: i for i, col in enumerate(order)}

    assert idx["id"] < idx["question"]
    assert idx["question"] < idx["answer"]
    assert idx["answer"] < idx["combined"]

    # Task counts
    counts = graph.compute_task_count(num_records=10, buffer_size=3)
    assert counts["id"] == math.ceil(10 / 3)
    assert counts["question"] == 10
    assert counts["answer"] == 10
    assert counts["combined"] == math.ceil(10 / 3)


# -- Test checkpoint correctness -----------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_checkpoint_produces_correct_parquet_calls() -> None:
    """Verify checkpoint writes parquet for each row group."""

    provider = _mock_provider()
    seed_gen = MockSeed(config=_expr_config("seed"), resource_provider=provider)

    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["X"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    gen_map = {"seed": seed_gen}

    graph = ExecutionGraph.create(configs, strategies)
    row_groups = [(0, 3), (1, 2)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    buffer_manager = RowGroupBufferManager(storage)

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        on_finalize_row_group=lambda rg_id: buffer_manager.checkpoint_row_group(rg_id),
    )
    await scheduler.run()

    # Two row groups → two write_batch_to_parquet_file calls
    assert storage.write_batch_to_parquet_file.call_count == 2
    assert storage.move_partial_result_to_final_file_path.call_count == 2
    assert buffer_manager.actual_num_records == 5


# -- Partial completion warning ------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_dropped_rows_reduce_actual_record_count() -> None:
    """When all rows in a row group are dropped, actual_num_records reflects the shortfall
    and write_metadata records the correct actual vs target counts."""
    provider = _mock_provider()
    seed_gen = MockSeed(config=_expr_config("seed"), resource_provider=provider)

    configs = [SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["X"]})]
    strategies = {"seed": GenerationStrategy.FULL_COLUMN}
    gen_map = {"seed": seed_gen}

    graph = ExecutionGraph.create(configs, strategies)
    num_records = 6
    row_groups = [(0, 3), (1, 3)]
    tracker = CompletionTracker.with_graph(graph, row_groups)

    storage = MagicMock()
    storage.dataset_name = "test"
    storage.get_file_paths.return_value = {}
    storage.write_batch_to_parquet_file.return_value = "/fake.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake_final.parquet"

    buffer_manager = RowGroupBufferManager(storage)

    def drop_all_in_rg1(rg_id: int, rg_size: int) -> FrontierDelta:
        deltas: list[FrontierDelta] = []
        if rg_id == 1:
            for ri in range(rg_size):
                deltas.append(tracker.drop_row(rg_id, ri))
                buffer_manager.drop_row(rg_id, ri)
        return FrontierDelta(
            added=tuple(task for delta in deltas for task in delta.added),
            removed=tuple(task for delta in deltas for task in delta.removed),
        )

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        on_finalize_row_group=lambda rg_id: buffer_manager.checkpoint_row_group(rg_id),
        on_seeds_complete=drop_all_in_rg1,
    )
    await scheduler.run()

    assert buffer_manager.actual_num_records < num_records

    buffer_manager.write_metadata(target_num_records=num_records, buffer_size=3)
    written = storage.write_metadata.call_args[0][0]
    assert written["actual_num_records"] == buffer_manager.actual_num_records
    assert written["target_num_records"] == num_records
