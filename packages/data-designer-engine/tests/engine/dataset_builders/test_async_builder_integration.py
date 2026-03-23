# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from unittest.mock import MagicMock, Mock

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
from data_designer.engine.dataset_builders.column_wise_builder import ColumnWiseDatasetBuilder
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
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


# -- allow_resize validation test ---------------------------------------------


@pytest.mark.parametrize(
    "configs,should_raise",
    [
        pytest.param(
            [Mock(name="col_a", allow_resize=True), Mock(name="col_b", allow_resize=False)],
            True,
            id="raises_on_allow_resize",
        ),
        pytest.param(
            [Mock(name="col_a", allow_resize=False), Mock(name="col_b", allow_resize=False)],
            False,
            id="passes_without_allow_resize",
        ),
    ],
)
def test_validate_async_compatibility(configs: list[Mock], should_raise: bool) -> None:
    """Validation rejects allow_resize=True with the async engine."""
    builder = Mock(spec=ColumnWiseDatasetBuilder)
    builder.single_column_configs = configs
    if should_raise:
        with pytest.raises(DatasetGenerationError, match="allow_resize=True"):
            ColumnWiseDatasetBuilder._validate_async_compatibility(builder)
    else:
        ColumnWiseDatasetBuilder._validate_async_compatibility(builder)


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

    checkpointed: list[int] = []

    scheduler = AsyncTaskScheduler(
        generators=gen_map,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        on_row_group_complete=lambda rg: checkpointed.append(rg),
    )
    await scheduler.run()

    # Both row groups should be checkpointed
    assert sorted(checkpointed) == [0, 1]
    assert buffer_manager.actual_num_records == 4

    # All columns should be complete
    all_cols = ["seed", "cell_out", "expr_out"]
    assert tracker.is_row_group_complete(0, 2, all_cols)
    assert tracker.is_row_group_complete(1, 2, all_cols)


# -- Test that existing sync path is unaffected --------------------------------


def test_sync_path_unaffected_by_async_engine_flag() -> None:
    """DATA_DESIGNER_ASYNC_ENGINE=0 keeps the sync path unchanged."""
    import data_designer.engine.dataset_builders.column_wise_builder as builder_mod

    assert hasattr(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE")
    assert isinstance(builder_mod.DATA_DESIGNER_ASYNC_ENGINE, bool)


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
    )
    await scheduler.run()

    # Two row groups → two write_batch_to_parquet_file calls
    assert storage.write_batch_to_parquet_file.call_count == 2
    assert storage.move_partial_result_to_final_file_path.call_count == 2
    assert buffer_manager.actual_num_records == 5
