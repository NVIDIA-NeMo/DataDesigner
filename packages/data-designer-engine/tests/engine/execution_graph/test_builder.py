# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph builder."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.execution_graph.builder import GraphBuilder
from data_designer.engine.execution_graph.traits import ExecutionTraits

if TYPE_CHECKING:
    from data_designer.config.data_designer_config import DataDesignerConfig


def create_mock_single_column_config(
    name: str,
    required_columns: list[str] | None = None,
    side_effect_columns: list[str] | None = None,
) -> Mock:
    """Create a mock single column config."""
    config = Mock()
    config.name = name
    config.required_columns = required_columns or []
    config.side_effect_columns = side_effect_columns or []
    return config


def create_mock_multi_column_config(column_names: list[str]) -> Mock:
    """Create a mock multi-column config."""
    from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig

    config = Mock(spec=MultiColumnConfig)
    config.columns = [Mock(name=name) for name in column_names]
    for i, col in enumerate(config.columns):
        col.name = column_names[i]
    return config


def create_mock_generator_class(
    can_generate_from_scratch: bool = False,
    generation_strategy: GenerationStrategy = GenerationStrategy.FULL_COLUMN,
    is_row_streamable: bool = False,
) -> type:
    """Create a mock generator class with specified traits.

    Returns an actual class (not a Mock) so that getattr works correctly
    for property access.
    """
    # Capture variables in local scope
    _can_gen = can_generate_from_scratch
    _is_stream = is_row_streamable
    _gen_strategy = generation_strategy

    class MockGenerator:
        pass

    # Set class-level attributes that getattr will find
    MockGenerator.can_generate_from_scratch = _can_gen
    MockGenerator.is_row_streamable = _is_stream

    # Create a static method that returns the captured strategy
    def get_generation_strategy() -> GenerationStrategy:
        return _gen_strategy

    MockGenerator.get_generation_strategy = staticmethod(get_generation_strategy)

    return MockGenerator


def create_mock_registry(generator_map: dict[type, Mock]) -> Mock:
    """Create a mock registry that returns generators based on config type."""
    registry = Mock()
    registry.get_for_config_type.side_effect = lambda config_type: generator_map.get(
        config_type, create_mock_generator_class()
    )
    return registry


def create_mock_data_designer_config(column_configs: list[Mock]) -> DataDesignerConfig:
    """Create a mock DataDesignerConfig."""
    config = Mock()
    config.columns = column_configs
    return config


# --- Build tests ---


@patch("data_designer.engine.execution_graph.builder.compile_dataset_builder_column_configs")
def test_build_with_single_start_column(mock_compile: Mock) -> None:
    col_config = create_mock_single_column_config("starter")
    gen_cls = create_mock_generator_class(
        can_generate_from_scratch=True,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
    )
    registry = create_mock_registry({type(col_config): gen_cls})
    config = create_mock_data_designer_config([col_config])

    # Mock the compiler to return the configs directly
    mock_compile.return_value = [col_config]

    builder = GraphBuilder(registry)
    graph = builder.build(config, num_records=100)

    assert graph.num_records == 100
    assert graph.num_columns == 1
    assert "starter" in graph.start_columns


@patch("data_designer.engine.execution_graph.builder.compile_dataset_builder_column_configs")
def test_build_requires_at_least_one_start_column(mock_compile: Mock) -> None:
    col_config = create_mock_single_column_config("no_start")
    gen_cls = create_mock_generator_class(
        can_generate_from_scratch=False,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
        is_row_streamable=False,
    )
    registry = create_mock_registry({type(col_config): gen_cls})
    config = create_mock_data_designer_config([col_config])

    # Mock the compiler to return the configs directly
    mock_compile.return_value = [col_config]

    builder = GraphBuilder(registry)
    with pytest.raises(ValueError, match="generate from scratch"):
        builder.build(config, num_records=100)


@patch("data_designer.engine.execution_graph.builder.compile_dataset_builder_column_configs")
def test_build_with_multi_column_config(mock_compile: Mock) -> None:
    multi_config = create_mock_multi_column_config(["person_name", "person_email", "person_phone"])
    gen_cls = create_mock_generator_class(
        can_generate_from_scratch=True,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
        is_row_streamable=True,
    )
    registry = create_mock_registry({type(multi_config): gen_cls})
    config = create_mock_data_designer_config([multi_config])

    # Mock the compiler to return the multi-column config directly
    mock_compile.return_value = [multi_config]

    builder = GraphBuilder(registry)
    graph = builder.build(config, num_records=100)

    # Should have one descriptor with first column as primary
    assert graph.num_columns == 1
    desc = graph.get_column_descriptor("person_name")
    assert desc.name == "person_name"
    assert desc.side_effects == ["person_email", "person_phone"]


# --- Trait inference tests ---


def test_infer_start_trait() -> None:
    gen_cls = create_mock_generator_class(
        can_generate_from_scratch=True,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    builder = GraphBuilder(Mock())
    traits = builder._infer_traits(gen_cls)

    assert bool(traits & ExecutionTraits.START)


def test_infer_cell_by_cell_trait() -> None:
    gen_cls = create_mock_generator_class(
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
    )

    builder = GraphBuilder(Mock())
    traits = builder._infer_traits(gen_cls)

    assert bool(traits & ExecutionTraits.CELL_BY_CELL)
    assert bool(traits & ExecutionTraits.ROW_STREAMABLE)


def test_infer_row_streamable_full_column() -> None:
    gen_cls = create_mock_generator_class(
        generation_strategy=GenerationStrategy.FULL_COLUMN,
        is_row_streamable=True,
    )

    builder = GraphBuilder(Mock())
    traits = builder._infer_traits(gen_cls)

    assert bool(traits & ExecutionTraits.ROW_STREAMABLE)
    assert not bool(traits & ExecutionTraits.BARRIER)


def test_infer_barrier_trait() -> None:
    gen_cls = create_mock_generator_class(
        generation_strategy=GenerationStrategy.FULL_COLUMN,
        is_row_streamable=False,
    )

    builder = GraphBuilder(Mock())
    traits = builder._infer_traits(gen_cls)

    assert bool(traits & ExecutionTraits.BARRIER)
    assert not bool(traits & ExecutionTraits.ROW_STREAMABLE)


# --- Dependencies tests ---


@patch("data_designer.engine.execution_graph.builder.compile_dataset_builder_column_configs")
def test_single_column_dependencies(mock_compile: Mock) -> None:
    start_config = create_mock_single_column_config("starter")
    dep_config = create_mock_single_column_config(
        "dependent",
        required_columns=["starter"],
    )

    start_gen = create_mock_generator_class(
        can_generate_from_scratch=True,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
    )
    dep_gen = create_mock_generator_class(
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
    )

    registry = create_mock_registry(
        {
            type(start_config): start_gen,
            type(dep_config): dep_gen,
        }
    )
    config = create_mock_data_designer_config([start_config, dep_config])

    # Mock the compiler to return the configs directly
    mock_compile.return_value = [start_config, dep_config]

    builder = GraphBuilder(registry)
    graph = builder.build(config, num_records=10)

    dep_desc = graph.get_column_descriptor("dependent")
    assert dep_desc.dependencies == ["starter"]


@patch("data_designer.engine.execution_graph.builder.compile_dataset_builder_column_configs")
def test_side_effect_columns_captured(mock_compile: Mock) -> None:
    config = create_mock_single_column_config(
        "llm_output",
        side_effect_columns=["reasoning_trace"],
    )
    gen_cls = create_mock_generator_class(
        can_generate_from_scratch=True,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
    )
    registry = create_mock_registry({type(config): gen_cls})
    dd_config = create_mock_data_designer_config([config])

    # Mock the compiler to return the config directly
    mock_compile.return_value = [config]

    builder = GraphBuilder(registry)
    graph = builder.build(dd_config, num_records=10)

    desc = graph.get_column_descriptor("llm_output")
    assert desc.side_effects == ["reasoning_trace"]


# --- Strict mode tests ---


@patch("data_designer.engine.execution_graph.builder.compile_dataset_builder_column_configs")
def test_strict_mode_validates_dependencies(mock_compile: Mock) -> None:
    config = create_mock_single_column_config(
        "dependent",
        required_columns=["nonexistent"],
    )
    gen_cls = create_mock_generator_class(
        can_generate_from_scratch=True,  # To pass the start column check
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
    )
    registry = create_mock_registry({type(config): gen_cls})
    dd_config = create_mock_data_designer_config([config])

    # Mock the compiler to return the config directly
    mock_compile.return_value = [config]

    builder = GraphBuilder(registry)
    with pytest.raises(ValueError, match="unknown column"):
        builder.build(dd_config, num_records=10, strict=True)


@patch("data_designer.engine.execution_graph.builder.compile_dataset_builder_column_configs")
def test_non_strict_mode_skips_validation(mock_compile: Mock) -> None:
    config = create_mock_single_column_config(
        "dependent",
        required_columns=["nonexistent"],
    )
    gen_cls = create_mock_generator_class(
        can_generate_from_scratch=True,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
    )
    registry = create_mock_registry({type(config): gen_cls})
    dd_config = create_mock_data_designer_config([config])

    # Mock the compiler to return the config directly
    mock_compile.return_value = [config]

    builder = GraphBuilder(registry)
    # Should not raise
    graph = builder.build(dd_config, num_records=10, strict=False)
    assert graph.num_columns == 1
