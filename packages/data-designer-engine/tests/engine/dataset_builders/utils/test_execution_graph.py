# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    Score,
    ValidationColumnConfig,
)
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
from data_designer.engine.dataset_builders.utils.errors import DAGCircularDependencyError
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.task_model import SliceRef

MODEL_ALIAS = "stub-model-alias"


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture()
def simple_pipeline_configs() -> list:
    """topic (sampler) → question (llm) → answer (llm) → score (expression)."""
    return [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A", "B"]}),
        LLMTextColumnConfig(name="question", prompt="Ask about {{ topic }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="answer", prompt="Answer {{ question }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="score", expr="{{ answer }}"),
    ]


@pytest.fixture()
def simple_pipeline_strategies() -> dict[str, GenerationStrategy]:
    return {
        "topic": GenerationStrategy.FULL_COLUMN,
        "question": GenerationStrategy.CELL_BY_CELL,
        "answer": GenerationStrategy.CELL_BY_CELL,
        "score": GenerationStrategy.FULL_COLUMN,
    }


@pytest.fixture()
def simple_graph(
    simple_pipeline_configs: list,
    simple_pipeline_strategies: dict[str, GenerationStrategy],
) -> ExecutionGraph:
    return ExecutionGraph.create(simple_pipeline_configs, simple_pipeline_strategies)


# -- Graph construction tests ------------------------------------------------


def test_build_basic_graph(simple_graph: ExecutionGraph) -> None:
    assert simple_graph.columns == ["topic", "question", "answer", "score"]
    assert simple_graph.get_upstream_columns("topic") == set()
    assert simple_graph.get_upstream_columns("question") == {"topic"}
    assert simple_graph.get_upstream_columns("answer") == {"question"}
    assert simple_graph.get_upstream_columns("score") == {"answer"}


def test_get_downstream_columns(simple_graph: ExecutionGraph) -> None:
    assert simple_graph.get_downstream_columns("topic") == {"question"}
    assert simple_graph.get_downstream_columns("question") == {"answer"}
    assert simple_graph.get_downstream_columns("answer") == {"score"}
    assert simple_graph.get_downstream_columns("score") == set()


def test_strategy(simple_graph: ExecutionGraph) -> None:
    assert simple_graph.get_strategy("topic") == GenerationStrategy.FULL_COLUMN
    assert simple_graph.get_strategy("question") == GenerationStrategy.CELL_BY_CELL


def test_unknown_column_get_upstream_columns() -> None:
    graph = ExecutionGraph()
    assert graph.get_upstream_columns("nonexistent") == set()


def test_unknown_column_get_downstream_columns() -> None:
    graph = ExecutionGraph()
    assert graph.get_downstream_columns("nonexistent") == set()


# -- Side-effect resolution -------------------------------------------------


def test_side_effect_column_resolution() -> None:
    configs = [
        LLMTextColumnConfig(
            name="summary",
            prompt="Summarize",
            model_alias=MODEL_ALIAS,
            with_trace="last_message",
        ),
        ExpressionColumnConfig(name="trace_len", expr="{{ summary__trace }}"),
    ]
    strategies = {
        "summary": GenerationStrategy.CELL_BY_CELL,
        "trace_len": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)

    assert graph.get_upstream_columns("trace_len") == {"summary"}
    assert graph.get_downstream_columns("summary") == {"trace_len"}
    assert graph.get_required_columns("trace_len") == ["summary"]


def test_reasoning_content_side_effect() -> None:
    configs = [
        LLMTextColumnConfig(
            name="answer",
            prompt="Think step by step",
            model_alias=MODEL_ALIAS,
            extract_reasoning_content=True,
        ),
        ExpressionColumnConfig(name="reasoning", expr="{{ answer__reasoning_content }}"),
    ]
    strategies = {
        "answer": GenerationStrategy.CELL_BY_CELL,
        "reasoning": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)

    assert graph.get_upstream_columns("reasoning") == {"answer"}


def test_side_effect_name_collision_prefers_real_column() -> None:
    configs = [
        LLMTextColumnConfig(
            name="summary",
            prompt="Summarize",
            model_alias=MODEL_ALIAS,
            with_trace="last_message",
        ),
        SamplerColumnConfig(name="summary__trace", sampler_type=SamplerType.CATEGORY, params={"values": ["OVERRIDE"]}),
        ExpressionColumnConfig(name="trace_len", expr="{{ summary__trace }}"),
    ]
    strategies = {
        "summary": GenerationStrategy.CELL_BY_CELL,
        "summary__trace": GenerationStrategy.FULL_COLUMN,
        "trace_len": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)

    assert graph.get_upstream_columns("trace_len") == {"summary__trace"}
    assert graph.get_downstream_columns("summary__trace") == {"trace_len"}
    assert graph.get_downstream_columns("summary") == set()


# -- Validation tests -------------------------------------------------------


def test_circular_dependency_raises() -> None:
    configs = [
        LLMTextColumnConfig(name="col_a", prompt="{{ col_b }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="col_b", prompt="{{ col_a }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {
        "col_a": GenerationStrategy.CELL_BY_CELL,
        "col_b": GenerationStrategy.CELL_BY_CELL,
    }
    with pytest.raises(DAGCircularDependencyError):
        ExecutionGraph.create(configs, strategies)


def test_unknown_required_column_raises() -> None:
    configs = [
        LLMTextColumnConfig(name="col_a", prompt="{{ nonexistent }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {"col_a": GenerationStrategy.CELL_BY_CELL}
    with pytest.raises(ValueError, match="not a known producer"):
        ExecutionGraph.create(configs, strategies)


# -- Topological order ------------------------------------------------------


def test_topological_order(simple_graph: ExecutionGraph) -> None:
    order = simple_graph.get_topological_order()
    idx = {col: i for i, col in enumerate(order)}

    assert idx["topic"] < idx["question"]
    assert idx["question"] < idx["answer"]
    assert idx["answer"] < idx["score"]


def test_parallel_columns_topological_order() -> None:
    """Two independent columns after a shared root."""
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["X"]}),
        LLMTextColumnConfig(name="branch_a", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="branch_b", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="merge", expr="{{ branch_a }} {{ branch_b }}"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "branch_a": GenerationStrategy.CELL_BY_CELL,
        "branch_b": GenerationStrategy.CELL_BY_CELL,
        "merge": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)
    order = graph.get_topological_order()
    idx = {col: i for i, col in enumerate(order)}

    assert idx["seed"] < idx["branch_a"]
    assert idx["seed"] < idx["branch_b"]
    assert idx["branch_a"] < idx["merge"]
    assert idx["branch_b"] < idx["merge"]


# -- Critical path ----------------------------------------------------------


def test_get_longest_dependency_chain_empty_graph() -> None:
    graph = ExecutionGraph()
    assert graph.get_longest_dependency_chain() == []


def test_get_longest_dependency_chain(simple_graph: ExecutionGraph) -> None:
    path = simple_graph.get_longest_dependency_chain()
    assert path == ["topic", "question", "answer", "score"]


def test_get_longest_dependency_chain_diamond() -> None:
    """Diamond: seed → (a, b) → merge. Path is seed → a/b → merge (length 3)."""
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["X"]}),
        LLMTextColumnConfig(name="a", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="b", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="merge", expr="{{ a }} {{ b }}"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "a": GenerationStrategy.CELL_BY_CELL,
        "b": GenerationStrategy.CELL_BY_CELL,
        "merge": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)
    path = graph.get_longest_dependency_chain()

    assert len(path) == 3
    assert path[0] == "seed"
    assert path[-1] == "merge"


# -- Task count -------------------------------------------------------------


def test_task_count(simple_graph: ExecutionGraph) -> None:
    counts = simple_graph.compute_task_count(num_records=10, buffer_size=3)

    assert counts["topic"] == 4  # ceil(10/3) = 4 row groups
    assert counts["question"] == 10  # cell-by-cell
    assert counts["answer"] == 10  # cell-by-cell
    assert counts["score"] == 4  # full-column


def test_task_count_exact_divisor(simple_graph: ExecutionGraph) -> None:
    counts = simple_graph.compute_task_count(num_records=9, buffer_size=3)

    assert counts["topic"] == 3
    assert counts["question"] == 9


@pytest.mark.parametrize("buffer_size", [0, -1])
def test_task_count_invalid_buffer_size_raises(simple_graph: ExecutionGraph, buffer_size: int) -> None:
    with pytest.raises(ValueError, match="buffer_size"):
        simple_graph.compute_task_count(num_records=10, buffer_size=buffer_size)


def test_add_column_duplicate_raises() -> None:
    graph = ExecutionGraph()
    graph.add_column("col_a", GenerationStrategy.CELL_BY_CELL)
    with pytest.raises(ValueError, match="already registered"):
        graph.add_column("col_a", GenerationStrategy.FULL_COLUMN)


# -- Cell dependencies ------------------------------------------------------


def test_cell_deps_cell_by_cell_upstream(simple_graph: ExecutionGraph) -> None:
    """question depends on topic (full-column); answer depends on question (cell-by-cell)."""
    # answer[rg=0, row=2] should depend on question[rg=0, row=2]
    deps = simple_graph.compute_cell_dependencies("answer", row_group=0, row_index=2, row_group_size=5)
    assert deps == [SliceRef("question", 0, 2)]


def test_cell_deps_full_column_upstream(simple_graph: ExecutionGraph) -> None:
    """question depends on topic (full-column)."""
    deps = simple_graph.compute_cell_dependencies("question", row_group=0, row_index=1, row_group_size=5)
    assert deps == [SliceRef("topic", 0, None)]


def test_cell_deps_no_upstream(simple_graph: ExecutionGraph) -> None:
    """topic has no upstream."""
    deps = simple_graph.compute_cell_dependencies("topic", row_group=0, row_index=None, row_group_size=5)
    assert deps == []


def test_cell_deps_full_column_downstream_of_cell_by_cell(simple_graph: ExecutionGraph) -> None:
    """score (full-column) depends on answer (cell-by-cell) → needs ALL rows."""
    deps = simple_graph.compute_cell_dependencies("score", row_group=0, row_index=None, row_group_size=3)
    assert sorted(deps) == [SliceRef("answer", 0, 0), SliceRef("answer", 0, 1), SliceRef("answer", 0, 2)]


# -- Mermaid output ----------------------------------------------------------


def test_to_mermaid(simple_graph: ExecutionGraph) -> None:
    mermaid = simple_graph.to_mermaid()

    assert "graph TD" in mermaid
    assert 'topic["topic [full_column]"]' in mermaid
    assert 'question["question [cell_by_cell]"]' in mermaid
    assert "topic --> question" in mermaid
    assert "question --> answer" in mermaid
    assert "answer --> score" in mermaid


# -- MultiColumnConfig -------------------------------------------------------


def test_multi_column_config() -> None:
    """Multi-column sampler config: all sub-columns share the same strategy."""
    multi = SamplerMultiColumnConfig(
        columns=[
            SamplerColumnConfig(name="first_name", sampler_type=SamplerType.CATEGORY, params={"values": ["Alice"]}),
            SamplerColumnConfig(name="last_name", sampler_type=SamplerType.CATEGORY, params={"values": ["Smith"]}),
        ]
    )
    configs = [multi]
    strategies = {
        "first_name": GenerationStrategy.FULL_COLUMN,
        "last_name": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)

    assert set(graph.columns) == {"first_name", "last_name"}
    assert graph.get_upstream_columns("first_name") == set()
    assert graph.get_upstream_columns("last_name") == set()


def test_multi_column_with_downstream_dependency() -> None:
    multi = SamplerMultiColumnConfig(
        columns=[
            SamplerColumnConfig(name="first_name", sampler_type=SamplerType.CATEGORY, params={"values": ["Alice"]}),
            SamplerColumnConfig(name="last_name", sampler_type=SamplerType.CATEGORY, params={"values": ["Smith"]}),
        ]
    )
    greeting = LLMTextColumnConfig(
        name="greeting",
        prompt="Hello {{ first_name }} {{ last_name }}",
        model_alias=MODEL_ALIAS,
    )
    configs = [multi, greeting]
    strategies = {
        "first_name": GenerationStrategy.FULL_COLUMN,
        "last_name": GenerationStrategy.FULL_COLUMN,
        "greeting": GenerationStrategy.CELL_BY_CELL,
    }
    graph = ExecutionGraph.create(configs, strategies)

    assert graph.get_upstream_columns("greeting") == {"first_name", "last_name"}


# -- Validation column dependency -------------------------------------------


def test_validation_column_dependency() -> None:
    configs = [
        LLMCodeColumnConfig(
            name="code",
            prompt="Write code",
            code_lang=CodeLang.PYTHON,
            model_alias=MODEL_ALIAS,
        ),
        ValidationColumnConfig(
            name="validation",
            target_columns=["code"],
            validator_type="code",
            validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        ),
    ]
    strategies = {
        "code": GenerationStrategy.CELL_BY_CELL,
        "validation": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)

    assert graph.get_upstream_columns("validation") == {"code"}
    assert graph.get_downstream_columns("code") == {"validation"}


# -- Immutability tests -----------------------------------------------------


def test_mutating_columns_does_not_affect_graph(simple_graph: ExecutionGraph) -> None:
    cols = simple_graph.columns
    cols.append("injected")
    assert "injected" not in simple_graph.columns


def test_mutating_upstream_does_not_affect_graph(simple_graph: ExecutionGraph) -> None:
    ups = simple_graph.get_upstream_columns("question")
    ups.add("injected")
    assert "injected" not in simple_graph.get_upstream_columns("question")


def test_mutating_downstream_does_not_affect_graph(simple_graph: ExecutionGraph) -> None:
    downs = simple_graph.get_downstream_columns("topic")
    downs.add("injected")
    assert "injected" not in simple_graph.get_downstream_columns("topic")


def test_mutating_topological_order_does_not_affect_cache(simple_graph: ExecutionGraph) -> None:
    order1 = simple_graph.get_topological_order()
    order1.reverse()
    order2 = simple_graph.get_topological_order()
    assert order2[0] == "topic"


# -- Judge column dependency ------------------------------------------------


def test_judge_column_dependency() -> None:
    configs = [
        LLMTextColumnConfig(name="text", prompt="Write something", model_alias=MODEL_ALIAS),
        LLMJudgeColumnConfig(
            name="judge",
            prompt="Judge {{ text }}",
            scores=[Score(name="quality", description="Quality", options={0: "Bad", 1: "Good"})],
            model_alias=MODEL_ALIAS,
        ),
    ]
    strategies = {
        "text": GenerationStrategy.CELL_BY_CELL,
        "judge": GenerationStrategy.CELL_BY_CELL,
    }
    graph = ExecutionGraph.create(configs, strategies)

    assert graph.get_upstream_columns("judge") == {"text"}
