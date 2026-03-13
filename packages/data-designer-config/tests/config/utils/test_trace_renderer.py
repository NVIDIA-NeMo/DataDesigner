# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.trace_renderer import TraceMessage, TraceRenderer
from data_designer.config.utils.trace_type import TraceType
from data_designer.config.utils.visualization import display_sample_record

# --- Fixtures ---


@pytest.fixture
def renderer() -> TraceRenderer:
    return TraceRenderer()


@pytest.fixture
def simple_trace() -> list[TraceMessage]:
    """Trace with system + user + assistant (no tool calls)."""
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "2+2 equals 4."}]},
    ]


@pytest.fixture
def tool_call_trace() -> list[TraceMessage]:
    """Trace with a single tool call round-trip."""
    return [
        {"role": "system", "content": [{"type": "text", "text": "Use tools to answer."}]},
        {"role": "user", "content": [{"type": "text", "text": "What is the population of Tokyo?"}]},
        {
            "role": "assistant",
            "content": [],
            "reasoning_content": "I need to search for current Tokyo population data.",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "tavily_search",
                        "arguments": '{"query": "current population of Tokyo", "max_results": 5}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": '{"results": [{"title": "Tokyo population", "content": "36.9 million"}]}'}
            ],
            "tool_call_id": "call_001",
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Tokyo has approximately 36.9 million people."}],
        },
    ]


@pytest.fixture
def multi_turn_tool_trace() -> list[TraceMessage]:
    """Trace with multiple tool call turns."""
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a researcher."}]},
        {"role": "user", "content": [{"type": "text", "text": "Compare populations of Tokyo and NYC."}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "Tokyo population"}'},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "NYC population"}'},
                },
            ],
        },
        {"role": "tool", "content": [{"type": "text", "text": "36.9 million"}], "tool_call_id": "call_a"},
        {"role": "tool", "content": [{"type": "text", "text": "8.3 million"}], "tool_call_id": "call_b"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Tokyo: 36.9M, NYC: 8.3M."}],
        },
    ]


# --- render_rich tests ---


def test_render_rich_simple_trace(renderer: TraceRenderer, simple_trace: list[TraceMessage]) -> None:
    """Simple trace without tool calls renders a Panel."""
    panel = renderer.render_rich(simple_trace, "answer__trace")
    assert panel.title is not None
    assert "answer__trace" in str(panel.title)


def test_render_rich_single_tool_call(renderer: TraceRenderer, tool_call_trace: list[TraceMessage]) -> None:
    """Trace with one tool call shows tool call count in summary."""
    panel = renderer.render_rich(tool_call_trace, "answer__trace")
    rendered = _render_panel_to_text(panel)
    assert "tool call" in rendered.lower()
    assert "tavily_search" in rendered


def test_render_rich_multi_turn_tool_calls(renderer: TraceRenderer, multi_turn_tool_trace: list[TraceMessage]) -> None:
    """Trace with multiple tool calls shows correct count."""
    panel = renderer.render_rich(multi_turn_tool_trace, "col__trace")
    rendered = _render_panel_to_text(panel)
    assert "2 tool calls" in rendered


def test_render_rich_reasoning_content(renderer: TraceRenderer, tool_call_trace: list[TraceMessage]) -> None:
    """Reasoning content is rendered when present."""
    panel = renderer.render_rich(tool_call_trace, "answer__trace")
    rendered = _render_panel_to_text(panel)
    assert "reasoning" in rendered.lower()
    assert "search for current Tokyo population" in rendered


def test_render_rich_empty_trace(renderer: TraceRenderer) -> None:
    """Empty trace list produces a panel with no crash."""
    panel = renderer.render_rich([], "empty__trace")
    assert panel.title is not None


def test_render_rich_no_tool_calls_no_summary(renderer: TraceRenderer, simple_trace: list[TraceMessage]) -> None:
    """When there are no tool calls, no summary line appears."""
    panel = renderer.render_rich(simple_trace, "answer__trace")
    rendered = _render_panel_to_text(panel)
    assert "tool call" not in rendered.lower()


def test_render_rich_html_special_chars_in_content(renderer: TraceRenderer) -> None:
    """Content with HTML special characters doesn't break rendering."""
    trace: list[TraceMessage] = [
        {"role": "user", "content": [{"type": "text", "text": "Show <script>alert('xss')</script> & \"quotes\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Here is <b>bold</b> & 'quoted'"}]},
    ]
    panel = renderer.render_rich(trace, "test__trace")
    rendered = _render_panel_to_text(panel)
    assert "<script>" in rendered
    assert "&" in rendered


def test_render_rich_malformed_tool_call_args(renderer: TraceRenderer) -> None:
    """Malformed JSON in tool call arguments doesn't crash."""
    trace: list[TraceMessage] = [
        {"role": "user", "content": [{"type": "text", "text": "test"}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "my_tool", "arguments": "not valid json {{{"},
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    panel = renderer.render_rich(trace, "test__trace")
    rendered = _render_panel_to_text(panel)
    assert "my_tool" in rendered
    assert "not valid json" in rendered


# --- render_notebook_html tests (outside notebook, should return False) ---


def test_render_notebook_html_returns_false_outside_notebook(
    renderer: TraceRenderer, simple_trace: list[TraceMessage]
) -> None:
    """Outside a Jupyter notebook, render_notebook_html returns False."""
    result = renderer.render_notebook_html(simple_trace, "answer__trace")
    assert result is False


# --- Integration with display_sample_record ---


def test_display_sample_record_with_trace_no_errors(
    stub_model_configs: list[Any], tool_call_trace: list[TraceMessage]
) -> None:
    """display_sample_record renders trace columns without errors."""
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(
        name="answer",
        column_type="llm-text",
        prompt="Answer: {{ question }}",
        model_alias="default",
        with_trace=TraceType.ALL_MESSAGES,
    )

    record = lazy.pd.Series(
        {
            "question": "What is 2+2?",
            "answer": "4",
            "answer__trace": tool_call_trace,
        }
    )

    display_sample_record(record, builder)


def test_display_sample_record_with_trace_in_saved_html(
    stub_model_configs: list[Any], tool_call_trace: list[TraceMessage], tmp_path: Path
) -> None:
    """Trace panel content appears in saved HTML output."""
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(
        name="answer",
        column_type="llm-text",
        prompt="Answer: {{ question }}",
        model_alias="default",
        with_trace=TraceType.ALL_MESSAGES,
    )

    record = lazy.pd.Series(
        {
            "question": "What is 2+2?",
            "answer": "4",
            "answer__trace": tool_call_trace,
        }
    )

    save_path = tmp_path / "output.html"
    display_sample_record(record, builder, save_path=save_path)

    content = save_path.read_text()
    assert "answer__trace" in content
    assert "tavily_search" in content


def test_display_sample_record_include_traces_false(
    stub_model_configs: list[Any], tool_call_trace: list[TraceMessage], tmp_path: Path
) -> None:
    """When include_traces=False, trace panels are not rendered."""
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(
        name="answer",
        column_type="llm-text",
        prompt="Answer: {{ question }}",
        model_alias="default",
        with_trace=TraceType.ALL_MESSAGES,
    )

    record = lazy.pd.Series(
        {
            "question": "What is 2+2?",
            "answer": "4",
            "answer__trace": tool_call_trace,
        }
    )

    save_path = tmp_path / "output.html"
    display_sample_record(record, builder, include_traces=False, save_path=save_path)

    content = save_path.read_text()
    assert "Trace:" not in content
    assert "tavily_search" not in content


def test_display_sample_record_no_trace_column_in_record(stub_model_configs: list[Any]) -> None:
    """When trace is configured but not in the record, no crash occurs."""
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(
        name="answer",
        column_type="llm-text",
        prompt="Answer: {{ question }}",
        model_alias="default",
        with_trace=TraceType.ALL_MESSAGES,
    )

    record = lazy.pd.Series(
        {
            "question": "What is 2+2?",
            "answer": "4",
        }
    )

    display_sample_record(record, builder)


def test_existing_display_sample_record_unaffected(
    stub_model_configs: list[Any],
) -> None:
    """Existing display_sample_record behavior is unaffected by the new parameter."""
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(
        name="answer",
        column_type="llm-text",
        prompt="Answer: {{ question }}",
        model_alias="default",
    )

    record = lazy.pd.Series(
        {
            "question": "What is 2+2?",
            "answer": "4",
        }
    )

    display_sample_record(record, builder)


# --- Helpers ---


def _render_panel_to_text(panel: Any) -> str:
    """Render a Rich Panel to plain text for assertion."""
    from rich.console import Console

    console = Console(width=120, file=__import__("io").StringIO())
    console.print(panel)
    return console.file.getvalue()
