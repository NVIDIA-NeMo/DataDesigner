# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from data_designer.config import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    LocalStdioMCPProvider,
    SamplerColumnConfig,
    SamplerType,
    ToolConfig,
    TraceType,
)
from data_designer.interface import DataDesigner


def test_mcp_server_tool_usage_with_nvidia_text(tmp_path: Path) -> None:
    if os.environ.get("NVIDIA_API_KEY") is None:
        pytest.skip("NVIDIA_API_KEY must be set to run the MCP demo with nvidia-text.")

    e2e_root = Path(__file__).resolve().parents[1]
    e2e_src = e2e_root / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(e2e_src) if not existing_pythonpath else f"{e2e_src}{os.pathsep}{existing_pythonpath}"

    log_path = tmp_path / "mcp_tool_calls.jsonl"

    mcp_provider = LocalStdioMCPProvider(
        name="demo-mcp",
        command=sys.executable,
        args=["-m", "data_designer_e2e_tests.mcp_demo_server"],
        env={
            "PYTHONPATH": pythonpath,
            "MCP_DEMO_LOG_PATH": str(log_path),
        },
    )

    data_designer = DataDesigner(mcp_providers=[mcp_provider])

    tool_config = ToolConfig(
        tool_alias="demo-tools",
        providers=["demo-mcp"],
        allow_tools=["get_fact", "add_numbers"],
    )

    config_builder = DataDesignerConfigBuilder(tool_configs=[tool_config])
    config_builder.add_column(
        SamplerColumnConfig(
            name="topic",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["MCP", "Data-Designer"]),
        )
    )
    config_builder.add_column(
        LLMTextColumnConfig(
            name="summary",
            prompt="Call get_fact for {{ topic }}, then call add_numbers with a=2 and b=3.",
            system_prompt=(
                "You must call the tools in sequence: first get_fact, then add_numbers. "
                "Do not answer before calling both tools."
            ),
            model_alias="nvidia-text",
            tool_alias="demo-tools",
            with_trace=TraceType.ALL_MESSAGES,
        )
    )

    preview = data_designer.preview(config_builder, num_records=1)

    expected_facts = {
        "MCP": "MCP lets models call tools over standardized transports.",
        "Data-Designer": "Data Designer generates structured synthetic datasets.",
    }

    assert preview.dataset is not None
    assert "summary" in preview.dataset.columns
    assert "summary__trace" in preview.dataset.columns
    assert "topic" in preview.dataset.columns

    for _, row in preview.dataset.iterrows():
        summary = row["summary"]
        assert summary is not None
        assert str(summary).strip()

        trace = row["summary__trace"]
        assert isinstance(trace, list)
        assert trace

        tool_call_messages = [
            msg for msg in trace if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls")
        ]
        assert tool_call_messages

        tool_calls: list[dict[str, object]] = []
        tool_call_indices: dict[str, int] = {}
        for msg_index, msg in enumerate(trace):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            for tool_call in msg.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_calls.append(tool_call)
                function = tool_call.get("function") or {}
                if isinstance(function, dict):
                    name = function.get("name")
                    if isinstance(name, str) and name not in tool_call_indices:
                        tool_call_indices[name] = msg_index

        assert tool_call_indices.get("get_fact") is not None
        assert tool_call_indices.get("add_numbers") is not None
        assert tool_call_indices["get_fact"] < tool_call_indices["add_numbers"]

        def _tool_call_to_name_args(tool_call: dict[str, object]) -> tuple[str | None, dict[str, object]]:
            function = tool_call.get("function")
            if not isinstance(function, dict):
                return None, {}
            name = function.get("name")
            if not isinstance(name, str):
                return None, {}
            raw_args = function.get("arguments")
            if isinstance(raw_args, str) and raw_args.strip():
                return name, json.loads(raw_args)
            return name, {}

        expected_topic = str(row["topic"])
        observed_calls: dict[str, dict[str, object]] = {}
        tool_call_ids: dict[str, str] = {}
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            if isinstance(tool_call_id, str):
                name, args = _tool_call_to_name_args(tool_call)
                if name is not None and name not in observed_calls:
                    observed_calls[name] = args
                    tool_call_ids[name] = tool_call_id

        assert observed_calls.get("get_fact") == {"topic": expected_topic}
        assert observed_calls.get("add_numbers") == {"a": 2, "b": 3}

        tool_messages = [msg for msg in trace if isinstance(msg, dict) and msg.get("role") == "tool"]
        tool_message_ids = {msg.get("tool_call_id") for msg in tool_messages}
        assert tool_call_ids["get_fact"] in tool_message_ids
        assert tool_call_ids["add_numbers"] in tool_message_ids

    assert log_path.exists()
    log_entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert log_entries

    fact_entries = [entry for entry in log_entries if entry.get("tool") == "get_fact"]
    assert fact_entries

    add_entries = [entry for entry in log_entries if entry.get("tool") == "add_numbers"]
    assert add_entries

    first_fact_index = next(index for index, entry in enumerate(log_entries) if entry.get("tool") == "get_fact")
    first_add_index = next(index for index, entry in enumerate(log_entries) if entry.get("tool") == "add_numbers")
    assert first_fact_index < first_add_index

    observed_topics: set[str] = set()
    for entry in fact_entries:
        arguments = entry.get("arguments") or {}
        topic = arguments.get("topic")
        assert isinstance(topic, str)
        assert topic in expected_facts
        observed_topics.add(topic)
        assert entry.get("result") == expected_facts[topic]

    for entry in add_entries:
        arguments = entry.get("arguments") or {}
        a_value = arguments.get("a")
        b_value = arguments.get("b")
        assert isinstance(a_value, int)
        assert isinstance(b_value, int)
        assert entry.get("result") == a_value + b_value

    generated_topics = set(preview.dataset["topic"].astype(str).tolist())
    assert generated_topics.issubset(observed_topics)
