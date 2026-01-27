# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    MCPServerConfig,
    MCPToolConfig,
    SamplerColumnConfig,
    SamplerType,
)


def test_mcp_server_tool_usage_with_nvidia_text() -> None:
    if os.environ.get("NVIDIA_API_KEY") is None:
        pytest.skip("NVIDIA_API_KEY must be set to run the MCP demo with nvidia-text.")

    e2e_root = Path(__file__).resolve().parents[1]
    e2e_src = e2e_root / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(e2e_src) if not existing_pythonpath else f"{e2e_src}{os.pathsep}{existing_pythonpath}"

    mcp_server = MCPServerConfig(
        name="demo-mcp",
        command=sys.executable,
        args=["-m", "data_designer_e2e_tests.mcp_demo_server"],
        env={"PYTHONPATH": pythonpath},
    )

    data_designer = DataDesigner(mcp_servers=[mcp_server])

    config_builder = DataDesignerConfigBuilder()
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
            prompt="Use the get_fact tool to fetch a fact about {{ topic }}. Respond with one sentence.",
            system_prompt="You must call the get_fact tool exactly once before answering.",
            model_alias="nvidia-text",
            tool_config=MCPToolConfig(server_name="demo-mcp", tool_names=["get_fact"]),
        )
    )

    preview = data_designer.preview(config_builder, num_records=2)

    assert "summary" in preview.dataset.columns
