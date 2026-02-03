# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "mcp",
# ]
# ///
"""Basic MCP Recipe: Simple Tool Use Example

This recipe demonstrates the minimal MCP tool-calling workflow with Data Designer:

1) Define a simple MCP server with basic tools (get_fact, add_numbers)
2) Configure Data Designer to use the MCP tools
3) Generate data that requires tool calls to complete

Prerequisites:
    - OPENAI_API_KEY environment variable for OpenAI provider model aliases.
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases (default model alias is "nvidia-text").

Run:
    # Basic usage (generates 2 records by default)
    uv run basic_mcp.py

    # For help message and available options
    uv run basic_mcp.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import data_designer.config as dd
from data_designer.interface import DataDesigner

MCP_SERVER_NAME = "basic-tools"


# =============================================================================
# MCP Server Definition
# =============================================================================

mcp_server = FastMCP(MCP_SERVER_NAME)

# Simple knowledge base for the get_fact tool
FACTS = {
    "python": "Python was created by Guido van Rossum and first released in 1991.",
    "earth": "Earth is the third planet from the Sun and has one natural satellite, the Moon.",
    "water": "Water (H2O) freezes at 0°C (32°F) and boils at 100°C (212°F) at sea level.",
    "light": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
}


@mcp_server.tool()
def get_fact(topic: str) -> str:
    """Get a fact about a topic from the knowledge base.

    Args:
        topic: The topic to look up (e.g., "python", "earth", "water", "light")

    Returns:
        A fact about the topic, or an error message if not found.
    """
    topic_lower = topic.lower()
    if topic_lower in FACTS:
        return json.dumps({"topic": topic, "fact": FACTS[topic_lower]})
    return json.dumps({"error": f"No fact found for topic: {topic}", "available_topics": list(FACTS.keys())})


@mcp_server.tool()
def add_numbers(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of the two numbers.
    """
    result = a + b
    return json.dumps({"a": a, "b": b, "sum": result})


@mcp_server.tool()
def list_topics() -> str:
    """List all available topics in the knowledge base.

    Returns:
        List of available topics.
    """
    return json.dumps({"topics": list(FACTS.keys())})


# =============================================================================
# Data Designer Configuration
# =============================================================================


def build_config(model_alias: str, provider_name: str) -> dd.DataDesignerConfigBuilder:
    """Build the Data Designer configuration for basic tool use."""
    tool_config = dd.ToolConfig(
        tool_alias="basic-tools",
        providers=[provider_name],
        allow_tools=["get_fact", "add_numbers", "list_topics"],
        max_tool_call_turns=5,
        timeout_sec=30.0,
    )

    config_builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

    # Add a seed column with topics to look up
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["python", "earth", "water", "light"]),
        )
    )

    # Add a column that uses the get_fact tool
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="fact_response",
            model_alias=model_alias,
            prompt=(
                "Use the get_fact tool to look up information about '{{ topic }}', "
                "then provide a one-sentence summary of what you learned."
            ),
            system_prompt="You must call the get_fact tool before answering. Only use information from tool results.",
            tool_alias="basic-tools",
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )

    # Add a column that uses the add_numbers tool
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="num_a",
            sampler_type=dd.SamplerType.UNIFORM,
            params=dd.UniformSamplerParams(low=1, high=100),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="num_b",
            sampler_type=dd.SamplerType.UNIFORM,
            params=dd.UniformSamplerParams(low=1, high=100),
        )
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="math_response",
            model_alias=model_alias,
            prompt=(
                "Use the add_numbers tool to calculate {{ num_a }} + {{ num_b }}, "
                "then report the result in a complete sentence."
            ),
            system_prompt="You must call the add_numbers tool to perform the calculation. Report the exact result.",
            tool_alias="basic-tools",
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )

    return config_builder


# =============================================================================
# Main Entry Points
# =============================================================================


def serve() -> None:
    """Run the MCP server (called when launched as subprocess by Data Designer)."""
    mcp_server.run()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic MCP tool use example with Data Designer.")
    subparsers = parser.add_subparsers(dest="command")

    # 'serve' subcommand for running the MCP server
    subparsers.add_parser("serve", help="Run the MCP server (used by Data Designer)")

    # Default command arguments (demo mode)
    parser.add_argument("--model-alias", type=str, default="nvidia-text", help="Model alias to use for generation")
    parser.add_argument("--num-records", type=int, default=2, help="Number of records to generate")
    # For compatibility with Makefile test-run-recipes target (ignored in demo mode)
    parser.add_argument("--artifact-path", type=str, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> None:
    """Main entry point for the demo."""
    args = parse_args()

    # Handle 'serve' subcommand
    if args.command == "serve":
        serve()
        return

    # Demo mode: run Data Designer with the MCP server
    if os.environ.get("NVIDIA_API_KEY") is None and args.model_alias.startswith("nvidia"):
        raise RuntimeError("NVIDIA_API_KEY must be set when using NVIDIA model aliases.")

    # Configure MCP provider to run via stdio transport (local subprocess)
    mcp_provider = dd.LocalStdioMCPProvider(
        name=MCP_SERVER_NAME,
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "serve"],
    )

    config_builder = build_config(
        model_alias=args.model_alias,
        provider_name=MCP_SERVER_NAME,
    )

    data_designer = DataDesigner(mcp_providers=[mcp_provider])
    preview_results = data_designer.preview(config_builder, num_records=args.num_records)

    # Display results
    print("\n" + "=" * 60)
    print("GENERATED DATA")
    print("=" * 60)
    preview_results.display_sample_record()


if __name__ == "__main__":
    main()
