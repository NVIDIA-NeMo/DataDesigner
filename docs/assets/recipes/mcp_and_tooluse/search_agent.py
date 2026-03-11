# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""MCP + Tool Use Recipe: Search Agent Trajectories with Tavily Web Search

Generate multi-turn search agent trajectories where an LLM iteratively
searches the web, reads results, reasons about evidence, and synthesizes
answers -- the kind of data needed to train BrowseComp-style search agents.

The pipeline:
  1. Seeds from Wikidata knowledge graph paths (parquet or built-in demo)
  2. Generates multi-hop search riddles from the paths (draft + obfuscation)
  3. Runs a tool-using search agent with live Tavily web search
  4. Captures full tool-call traces for SFT training data

Based on the "Search Agent SFT Data: Teaching LLMs to Browse the Web" dev
note, which produced ~7,000 high-quality tool-use trajectories for Nemotron
post-training starting from 50,000 Wikidata seeds.

Prerequisites:
    - TAVILY_API_KEY environment variable (get a free key at https://tavily.com)
    - OPENAI_API_KEY environment variable for OpenAI provider model aliases.
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases (default model alias is "nvidia-text").

Run:
    # Basic usage with built-in demo seeds (generates 2 trajectories)
    uv run search_agent.py

    # Use a custom seed parquet
    uv run search_agent.py --seed-path /path/to/seeds.parquet --num-records 10

    # For help message and available options
    uv run search_agent.py --help
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner

# =============================================================================
# Data Designer Configuration
# =============================================================================

QUERY_DRAFT_PROMPT = """\
You are an expert Search Evaluator designing Grandmaster-Level search tests.
Create a complex, multi-step search riddle based on this knowledge path:

{{ readable_path }}

Start Entity: {{ seed_entity }}
Final Answer Entity: {{ final_answer_entity }}

RULES:
1. DO NOT name the intermediate nodes. Hide them behind descriptions.
2. DO NOT name the Final Answer.
3. Chain the clues logically -- describe each step relative to the previous one.
4. If a step is weak or nonsensical, IGNORE IT.
5. Output INVALID_PATH if the path is unsalvageable.

Return ONLY the question string (or INVALID_PATH).\
"""

OBFUSCATE_PROMPT = """\
Rewrite this search riddle to better match BrowseComp-style tasks.

Original Riddle: {{ user_query_draft }}

Secret Path (do not leak entities): {{ readable_path }}
Start Entity: {{ seed_entity }}
Final Answer (do not leak): {{ final_answer_entity }}

HARD REQUIREMENTS:
1. NEVER mention the final answer or any intermediate entity by name.
2. NO breadcrumb chains (avoid "X leads to Y leads to Z").
3. Use descriptive clues that require reasoning.
4. 1-2 sentences max, sounding like a natural web search query.
5. If original == "INVALID_PATH", output exactly "INVALID_PATH".

Return ONLY the rewritten question string (or INVALID_PATH).\
"""

AGENT_SYSTEM_PROMPT = """\
You are an expert search agent that uses web search to answer questions accurately.

You MUST output ONLY valid JSON matching this schema:

{
  "final_answer": "string - the specific answer entity",
  "supporting_urls": ["url1", "url2"],
  "short_justification": "string - brief 1-2 sentence explanation"
}

AVAILABLE TOOLS:
You have access to "tavily_search" with parameter: query (string, required).

TOOL USAGE RULES:
1. Always use "tavily_search" -- only send {"query": "..."}.
2. Maximum 15 tool calls. Budget your searches wisely.
3. Start with broad queries, then refine to specific entities.
4. Cross-verify facts across multiple sources.
5. After searches, output ONLY the JSON object.\
"""


def build_config(model_alias: str) -> tuple[dd.DataDesignerConfigBuilder, dd.MCPProvider]:
    """Build the Data Designer configuration for search agent trajectory generation.

    Returns:
        A tuple of (config_builder, mcp_provider).
    """
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    mcp_provider = dd.MCPProvider(
        name="tavily",
        endpoint=f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        provider_type="streamable_http",
    )

    tool_config = dd.ToolConfig(
        tool_alias="tavily-search",
        providers=["tavily"],
        allow_tools=["tavily_search"],
        max_tool_call_turns=15,
        timeout_sec=300.0,
    )

    config_builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

    # Stage 1: Draft question from knowledge path
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="user_query_draft",
            model_alias=model_alias,
            prompt=QUERY_DRAFT_PROMPT,
        )
    )

    # Stage 2: BrowseComp-style obfuscation
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="user_query_obfuscated",
            model_alias=model_alias,
            prompt=OBFUSCATE_PROMPT,
        )
    )

    # Stage 3: Agent trajectory with MCP tool calling
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="agent_solution_raw",
            model_alias=model_alias,
            system_prompt=AGENT_SYSTEM_PROMPT,
            prompt="Problem: {{ user_query_obfuscated }}",
            tool_alias="tavily-search",
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )

    return config_builder, mcp_provider


# =============================================================================
# Demo Seed Data
# =============================================================================

DEMO_SEEDS = [
    {
        "seed_entity": "NVIDIA",
        "final_answer_entity": "Thomas Hart Benton",
        "readable_path": (
            "START ENTITY: NVIDIA (Q182477)\n"
            "  ⬇ [chief executive officer (P169)]\n"
            "  NODE: Jensen Huang (Q332838)\n"
            "  ⬇ [educated at (P69)]\n"
            "  NODE: Oregon State University (Q861888)\n"
            "  ⬇ [located in the administrative territorial entity (P131)]\n"
            "  NODE: Benton County (Q115372)\n"
            "  ⬇ [named after (P138)]\n"
            "  NODE: Thomas Hart Benton (Q178712)"
        ),
        "num_hops_in_graph": 4,
        "ground_truth": "Thomas Hart Benton",
    },
    {
        "seed_entity": "Python",
        "final_answer_entity": "Centrum Wiskunde & Informatica",
        "readable_path": (
            "START ENTITY: Python (Q28865)\n"
            "  ⬇ [developer (P178)]\n"
            "  NODE: Guido van Rossum (Q19845)\n"
            "  ⬇ [employer (P108)]\n"
            "  NODE: Centrum Wiskunde & Informatica (Q1060645)"
        ),
        "num_hops_in_graph": 2,
        "ground_truth": "Centrum Wiskunde & Informatica",
    },
]


def write_demo_seeds(output_dir: Path) -> Path:
    """Write demo seed data to a JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "demo_seeds.jsonl"
    with open(seed_path, "w", encoding="utf-8") as f:
        for seed in DEMO_SEEDS:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")
    return seed_path


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate search agent trajectories using Tavily web search via MCP.")
    parser.add_argument("--model-alias", type=str, default="nvidia-text", help="Model alias to use for generation")
    parser.add_argument("--num-records", type=int, default=2, help="Number of trajectories to generate")
    parser.add_argument("--seed-path", type=str, default=None, help="Path to seed parquet or JSONL file")
    parser.add_argument("--artifact-path", type=str, default=None, help="Path to save artifacts")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the demo."""
    args = parse_args()

    if os.environ.get("TAVILY_API_KEY") is None:
        raise RuntimeError("TAVILY_API_KEY must be set. Get a free key at https://tavily.com")

    if os.environ.get("NVIDIA_API_KEY") is None and args.model_alias.startswith("nvidia"):
        raise RuntimeError("NVIDIA_API_KEY must be set when using NVIDIA model aliases.")

    # Use provided seed path or generate demo data
    if args.seed_path:
        seed_path = args.seed_path
    else:
        demo_dir = Path(tempfile.mkdtemp(prefix="search_agent_demo_"))
        seed_path = str(write_demo_seeds(demo_dir))
        print(f"Using demo seeds in: {demo_dir}")

    config_builder, mcp_provider = build_config(model_alias=args.model_alias)
    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.SHUFFLE,
    )

    data_designer = DataDesigner(artifact_path=args.artifact_path, mcp_providers=[mcp_provider])
    preview_results = data_designer.preview(config_builder, num_records=args.num_records)

    print("\n" + "=" * 60)
    print("GENERATED SEARCH AGENT TRAJECTORIES")
    print("=" * 60)
    preview_results.display_sample_record()


if __name__ == "__main__":
    main()
