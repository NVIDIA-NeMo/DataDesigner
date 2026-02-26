"""Utilities for converting generated records to NeMo Gym JSONL format."""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd


def convert_to_nemo_gym_format(
    row: dict[str, Any],
    idx: int,
    tools: list[dict[str, Any]],
    system_prompt: str,
    environment_name: str = "workplace_assistant",
) -> dict[str, Any]:
    """Convert a generated row to NeMo Gym rollout format."""
    trajectory = row.get("trajectory", {})
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)

    ground_truth = []
    for step in trajectory.get("reasoning_trace", []):
        tool_call = step.get("tool_call", {})
        ground_truth.append(
            {
                "name": tool_call.get("name", ""),
                "arguments": tool_call.get("arguments", "{}"),
            }
        )

    cleaned_tools = [
        {
            "type": tool.get("type"),
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("parameters"),
            "strict": tool.get("strict"),
        }
        for tool in tools
    ]

    return {
        "id": idx,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row.get("user_query", "")},
            ],
            "tools": cleaned_tools,
            "parallel_tool_calls": False,
            "temperature": 1.0,
        },
        "ground_truth": ground_truth,
        "category": f"workplace_assistant_{row.get('category', 'general')}",
        "environment_name": environment_name,
        "user_query_judge": row.get("user_query_judge", {}),
        "trajectory_judge": row.get("trajectory_judge", {}),
        "pattern": row.get("pattern", ""),
    }


def save_for_nemo_gym(
    df: pd.DataFrame,
    output_path: str,
    convert_fn: Callable[[dict[str, Any], int], dict[str, Any]],
) -> None:
    """Save records as JSONL for NeMo Gym."""
    with open(output_path, "w") as f:
        for idx, row in df.iterrows():
            record = convert_fn(row.to_dict(), idx)
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(df)} examples to {output_path}")
