# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "pydantic",
# ]
# ///
"""Claude Code Trace Distillation Recipe

Read Claude Code session traces from disk and turn them into a structured dataset
of reusable workflow records.

This recipe demonstrates:
    - ingesting Claude Code sessions with `TraceSeedSource`
    - using normalized trace metadata as seed columns in a workflow
    - conditioning structured generation on imported `messages`

Prerequisites:
    - OPENAI_API_KEY environment variable for OpenAI provider model aliases (default model alias is "openai-text").
    - A directory containing Claude Code session JSONL files, such as a project folder under `~/.claude/projects/`.

Run:
    uv run claude_code_sessions.py --trace-dir ~/.claude/projects/my-project
    uv run claude_code_sessions.py --trace-dir ~/.claude/projects/my-project --shuffle --num-records 20
    uv run claude_code_sessions.py --trace-dir ~/.claude/projects/my-project --num-records 32 --preview
    uv run claude_code_sessions.py --trace-dir ~/.claude/projects/my-project --partition-index 0 --num-partitions 8
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults
from data_designer.interface import DataDesigner, DatasetCreationResults


class ClaudeCodeWorkflowRecord(BaseModel):
    user_goal: str = Field(..., description="Concise description of the user objective in the trace.")
    repository_context: str = Field(
        ...,
        description="Relevant repo or environment context that shaped the work in this session.",
    )
    work_completed: str = Field(..., description="Summary of what the assistant actually did or attempted.")
    notable_actions: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Important tools, actions, or substeps that defined the workflow.",
    )
    recommended_follow_up: str = Field(
        ...,
        description="A realistic next task that would continue this workflow productively.",
    )


def build_config(
    trace_dir: Path,
    model_alias: str,
    *,
    sampling_strategy: dd.SamplingStrategy,
    selection_strategy: dd.PartitionBlock | None,
) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(
        dd.TraceSeedSource(
            path=str(trace_dir),
            format=dd.TraceSeedFormat.CLAUDE_CODE_DIR,
        ),
        sampling_strategy=sampling_strategy,
        selection_strategy=selection_strategy,
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="workflow_record",
            model_alias=model_alias,
            output_format=ClaudeCodeWorkflowRecord,
            prompt=(
                "You are distilling Claude Code sessions into reusable workflow records for synthetic data generation.\n\n"
                "<trace_metadata>\n"
                "trace_id: {{ trace_id }}\n"
                "root_session_id: {{ root_session_id }}\n"
                "agent_id: {{ agent_id }}\n"
                "is_sidechain: {{ is_sidechain }}\n"
                "project_path: {{ project_path }}\n"
                "cwd: {{ cwd }}\n"
                "git_branch: {{ git_branch }}\n"
                "message_count: {{ message_count }}\n"
                "tool_call_count: {{ tool_call_count }}\n"
                "</trace_metadata>\n\n"
                "<final_assistant_message>\n"
                "{{ final_assistant_message }}\n"
                "</final_assistant_message>\n\n"
                "<messages>\n"
                "{{ messages }}\n"
                "</messages>\n\n"
                "Return a concise, high-signal workflow record.\n"
                "Focus on:\n"
                "- what the user wanted\n"
                "- what the assistant did in the repo\n"
                "- the most important actions or tools used\n"
                "- one strong next follow-up task\n"
                "If this is a subagent trace, reflect that in the summary rather than pretending it was the root task."
            ),
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="assistant_outcome",
            expr="{{ final_assistant_message }}",
        )
    )

    return config_builder


def run_recipe(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    num_records: int,
    artifact_path: Path | str | None = None,
    dataset_name: str = "claude_code_trace_workflows",
    preview: bool = False,
) -> DatasetCreationResults | PreviewResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    if preview:
        return data_designer.preview(config_builder, num_records=num_records)
    return data_designer.create(config_builder, num_records=num_records, dataset_name=dataset_name)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Path to a Claude Code project session directory containing JSONL session traces.",
    )
    parser.add_argument("--model-alias", type=str, default="openai-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="claude_code_trace_workflows")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run the recipe in preview mode and keep the generated dataset in memory.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the normalized trace rows before sampling.",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=None,
        help="Optional partition index for large trace corpora.",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=None,
        help="Optional total number of partitions for large trace corpora.",
    )
    return parser


def resolve_selection_strategy(
    partition_index: int | None,
    num_partitions: int | None,
) -> dd.PartitionBlock | None:
    if partition_index is None and num_partitions is None:
        return None
    if partition_index is None or num_partitions is None:
        raise ValueError("--partition-index and --num-partitions must be provided together.")
    return dd.PartitionBlock(index=partition_index, num_partitions=num_partitions)


def main() -> None:
    args = parse_args().parse_args()
    trace_dir = args.trace_dir.expanduser().resolve()
    sampling_strategy = dd.SamplingStrategy.SHUFFLE if args.shuffle else dd.SamplingStrategy.ORDERED
    selection_strategy = resolve_selection_strategy(args.partition_index, args.num_partitions)

    config_builder = build_config(
        trace_dir=trace_dir,
        model_alias=args.model_alias,
        sampling_strategy=sampling_strategy,
        selection_strategy=selection_strategy,
    )
    results = run_recipe(
        config_builder,
        num_records=args.num_records,
        artifact_path=args.artifact_path,
        dataset_name=args.dataset_name,
        preview=args.preview,
    )

    if args.preview:
        print(f"Preview generated {len(results.dataset)} rows in memory.")
    else:
        print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    results.display_sample_record()


if __name__ == "__main__":
    main()
