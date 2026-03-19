# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "pydantic",
# ]
# ///
"""Agent Rollout Trace Distillation Recipe

Read agent rollout traces from disk and turn them into a practical
supervised fine-tuning dataset for coding assistants.

This recipe demonstrates:
    - ingesting built-in agent rollout formats with `AgentRolloutSeedSource`
    - distilling traces into compact task digests
    - generating standalone instruction-response training examples
    - scoring each candidate for SFT utility with an LLM judge
    - flattening the result into convenient `sft_instruction` / `sft_response` columns

Prerequisites:
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases (default model alias is "nvidia-super").
    - Agent rollout JSONL files for one of the built-in formats. `claude_code` and `codex` can use their
      default locations when `--trace-dir` is omitted.

Run:
    uv run agent_rollout_distillation.py --format claude_code
    uv run agent_rollout_distillation.py --format codex --shuffle --num-records 20
    uv run agent_rollout_distillation.py --format claude_code --num-records 32 --preview
    uv run agent_rollout_distillation.py --format codex --partition-index 0 --num-partitions 8
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults
from data_designer.interface import DataDesigner, DatasetCreationResults


class AgentRolloutTraceDigest(BaseModel):
    user_goal: str = Field(..., description="Standalone summary of the concrete user or delegated agent task.")
    repository_context: str = Field(
        ...,
        description="The repo, codebase, or environment context that materially shaped the task.",
    )
    task_type: str = Field(..., description="Short label for the kind of work in the trace.")
    notable_actions: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Most important assistant actions, tools, or repo operations from the trace.",
    )
    useful_outcome: str = Field(
        ...,
        description="The most useful result, conclusion, or next-step learned from the trace.",
    )
    training_value: Literal["high", "medium", "low"] = Field(
        ...,
        description="Assessment of whether this trace is a good source for assistant fine-tuning.",
    )
    quality_notes: str = Field(
        ...,
        description="Short note about anything that makes the trace especially useful, narrow, noisy, or partial.",
    )


class AgentRolloutFinetuningRecord(BaseModel):
    instruction: str = Field(
        ...,
        description="A standalone user request suitable for supervised fine-tuning of a coding assistant.",
    )
    response: str = Field(
        ...,
        description="A grounded assistant response that helps with the instruction without inventing unsupported details.",
    )
    skill_tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Short tags describing the skills demonstrated in the example.",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Approximate difficulty of the resulting training example.",
    )


AgentRolloutTraceDigest.model_rebuild()
AgentRolloutFinetuningRecord.model_rebuild()


TRACE_DIGEST_SYSTEM_PROMPT = """\
You are curating real coding-assistant traces into training data for supervised fine-tuning.
Extract the practical substance of the task without copying long code blocks, logs, or markdown verbatim.
Prefer concrete repo work over generic chatter. If the trace is a sidechain, capture the delegated subtask accurately.
"""


TRACE_DIGEST_PROMPT = """\
Create a compact trace digest from this agent rollout seed row.

<trace_metadata>
trace_id: {{ trace_id }}
source_kind: {{ source_kind }}
root_session_id: {{ root_session_id }}
agent_id: {{ agent_id }}
is_sidechain: {{ is_sidechain }}
project_path: {{ project_path }}
cwd: {{ cwd }}
git_branch: {{ git_branch }}
message_count: {{ message_count }}
tool_call_count: {{ tool_call_count }}
source_meta: {{ source_meta }}
</trace_metadata>

<trace_opening_messages>
{{ messages[:4] }}
</trace_opening_messages>

<trace_closing_messages>
{{ messages[-4:] }}
</trace_closing_messages>

<final_assistant_message>
{{ final_assistant_message }}
</final_assistant_message>

Requirements:
- Summarize; do not paste long code, logs, or markdown sections.
- Focus on the actual task, the repo context, the key actions, and the useful outcome.
- Mark `training_value` as `high` only when the trace teaches a concrete, reusable assistant behavior.
- Use `medium` when the trace is somewhat useful but noisy or partial.
- Use `low` when the trace is mostly bookkeeping, suggestion-mode filler, or too trace-specific to teach well.
"""


SFT_RECORD_SYSTEM_PROMPT = """\
You create high-quality supervised fine-tuning examples for coding assistants.
Produce standalone instruction-response pairs that teach useful technical behavior.
The trace digest is authoritative. Do not invent file paths, commands, config keys, package names, APIs, or code that are not clearly supported by it.
If the digest suggests there was a strong implementation example but does not provide its exact contents, give grounded guidance and structure rather than fabricated snippets.
Prefer plain-language implementation guidance over code blocks, config fragments, or shell commands.
"""


SFT_RECORD_PROMPT = """\
Transform this trace digest into one strong supervised fine-tuning example for a coding assistant.

<trace_digest>
{{ trace_digest }}
</trace_digest>

Requirements:
- The instruction must be self-contained and realistic.
- Do not mention the trace, session, seed row, or that this was distilled from prior work.
- Preserve repo context only when it materially helps the task.
- The response should answer the instruction as a strong assistant would, not narrate what happened in the trace.
- Prefer actionable technical help over retrospective summaries.
- Avoid placeholders like TODO, <path>, or "I would".
- If the original trace was partial or blocked, write the best next-step assistant response to move the task forward.
- Do not fabricate commands, file paths, config blocks, code, package names, or API names unless they are explicitly justified by the digest.
- If the digest only supports high-level guidance, return a high-level answer with concrete checks, structure, and cautions rather than made-up implementation details.
- Prefer short numbered or bulleted steps in plain language. Avoid code fences and command examples unless the digest explicitly contains those exact details.
- If the digest mentions that a preview or validation run happened but does not provide the exact invocation, describe that step generically instead of inventing the command.
- Keep the response concise and high-signal, ideally under 220 words.
"""


SFT_JUDGE_SYSTEM_PROMPT = """\
You are a strict curator for coding-assistant supervised fine-tuning data.
Use the trace digest as the source of truth and score whether the candidate example is worth keeping.
Invented implementation details are a serious defect. If the response fabricates commands, code, config keys, file names, APIs, or package details not supported by the digest, score it harshly.
"""


SFT_JUDGE_PROMPT = """\
Evaluate this candidate supervised fine-tuning example for a coding assistant.

Trace digest:
{{ trace_digest }}

Candidate instruction:
{{ sft_record.instruction }}

Candidate response:
{{ sft_record.response }}

Hard rules:
- Penalize invented commands, code, config keys, file names, APIs, or package details that are not explicitly justified by the digest.
- Prefer grounded advisory answers over fabricated implementation snippets.
"""


SFT_JUDGE_SCORES = [
    dd.Score(
        name="groundedness",
        description="Is the candidate example clearly grounded in the trace digest rather than generic filler?",
        options={
            4: "Strongly grounded in the trace digest with concrete task fidelity.",
            3: "Mostly grounded but slightly generic or overgeneralized.",
            2: "Partially grounded but missing important trace-specific substance.",
            1: "Weakly grounded and mostly generic.",
            0: "Not grounded in the trace digest.",
        },
    ),
    dd.Score(
        name="standalone_task",
        description="Would a new reader understand the instruction without seeing the underlying trace?",
        options={
            4: "Fully standalone and immediately understandable.",
            3: "Mostly standalone with minor missing context.",
            2: "Understandable but noticeably dependent on hidden trace context.",
            1: "Hard to understand without the trace.",
            0: "Not standalone.",
        },
    ),
    dd.Score(
        name="response_quality",
        description="How helpful, technically specific, and instruction-following is the assistant response?",
        options={
            4: "Highly useful, technically specific, and directly responsive.",
            3: "Useful overall with minor omissions or verbosity.",
            2: "Partially helpful but shallow, vague, or uneven.",
            1: "Low-quality response with major gaps.",
            0: "Unhelpful or incorrect response.",
        },
    ),
    dd.Score(
        name="faithfulness",
        description="Does the candidate avoid inventing unsupported implementation details beyond what the trace digest justifies?",
        options={
            4: "Faithful to the digest; no meaningful unsupported details are invented.",
            3: "Mostly faithful with minor speculative details.",
            2: "Noticeable invented details or overconfident extrapolation.",
            1: "Many unsupported implementation details are fabricated.",
            0: "Severely unfaithful to the digest.",
        },
    ),
    dd.Score(
        name="training_utility",
        description="Would this example be worth keeping in an SFT dataset for a coding assistant?",
        options={
            4: "Very strong SFT example worth keeping.",
            3: "Reasonably useful SFT example.",
            2: "Marginal example; probably not worth the tokens.",
            1: "Poor SFT example.",
            0: "Should not be kept.",
        },
    ),
]


MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"


def build_config(
    trace_dir: Path | None,
    rollout_format: dd.AgentRolloutFormat,
    model_alias: str,
    *,
    sampling_strategy: dd.SamplingStrategy,
    selection_strategy: dd.PartitionBlock | None,
) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.add_model_config(
        dd.ModelConfig(
            alias=model_alias,
            model=MODEL_NAME,
            provider="nvidia",
        )
    )
    seed_source = build_seed_source(trace_dir=trace_dir, rollout_format=rollout_format)
    config_builder.with_seed_dataset(
        seed_source, sampling_strategy=sampling_strategy, selection_strategy=selection_strategy
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="trace_digest",
            model_alias=model_alias,
            output_format=AgentRolloutTraceDigest,
            system_prompt=TRACE_DIGEST_SYSTEM_PROMPT,
            prompt=TRACE_DIGEST_PROMPT,
        )
    )
    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="sft_record",
            model_alias=model_alias,
            output_format=AgentRolloutFinetuningRecord,
            system_prompt=SFT_RECORD_SYSTEM_PROMPT,
            prompt=SFT_RECORD_PROMPT,
        )
    )
    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="sft_quality_judge_result",
            model_alias=model_alias,
            system_prompt=SFT_JUDGE_SYSTEM_PROMPT,
            prompt=SFT_JUDGE_PROMPT,
            scores=SFT_JUDGE_SCORES,
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="sft_instruction",
            expr="{{ sft_record.instruction }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="sft_response",
            expr="{{ sft_record.response }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="sft_skill_tags",
            expr="{{ sft_record.skill_tags }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="groundedness_score",
            expr="{{ sft_quality_judge_result.groundedness.score if sft_quality_judge_result.groundedness.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="standalone_task_score",
            expr="{{ sft_quality_judge_result.standalone_task.score if sft_quality_judge_result.standalone_task.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="response_quality_score",
            expr="{{ sft_quality_judge_result.response_quality.score if sft_quality_judge_result.response_quality.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="faithfulness_score",
            expr="{{ sft_quality_judge_result.faithfulness.score if sft_quality_judge_result.faithfulness.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="training_utility_score",
            expr="{{ sft_quality_judge_result.training_utility.score if sft_quality_judge_result.training_utility.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="trace_training_value",
            expr="{{ trace_digest.training_value }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="recommended_for_sft",
            expr=(
                "{{ "
                "groundedness_score >= 4 and "
                "standalone_task_score >= 4 and "
                "response_quality_score >= 4 and "
                "faithfulness_score >= 4 and "
                "training_utility_score >= 4 and "
                "trace_training_value == 'high' "
                "}}"
            ),
            dtype="bool",
        )
    )

    return config_builder


def run_recipe(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    num_records: int,
    artifact_path: Path | str | None = None,
    dataset_name: str = "agent_rollout_trace_workflows",
    preview: bool = False,
) -> DatasetCreationResults | PreviewResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    if preview:
        return data_designer.preview(config_builder, num_records=num_records)
    return data_designer.create(config_builder, num_records=num_records, dataset_name=dataset_name)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=[rollout_format.value for rollout_format in dd.AgentRolloutFormat],
        help="Built-in rollout format to read.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing rollout JSONL files. When omitted, `claude_code` defaults to "
            "~/.claude/projects and `codex` defaults to ~/.codex/sessions."
        ),
    )
    parser.add_argument("--model-alias", type=str, default="nvidia-super")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="agent_rollout_trace_workflows")
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


def build_seed_source(
    trace_dir: Path | None,
    rollout_format: dd.AgentRolloutFormat,
) -> dd.AgentRolloutSeedSource:
    seed_source_kwargs: dict[str, str | dd.AgentRolloutFormat] = {"format": rollout_format}
    if trace_dir is not None:
        seed_source_kwargs["path"] = str(trace_dir)
    return dd.AgentRolloutSeedSource(**seed_source_kwargs)


def main() -> None:
    args = build_arg_parser().parse_args()
    rollout_format = dd.AgentRolloutFormat(args.format)
    trace_dir = args.trace_dir.expanduser().resolve() if args.trace_dir is not None else None
    sampling_strategy = dd.SamplingStrategy.SHUFFLE if args.shuffle else dd.SamplingStrategy.ORDERED
    selection_strategy = resolve_selection_strategy(args.partition_index, args.num_partitions)

    config_builder = build_config(
        trace_dir=trace_dir,
        rollout_format=rollout_format,
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
