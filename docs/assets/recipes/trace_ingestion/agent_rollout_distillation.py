# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "pydantic",
# ]
# ///
"""Agent Rollout Plan Generation Recipe

Read agent rollout traces from disk and generate agentic workflow plans —
structured reasoning strategies that would lead a coding assistant to the
same successful outcome observed in the trace.

This recipe demonstrates:
    - ingesting built-in agent rollout formats with `AgentRolloutSeedSource`
    - distilling traces into compact task digests
    - generating agentic workflow plans (task understanding, approach, decision points, verification)
    - scoring each plan for quality with an LLM judge
    - flattening the result into `plan_instruction` / `plan_response` SFT columns

Prerequisites:
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases (default model alias is "nvidia-super").
    - Agent rollout files for one of the built-in formats. `atif` expects standalone JSON trajectory files and
      requires `--trace-dir`. `claude_code`, `codex`, and `hermes_agent` can use their default locations when
      `--trace-dir` is omitted.

Run:
    uv run agent_rollout_distillation.py --format atif --trace-dir ./atif_traces
    uv run agent_rollout_distillation.py --format claude_code
    uv run agent_rollout_distillation.py --format codex --shuffle --num-records 20
    uv run agent_rollout_distillation.py --format hermes_agent --num-records 20
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


class DecisionPoint(BaseModel):
    question: str = Field(..., description="A concrete decision the agent must make during execution.")
    recommendation: str = Field(..., description="The recommended choice, grounded in what the trace shows worked.")


class VerificationStep(BaseModel):
    check: str = Field(..., description="What to verify after execution.")
    expected: str = Field(..., description="What a successful result looks like.")


class AgentWorkflowPlan(BaseModel):
    task_description: str = Field(
        ...,
        description="A standalone description of the task, understandable without seeing the original trace.",
    )
    task_understanding: str = Field(
        ...,
        description="1-3 sentence analysis of what the user needs and why, including key constraints or context.",
    )
    approach: list[str] = Field(
        ...,
        min_length=2,
        max_length=8,
        description=(
            "Ordered steps the agent should take. Each step should name the action, the target "
            "(file, tool, API), and what information it produces for subsequent steps."
        ),
    )
    decision_points: list[DecisionPoint] = Field(
        ...,
        min_length=0,
        max_length=4,
        description="Key decisions the agent faces during execution. Omit if the task is straightforward.",
    )
    verification: list[VerificationStep] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="How to confirm the task was completed successfully.",
    )
    skill_tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Short tags describing the skills exercised in this workflow.",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Approximate difficulty of the task.",
    )


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


PLAN_SYSTEM_PROMPT = """\
You generate agentic workflow plans from real coding-assistant trace digests.
A workflow plan captures the reasoning strategy, tool-use patterns, decision points, and verification
steps that would lead an agent to the successful outcome observed in the trace.
The trace digest is authoritative — do not invent file paths, commands, APIs, or details not supported by it.
"""


PLAN_PROMPT = """\
Generate an agentic workflow plan from this trace digest. The plan should describe the strategy
an agent should follow to achieve the same outcome, written as if the agent is about to start the task.

<trace_digest>
{{ trace_digest }}
</trace_digest>

Requirements:
- `task_description` must be standalone — a reader should understand the task without seeing the trace.
- `task_understanding` should analyze the user's need, key constraints, and relevant context.
- `approach` steps should be concrete and ordered: name the action, the target (file, tool, API), and
  what information the step produces. Avoid vague steps like "understand the codebase".
- `decision_points` should capture real choices the agent faces. Omit this if the task is linear.
- `verification` steps should describe observable checks, not aspirational goals.
- Do not fabricate file paths, commands, config keys, or API details not justified by the digest.
- If the digest describes a partial or failed trace, write the plan that would lead to success.
- Do not mention the trace, session, or that this was derived from prior work.
"""


PLAN_JUDGE_SYSTEM_PROMPT = """\
You evaluate agentic workflow plans for coding assistants.
Use the trace digest as the source of truth. A good plan is one that an agent could follow to
reach the outcome described in the digest, using concrete and faithful steps.
"""


PLAN_JUDGE_PROMPT = """\
Evaluate this agentic workflow plan for a coding assistant.

Trace digest:
{{ trace_digest }}

Task description:
{{ workflow_plan.task_description }}

Task understanding:
{{ workflow_plan.task_understanding }}

Approach:
{{ workflow_plan.approach }}

Decision points:
{{ workflow_plan.decision_points }}

Verification:
{{ workflow_plan.verification }}

Hard rules:
- Penalize invented file paths, commands, config keys, APIs, or implementation details not justified by the digest.
- Penalize vague steps that don't name a concrete action or target.
- Reward plans where following the steps would plausibly reach the digest's outcome.
"""


PLAN_JUDGE_SCORES = [
    dd.Score(
        name="actionability",
        description="Could an agent follow this plan step-by-step to make progress on the task?",
        options={
            4: "Every step names a concrete action and target; an agent could execute immediately.",
            3: "Most steps are concrete with minor vagueness in one or two.",
            2: "Several steps are too vague to act on without further clarification.",
            1: "Mostly aspirational; an agent would need to re-plan before acting.",
            0: "Not actionable.",
        },
    ),
    dd.Score(
        name="completeness",
        description="Does the plan cover the full path from task start to verified completion?",
        options={
            4: "Covers investigation, implementation, and verification with no major gaps.",
            3: "Covers the main path with minor omissions.",
            2: "Missing a significant phase (e.g., no verification, no investigation).",
            1: "Covers only a fragment of the task.",
            0: "Incomplete or empty plan.",
        },
    ),
    dd.Score(
        name="faithfulness",
        description="Does the plan avoid inventing details not supported by the trace digest?",
        options={
            4: "Faithful to the digest; no unsupported details.",
            3: "Mostly faithful with minor speculative details.",
            2: "Noticeable invented details or overconfident extrapolation.",
            1: "Many unsupported implementation details fabricated.",
            0: "Severely unfaithful to the digest.",
        },
    ),
    dd.Score(
        name="reasoning_quality",
        description="Does the plan show good task understanding, sensible ordering, and awareness of decision points?",
        options={
            4: "Strong reasoning: correct task analysis, logical ordering, decision points identified.",
            3: "Reasonable reasoning with minor ordering or analysis issues.",
            2: "Shallow reasoning; steps could be in any order or key decisions are missed.",
            1: "Poor reasoning; the plan doesn't reflect understanding of the task.",
            0: "No meaningful reasoning.",
        },
    ),
    dd.Score(
        name="training_utility",
        description="Would this plan be valuable as an SFT example teaching an agent how to approach coding tasks?",
        options={
            4: "Excellent SFT example — teaches a reusable reasoning pattern.",
            3: "Good SFT example with minor limitations.",
            2: "Marginal; too task-specific or shallow to generalize.",
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
            name="workflow_plan",
            model_alias=model_alias,
            output_format=AgentWorkflowPlan,
            system_prompt=PLAN_SYSTEM_PROMPT,
            prompt=PLAN_PROMPT,
        )
    )
    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="plan_judge_result",
            model_alias=model_alias,
            system_prompt=PLAN_JUDGE_SYSTEM_PROMPT,
            prompt=PLAN_JUDGE_PROMPT,
            scores=PLAN_JUDGE_SCORES,
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="plan_instruction",
            expr="{{ workflow_plan.task_description }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="plan_response",
            expr=(
                "## Task Understanding\n{{ workflow_plan.task_understanding }}\n\n"
                "## Approach\n{% for step in workflow_plan.approach %}{{ loop.index }}. {{ step }}\n{% endfor %}\n"
                "{% if workflow_plan.decision_points %}"
                "## Decision Points\n{% for dp in workflow_plan.decision_points %}"
                "- **{{ dp.question }}** — {{ dp.recommendation }}\n{% endfor %}\n"
                "{% endif %}"
                "## Verification\n{% for v in workflow_plan.verification %}"
                "- {{ v.check }}: {{ v.expected }}\n{% endfor %}"
            ),
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="plan_skill_tags",
            expr="{{ workflow_plan.skill_tags }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="actionability_score",
            expr="{{ plan_judge_result.actionability.score if plan_judge_result.actionability.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="completeness_score",
            expr="{{ plan_judge_result.completeness.score if plan_judge_result.completeness.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="faithfulness_score",
            expr="{{ plan_judge_result.faithfulness.score if plan_judge_result.faithfulness.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="reasoning_quality_score",
            expr="{{ plan_judge_result.reasoning_quality.score if plan_judge_result.reasoning_quality.score is not none else 0 }}",
            dtype="int",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="training_utility_score",
            expr="{{ plan_judge_result.training_utility.score if plan_judge_result.training_utility.score is not none else 0 }}",
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
            name="recommended",
            expr=(
                "{{ "
                "actionability_score >= 3 and "
                "completeness_score >= 3 and "
                "faithfulness_score >= 3 and "
                "reasoning_quality_score >= 3 and "
                "training_utility_score >= 3 and "
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
    dataset_name: str = "agent_rollout_workflow_plans",
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
            "Optional directory containing rollout trace files. `atif` expects standalone JSON trajectory files "
            "and requires `--trace-dir`. When omitted, `claude_code` defaults to ~/.claude/projects, "
            "`codex` defaults to ~/.codex/sessions, and `hermes_agent` defaults to ~/.hermes/sessions."
        ),
    )
    parser.add_argument("--model-alias", type=str, default="nvidia-super")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="agent_rollout_workflow_plans")
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
    if rollout_format == dd.AgentRolloutFormat.ATIF and trace_dir is None:
        raise ValueError("--trace-dir is required when --format atif.")
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
