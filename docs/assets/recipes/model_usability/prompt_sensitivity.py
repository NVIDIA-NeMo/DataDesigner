# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "pandas",
# ]
# ///
"""Nemotron Nano Prompt Sensitivity Recipe: Regex-Verified Preamble Generation

Generate diverse prompt preambles for RL training that pair natural-language
instructions with regex-verified output format specifications.  Each record
contains a paraphrased preamble, a format instruction aligned to a regex
pattern, and a composed user prompt -- all scored by four LLM judges.

This recipe implements the preamble generation stage of the prompt sensitivity
pipeline used for Nemotron Nano training.  The key idea: seed the pipeline with
10 regex-based answer formats (boxed, double-parens, angle brackets, XML tags,
etc.), then use an LLM to paraphrase both the instruction preamble and the
format specification while preserving the regex contract.  Placement order
variants control where the preamble, format instruction, and {problem}
placeholder appear in the final prompt.

Pipeline architecture:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │               SEED DATA: 10 regex formats x 30 preambles              │
    │  Each row: format_key, output_regex, seed_preamble,                   │
    │            seed_format_instruction                                     │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 1: DIVERSITY SAMPLERS (7 columns)                  │
    │  sentence_length, sentence_type, tone, strictness_level,              │
    │  verbosity_level, domain_context, preamble_format_order               │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 2: PREAMBLE GENERATION (LLM)                       │
    │  Paraphrase seed preamble respecting sampled controls.                │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 3: FORMAT INSTRUCTION GENERATION (LLM)             │
    │  Paraphrase format instruction preserving regex intent.               │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 4: USER PROMPT COMPOSITION (LLM)                   │
    │  Assemble preamble + format instruction + {problem} placeholder.      │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 5: QUALITY SCORING (4 LLM judges)                  │
    │  format_compliance (0-2), regex_alignment (0-1),                      │
    │  order_coherence (0-1), preamble_quality (0-3)                        │
    └────────────────────────────────────────────────────────────────────────┘

Prerequisites:
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases.

Run:
    uv run prompt_sensitivity.py

    uv run prompt_sensitivity.py --num-records 200

    uv run prompt_sensitivity.py --help
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

# =============================================================================
# Seed data: regex format templates + seed preambles
# =============================================================================

FORMAT_TEMPLATES = [
    {
        "format_key": "fmt_00",
        "output_regex": r"\boxed{([.*?])}",
        "seed_format_instruction": "Make sure to put the answer (and only answer) inside \\boxed{}.",
    },
    {
        "format_key": "fmt_01",
        "output_regex": r"\(\((.*?)\)\)",
        "seed_format_instruction": "Your final answer (and only the answer) should be enclosed within double parentheses.",
    },
    {
        "format_key": "fmt_02",
        "output_regex": r"Answer is\s*\[([A-Za-z])\]",
        "seed_format_instruction": "Wrap answer in square brackets at the end: 'Answer is [X]'.",
    },
    {
        "format_key": "fmt_03",
        "output_regex": r"\(Answer:\s*([A-Za-z])\)",
        "seed_format_instruction": "Conclude with (Answer: X), where X is the final answer.",
    },
    {
        "format_key": "fmt_04",
        "output_regex": r"Final Answer:\s*\|\|(.*?)\|\|",
        "seed_format_instruction": "End your response with Final Answer: ||X||, where X is the final answer (and only the answer).",
    },
    {
        "format_key": "fmt_05",
        "output_regex": r"\[Answer:\s*([A-Za-z])\)",
        "seed_format_instruction": "Remember to end with [Answer: X], where X is the final answer.",
    },
    {
        "format_key": "fmt_06",
        "output_regex": r"<<(.*?)>>",
        "seed_format_instruction": "Put your final answer in double angle brackets.",
    },
    {
        "format_key": "fmt_07",
        "output_regex": r"\s*\*\*(.*?)\*\*",
        "seed_format_instruction": "Give the answer at the end in this format -> **X**, where X is final answer.",
    },
    {
        "format_key": "fmt_08",
        "output_regex": r"<final_answer>\s*([.*?])\s*</final_answer>",
        "seed_format_instruction": "Wrap your final answer in XML-style tags like this: <final_answer>X</final_answer>.",
    },
    {
        "format_key": "fmt_09",
        "output_regex": r"\boxed{([.*?])}",
        "seed_format_instruction": "The final answer must be in \\boxed{} format. It's crucial to follow this format.",
    },
]

SEED_PREAMBLES = [
    "Solve the following problem step by step.",
    "Provide a comprehensive solution to the problem below.",
    "Think carefully and solve the following.",
    "Analyze the problem and provide your answer.",
    "Work through the following problem methodically.",
    "Read the problem carefully and provide a detailed solution.",
    "Consider all aspects of the problem before answering.",
    "Break down the problem into steps and solve it.",
    "Explain your reasoning as you solve the following.",
    "Solve the problem, then provide the answer.",
    "Provide your reasoning for the answer and state the final result.",
    "Think step-by-step before giving your final answer.",
    "Carefully analyze the following and provide your solution.",
    "Show your work as you solve the following problem.",
    "Reason through the problem below and give your answer.",
    "Solve the following. Show your reasoning.",
    "Provide a clear and complete solution.",
    "Answer the following question with full explanation.",
    "Walk through the solution step by step.",
    "Read the following problem and solve it completely.",
    "Give a thorough answer to the problem below.",
    "Determine the answer to the following problem.",
    "Present your solution with supporting reasoning.",
    "Evaluate the following and provide your answer.",
    "Think about the problem below and respond with your solution.",
    "Provide a well-reasoned answer to the following.",
    "Solve this problem. Explain each step.",
    "Work out the answer to the following question.",
    "Address the problem below with a complete solution.",
    "Respond to the following with a clear answer.",
]

# =============================================================================
# Placement order variants
# =============================================================================

PLACEMENT_ORDERS = [
    "P + F + {problem}",
    "F + P + {problem}",
    "P + {problem} + F",
    "F + {problem} + P",
    "{problem} + P + F",
    "{problem} + F + P",
    "PF + {problem}",
    "{problem} + PF",
]

# =============================================================================
# LLM prompts
# =============================================================================

PREAMBLE_GEN_PROMPT = """\
You are rewriting a seed preamble for an open-ended question prompt.

Seed preamble: {{ seed_preamble }}

Constraints:
- Sentence length: {{ sentence_length }}
- Sentence type: {{ sentence_type }}
- Tone: {{ tone }}
- Strictness: {{ strictness_level }}
- Verbosity: {{ verbosity_level }}
- Domain: {{ domain_context }}

Instructions:
- Paraphrase the seed preamble (do NOT copy it verbatim).
- Produce a concise instruction line for generic open-ended questions.
- Keep it neutral and generic; do NOT include output formatting requirements.
- Respect the constraints above (length, type, tone, strictness, verbosity).
- Output ONLY the rewritten preamble, nothing else.
"""

FORMAT_INSTRUCTION_GEN_PROMPT = """\
You are rewriting a format instruction that tells the user how to present their final answer.

Seed format instruction: {{ seed_format_instruction }}
Output regex pattern: {{ output_regex }}

Constraints:
- Sentence length: {{ sentence_length }}
- Tone: {{ tone }}

Instructions:
- Paraphrase the seed format instruction while preserving its intent.
- The instruction must unambiguously specify how the final answer should be formatted.
- The answer must be required at the end of the response.
- Do NOT refer to the type of answer (sentence, paragraph, math expression).
- Respect the sentence length and tone constraints.
- Output ONLY the rewritten format instruction, nothing else.
"""

USER_PROMPT_COMPOSITION_PROMPT = """\
Compose a final user prompt from the following parts.

Preamble (P): {{ preamble }}
Format instruction (F): {{ format_instruction }}
Placement order: {{ preamble_format_order }}

Instructions:
- Concatenate the parts in the order specified by "Placement order".
- Use {problem} as a literal placeholder for the question text.
- For "PF" or "FP" merged orders, combine P and F into a single natural sentence.
- Ensure {problem} has newlines before and after it for readability.
- Preserve the exact text of P and F; do NOT abbreviate or add new content.
- Output ONLY the composed user prompt, nothing else.
"""

# =============================================================================
# Judge rubrics
# =============================================================================

FORMAT_COMPLIANCE_SCORES = [
    dd.Score(
        name="Format Compliance",
        description="Does the format instruction unambiguously enforce the intended output format and require the answer at end of response?",
        options={
            "2": "Explicit, unambiguous, requires ending with answer in specified format.",
            "1": "Mentions format but leaves room for trailing text after the answer.",
            "0": "Ambiguous, doesn't mention format, or specifies an alternative format.",
        },
    ),
]

REGEX_ALIGNMENT_SCORES = [
    dd.Score(
        name="Regex Alignment",
        description="Does the format instruction semantically and structurally align with the output_regex pattern?",
        options={
            "1": "Instruction matches the regex pattern intent.",
            "0": "Instruction conflicts with or deviates from the regex intent.",
        },
    ),
]

ORDER_COHERENCE_SCORES = [
    dd.Score(
        name="Order Coherence",
        description="Is the composed user prompt coherent with respect to the ordering of preamble, format instruction, and {problem} placeholder?",
        options={
            "1": "Makes sense given the part ordering.",
            "0": "Confusing or contradictory ordering.",
        },
    ),
]

PREAMBLE_QUALITY_SCORES = [
    dd.Score(
        name="Preamble Quality",
        description="Assess the preamble for clarity, concision, generic tone, and adherence to the sampled controls.",
        options={
            "3": "Clear, concise, generic, adheres to all controls.",
            "2": "Good with minor issues in tone or length.",
            "1": "Fair with noticeable issues.",
            "0": "Poor, unclear, or conflicts with controls.",
        },
    ),
]

# =============================================================================
# Judge prompts
# =============================================================================

FORMAT_COMPLIANCE_JUDGE_PROMPT = """\
Evaluate the format instruction for compliance.

Format instruction: {{ format_instruction }}
Output regex: {{ output_regex }}
Seed format instruction: {{ seed_format_instruction }}
"""

REGEX_ALIGNMENT_JUDGE_PROMPT = """\
Evaluate whether the format instruction aligns with the regex pattern.

Format instruction: {{ format_instruction }}
Output regex: {{ output_regex }}
"""

ORDER_COHERENCE_JUDGE_PROMPT = """\
Evaluate whether the composed user prompt is coherent given the placement order.

User prompt: {{ user_prompt }}
Preamble: {{ preamble }}
Format instruction: {{ format_instruction }}
Placement order: {{ preamble_format_order }}
"""

PREAMBLE_QUALITY_JUDGE_PROMPT = """\
Evaluate the preamble for quality.

Preamble: {{ preamble }}
Seed preamble: {{ seed_preamble }}
Sentence length: {{ sentence_length }}
Sentence type: {{ sentence_type }}
Tone: {{ tone }}
Strictness: {{ strictness_level }}
Verbosity: {{ verbosity_level }}
"""


# =============================================================================
# Seed data builder
# =============================================================================


def build_seed_dataframe() -> pd.DataFrame:
    """Build the seed DataFrame as the cross product of formats x preambles."""
    rows = []
    for fmt in FORMAT_TEMPLATES:
        for preamble in SEED_PREAMBLES:
            rows.append(
                {
                    "format_key": fmt["format_key"],
                    "output_regex": fmt["output_regex"],
                    "seed_format_instruction": fmt["seed_format_instruction"],
                    "seed_preamble": preamble,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Pipeline builder
# =============================================================================


def build_config(model_alias: str) -> tuple[dd.DataDesignerConfigBuilder, Path]:
    config_builder = dd.DataDesignerConfigBuilder()

    # ── Seed data ────────────────────────────────────────────────────────
    seed_df = build_seed_dataframe()
    seed_path = Path(tempfile.mkdtemp()) / "prompt_sensitivity_seed.csv"
    seed_df.to_csv(seed_path, index=False)

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=dd.SamplingStrategy.SHUFFLE,
    )

    # ── Stage 1: Diversity samplers ──────────────────────────────────────

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="sentence_length",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["short", "medium"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="sentence_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["imperative", "declarative", "interrogative"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="tone",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["formal", "neutral", "concise", "informal", "strict"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="strictness_level",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["low", "medium", "high"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="verbosity_level",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["concise", "standard"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="domain_context",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["general"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="preamble_format_order",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=PLACEMENT_ORDERS),
        )
    )

    # ── Stage 2: Preamble generation ─────────────────────────────────────

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="preamble",
            prompt=PREAMBLE_GEN_PROMPT,
            model_alias=model_alias,
        )
    )

    # ── Stage 3: Format instruction generation ───────────────────────────

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="format_instruction",
            prompt=FORMAT_INSTRUCTION_GEN_PROMPT,
            model_alias=model_alias,
        )
    )

    # ── Stage 4: User prompt composition ─────────────────────────────────

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="user_prompt",
            prompt=USER_PROMPT_COMPOSITION_PROMPT,
            model_alias=model_alias,
        )
    )

    # ── Stage 5: Quality scoring ─────────────────────────────────────────

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="format_compliance_result",
            prompt=FORMAT_COMPLIANCE_JUDGE_PROMPT,
            scores=FORMAT_COMPLIANCE_SCORES,
            model_alias=model_alias,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="regex_alignment_result",
            prompt=REGEX_ALIGNMENT_JUDGE_PROMPT,
            scores=REGEX_ALIGNMENT_SCORES,
            model_alias=model_alias,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="order_coherence_result",
            prompt=ORDER_COHERENCE_JUDGE_PROMPT,
            scores=ORDER_COHERENCE_SCORES,
            model_alias=model_alias,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="preamble_quality_result",
            prompt=PREAMBLE_QUALITY_JUDGE_PROMPT,
            scores=PREAMBLE_QUALITY_SCORES,
            model_alias=model_alias,
        )
    )

    return config_builder, seed_path


# =============================================================================
# Dataset creation
# =============================================================================


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    results = data_designer.create(config_builder, num_records=num_records)
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Nemotron Nano Prompt Sensitivity Recipe")
    parser.add_argument("--model-alias", type=str, default="nvidia-text")
    parser.add_argument("--num-records", type=int, default=10)
    parser.add_argument("--artifact-path", type=str, default=None)
    args = parser.parse_args()

    config_builder, _seed_path = build_config(model_alias=args.model_alias)
    results = create_dataset(
        config_builder,
        num_records=args.num_records,
        artifact_path=args.artifact_path,
    )

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    results.load_analysis().to_report()
