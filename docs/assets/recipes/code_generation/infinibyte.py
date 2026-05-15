# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "datasets",
#     "pandas",
# ]
# ///
"""Nemotron Nano InfiniByte Recipe: Cross-Source Problem Generation

Generate more diverse and complex training problems by cross-breeding two
source datasets. Each record pairs a "problem A" (e.g. a coding problem) with
a "problem B" (e.g. a math or science problem), then uses an LLM to create new
problems that incorporate concepts from both sources through either obfuscation
(adding plausible but irrelevant complexity) or complication (genuinely
increasing difficulty).

This recipe implements the InfiniByte pipeline used for Nemotron Nano
post-training data. The key idea: rather than generating problems from scratch,
cross-join two existing problem datasets, then augment problem A with concepts
from problem B to produce novel, harder problems.

Pipeline architecture:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │               PRE-PROCESSING (outside Data Designer)                   │
    │  Download 2 HuggingFace datasets, cross-join with random sampling,    │
    │  save as CSV seed file.                                                │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 1: SEED + SAMPLER                                  │
    │  Seed: cross-joined CSV (problem_a, problem_b pairs)                  │
    │  Sampler: combination_type (obfuscation / complication)               │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 2: CANDIDATE GENERATION (LLM Structured)           │
    │  Generate 2 candidate problems augmenting A with concepts from B.     │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 3: BEST PROBLEM SELECTION (LLM Structured)         │
    │  Select the best candidate based on adherence, difficulty, clarity.   │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 4: EVALUATION (LLM Structured)                     │
    │  Score difficulty (1-3), clarity (1-3), adherence (1-3).              │
    ├────────────────────────────────────────────────────────────────────────┤
    │               STAGE 5: SOLUTION GENERATION (LLM Text)                  │
    │  Solve the new problem.                                               │
    └────────────────────────────────────────────────────────────────────────┘

Prerequisites:
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases.
    - Internet access for downloading HuggingFace datasets.

Run:
    # Basic usage (downloads OpenCodeReasoning + OpenMathReasoning, 100 records)
    uv run infinibyte.py

    # Customize dataset strategy and record count
    uv run infinibyte.py --strategy ocr_omr --num-records 500 --limit 10000

    # For help message and available options
    uv run infinibyte.py --help
"""

from __future__ import annotations

import hashlib
import random
import tempfile
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

# =============================================================================
# Pydantic models for structured LLM outputs
# =============================================================================


class NewProblem(BaseModel):
    added_concepts: str = Field(
        ..., description="Specific new concepts incorporated into the original problem"
    )
    problem: str = Field(
        ..., description="A new problem statement incorporating new concepts from the provided problems."
    )


class NewProblemList(BaseModel):
    problems: list[NewProblem] = Field(
        default_factory=list, description="List of new problems"
    )


class NewProblemWithReasoning(NewProblem):
    reasoning: str = Field(
        ..., description="Concise explanation for selecting this specific new problem"
    )


class NewProblemEvals(BaseModel):
    difficulty: int = Field(
        ...,
        description=(
            "Evaluation of problem difficulty relative to original. "
            "3: Significantly more challenging. "
            "2: Moderately more difficult. "
            "1: Similar to or easier than the original."
        ),
    )
    clarity: int = Field(
        ...,
        description=(
            "Evaluation of clarity and coherence. "
            "3: Exceptionally clear and well-structured. "
            "2: Generally clear with minor issues. "
            "1: Unclear or poorly structured."
        ),
    )
    adherence: int = Field(
        ...,
        description=(
            "Evaluation of adherence to the combination goal. "
            "3: Excellent adherence. "
            "2: Adequate adherence. "
            "1: Poor adherence."
        ),
    )


# =============================================================================
# Dataset download and cross-join
# =============================================================================

DATASET_CONFIGS = {
    "ocr": {
        "name": "nvidia/OpenCodeReasoning",
        "config_name": "split_0",
        "input_column": "input",
        "output_column": "problem_a",
        "output_id_column": "id_a",
        "deduplicate": True,
        "max_records": 500,
        "random_seed": 27,
    },
    "omr": {
        "name": "nvidia/OpenMathReasoning",
        "split": "tir",
        "input_column": "problem",
        "output_column": "problem_b",
        "output_id_column": "id_b",
        "deduplicate": True,
        "max_records": 500,
        "random_seed": 7,
    },
}

STRATEGIES = {
    "ocr_omr": ("ocr", "omr"),
}


def fetch_hf_dataset_to_df(dataset_config: dict) -> pd.DataFrame:
    """Load a HuggingFace dataset via streaming into a DataFrame with id and text columns.

    Uses HF streaming to avoid downloading the full dataset, collecting up to
    ``max_records`` deduplicated rows.
    """
    from datasets import load_dataset

    dataset_name = dataset_config["name"]
    config_name = dataset_config.get("config_name")
    split = dataset_config.get("split")
    input_col = dataset_config.get("input_column", "text")
    output_col = dataset_config.get("output_column", input_col)
    output_id_col = dataset_config.get("output_id_column", "id")
    dedupe = dataset_config.get("deduplicate", False)
    max_records = dataset_config.get("max_records", 500)
    random_seed = dataset_config.get("random_seed", 42)

    print(f"\n=== Streaming {dataset_name} (config={config_name}, split={split or 'all'}) ===")

    load_kwargs: dict = {"path": dataset_name, "streaming": True}
    if config_name:
        load_kwargs["name"] = config_name
    if split:
        load_kwargs["split"] = split

    ds = load_dataset(**load_kwargs)

    # For DatasetDict (no split specified), use the first available split
    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        print(f"  Using split: {split_name}")
        ds = ds[split_name]

    # Shuffle the stream for diversity, then collect up to max_records
    ds = ds.shuffle(seed=random_seed)

    rows = []
    seen: set[str] = set()
    scanned = 0
    for rec in ds:
        if input_col not in rec:
            continue
        text = rec[input_col]
        scanned += 1
        if dedupe:
            if text in seen:
                continue
            seen.add(text)
        rec_id = rec.get("id") or hashlib.md5(text.encode("utf-8")).hexdigest()
        rows.append({output_id_col: rec_id, output_col: text})

        if len(rows) >= max_records:
            break

        if scanned % 10_000 == 0:
            print(f"  Scanned {scanned} records, collected {len(rows)}...")

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} rows (scanned={scanned}, deduplicated={dedupe})")

    return df


def cross_join_with_limit(df1: pd.DataFrame, df2: pd.DataFrame, limit: int = 10_000) -> pd.DataFrame:
    """Randomly sample pairs from the cartesian product of two DataFrames."""
    n1, n2 = len(df1), len(df2)
    total = n1 * n2
    actual_limit = min(limit, total)

    print(f"\nCross-joining {n1} x {n2} = {total} possible pairs, sampling {actual_limit}")

    flat_indices = random.sample(range(total), actual_limit)
    idx1 = [k // n2 for k in flat_indices]
    idx2 = [k % n2 for k in flat_indices]

    sub1 = df1.iloc[idx1].reset_index(drop=True)
    sub2 = df2.iloc[idx2].reset_index(drop=True)

    return pd.concat([sub1, sub2], axis=1)


def prepare_seed_data(strategy: str = "ocr_omr", limit: int = 10_000) -> Path:
    """Download datasets, cross-join, and save as CSV. Returns the CSV path."""
    ds_a_key, ds_b_key = STRATEGIES[strategy]

    df_a = fetch_hf_dataset_to_df(DATASET_CONFIGS[ds_a_key])
    df_b = fetch_hf_dataset_to_df(DATASET_CONFIGS[ds_b_key])

    cross_joined = cross_join_with_limit(df_a, df_b, limit=limit)

    seed_path = Path(tempfile.mkdtemp()) / "infinibyte_seed.csv"
    cross_joined.to_csv(seed_path, index=False)
    print(f"\nSeed data saved to: {seed_path} ({len(cross_joined)} rows)")

    return seed_path


# =============================================================================
# LLM prompts
# =============================================================================

PROBLEM_SYSTEM_PROMPT = """\
You are an experienced competitive programmer, well versed in algorithms, \
data structures, mathematics, physics, chemistry, biology and other sciences. \
You excel in crafting problems that combine multiple concepts into a cohesive \
problem statement.
"""

SOLUTION_SYSTEM_PROMPT = """\
You are a helpful and harmless code assistant, well versed in competitive \
coding problems and STEM subjects. You should think step-by-step before \
responding to any instruction.

You must use python programming language when generating code.
You must use the python code block for just the final solution with the \
following format:
```python
# Your final solution goes here
```
"""

CANDIDATE_GENERATION_PROMPT = """\
### Problem A:
{{ problem_a }}

### Problem B:
{{ problem_b }}

Carefully examine problems A and B above. Then formulate TWO new problems by \
augmenting Problem A with concepts from Problem B.

DO NOT REPEAT PROBLEM B VERBATIM WHEN AUGMENTING. INCORPORATE JUST THE \
CONCEPTS FROM IT.

{% if combination_type == 'obfuscation' %}
Focus on obfuscation: Add concepts from Problem B to Problem A in a way that \
makes the new problem seem more complex, but doesn't actually change the \
solution. The added information should appear relevant but be effectively \
irrelevant to solving the core problem. The goal is to create a problem that \
appears more complicated than it actually is.

NEVER DISCLOSE THAT ADDED INFORMATION IS IRRELEVANT OR THAT IT DOESN'T \
AFFECT THE PROBLEM.
IF A QUESTION IS POSED IN PROBLEM A, THAT QUESTION MUST REMAIN THE SAME.
IF INPUT, OUTPUT, AND EXAMPLES ARE PRESENT IN PROBLEM A, INCLUDE THEM IN THE \
FINAL PROBLEM AS WELL.

{% elif combination_type == 'complication' %}
Focus on complication: Integrate concepts from Problem B into Problem A to \
genuinely increase the complexity. The solution should require understanding \
and applying elements from both problems. The new problem should be more \
challenging but still logically coherent and solvable.

GIVEN THAT IT'S A NEW PROBLEM, DO NOT INCLUDE EXEMPLARY INPUT AND OUTPUT FROM \
THE ORIGINAL PROBLEM.
{% endif %}

MAKE SURE TO INCORPORATE CONCEPTS ONLY FROM PROBLEM B.
Your augmented problem should be believable and appear as a natural, cohesive \
question without artificial divisions between the original elements.
A reader should not be able to easily identify which parts came from Problem A \
versus Problem B.

DO NOT USE WORDS "PROBLEM A" OR "PROBLEM B" IN YOUR RESPONSE. INSTEAD, \
PROVIDE A COMPLETE PROBLEM STATEMENT.
"""

BEST_SELECTION_PROMPT = """\
### Original problem:
{{ problem_a }}

Examine candidate problems below which were created with the goal of \
{% if combination_type == 'obfuscation' %}
adding information to make the original problem seem more complex, without \
actually changing the solution. The added information should be effectively \
irrelevant to solving the core problem.
{% elif combination_type == 'complication' %}
genuinely increasing the complexity of the original problem by incorporating \
new concepts that are logically coherent and solvable. Solving the new problem \
should require understanding and applying newly introduced concepts.
{% endif %}

### Candidate problems:
{{ problem_candidates }}

Select the BEST problem based on the following criteria:
1. Goal adherence (1-3): How well does the new problem adhere to the goal of \
{% if combination_type == 'obfuscation' %}
obfuscating the original problem without actually changing the solution
{% elif combination_type == 'complication' %}
increasing the complexity of the original problem by incorporating new \
concepts that are logically coherent and solvable
{% endif %}
2. Difficulty (1-3): How challenging is the problem to answer, compared to the \
original problem.
3. Clarity (1-3): Is the problem clearly formulated and coherent?
"""

EVALUATION_PROMPT = """\
### Original problem:
{{ problem_a }}

### New problem:
{{ new_problem }}

The new problem was created with the goal of \
{% if combination_type == 'obfuscation' %}
adding information to make the original problem seem more complex, without \
actually changing the solution. The added information should be effectively \
irrelevant to solving the core problem.
{% elif combination_type == 'complication' %}
genuinely increasing the complexity of the original problem by incorporating \
new concepts that are logically coherent and solvable. Solving the new problem \
should require understanding and applying newly introduced concepts.
{% endif %}

## Instructions:
1. Carefully examine and compare the new problem to the original problem.
2. Evaluate the new problem on goal adherence, difficulty and clarity.
"""


# =============================================================================
# Pipeline builder
# =============================================================================


def build_config(model_alias: str, seed_path: Path) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    # ── Seed data ────────────────────────────────────────────────────────

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=dd.SamplingStrategy.SHUFFLE,
    )

    # ── Stage 1: Combination type sampler ────────────────────────────────

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="combination_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["obfuscation", "complication"]),
        )
    )

    # ── Stage 2: Candidate problem generation ────────────────────────────

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="problem_candidates",
            prompt=CANDIDATE_GENERATION_PROMPT,
            system_prompt=PROBLEM_SYSTEM_PROMPT,
            output_format=NewProblemList,
            model_alias=model_alias,
        )
    )

    # ── Stage 3: Best problem selection ──────────────────────────────────

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="best_problem_json",
            prompt=BEST_SELECTION_PROMPT,
            system_prompt=PROBLEM_SYSTEM_PROMPT,
            output_format=NewProblemWithReasoning,
            model_alias=model_alias,
        )
    )

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="new_problem",
            expr="{{ best_problem_json.problem }}",
        )
    )

    # ── Stage 4: Evaluation ──────────────────────────────────────────────

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="evals",
            prompt=EVALUATION_PROMPT,
            system_prompt=PROBLEM_SYSTEM_PROMPT,
            output_format=NewProblemEvals,
            model_alias=model_alias,
        )
    )

    # ── Stage 5: Solution generation ─────────────────────────────────────
    # NOTE: The evals column above already contains difficulty, clarity, and
    # adherence scores as structured fields (e.g. {{ evals.difficulty }}).

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="solution",
            prompt="Solve the following problem: {{ new_problem }}",
            system_prompt=SOLUTION_SYSTEM_PROMPT,
            model_alias=model_alias,
        )
    )

    return config_builder


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

    parser = ArgumentParser(description="Nemotron Nano InfiniByte Recipe")
    parser.add_argument("--model-alias", type=str, default="nvidia-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default="ocr_omr",
        choices=list(STRATEGIES.keys()),
        help="Cross-join strategy: which two datasets to combine (default: ocr_omr)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Maximum number of cross-joined pairs to sample (default: 10000)",
    )
    args = parser.parse_args()

    print("=== Step 1: Preparing seed data ===")
    seed_path = prepare_seed_data(strategy=args.strategy, limit=args.limit)

    print("\n=== Step 2: Building pipeline and generating data ===")
    config_builder = build_config(model_alias=args.model_alias, seed_path=seed_path)
    results = create_dataset(
        config_builder,
        num_records=args.num_records,
        artifact_path=args.artifact_path,
    )

    print(f"\nDataset saved to: {results.artifact_storage.final_dataset_path}")
    results.load_analysis().to_report()
