# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Nemotron Nano Structured Data Recipe: Multi-Format Schema Generation

Generate synthetic structured data across multiple output formats (JSON, YAML,
XML, Markdown) with controlled schema complexity, conversational grounding,
and best-of-3 candidate generation.

This recipe implements the pipeline used to produce structured-data SFT records
for Nemotron Nano training. Each record contains a generated schema, a natural
user request, grounding Q&A conversation pairs, and three candidate structured
outputs that conform to the schema.

Pipeline architecture:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   STAGE 1: SAMPLING (DIVERSITY CONTROLS)               │
    │                                                                        │
    │  Format Controls          Schema Controls        Conversation Controls │
    │  ├─ output_format         ├─ schema_rigidity     ├─ num_turns          │
    │  └─ topic (conditional)   ├─ schema_fields_count ├─ tone               │
    │                           ├─ schema_complexity   └─ detail_level       │
    │                           └─ nesting_depth                             │
    ├────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 2: SCHEMA GENERATION (LLM)                     │
    │  Generates format-specific schema from sampled controls + topic.       │
    ├────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 3: USER PROMPT GENERATION (LLM)                │
    │  Natural-language request matching tone and detail level.              │
    ├────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 4: CONVERSATION PAIRS (LLM)                    │
    │  Q&A pairs covering schema facts for grounding.                        │
    ├────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 5: STRUCTURED OUTPUT (LLM, best-of-3)          │
    │  Three candidate structured outputs conforming to the schema.          │
    └────────────────────────────────────────────────────────────────────────┘

Prerequisites:
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases.

Run:
    uv run structured_data.py

    uv run structured_data.py --num-records 100 --output-format json

    uv run structured_data.py --help
"""

from __future__ import annotations

from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

# =============================================================================
# Topics: representative subset of categories and subtopics
# =============================================================================

TOPICS: dict[str, list[str]] = {
    "Leisure Activities": [
        "Outdoor Recreation",
        "Board Games",
        "DIY Crafts",
        "Photography",
        "Gardening",
    ],
    "Daily Life": [
        "Morning Routines",
        "Grocery Shopping",
        "Commuting",
        "Household Chores",
        "Meal Planning",
    ],
    "Education and Learning": [
        "Online Courses",
        "Study Techniques",
        "Language Learning",
        "STEM Education",
        "Library Systems",
    ],
    "Technology and Gadgets": [
        "Smartphones",
        "Smart Home Devices",
        "Wearable Tech",
        "Cloud Computing",
        "Cybersecurity Basics",
    ],
    "Health and Wellness": [
        "Nutrition Planning",
        "Mental Health",
        "Exercise Routines",
        "Sleep Hygiene",
        "Preventive Care",
    ],
    "Finance and Money": [
        "Personal Budgeting",
        "Investment Basics",
        "Tax Preparation",
        "Credit Management",
        "Retirement Planning",
    ],
    "Food and Cooking": [
        "Baking Techniques",
        "Meal Prep",
        "International Cuisines",
        "Dietary Restrictions",
        "Kitchen Equipment",
    ],
    "Travel and Transportation": [
        "Trip Planning",
        "Public Transit",
        "Road Trips",
        "Travel Insurance",
        "Packing Strategies",
    ],
    "Arts and Culture": [
        "Music Theory",
        "Film Analysis",
        "Theater Production",
        "Contemporary Art",
        "Creative Writing",
    ],
    "Work and Careers": [
        "Resume Building",
        "Interview Preparation",
        "Remote Work",
        "Project Management",
        "Career Transitions",
    ],
}

# =============================================================================
# Prompts
# =============================================================================

SCHEMA_GENERATION_PROMPT = """\
Create a schema for a structured object response in the format {{ output_format }}.

Controls:
- Rigidity: {{ schema_rigidity }}
- Top-level properties: {{ schema_fields_count }}
- Complexity: {{ schema_complexity }}
- Nesting depth: {{ nesting_depth }}
- Topic: {{ topic_category }} / {{ topic_subtopic }}

Instructions:
- Output only an object with keys: "name", "schema", and "strict", formatted as \
{{ output_format }}.
- "name" must be appropriate with the Topic: {{ topic_category }} / {{ topic_subtopic }}
- "schema" should be a valid structured schema as specified in {{ output_format }}.
- Use {{ schema_fields_count }} top-level properties, relevant to the topic.
- Include at least one boolean and, if appropriate, one enum.
- All top-level properties must be listed in "required".
- Set "additionalProperties": false at every object level.
- If {{ schema_complexity }} is "complex", make the schema deeply nested: at least two \
levels of nested objects, with at least one object nested three levels deep. Keep nesting \
relevant to the topic.
- If "simple", keep nesting minimal or flat.
- "strict" must be true.

Formatting by output_format:
- "json": Output a valid JSON object, no code fences or comments.
- "yaml": Output a valid YAML object, no code fences or comments.
- "xml": Output a valid XML document with root "root" and child elements "name", "schema", \
and "strict". "schema" can be a string or nested XML.
- "markdown": Output a Markdown code block with the JSON object, using triple backticks and \
"json" as the language, no extra text.

Output only the object in the specified format. No explanations or extra text.
"""

USER_PROMPT_GENERATION = """\
You are a human user asking an AI assistant to produce a structured output. Write a natural, \
concise request that would lead to filling in a schema about {{ topic_category }} / \
{{ topic_subtopic }}.

The request should:
- Sound like something a real person would type or say
- Describe what data they want without exposing the schema itself
- Mention the desired output format: {{ output_format }}
- Match the tone: {{ tone }} and detail level: {{ detail_level }}

Do not include the schema, code fences, or technical formatting. Just the user request.
"""

CONVERSATION_PROMPT = """\
Write a short Q&A conversation about the following topic. Follow the selected JSON Schema \
fields as the underlying facts to cover, but DO NOT output JSON here.

Topic context:
- Category: {{ topic_category }}
- Subtopic: {{ topic_subtopic }}

Constraints:
- Number of Q&A pairs: {{ num_turns }}
- Tone: {{ tone }}
- Detail level: {{ detail_level }}

Write alternating question/answer pairs that make these facts unambiguous for the chosen \
schema: {{ structured_schema }}
Return only a Python list of [question, answer] pairs (no extra text).
"""

STRUCTURED_OUTPUT_PROMPT = """\
You will produce a {{ output_format }} instance that conforms strictly to the following \
schema (no extra keys).

Schema:
{{ structured_schema }}

You are given a Python list of [question, answer] pairs:
{{ conversation_pairs }}

Instructions:
- Derive values only from the answers given.
- Render ONLY the {{ output_format }} instance, with no commentary.
- Formatting rules:
  - If output_format is "json", output a single JSON object (no code fences).
  - If output_format is "yaml", output a YAML mapping (no code fences).
  - If output_format is "xml", output an XML document with root <scene_response>.
  - If output_format is "markdown", output a fenced code block with ```json.
- Ensure the content validates against the schema when parsed back to JSON.
"""

# =============================================================================
# Supported output formats
# =============================================================================

OUTPUT_FORMATS = ["json", "yaml", "xml", "markdown"]


# =============================================================================
# Pipeline builder
# =============================================================================


def build_config(
    model_alias: str,
    output_format: str | None = None,
) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    # ── Stage 1: Sampling ────────────────────────────────────────────────

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="record_id",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(prefix="SD-", short_form=True, uppercase=True),
        )
    )

    formats = [output_format] if output_format else OUTPUT_FORMATS
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="output_format",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=formats),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic_category",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=list(TOPICS.keys())),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic_subtopic",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="topic_category",
                values=TOPICS,
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="schema_rigidity",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["strict", "moderate"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="schema_fields_count",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="schema_complexity",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["simple", "complex"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="nesting_depth",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["1", "2", "3", "4"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="num_turns",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["2", "3", "4", "5", "6", "7", "8"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="tone",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["neutral", "enthusiastic", "factual"]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="detail_level",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["brief", "standard", "detailed", "super verbose"],
            ),
        )
    )

    # ── Stage 2: Schema generation ───────────────────────────────────────

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="structured_schema",
            prompt=SCHEMA_GENERATION_PROMPT,
            model_alias=model_alias,
        )
    )

    # ── Stage 3: User prompt generation ──────────────────────────────────

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="user_prompt",
            prompt=USER_PROMPT_GENERATION,
            model_alias=model_alias,
        )
    )

    # ── Stage 4: Conversation pairs ──────────────────────────────────────

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="conversation_pairs",
            prompt=CONVERSATION_PROMPT,
            model_alias=model_alias,
        )
    )

    # ── Stage 5: Best-of-3 structured output ─────────────────────────────

    for i in range(3):
        config_builder.add_column(
            dd.LLMTextColumnConfig(
                name=f"structured_output_{i}",
                prompt=STRUCTURED_OUTPUT_PROMPT,
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

    parser = ArgumentParser(description="Nemotron Nano Structured Data Recipe")
    parser.add_argument("--model-alias", type=str, default="nvidia-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument(
        "--output-format",
        type=str,
        default=None,
        choices=OUTPUT_FORMATS,
        help="Generate for a single output format (default: all formats)",
    )
    args = parser.parse_args()

    config_builder = build_config(
        model_alias=args.model_alias,
        output_format=args.output_format,
    )
    results = create_dataset(
        config_builder,
        num_records=args.num_records,
        artifact_path=args.artifact_path,
    )

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    results.load_analysis().to_report()
