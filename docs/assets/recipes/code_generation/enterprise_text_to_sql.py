# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Enterprise Text-to-SQL Recipe: Distractors, Dirty Data, and Multi-Judge Scoring

Generate enterprise-grade text-to-SQL training data with dialect-specific SQL
(SQLite, MySQL, PostgreSQL), distractor table/column injection, dirty data
handling, conditional sampling, and multi-dimensional LLM judge scoring.

Each record flows through five stages:
  1. Seeding & diversification (industry, complexity, data quality, prompt style)
  2. Natural-language prompt generation (no SQL jargon)
  3. Schema + sample data generation with distractor injection
  4. Dialect-specific SQL generation
  5. Syntax validation + two LLM judges (prompt quality, SQL quality)

Based on the pipeline described in the "Engineering an Enterprise-Grade
Text-to-SQL Dataset" dev note, which produced 96.5k validated records across
PostgreSQL, MySQL, and SQLite for Nemotron Super v3 SFT training.

Prerequisites:
    - OPENAI_API_KEY environment variable for OpenAI provider model aliases (default model alias is "openai-text").
    - NVIDIA_API_KEY environment variable for NVIDIA provider model aliases.

Run:
    # Basic usage (generates 5 records by default, SQLite dialect)
    uv run enterprise_text_to_sql.py

    # Generate for a specific dialect
    uv run enterprise_text_to_sql.py --dialect postgres

    # For help message and available options
    uv run enterprise_text_to_sql.py --help
"""

from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

SQL_DIALECTS = {
    "sqlite": dd.CodeLang.SQL_SQLITE,
    "mysql": dd.CodeLang.SQL_MYSQL,
    "postgres": dd.CodeLang.SQL_POSTGRES,
}


def build_config(model_alias: str, dialect: str = "sqlite") -> dd.DataDesignerConfigBuilder:
    code_lang = SQL_DIALECTS[dialect]
    config_builder = dd.DataDesignerConfigBuilder()

    # --- Stage 1: Seeding & diversification ---

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="sql_dialect",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=[dialect]),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="industry_sector",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "Healthcare",
                    "Financial Services",
                    "Retail",
                    "Technology",
                    "Manufacturing",
                ],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="industry_sector",
                values={
                    "Healthcare": [
                        "Electronic Health Records",
                        "Telemedicine Platforms",
                        "Clinical Trials",
                        "Patient Scheduling",
                        "Insurance Claims",
                    ],
                    "Financial Services": [
                        "Fraud Detection",
                        "Trading Systems",
                        "Risk Assessment",
                        "Portfolio Management",
                        "Regulatory Compliance",
                    ],
                    "Retail": [
                        "Inventory Management",
                        "Customer Segmentation",
                        "Pricing Optimization",
                        "Supply Chain",
                        "Returns Processing",
                    ],
                    "Technology": [
                        "Cloud Platforms",
                        "ML Pipelines",
                        "DevOps Tools",
                        "API Gateway Logs",
                        "User Analytics",
                    ],
                    "Manufacturing": [
                        "Quality Control",
                        "Production Scheduling",
                        "Equipment Maintenance",
                        "Supply Chain Optimization",
                        "Safety Compliance",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="sql_complexity",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Beginner", "Intermediate", "Advanced"],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="sql_concept",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="sql_complexity",
                values={
                    "Beginner": [
                        "Basic SELECT Statements",
                        "WHERE Clauses",
                        "Simple Aggregations",
                        "Basic JOINs",
                    ],
                    "Intermediate": [
                        "Window Functions",
                        "Correlated Subqueries",
                        "Multiple JOINs with Aggregations",
                        "CASE Expressions",
                    ],
                    "Advanced": [
                        "Recursive CTEs",
                        "Frame Clauses",
                        "Pivot/Unpivot Patterns",
                        "Complex Analytical Functions",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="data_quality_challenge",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "Type Mismatches",
                    "Temporal Drift",
                    "Embedded Special Characters",
                    "Mixed Formats",
                    "NULL Handling",
                ],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="instruction_style",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["imperative", "declarative", "interrogative", "contextual", "abbreviated"],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="linguistic_register",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["formal", "conversational", "technical", "academic", "direct"],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="politeness_level",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["none", "minimal", "polite", "very polite"],
            ),
        )
    )

    # --- Stage 2: Natural-language prompt generation ---

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="sql_prompt",
            model_alias=model_alias,
            system_prompt=PROMPT_GEN_SYSTEM_PROMPT,
            prompt=PROMPT_GEN_TEXT,
        )
    )

    # --- Stage 3: Schema + data with distractor injection ---

    config_builder.add_column(
        dd.LLMCodeColumnConfig(
            name="sql_context",
            model_alias=model_alias,
            system_prompt="You are an expert SQL database architect who designs well-structured, normalized schemas.",
            prompt=SCHEMA_GEN_TEXT,
            code_lang=code_lang,
        )
    )

    # --- Stage 4: Dialect-specific SQL generation ---

    config_builder.add_column(
        dd.LLMCodeColumnConfig(
            name="sql",
            model_alias=model_alias,
            system_prompt="You are an expert SQL programmer. Return only the final SQL query.",
            prompt=SQL_GEN_TEXT,
            code_lang=code_lang,
        )
    )

    # --- Stage 5: Validation + judges ---

    config_builder.add_column(
        dd.ValidationColumnConfig(
            name="sql_validity_result",
            target_columns=["sql"],
            validator_type=dd.ValidatorType.CODE,
            validator_params=dd.CodeValidatorParams(code_lang=code_lang),
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="prompt_judge_result",
            model_alias=model_alias,
            prompt=PROMPT_JUDGE_TEXT,
            scores=prompt_scoring,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="sql_judge_result",
            model_alias=model_alias,
            prompt=SQL_JUDGE_TEXT,
            scores=sql_scoring,
        )
    )

    # --- Score extraction ---

    for judge_name, rubric_names in [
        ("prompt_judge_result", ["naturalness", "specificity"]),
        ("sql_judge_result", ["relevance", "readability", "data_cleaning"]),
    ]:
        prefix = judge_name.replace("_judge_result", "")
        for rubric in rubric_names:
            config_builder.add_column(
                dd.ExpressionColumnConfig(
                    name=f"{prefix}_{rubric}_score",
                    expr=(f"{{{{ {judge_name}.{rubric}.score if {judge_name}.{rubric}.score is not none else '' }}}}"),
                )
            )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    results = data_designer.create(config_builder, num_records=num_records)
    return results


# =============================================================================
# Prompt Templates
# =============================================================================

PROMPT_GEN_SYSTEM_PROMPT = (
    "You write natural-language requests to a data assistant. "
    "You adapt your writing style based on the specified instruction style, "
    "linguistic register, and politeness level."
)

PROMPT_GEN_TEXT = (
    "Write a single natural-language request to a data assistant.\n\n"
    "## Style Requirements\n"
    "* Instruction Style: {{ instruction_style }}\n"
    "* Linguistic Register: {{ linguistic_register }}\n"
    "* Politeness Level: {{ politeness_level }}\n\n"
    "## Grounding Requirements\n"
    "* Industry: {{ industry_sector }} / {{ topic }}\n"
    "* SQL Complexity: {{ sql_complexity }} ({{ sql_concept }})\n"
    "* Data Quality Challenge: {{ data_quality_challenge }}\n\n"
    "The request must describe a business problem without any SQL jargon. "
    "Do not mention SQL, queries, tables, columns, JOINs, or any database terminology. "
    "Use realistic thresholds appropriate for small datasets (5-10 rows per table)."
)

SCHEMA_GEN_TEXT = (
    "Generate {{ sql_dialect }} DDL and sample data for tables relevant to the instruction.\n"
    "Instruction: {{ sql_prompt }}\n\n"
    "Requirements:\n"
    "* Include 3-5 core tables for {{ industry_sector }}/{{ topic }} with PKs, FKs, and constraints\n"
    "* Include 1-2 distractor tables (plausible for the domain but NOT needed for the instruction), "
    "each with FK links to core tables and 5-10 rows of realistic data\n"
    "* Include 3-5 distractor columns per table (created_at, updated_by, description, is_active, etc.)\n"
    "* Introduce {{ data_quality_challenge }} dirty data issues in TEXT/VARCHAR columns\n"
    "* Use section headers: -- Core Tables, -- Distractor Tables, "
    "-- Sample Data for Core Tables, -- Sample Data for Distractor Tables\n"
    "* No NOW()/CURRENT_DATE in INSERT statements\n"
    "* 5-10 rows per table with realistic values"
)

SQL_GEN_TEXT = (
    "Write {{ sql_dialect }} SQL for the instruction using only the provided database context.\n"
    "Instruction: {{ sql_prompt }}\n\n"
    "Database Context:\n{{ sql_context }}\n\n"
    "Requirements:\n"
    "* Only reference tables and columns that exist in the context\n"
    "* Handle {{ data_quality_challenge }} issues with cleaning logic (CAST, REPLACE, SUBSTR, etc.)\n"
    "* Match {{ sql_complexity }} level using {{ sql_concept }}\n"
    "* Do NOT join distractor tables or select distractor columns\n"
    "* Anchor relative time to max date in sample data, not CURRENT_DATE/NOW()"
)

PROMPT_JUDGE_TEXT = """\
Evaluate the quality of this natural-language data request.

Request: {{ sql_prompt }}

The request should describe a business problem naturally, without SQL jargon,
and be specific enough that a database expert could derive a unique SQL query from it.
"""

SQL_JUDGE_TEXT = """\
Grade the SQL quality given the prompt, database context, and generated query.

Prompt: {{ sql_prompt }}
Context: {{ sql_context }}
SQL: {{ sql }}

The database context includes distractor tables that look relevant but are not needed.
Penalize queries that unnecessarily join or reference these distractor tables.
The SQL should handle dirty data issues present in the context with appropriate cleaning logic.
"""

# =============================================================================
# Scoring Rubrics
# =============================================================================

prompt_scoring = [
    dd.Score(
        name="naturalness",
        description="Natural business language without SQL jargon",
        options={
            "4": "Completely natural, no technical SQL terms",
            "3": "Mostly natural with minor technical leakage",
            "2": "Some SQL jargon present",
            "1": "Reads like a SQL specification",
            "0": "Pure SQL/database description",
        },
    ),
    dd.Score(
        name="specificity",
        description="Clear and specific enough to derive a unique query",
        options={
            "4": "Unambiguous, specific business request",
            "3": "Clear with minor ambiguity",
            "2": "Somewhat vague",
            "1": "Very vague or generic",
            "0": "Impossible to derive a query",
        },
    ),
]

sql_scoring = [
    dd.Score(
        name="relevance",
        description="Uses only necessary tables/columns, ignores distractors",
        options={
            "4": "Perfect -- only necessary tables/columns used",
            "3": "Minor extras but core logic correct",
            "2": "Unnecessary joins to distractor tables",
            "1": "Largely irrelevant table usage",
            "0": "Wrong tables entirely",
        },
    ),
    dd.Score(
        name="readability",
        description="Code clarity, formatting, and maintainability",
        options={
            "4": "Excellent formatting, clear aliases, well-structured",
            "3": "Good readability with minor issues",
            "2": "Adequate but messy",
            "1": "Poor formatting, confusing aliases",
            "0": "Unreadable",
        },
    ),
    dd.Score(
        name="data_cleaning",
        description="Correct handling of dirty data issues in the schema",
        options={
            "4": "All dirty data properly cleaned before use",
            "3": "Most cleaning handled correctly",
            "2": "Some cleaning missing or incorrect",
            "1": "Minimal cleaning attempted",
            "0": "No cleaning, dirty data used directly",
        },
    ),
]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-alias", type=str, default="openai-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument(
        "--dialect",
        type=str,
        default="sqlite",
        choices=list(SQL_DIALECTS.keys()),
        help="SQL dialect to generate for (default: sqlite)",
    )
    args = parser.parse_args()

    config_builder = build_config(model_alias=args.model_alias, dialect=args.dialect)
    results = create_dataset(config_builder, num_records=args.num_records, artifact_path=args.artifact_path)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()
