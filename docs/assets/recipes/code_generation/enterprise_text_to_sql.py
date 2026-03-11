# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Nemotron Super Text-to-SQL Recipe: Distractors, Dirty Data, and Multi-Judge Scoring

Generate enterprise-grade text-to-SQL training data with dialect-specific SQL
(SQLite, MySQL, PostgreSQL), distractor table/column injection, dirty data
handling, conditional sampling, and multi-dimensional LLM judge scoring.

This recipe implements the pipeline used to produce 96.5k validated text-to-SQL
records for Nemotron Super v3 SFT training, which raised BIRD benchmark
execution accuracy from 26.77% to 41.80%.

Pipeline architecture:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   STAGE 1: SEEDING & DIVERSIFICATION                    │
    │                                                                         │
    │  Domain Controls          SQL Controls           Prompt Controls        │
    │  ├─ industry_sector       ├─ sql_complexity      ├─ instruction_style   │
    │  ├─ topic (conditional)   ├─ sql_concept         ├─ linguistic_register │
    │  ├─ data_quality_challenge├─ sql_task_type       └─ politeness_level    │
    │  ├─ data_quality_concept  │   (conditional)                             │
    │  ├─ knowledge_dependency  └─ sql_task_concept                           │
    │  └─ knowledge_concept                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 2: PROMPT GENERATION (LLM)                      │
    │  Natural-language request grounded in metadata; no SQL jargon.          │
    │  Style adapts to instruction_style × register × politeness.             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 3: SCHEMA + DATA GENERATION (LLM)               │
    │  Dialect-specific DDL + INSERT with 3-5 core tables, 1-2 distractor     │
    │  tables, 3-5 distractor columns per table, dirty data injection.        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 4: SQL GENERATION (LLM)                         │
    │  Dialect-specific SQL; ignores distractors; handles dirty data.         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 5: VALIDATION + QUALITY SCORING                 │
    │                                                                         │
    │  Syntax Validator            5 LLM Judges (0-4 scores)                  │
    │  ├─ SQL_SQLITE               ├─ Prompt: naturalness, specificity,       │
    │  ├─ SQL_MYSQL                │   absence of SQL jargon                  │
    │  └─ SQL_POSTGRES             ├─ SQL: relevance, readability,            │
    │                              │   scalability, standards                 │
    │                              ├─ Context: relevance, readability,        │
    │                              │   scalability, standards                 │
    │                              ├─ Data Quality: cleaning correctness,     │
    │                              │   efficiency                             │
    │                              └─ Knowledge: application correctness,     │
    │                                  clarity of inference                   │
    │                                                                         │
    │  15 score columns extracted for downstream filtering                    │
    └─────────────────────────────────────────────────────────────────────────┘

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

    # =========================================================================
    # Stage 1: Seeding & diversification
    # =========================================================================

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
                    "Aerospace",
                    "Energy",
                    "Telecommunications",
                    "Transportation",
                    "Education",
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
                    "Aerospace": [
                        "Flight Operations",
                        "Satellite Systems",
                        "Parts Procurement",
                        "Maintenance Scheduling",
                        "Crew Management",
                    ],
                    "Energy": [
                        "Grid Management",
                        "Renewable Forecasting",
                        "Asset Monitoring",
                        "Trading and Markets",
                        "Regulatory Reporting",
                    ],
                    "Telecommunications": [
                        "Network Operations",
                        "Customer Billing",
                        "Service Provisioning",
                        "Call Detail Records",
                        "Churn Prediction",
                    ],
                    "Transportation": [
                        "Fleet Management",
                        "Route Optimization",
                        "Freight Tracking",
                        "Driver Scheduling",
                        "Maintenance Records",
                    ],
                    "Education": [
                        "Student Records",
                        "Course Enrollment",
                        "Learning Analytics",
                        "Financial Aid",
                        "Faculty Management",
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
                        "INSERT, UPDATE, DELETE",
                        "ORDER BY and LIMIT",
                    ],
                    "Intermediate": [
                        "Window Functions",
                        "Correlated Subqueries",
                        "Multiple JOINs with Aggregations",
                        "CASE Expressions",
                        "GROUP BY with HAVING",
                        "Set Operations (UNION, INTERSECT, EXCEPT)",
                    ],
                    "Advanced": [
                        "Recursive CTEs",
                        "Frame Clauses",
                        "Pivot/Unpivot Patterns",
                        "Complex Analytical Functions",
                        "Self-Joins for Hierarchies",
                        "Conditional Aggregation",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="sql_task_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "Foundational Queries & DML",
                    "Data Quality & Validation",
                    "Advanced Analytics & Windowing",
                    "Schema, DDL & Performance",
                ],
            ),
            conditional_params={
                "sql_complexity == 'Beginner'": dd.CategorySamplerParams(
                    values=["Foundational Queries & DML", "Data Quality & Validation"],
                ),
                "sql_complexity == 'Intermediate'": dd.CategorySamplerParams(
                    values=[
                        "Foundational Queries & DML",
                        "Data Quality & Validation",
                        "Advanced Analytics & Windowing",
                    ],
                ),
                "sql_complexity == 'Advanced'": dd.CategorySamplerParams(
                    values=[
                        "Advanced Analytics & Windowing",
                        "Schema, DDL & Performance",
                        "Data Quality & Validation",
                    ],
                ),
            },
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="sql_task_concept",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="sql_task_type",
                values={
                    "Foundational Queries & DML": [
                        "Multi-table retrieval",
                        "Filtered aggregation",
                        "Conditional insert/update",
                        "Ranked retrieval",
                    ],
                    "Data Quality & Validation": [
                        "NULL detection and handling",
                        "Duplicate detection",
                        "Data type casting and cleanup",
                        "Referential integrity check",
                    ],
                    "Advanced Analytics & Windowing": [
                        "Running totals and moving averages",
                        "Ranking and percentile computation",
                        "Gap and island detection",
                        "Year-over-year comparison",
                    ],
                    "Schema, DDL & Performance": [
                        "Index-aware query optimization",
                        "Partitioned query design",
                        "Constraint-based validation",
                        "Schema migration pattern",
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
            name="data_quality_concept",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="data_quality_challenge",
                values={
                    "Type Mismatches": [
                        "Currency stored as text with symbols ($57,500)",
                        "Boolean stored as string (yes/no/true/false/1/0)",
                    ],
                    "Temporal Drift": [
                        "Dates stored as text in mixed formats (01-Jan-2023 vs 2023/01/01)",
                        "Timestamps with inconsistent timezone handling",
                    ],
                    "Embedded Special Characters": [
                        "Newlines or tabs inside text fields",
                        "Unicode or accented characters in names",
                    ],
                    "Mixed Formats": [
                        "Phone numbers in mixed formats (555-1234 vs (555) 123-4567)",
                        "Addresses with inconsistent abbreviations",
                    ],
                    "NULL Handling": [
                        "NULLs disguised as empty strings or sentinel values (-1, N/A)",
                        "Optional FKs with NULL references",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="knowledge_dependency",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Domain Knowledge", "Implicit Logic", "Common Sense"],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="knowledge_concept",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="knowledge_dependency",
                values={
                    "Domain Knowledge": [
                        "Industry-specific business rules",
                        "Regulatory thresholds and compliance criteria",
                        "Domain-specific KPI definitions",
                    ],
                    "Implicit Logic": [
                        "Fiscal year vs calendar year reasoning",
                        "Business-day exclusion logic",
                        "Implied sort/filter criteria from context",
                    ],
                    "Common Sense": [
                        "Unit conversion (e.g., cents to dollars)",
                        "Age or duration calculation from dates",
                        "Geographic or hierarchical inference",
                    ],
                },
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

    # =========================================================================
    # Stage 2: Prompt generation
    # =========================================================================

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="sql_prompt",
            model_alias=model_alias,
            system_prompt=PROMPT_GEN_SYSTEM_PROMPT,
            prompt=PROMPT_GEN_TEXT,
        )
    )

    # =========================================================================
    # Stage 3: Schema + data with distractor injection
    # =========================================================================

    config_builder.add_column(
        dd.LLMCodeColumnConfig(
            name="sql_context",
            model_alias=model_alias,
            system_prompt="You are an expert SQL database architect who designs well-structured, normalized schemas.",
            prompt=SCHEMA_GEN_PROMPTS[dialect],
            code_lang=code_lang,
        )
    )

    # =========================================================================
    # Stage 4: Dialect-specific SQL generation
    # =========================================================================

    config_builder.add_column(
        dd.LLMCodeColumnConfig(
            name="sql",
            model_alias=model_alias,
            system_prompt="You are an expert SQL programmer who solves problems with clean, efficient, and perfectly structured queries. Return only the final SQL.",
            prompt=SQL_GEN_PROMPTS[dialect],
            code_lang=code_lang,
        )
    )

    # =========================================================================
    # Stage 5: Validation + 5 LLM judges
    # =========================================================================

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
            name="sql_prompt_judge_result",
            model_alias=model_alias,
            prompt=PROMPT_JUDGE_TEXT,
            scores=PROMPT_SCORES,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="sql_judge_result",
            model_alias=model_alias,
            prompt=SQL_JUDGE_TEXT,
            scores=SQL_SCORES,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="sql_context_judge_result",
            model_alias=model_alias,
            prompt=CONTEXT_JUDGE_PROMPTS[dialect],
            scores=SQL_SCORES,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="sql_data_quality_judge_result",
            model_alias=model_alias,
            prompt=DATA_QUALITY_JUDGE_TEXT,
            scores=DATA_QUALITY_SCORES,
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="sql_knowledge_judge_result",
            model_alias=model_alias,
            prompt=KNOWLEDGE_JUDGE_TEXT,
            scores=KNOWLEDGE_SCORES,
        )
    )

    # =========================================================================
    # Score extraction (15 flat columns for downstream filtering)
    # =========================================================================

    for judge_name, rubric_names in SCORE_EXTRACTION_MAP:
        prefix = judge_name.replace("_judge_result", "").replace("sql_", "")
        for rubric in rubric_names:
            config_builder.add_column(
                dd.ExpressionColumnConfig(
                    name=f"{prefix}_{rubric}_score",
                    expr=f"{{{{ {judge_name}.{rubric}.score if {judge_name}.{rubric}.score else ' ' }}}}",
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

PROMPT_GEN_SYSTEM_PROMPT = """\
You write natural-language requests to a data assistant. You adapt your writing \
style based on the specified instruction style, linguistic register, and politeness level. \
Requests avoid meta-instructions, code, or explicit SQL jargon unless it's common-sense terminology.\
"""

PROMPT_GEN_TEXT = """\
Write a single-sentence, natural-language request to a data assistant or agent to solve a specific SQL problem.

## Style Requirements

* **Instruction Style:** Use a {{ instruction_style }} style.
* **Linguistic Register:** Use a {{ linguistic_register }} register.
* **Politeness Level:** Apply {{ politeness_level }} politeness.

## Content Constraints

* Do NOT use explicit SQL keywords or technical jargon. Describe the **business problem**.
* Keep the intent specific; mention outputs, filters, and aggregations clearly.
* Do not include code, backticks, or any fenced blocks.
* Realistic Thresholds: The sample data is small (5-10 rows per table). Keep thresholds small.
* Relative Time: It's okay to say "recent", "last year", "past few months" instead of exact dates.

## Grounding Requirements

* The request must pertain to the {{ industry_sector }} sector and {{ topic }} topic.
* The request must implicitly require SQL at the {{ sql_complexity }} level involving {{ sql_concept }}.
* The request must require a {{ sql_task_type }} task, specifically "{{ sql_task_concept }}".
* The problem must implicitly require handling "{{ data_quality_concept }}".
* The problem must implicitly require "{{ knowledge_concept }}".\
"""

_SCHEMA_GEN_TEMPLATE = """\
Generate {dialect_label} DDL and sample data for tables relevant to the instruction.
Instruction: {{{{ sql_prompt }}}}

Requirements:

* Scope: Provide only CREATE TABLE and INSERT statements.
* Integrity: Define PRIMARY KEYs and FOREIGN KEYs with consistent data types. Use snake_case names.
* **Section Headers (REQUIRED):**
  - `-- Core Tables`
  - `-- Distractor Tables`
  - `-- Sample Data for Core Tables`
  - `-- Sample Data for Distractor Tables`
  - Do NOT include any other comments.
* Coverage: Include 3-5 core tables for {{{{ industry_sector }}}}/{{{{ topic }}}} connected via FKs.
* **Distractor Tables:** Include 1-2 additional tables plausible for the domain but NOT needed \
for the instruction. Each with FK links to core tables and 5-10 rows of realistic data.
* Realism: Include 3-5 distractor columns per table (created_at, updated_by, description, is_active).
* **Dirty Data:** Introduce "{{{{ data_quality_concept }}}}" issues. Dirty columns MUST be {text_type}.
* Sample Data: 5-10 realistic rows per table. Mix clean and dirty rows.
* **No Data Comments:** Do NOT explain which rows are dirty.
* **Determinism:** No NOW()/CURRENT_DATE in INSERT statements. Use explicit literal dates.
* Executability: End each statement with a semicolon. Use {dialect_label} syntax.
* Do not include meta-instructions or reasoning traces.\
"""

SCHEMA_GEN_PROMPTS = {
    "sqlite": _SCHEMA_GEN_TEMPLATE.format(dialect_label="SQLite", text_type="TEXT"),
    "mysql": _SCHEMA_GEN_TEMPLATE.format(dialect_label="MySQL", text_type="VARCHAR or TEXT"),
    "postgres": _SCHEMA_GEN_TEMPLATE.format(dialect_label="PostgreSQL", text_type="TEXT or VARCHAR"),
}

_SQL_GEN_BASE = """\
Write {dialect_label} SQL for the instruction using only the provided database context.
Instruction: {{{{ sql_prompt }}}}

Database Context:
{{{{ sql_context }}}}

Requirements:

* Validity: You are strictly forbidden from referencing any table or column not in the context.
* Handle Data Quality: Correctly handle "{{{{ data_quality_concept }}}}" using appropriate cleaning functions.
* Apply Knowledge: Apply "{{{{ knowledge_concept }}}}" even if it requires inferring unstated logic.
* Grounding: The SQL must demonstrate {{{{ sql_concept }}}} and {{{{ sql_task_type }}}}.
* Precision: Avoid SELECT *. Explicitly list columns; alias computed columns descriptively.
* Alignment: Match the {{{{ sql_complexity }}}} level.
* **Relative Time Anchoring:** Do NOT use CURRENT_DATE or real-time functions. Anchor to max date in data.
* **No Unasked Joins:** Do NOT join distractor tables or select distractor columns.
* **Logic:** Prefer CTEs to clean/normalize first, then compute/aggregate.
* Comments: Do not include inline comments.
* Formatting: Terminate with semicolons.\
"""

_SQLITE_EXTRAS = """
* Use SQLite syntax: strftime for dates, LIMIT instead of TOP.
* Do NOT use LATERAL joins or REGEXP_REPLACE. Use REPLACE()/SUBSTR() for cleaning.
* Date Parsing: Normalize with REPLACE(date_col, '/', '-') inside date()/strftime().\
"""

_MYSQL_EXTRAS = """
* Use MySQL syntax: DATE_ADD, DATEDIFF for dates, LIMIT for pagination, backticks for identifiers.
* Do NOT use REGEXP_REPLACE or CONVERT_TZ. Use REPLACE(), TRIM(), SUBSTRING().\
"""

_POSTGRES_EXTRAS = """
* Use PostgreSQL syntax: :: for casting, ILIKE for case-insensitive matching, LIMIT and OFFSET.
* regexp_replace is available for cleaning.\
"""

SQL_GEN_PROMPTS = {
    "sqlite": _SQL_GEN_BASE.format(dialect_label="SQLite") + _SQLITE_EXTRAS,
    "mysql": _SQL_GEN_BASE.format(dialect_label="MySQL") + _MYSQL_EXTRAS,
    "postgres": _SQL_GEN_BASE.format(dialect_label="PostgreSQL") + _POSTGRES_EXTRAS,
}

# =============================================================================
# Judge Prompts
# =============================================================================

PROMPT_JUDGE_TEXT = """\
You are an expert product analyst who writes and reviews natural, human-like data requests.
Evaluate the **NL Prompt** quality.

## NL Prompt

{{ sql_prompt }}\
"""

SQL_JUDGE_TEXT = """\
You are a SQL data expert. Grade the quality of **Generated SQL** based on the prompt and context.

Natural Language Prompt:
{{ sql_prompt }}

Database Context:
{{ sql_context }}

Generated SQL:
{{ sql }}

When scoring, pay special attention to:
- **Minimal Table Usage:** Penalize queries that unnecessarily join distractor tables.
- **Minimal Column Usage:** Distractor columns should be ignored unless explicitly needed.
- **Correctness:** The query must produce the correct result.
- **Efficiency:** Prefer simple, readable solutions over unnecessarily complex ones.\
"""

_CONTEXT_JUDGE_TEMPLATE = """\
You are a SQL database architect. Evaluate the **Generated Database Context** quality.

Natural Language Prompt:
{{{{ sql_prompt }}}}

Generated Database Context ({dialect}):
{{{{ sql_context }}}}

When scoring, verify:
- **Sufficient Tables:** 3-5 core tables plus 1-2 distractor tables. Penalize bare-minimum schemas.
- **Distractor Columns:** Each table should include realistic columns beyond those needed for the query.
- **Realistic Relationships:** Appropriate PK/FK relationships. Distractor tables should have logical FK links.
- **Sample Data Quality:** Realistic, varied INSERT data.
- **Executability:** Syntactically correct for {dialect}.\
"""

CONTEXT_JUDGE_PROMPTS = {
    "sqlite": _CONTEXT_JUDGE_TEMPLATE.format(dialect="SQLite"),
    "mysql": _CONTEXT_JUDGE_TEMPLATE.format(dialect="MySQL"),
    "postgres": _CONTEXT_JUDGE_TEMPLATE.format(dialect="PostgreSQL"),
}

DATA_QUALITY_JUDGE_TEXT = """\
You are an expert in data quality and validation. Score the SQL's handling of messy data.

## Natural Language Prompt
{{ sql_prompt }}

## Data Quality Challenge
{{ data_quality_challenge }} / {{ data_quality_concept }}

## Database Context
{{ sql_context }}

## Generated SQL
{{ sql }}\
"""

KNOWLEDGE_JUDGE_TEXT = """\
You are an expert in business intelligence and semantic interpretation. \
Score the SQL's application of implicit business knowledge.

## Natural Language Prompt
{{ sql_prompt }}

## Knowledge Dependency
{{ knowledge_dependency }} / {{ knowledge_concept }}

## Database Context
{{ sql_context }}

## Generated SQL
{{ sql }}\
"""

# =============================================================================
# Scoring Rubrics (5 judges, 15 dimensions)
# =============================================================================

SQL_SCORES = [
    dd.Score(
        name="relevance",
        description="Uses only necessary tables/columns; ignores distractors",
        options={
            "4": "Perfectly meets all requirements; uses only strictly necessary tables and columns.",
            "3": "Meets most requirements with minor deviations; may include a slightly unnecessary column.",
            "2": "Moderate deviation; joins an unnecessary table or selects several irrelevant columns.",
            "1": "Significant deviations; multiple unnecessary table joins or largely irrelevant output.",
            "0": "Does not adhere to instructions; query is unrelated or joins many unnecessary tables.",
        },
    ),
    dd.Score(
        name="readability",
        description="Formatting, clarity, and maintainability",
        options={
            "4": "Excellently formatted, meaningful aliases, high readability and ease of maintenance.",
            "3": "Well-formatted, relatively easy to understand; uses aliases with some organization.",
            "2": "Somewhat readable with basic formatting but needs improvement.",
            "1": "Minimal formatting, hard to understand; lacks meaningful names.",
            "0": "Unreadable, no attempt at formatting.",
        },
    ),
    dd.Score(
        name="scalability",
        description="Scales well with larger datasets; avoids inefficient patterns",
        options={
            "4": "Highly scalable; avoids Cartesian joins and unnecessary table joins.",
            "3": "Scales well; minor areas for optimization such as an extra join.",
            "2": "Moderately scalable; includes unnecessary joins or suboptimal access patterns.",
            "1": "Poor scalability; joins multiple unnecessary tables or uses inefficient patterns.",
            "0": "Does not scale; overlooks fundamental scalability practices.",
        },
    ),
    dd.Score(
        name="standards",
        description="Compliance with SQL standards and best practices",
        options={
            "4": "Strictly adheres to SQL standards and best practices.",
            "3": "Closely follows SQL standards and many best practices.",
            "2": "Generally follows standards but has room for better alignment.",
            "1": "Loosely follows standards, with several deviations.",
            "0": "Does not follow standards; uses deprecated or non-standard syntax.",
        },
    ),
]

PROMPT_SCORES = [
    dd.Score(
        name="naturalness_of_wording",
        description="How human-like, colloquial, and non-robotic the phrasing is",
        options={
            "4": "Reads like a native speaker; concise, fluent, and natural.",
            "3": "Generally natural; minor stiffness or formalism.",
            "2": "Somewhat stilted or templated; noticeable artifacts.",
            "1": "Robotic or awkward; obviously machine-generated.",
            "0": "Unnatural and hard to read.",
        },
    ),
    dd.Score(
        name="specificity_and_clarity",
        description="Is the request specific about outputs, filters, and operations?",
        options={
            "4": "Very specific and clear outputs/filters/aggregations; minimal ambiguity.",
            "3": "Mostly specific; minor ambiguity remains.",
            "2": "Partially specific; key details are missing.",
            "1": "Vague; unclear what should be returned or computed.",
            "0": "Completely ambiguous.",
        },
    ),
    dd.Score(
        name="absence_of_sql_jargon",
        description="Avoids explicit SQL terms, table/column names, or schema hints",
        options={
            "4": "No SQL jargon at all; entirely tool-agnostic phrasing.",
            "3": "Tiny hints but no explicit SQL or schema leakage.",
            "2": "Occasional SQL terms or schema leakage present.",
            "1": "Frequent SQL jargon and schema references.",
            "0": "Reads like a SQL spec; heavy jargon.",
        },
    ),
]

DATA_QUALITY_SCORES = [
    dd.Score(
        name="correctness_of_cleaning_logic",
        description="Does the query correctly and fully clean the messy data?",
        options={
            "4": "Flawless cleaning logic; handles all transformations and edge cases perfectly.",
            "3": "Correctly cleans data for most cases but might miss minor edge cases.",
            "2": "Attempts to clean data, but logic is only partially correct.",
            "1": "Cleaning logic is fundamentally flawed.",
            "0": "No attempt to clean the data.",
        },
    ),
    dd.Score(
        name="efficiency_of_cleaning_method",
        description="Uses efficient, standard functions for cleaning?",
        options={
            "4": "Highly efficient, optimal SQL functions for the task.",
            "3": "Correct and standard functions, but a more performant approach exists.",
            "2": "Convoluted or inefficient method where a simpler one would suffice.",
            "1": "Very inefficient or non-standard method that would scale poorly.",
            "0": "Completely inappropriate or non-functional method.",
        },
    ),
]

KNOWLEDGE_SCORES = [
    dd.Score(
        name="correctness_of_knowledge_application",
        description="Does the query correctly translate the implicit knowledge into SQL logic?",
        options={
            "4": "Flawlessly translates abstract concept into precise, correct SQL logic.",
            "3": "Logic correctly reflects the knowledge concept but could be expressed more directly.",
            "2": "Partially applies the logic, misinterpreting some nuances.",
            "1": "Fundamentally misinterprets the knowledge concept.",
            "0": "No attempt to apply the required knowledge.",
        },
    ),
    dd.Score(
        name="clarity_of_inference",
        description="Is the applied logic clear and self-explanatory within the query?",
        options={
            "4": "Logic is immediately obvious through well-chosen aliases, CTEs, or clear filtering.",
            "3": "Logic is correct but requires some inspection to understand.",
            "2": "Logic is technically correct but obscure, using magic numbers or hard-to-read conditions.",
            "1": "Logic is convoluted and appears incorrect.",
            "0": "Query is completely opaque, with no discernible link to the required knowledge.",
        },
    ),
]

SCORE_EXTRACTION_MAP = [
    ("sql_judge_result", ["relevance", "readability", "scalability", "standards"]),
    ("sql_context_judge_result", ["relevance", "readability", "scalability", "standards"]),
    ("sql_prompt_judge_result", ["naturalness_of_wording", "specificity_and_clarity", "absence_of_sql_jargon"]),
    ("sql_data_quality_judge_result", ["correctness_of_cleaning_logic", "efficiency_of_cleaning_method"]),
    ("sql_knowledge_judge_result", ["correctness_of_knowledge_application", "clarity_of_inference"]),
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
