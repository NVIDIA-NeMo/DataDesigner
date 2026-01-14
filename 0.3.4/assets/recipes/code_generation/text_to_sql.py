from pathlib import Path

from data_designer.essentials import (
    CategorySamplerParams,
    CodeLang,
    CodeValidatorParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
    Score,
    SubcategorySamplerParams,
    ValidationColumnConfig,
    ValidatorType,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="industry_sector",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["Healthcare", "Finance", "Technology"],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="topic",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="industry_sector",
                values={
                    "Healthcare": [
                        "Electronic Health Records (EHR) Systems",
                        "Telemedicine Platforms",
                        "AI-Powered Diagnostic Tools",
                    ],
                    "Finance": [
                        "Fraud Detection Software",
                        "Automated Trading Systems",
                        "Personal Finance Apps",
                    ],
                    "Technology": [
                        "Cloud Computing Platforms",
                        "Artificial Intelligence and Machine Learning Platforms",
                        "DevOps and CI/CD Tools",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="sql_complexity",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["Beginner", "Intermediate", "Advanced"],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="sql_concept",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="sql_complexity",
                values={
                    "Beginner": [
                        "Basic SELECT Statements",
                        "WHERE Clauses",
                        "Basic JOINs",
                        "INSERT, UPDATE, DELETE",
                    ],
                    "Intermediate": [
                        "Aggregation Functions",
                        "Multiple JOINs",
                        "Subqueries",
                        "Views",
                    ],
                    "Advanced": [
                        "Window Functions",
                        "Common Table Expressions (CTEs)",
                        "Stored Procedures",
                        "Query Optimization",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="sql_task_type",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Data Retrieval",
                    "Data Manipulation",
                    "Analytics and Reporting",
                    "Data Transformation",
                ],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="instruction_phrase",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Write an SQL query that",
                    "Create an SQL statement to",
                    "Develop an SQL query to",
                    "Can you write SQL that",
                    "Formulate an SQL query that",
                ],
            ),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="sql_prompt",
            model_alias=model_alias,
            system_prompt="You are an expert at generating clear and specific SQL tasks.",
            prompt=SQL_PROMPT_TEXT,
        )
    )

    config_builder.add_column(
        LLMCodeColumnConfig(
            name="sql_context",
            model_alias=model_alias,
            code_lang=CodeLang.SQL_ANSI,
            system_prompt=(
                "You are an expert SQL database designer who creates clean, efficient, and "
                "well-structured database schemas."
            ),
            prompt=SQL_CONTEXT_TEXT,
        )
    )

    config_builder.add_column(
        LLMCodeColumnConfig(
            name="sql",
            model_alias=model_alias,
            code_lang=CodeLang.SQL_ANSI,
            system_prompt="You are an expert SQL programmer who writes clean, efficient, and well-structured queries.",
            prompt=SQL_CODE_TEXT,
        )
    )

    config_builder.add_column(
        ValidationColumnConfig(
            name="code_validity_result",
            validator_type=ValidatorType.CODE,
            target_columns=["sql"],
            validator_params=CodeValidatorParams(
                code_lang=CodeLang.SQL_ANSI,
            ),
            batch_size=100,
        )
    )

    config_builder.add_column(
        LLMJudgeColumnConfig(
            name="code_judge_result",
            model_alias=model_alias,
            prompt=TEXT_TO_SQL_JUDGE_TEMPLATE,
            scores=sql_scoring,
        )
    )

    return config_builder


def create_dataset(
    config_builder: DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    results = data_designer.create(config_builder, num_records=num_records)
    return results


SQL_PROMPT_TEXT = (
    "Generate an instruction to create SQL code that solves a specific problem.\n"
    "Each instruction should begin with one of the following phrases: {{instruction_phrase}}.\n\n"
    "Important Guidelines:\n"
    "* Industry Relevance: Ensure the instruction pertains to the {{industry_sector}} sector and {{topic}} topic.\n"
    "* SQL Complexity: Tailor the instruction to the {{sql_complexity}} level. Utilize relevant {{sql_concept}} "
    "where appropriate to match the complexity level.\n"
    "* Task Type: The instruction should involve a {{sql_task_type}} task.\n"
    "* Clarity and Specificity: Make the problem statement clear and unambiguous. Provide sufficient context to "
    "understand the requirements without being overly verbose.\n"
    "* Response Formatting: Do not include any markers such as ### Response ### in the instruction.\n"
)

SQL_CONTEXT_TEXT = (
    "Generate the SQL for creating database tables that would be relevant for the following instruction:\n"
    "Instruction: {{sql_prompt}}\n\n"
    "Important Guidelines:\n"
    "* Relevance: Ensure all tables are directly related to the {{industry_sector}} sector and {{topic}} topic.\n"
    "* Completeness: Include all essential columns with appropriate data types, primary/foreign keys, and necessary constraints.\n"
    "* Realism: Use realistic table structures typical for the specified industry.\n"
    "* Executable SQL: Provide complete CREATE TABLE statements that can be run without modification.\n"
    "* Consistency: Use consistent naming conventions (e.g., snake_case for table and column names).\n"
    "* Sample Data: Include INSERT statements with sample data that makes sense for the tables (at least 5-10 rows per table)."
)

SQL_CODE_TEXT = (
    "Write SQL code for the following instruction based on the provided database context:\n"
    "Instruction: {{sql_prompt}}\n\n"
    "Database Context:\n"
    "{{sql_context}}\n\n"
    "Important Guidelines:\n"
    "* Code Quality: Your SQL should be clean, complete, self-contained and accurate.\n"
    "* Code Validity: Please ensure that your SQL code is executable and does not contain any errors.\n"
    "* Context: Base your query on the provided database context. Only reference tables and columns that "
    "exist in the context.\n"
    "* Complexity & Concepts: The SQL should be written at a {{sql_complexity}} level, making use of "
    "concepts such as {{sql_concept}}.\n"
    "* Task Type: Ensure your solution implements the appropriate {{sql_task_type}} operation.\n"
    "* Comments: Include brief comments explaining the key parts of your query.\n"
)


TEXT_TO_SQL_JUDGE_TEMPLATE = """\
You are an expert in SQL with deep knowledge of relational modeling, query semantics,
and performance tuning across common dialects (e.g., PostgreSQL, MySQL, SQLite, SQL Server).
You think critically about correctness, readability, and efficiency.

Use the SQL Query Quality Rubric below to score the **Generated SQL Query** based on the INSTRUCTIONS.

#### INSTRUCTIONS
The Generated SQL Query should be a valid response to the Natural Language Prompt below

Natural Language Prompt:
{{ sql_prompt }}

Database Context:
{{ sql_context }}

Generated SQL Query
{{ sql }}
"""


sql_scoring = [
    Score(
        name="Relevance",
        description="Adherence to INSTRUCTIONS and CONTEXT",
        options={
            4: "Perfectly meets all specified requirements.",
            3: "Meets most requirements with minor deviations.",
            2: "Moderate deviation from the instructions.",
            1: "Significant deviations from the instructions.",
            0: "Does not adhere to the instructions.",
        },
    ),
    Score(
        name="SQL Correctness",
        description="Syntax and semantic correctness; returns the intended result",
        options={
            4: "Valid SQL with correct joins, filters, grouping/aggregation, and NULL handling; produces the intended result set under the stated/implicit dialect.",
            3: "Generally correct with minor issues (e.g., edge-case NULLs, minor grouping detail) but still likely yields the intended result.",
            2: "Partially correct; noticeable semantic mistakes (joins, grouping, filters) that may change results or fail in edge cases.",
            1: "Largely incorrect; major semantic or syntactic errors likely causing failure or wrong results.",
            0: "Invalid SQL or unrelated to the task; will not run or cannot produce a meaningful result.",
        },
    ),
    Score(
        name="Readability",
        description="Formatting, clarity, and maintainability",
        options={
            4: "Cleanly formatted (keywords/clauses consistently styled), clear structure (CTEs/subqueries where helpful), meaningful table/column aliases, and concise.",
            3: "Generally readable with consistent formatting and understandable aliases; could be organized slightly better.",
            2: "Somewhat readable but inconsistent formatting or confusing aliasing; structure is harder to follow.",
            1: "Poorly formatted and hard to read; unclear structure and aliasing.",
            0: "Unreadable or chaotic; no meaningful structure or styling.",
        },
    ),
    Score(
        name="Efficiency",
        description="Query performance best practices",
        options={
            4: "Uses sargable predicates, appropriate joins, selective filters early, avoids SELECT *, unnecessary DISTINCT, and wasteful subqueries; likely to use indexes effectively.",
            3: "Mostly efficient; minor opportunities for improvement (e.g., simplifying expressions, reducing data early).",
            2: "Moderate inefficiencies (e.g., non-sargable filters, unnecessary nested subqueries, broad SELECT *).",
            1: "Notably inefficient patterns likely causing large scans or poor plans.",
            0: "Highly inefficient; ignores basic best practices and likely to perform very poorly.",
        },
    ),
]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-alias", type=str, default="openai-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    args = parser.parse_args()

    config_builder = build_config(model_alias=args.model_alias)
    results = create_dataset(config_builder, num_records=args.num_records, artifact_path=args.artifact_path)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()
