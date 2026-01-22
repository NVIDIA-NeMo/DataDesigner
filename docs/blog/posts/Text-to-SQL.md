---
date: 2026-01-15
authors:
  - dhruv
  - yev
  - maarten
---

# Engineering an Enterprise-Grade Text-to-SQL Dataset with NeMo Data Designer

While Large Language Models (LLMs) have mastered generic coding, Text-to-SQL remains one of the hardest frontiers to crack in the enterprise.

<!-- more -->

The gap between academic benchmarks (like Spider or WikiSQL) and the messy reality of enterprise data warehouses is massive. Real databases have "dirty" data, inconsistent formats, hundreds of tables, and specific dialect quirks that generic SQL training data simply ignores. To train NVIDIA's state-of-the-art Nemotron-Super model, we couldn't just scrape GitHub. We needed a dataset that forces the model to act not just as a translator, but as a **Data Engineer**. Using NVIDIA Data Designer, we engineered a massive, high-fidelity dataset of **96,500 reasoning-heavy samples** (filtered down from 300,000) that covers PostgreSQL, MySQL, and SQLite. This dataset bridges the "last mile" of enterprise AI, enabling agents to navigate the ambiguity of real-world schemas with the precision of a seasoned expert.

## The "Real-World" Gap: Why Academic Data Wasn't Enough

Most open-source Text-to-SQL datasets assume a "happy path": clear column names, perfect data types, and simple questions. In the real world, a user might ask for "sales from last month" on a text column stored as `MM/DD/YYYY` in a MySQL database that doesn't support the same date functions as Postgres.

We needed a dataset that reflects these harsh realities:

**Dialect Specificity**: Generic "SQL" doesn't compile. We needed valid, executable code for MySQL, PostgreSQL, and SQLite that respects their unique syntax (e.g., `date('now')` in SQLite vs. `CURRENT_DATE` in Postgres).

**Dirty Data**: Real columns contain currency symbols (`$57,500`), mixed date formats, and JSON blobs. The model needs to learn how to clean this data inside the query using CTEs (Common Table Expressions) before aggregating it.

**Distractor Tables**: In production, you don't get just the 2 tables you need; you are more likely to get a schema with 50 tables. We injected "distractor" tables into every context to teach the model to identify relevant signals amidst the noise.

## The Engine: NVIDIA NeMo Data Designer

With Data Designer we can move from manual annotation to programmable data generation. Instead of writing SQL queries by hand, we defined "Semantic Blueprints" that controlled the distribution of the data.

**Semantic Sampling**: We didn't just random-walk through topics. We used seed taxonomies covering 60 industries (from Aerospace to Supply Chain) and 700 distinct topics to ensure the model has seen every corner of the enterprise world.

**Concept Buckets**: We targeted 90 specific SQL concepts, ranging from "Beginner" (basic SELECT) to "Advanced" (Recursive CTEs, Window Functions, and JSON extraction).

**Prompt Diversity**: We systematically varied the linguistic register (formal vs. colloquial), instruction style, and politeness level to robustly handle any user persona.

### Data Designer Workflow

Here's how we structured the generation pipeline using Data Designer's column-based approach:

```python
from data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
)

config_builder = DataDesignerConfigBuilder()

# Stage 1: Sample diversity dimensions
config_builder.add_column(
    SamplerColumnConfig(
        name="industry_sector",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Healthcare", "Finance", "Technology", "Retail", "Manufacturing"],
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="sql_dialect",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["PostgreSQL", "MySQL", "SQLite"],
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
                "Beginner": ["Basic SELECT", "WHERE Clauses", "Simple JOINs"],
                "Intermediate": ["Aggregations", "Multiple JOINs", "Subqueries", "CTEs"],
                "Advanced": ["Window Functions", "Recursive CTEs", "JSON Extraction"],
            },
        ),
    )
)
```

The power of Data Designer is that it handles the dependency graph automatically. When you reference `{{sql_dialect}}` or `{{industry_sector}}` in downstream columns, Data Designer ensures they're generated in the correct order.

```python
from data_designer.essentials import LLMTextColumnConfig, LLMCodeColumnConfig, CodeLang

# Stage 2: Generate natural language instruction
config_builder.add_column(
    LLMTextColumnConfig(
        name="instruction",
        model_alias="nvidia-text",
        prompt=(
            "Generate a natural language request for SQL code that solves a specific problem.\n"
            "Industry: {{industry_sector}}\n"
            "SQL Dialect: {{sql_dialect}}\n"
            "Complexity: {{sql_complexity}}\n"
            "Concept: {{sql_concept}}\n"
        ),
    )
)

# Stage 3: Generate database schema (with distractor tables)
config_builder.add_column(
    LLMCodeColumnConfig(
        name="schema",
        model_alias="nvidia-text",
        code_lang=CodeLang.SQL_ANSI,
        prompt=(
            "Generate CREATE TABLE statements for {{sql_dialect}} that are relevant for:\n"
            "{{instruction}}\n\n"
            "Include 3-5 relevant tables AND 2-3 distractor tables that are plausible but not needed.\n"
            "Include realistic dirty data patterns (text dates, currency symbols, mixed formats).\n"
        ),
    )
)

# Stage 4: Generate the SQL query with reasoning
config_builder.add_column(
    LLMCodeColumnConfig(
        name="sql_with_reasoning",
        model_alias="nvidia-reasoning",
        code_lang=CodeLang.SQL_ANSI,
        prompt=(
            "Instruction: {{instruction}}\n"
            "Database Schema: {{schema}}\n"
            "Target Dialect: {{sql_dialect}}\n\n"
            "First, think step-by-step about the approach (Chain of Thought).\n"
            "Then generate valid {{sql_dialect}} code.\n"
        ),
    )
)
```

### Rich Metadata for Precision Training

Unlike standard datasets that give you a black box of question → SQL, every single record in the Nemotron Text-to-SQL dataset is tagged with **rich, granular metadata**.

This allows researchers and engineers to "slice and dice" the training data with surgical precision. Want to fine-tune a model specifically for Finance analytics using Window Functions in PostgreSQL? You can filter for exactly that subset.

The dataset includes structured fields for:

- **SQL Complexity**: Categorized levels from Beginner (simple SELECT) to Advanced (nested subqueries)
- **Task Type**: Specific intents like Data Cleaning, Aggregation, Pattern Matching, or Ranking
- **Industry & Topic**: Granular tags covering 60 distinct industries (e.g., Healthcare → Patient Records)
- **Dialect**: Explicit labels for MySQL, PostgreSQL, and SQLite to prevent syntax leakage

This structured approach transforms the dataset from a simple training corpus into a **strategic asset** for targeted model improvement.

## The "Quality Waterfall": Our Brutal Filtering Pipeline

Generating 300,000 samples with this pipeline is straightforward. Ensuring they are correct is the hard part. We implemented a rigorous **Quality Waterfall** in Data Designer that rejected over 68% of the generated data. If a sample didn't pass every gate, it was dropped.

Our pipeline uses a multi-stage approach to ensure fidelity:

### Syntax Validator

Is this valid SQL for the specific target dialect? (e.g., rejecting T-SQL functions in a PostgreSQL prompt).

```python
from data_designer.essentials import (
    ValidationColumnConfig,
    ValidatorType,
    CodeValidatorParams,
    CodeLang,
)

config_builder.add_column(
    ValidationColumnConfig(
        name="sql_validation",
        validator_type=ValidatorType.CODE,
        target_columns=["sql_with_reasoning"],
        validator_params=CodeValidatorParams(
            code_lang=CodeLang.SQL_POSTGRES,  # Dialect-specific validation
        ),
    )
)
```

### LLM-as-a-Critic

A separate "Judge" model scores the prompt naturalness, schema realism, and SQL logic, filtering out nonsensical or ambiguous requests.

```python
from data_designer.essentials import LLMJudgeColumnConfig, Score

config_builder.add_column(
    LLMJudgeColumnConfig(
        name="quality_scores",
        model_alias="nvidia-reasoning",
        prompt=(
            "Evaluate the following Text-to-SQL pair:\n"
            "Instruction: {{instruction}}\n"
            "Schema: {{schema}}\n"
            "SQL: {{sql_with_reasoning}}\n"
        ),
        scores=[
            Score(
                name="Prompt_Naturalness",
                description="Is the natural language request realistic?",
                options={5: "Perfect", 4: "Good", 3: "Acceptable", 2: "Poor", 1: "Unusable"},
            ),
            Score(
                name="Schema_Realism",
                description="Does the schema reflect real-world complexity?",
                options={5: "Perfect", 4: "Good", 3: "Acceptable", 2: "Poor", 1: "Unusable"},
            ),
            Score(
                name="SQL_Correctness",
                description="Is the SQL semantically correct for the task?",
                options={5: "Perfect", 4: "Good", 3: "Acceptable", 2: "Poor", 1: "Unusable"},
            ),
        ],
    )
)
```

The result? We started with 300k raw records and distilled them down to **96.5k platinum-tier samples**. We don't want the model to learn from hallucinations; we want it to learn from code that is structurally sound and logically robust.

## Teaching Reasoning, Not Just Syntax

This dataset is designed for **Reasoning Models**. We didn't just want the model to output the final SQL; we wanted it to **show its work**.

The training data includes "Chain of Thought" (CoT) traces where the model explicitly plans its approach:

> "The user wants to filter by date, but the 'timestamp' column is stored as TEXT. I need to first normalize this column using STR_TO_DATE before I can apply the WHERE clause..."

This teaches Nemotron to think like a **Data Engineer**: decomposing complex problems, handling edge cases, and verifying logic before writing a single line of code.

## Looking Ahead: The Code Sandbox

To push evaluation even further, we are actively implementing Data Designer's **Code Sandbox**. While syntax validators ensure code compiles, the Sandbox allows us to validate **semantic correctness** by actually executing the generated SQL against a ground-truth database and comparing the results (execution-based evaluation). This is the next leap forward in ensuring our synthetic data isn't just syntactically correct, but functionally perfect.

## Iteration & The Future

One of the most powerful features of NVIDIA Data Designer is that it treats data generation as a **workflow**, not a one-off script. We didn't get this dataset right on the first try—and thanks to Data Designer, we didn't have to.

We defined our entire generation pipeline—prompts, schemas, validators, and filters—as **portable configuration files**. This allowed us to iterate rapidly: generate a small batch, analyze the "Waterfall" drop-offs, adjust our constraints (e.g., "SQLite doesn't support LATERAL joins"), and re-run immediately.

Because this workflow is encapsulated in Data Designer, it isn't locked away in a private notebook. The configuration can be shared with any team, allowing them to **fork our baseline**, swap in their own schemas or industry verticals, and generate a custom, high-fidelity dataset for their specific domain in hours, not months.

## Summary: The Nemotron Text-to-SQL Dataset

We are excited to share this resource internally to accelerate our research:

- **Dataset Link**: Nemotron-Text-to-SQL-Internal
- **Scale**: 96.5k filtered records (from 300k raw)
- **Dialects**: MySQL, PostgreSQL, SQLite
- **Diversity**: 60 Industries, 700 Topics, 90 SQL Concepts
- **Quality**: 100% Execution-Verified Schema & SQL

## A Team Effort

This milestone is the culmination of years of work in synthetic data. It builds on the foundation laid during our time at Gretel.ai (creators of the #1 trending synthetic text-to-sql dataset on Hugging Face). Today, we are proud to bring that DNA into NVIDIA, building the data infrastructure that powers the next generation of Nemotron models.

---

**Want to build your own enterprise-grade datasets?** Check out the [Text-to-SQL Recipe](../../recipes/code_generation/text_to_sql.md) and [Data Designer documentation](../../index.md) to get started.

