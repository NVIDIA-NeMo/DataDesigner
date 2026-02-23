---
date: 2026-02-18
authors:
  - dnathawani
  - ymeyer
  - mvansegbroeck
---

# **Engineering an Enterprise-Grade Text-to-SQL Dataset with NeMo Data Designer**

While LLMs have mastered generic coding, Text-to-SQL remains one of the most challenging frontiers in enterprise AI. Using NeMo Data Designer with conditional sampling, three-stage LLM generation, code validators, and multi-dimension judge scoring, we built a pipeline that generated 300,000 reasoning-heavy text-to-SQL samples --- filtered down to 96,500 --- across PostgreSQL, MySQL, and SQLite. This data powers Nemotron's SQL capabilities, targeting top-tier performance on benchmarks like [BIRD](https://bird-bench.github.io/) and [Spider 2.0](https://spider2-sql.github.io/).

<!-- more -->

---

## **The "Real-World" Gap: Why Academic Data Wasn't Enough**

The gap between academic benchmarks and the messy reality of enterprise data warehouses is massive. On academic benchmarks like Spider (where schemas are clean, tables are few, and queries are straightforward), frontier models score above 85%. On [BIRD](https://bird-bench.github.io/) (which introduces dirty data, larger schemas, and external knowledge requirements), the same models drop to 60-70%. On [Spider 2.0 Lite](https://spider2-sql.github.io/) (which uses real enterprise databases with hundreds of tables, multiple dialects, and complex business logic), even the best models score below 50%.

The problem isn't model capability --- it's **training data**. Most open-source text-to-SQL datasets assume a "happy path": intuitive column names, perfect data types, and straightforward questions. Production SQL is different:

- **Dialect specificity.** Generic "SQL" doesn't compile. We needed valid, executable code for MySQL, PostgreSQL, and SQLite that respects their unique syntax --- `date('now')` in SQLite vs. `CURRENT_DATE` in Postgres, `DISTINCT ON` in PostgreSQL vs. nested subqueries in MySQL.
- **Dirty data.** Real columns contain currency symbols (`$57,500`), mixed date formats, and JSON blobs. The model needs to learn *defensive SQL*: writing queries that use `CAST`, `STR_TO_DATE`, and string manipulation functions to clean data at query time before attempting any aggregation. We explicitly prompted the generation engine to introduce anti-patterns like storing dates as text (`'01-Jan-2023'`), including currency symbols in pricing columns, or burying critical flags inside JSON blobs.
- **Distractor tables and schema linking.** In production, you rarely get just the 2 tables you need; you're more likely to get a schema with 50 tables, many of which look identical. We injected semantically similar "distractor" tables into every context --- `sales_orders` vs. `sales_orders_archive`, `customer_leads` vs. `active_customers` --- forcing the model to perform schema linking based on column constraints and relationships, not just table names.
- **Industry-specific schemas.** Healthcare EHR tables look nothing like financial trading systems. The column names, relationships, and business logic are domain-specific.
- **Complexity gradients.** Junior analysts write simple SELECTs; senior engineers write recursive CTEs with window functions. Training data needs the full spectrum.

For Nemotron's SQL capabilities, we needed synthetic training data that mirrors production complexity. The key insight: **domain diversity and complexity coverage matter more than dataset size**.

---

## **Pipeline Architecture**

The pipeline uses Data Designer's conditional sampling (`SubcategorySamplerParams`) to create correlated diversity across 60 industries, 700 topics, and 90 SQL concepts. It then chains three LLM generation stages with a code validator and multi-dimension judge:

```
                                        TEXT-TO-SQL SDG PIPELINE
                                        =======================

             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                          STAGE 1: CONDITIONAL SAMPLERS                              │
             │                                                                                     │
             │   Domain Controls                   SQL Controls            Prompt Controls         │
             │   ├─ industry_sector (60)           ├─ sql_complexity       ├─ instruction_style    │
             │   └─ topic (700 subcategories)      │   Beginner / Inter-   │   imperative /        │
             │       ↳ conditioned on industry     │   mediate / Advanced  │   declarative /       │
             │                                     ├─ sql_concept (90)     │   interrogative /     │
             │                                     │   ↳ conditioned on    │   contextual          │
             │                                     │     complexity        └─ tone, register       │
             │                                     └─ sql_dialect                                  │
             │                                         PostgreSQL / MySQL / SQLite                 │
             └─────────────────────────────────────────┬───────────────────────────────────────────┘
                                                       │
                                                       ▼
             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                       STAGE 2: THREE-STAGE LLM GENERATION                           │
             │                                                                                     │
             │   sql_prompt ──────────► sql_context ──────────► sql                                │
             │   (natural language        (CREATE TABLE +         (SQL query with                  │
             │    business request)        INSERT statements       chain-of-thought                │
             │                             + distractor tables     reasoning trace)                │
             │                             + dirty data)                                           │
             └─────────────────────────────────────────┬───────────────────────────────────────────┘
                                                       │
                                                       ▼
             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                          STAGE 3: QUALITY WATERFALL                                 │
             │                                                                                     │
             │   Hard validation:                                                                  │
             │     SQLFluff syntax check (dialect-aware: ANSI / Postgres / MySQL / SQLite)         │
             │                                                                                     │
             │   LLM Judge (4 dimensions × 0-4 scale):                                             │
             │     1. Relevance ─── Does the query answer the business request?                    │
             │     2. Correctness ─ Valid joins, filters, grouping, NULL handling?                 │
             │     3. Readability ─ Formatting, aliases, CTEs where helpful?                       │
             │     4. Efficiency ── Sargable predicates, appropriate joins?                        │
             │                                                                                     │
             │   Filter: syntax valid AND all dimensions ≥ 3                                       │
             └─────────────────────────────────────────┬───────────────────────────────────────────┘
                                                       │
                                                       ▼
             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                              OUTPUT: 96,500 RECORDS                                 │
             │                                                                                     │
             │   300k generated → 96.5k after Quality Waterfall (68% rejection)                    │
             │   Dialects: PostgreSQL, MySQL, SQLite                                               │
             │   60 industries · 700 topics · 90 SQL concepts · 100% syntax-verified               │
             └─────────────────────────────────────────────────────────────────────────────────────┘
```

The critical feature is **two-level conditional sampling**: `topic` depends on `industry_sector`, and `sql_concept` depends on `sql_complexity`. This ensures coherent records --- you don't get "Window Functions" paired with "Beginner" complexity, or "Electronic Health Records" paired with a "Finance" industry.

---

## **Step 1: Semantic Sampling with SubcategorySamplerParams**

Standard categorical samplers draw independently from their value lists. `SubcategorySamplerParams` creates hierarchical dependencies, controlling the distribution of the data through what we call "Semantic Blueprints":

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

config = dd.DataDesignerConfigBuilder(model_configs=[
    dd.ModelConfig(
        alias="sql-gen",
        model="qwen/qwen3-235b-a22b",
        provider="nvidia",
    ),
])

# Industry → Topic (two-level conditional)
config.add_column(dd.SamplerColumnConfig(
    name="industry_sector",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=[
        "Healthcare", "Finance", "Technology", "Retail", "Manufacturing",
        "Aerospace", "Energy", "Telecommunications", "Transportation", "Education",
        # ... 60 industries total
    ]),
))

config.add_column(dd.SamplerColumnConfig(
    name="topic",
    sampler_type=dd.SamplerType.SUBCATEGORY,
    params=dd.SubcategorySamplerParams(
        category="industry_sector",
        values={
            "Healthcare":    ["Electronic Health Records", "Telemedicine Platforms",
                              "Clinical Trials", "Patient Scheduling", "Insurance Claims"],
            "Finance":       ["Fraud Detection", "Trading Systems", "Risk Assessment",
                              "Portfolio Management", "Regulatory Compliance"],
            "Technology":    ["Cloud Platforms", "ML Pipelines", "DevOps Tools",
                              "API Gateway Logs", "User Analytics"],
            "Retail":        ["Inventory Management", "Customer Segmentation",
                              "Pricing Optimization", "Supply Chain", "Returns Processing"],
            # ... 700 subcategories across all industries
        },
    ),
))
```

When `industry_sector` samples "Healthcare", `topic` is drawn only from healthcare-specific subcategories. This is the difference between realistic training data and random noise.

The same pattern controls SQL concepts and prompt diversity:

```python
# Complexity → SQL Concept (two-level conditional)
config.add_column(dd.SamplerColumnConfig(
    name="sql_complexity",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["Beginner", "Intermediate", "Advanced"]),
))

config.add_column(dd.SamplerColumnConfig(
    name="sql_concept",
    sampler_type=dd.SamplerType.SUBCATEGORY,
    params=dd.SubcategorySamplerParams(
        category="sql_complexity",
        values={
            "Beginner":     ["Basic SELECT", "WHERE Clauses", "Basic JOINs",
                             "ORDER BY", "LIMIT/OFFSET", "DISTINCT"],
            "Intermediate": ["Aggregation Functions", "Multiple JOINs", "Subqueries",
                             "Views", "CASE Expressions", "Date Functions", "String Functions"],
            "Advanced":     ["Window Functions", "Recursive CTEs", "Stored Procedures",
                             "Query Optimization", "JSON Extraction", "Lateral Joins",
                             "Pivoting", "Dynamic SQL"],
        },
    ),
))

# Dialect control
config.add_column(dd.SamplerColumnConfig(
    name="sql_dialect",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["PostgreSQL", "MySQL", "SQLite"]),
))

# Prompt diversity: linguistic register, instruction style, politeness
config.add_column(dd.SamplerColumnConfig(
    name="instruction_style",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=[
        "imperative", "declarative", "interrogative", "contextual",
    ]),
))
```

We systematically varied the linguistic register (formal vs. colloquial), instruction style, and politeness level to robustly handle any user persona. A CFO asking "Can you pull the Q3 numbers?" and an engineer saying "Write a query that joins sales on customer_id" should both produce correct SQL.

---

## **Step 2: Three-Stage LLM Generation**

The pipeline chains three LLM columns, each with a focused task. This decomposition is essential --- when you ask a single prompt to generate all three, the SQL tends to reference tables that don't exist in the schema, or the schema doesn't contain the columns the SQL needs.

**Stage 1 --- Natural language prompt:** Generate a business request that *implicitly* requires the sampled SQL concept, without using SQL jargon.

```python
config.add_column(dd.LLMTextColumnConfig(
    name="sql_prompt",
    model_alias="sql-gen",
    prompt=(
        "Generate a natural language data request from a {{ instruction_style }} "
        "business user in the {{ industry_sector }} industry, specifically about "
        "{{ topic }}. The request should implicitly require {{ sql_concept }} to answer, "
        "targeting {{ sql_dialect }} syntax.\n\n"
        "Do NOT use SQL terminology. Write it as a business person would ask it.\n\n"
        "Example: 'For the quarterly review, pull the patient records with their most "
        "recent lab test dates.'"
    ),
))
```

**Stage 2 --- Database context:** Generate `CREATE TABLE` and `INSERT` statements that provide the schema and sample data needed to answer the request. This is where dirty data and distractor tables are introduced:

```python
config.add_column(dd.LLMTextColumnConfig(
    name="sql_context",
    model_alias="sql-gen",
    prompt=(
        "Generate a realistic {{ sql_dialect }} database schema to answer this request:\n"
        "{{ sql_prompt }}\n\n"
        "Requirements:\n"
        "- Use {{ sql_dialect }}-specific syntax for CREATE TABLE and INSERT statements\n"
        "- Include 3-5 tables, with at least 1 distractor table that is semantically "
        "similar but NOT needed for the query\n"
        "- Include realistic data quality issues: dates stored as text, currency symbols "
        "in numeric fields, NULL values, inconsistent formats\n"
        "- Industry: {{ industry_sector }} / {{ topic }}\n"
        "- Include 5-10 sample INSERT rows per table\n\n"
        "Return ONLY the SQL DDL and INSERT statements."
    ),
))
```

**Stage 3 --- SQL query with reasoning:** Write the SQL that answers the request using the provided context, including a chain-of-thought trace:

```python
config.add_column(dd.LLMTextColumnConfig(
    name="sql",
    model_alias="sql-gen",
    prompt=(
        "Write a {{ sql_dialect }} query that answers this request:\n"
        "{{ sql_prompt }}\n\n"
        "Using this database:\n{{ sql_context }}\n\n"
        "First, explain your reasoning step by step:\n"
        "1. Which tables are relevant (and which are distractors)?\n"
        "2. What data quality issues need to be handled?\n"
        "3. What {{ sql_dialect }}-specific syntax is required?\n\n"
        "Then write the final SQL query. Use CTEs for complex logic."
    ),
))
```

The chain-of-thought traces teach the model to *think like a Data Engineer*: decomposing complex problems, handling edge cases, and verifying logic before writing a single line of code. A typical reasoning trace looks like:

> "The user wants to filter by date, but the 'timestamp' column is stored as TEXT. I need to first normalize this column using STR_TO_DATE before I can apply the WHERE clause..."

---

## **Step 3: The Quality Waterfall**

Generating 300,000 samples is straightforward. Ensuring they are correct is the hard part. We implemented a rigorous "Quality Waterfall" that rejected over 68% of the generated data.

### Hard Validation with SQLFluff

Data Designer's `ValidationColumnConfig` with `CodeValidatorParams` runs generated SQL through SQLFluff, a dialect-aware SQL linter:

```python
config.add_column(dd.ValidationColumnConfig(
    name="sql_validity",
    validator_type=dd.ValidatorType.CODE,
    target_columns=["sql"],
    validator_params=dd.CodeValidatorParams(code_lang=dd.CodeLang.SQL_ANSI),
    batch_size=10,
))
```

The validator returns `is_valid` (boolean) and `error_messages` (string). Records that fail parsing are flagged immediately. Supported dialects: `SQL_SQLITE`, `SQL_POSTGRES`, `SQL_MYSQL`, `SQL_TSQL`, `SQL_BIGQUERY`, `SQL_ANSI`.

### Multi-Dimension Judge Scoring

Beyond syntax validity, we evaluate SQL *quality* across four dimensions on a 0-4 scale:

```
     ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
     │    Relevance     │  │  SQL Correctness │  │   Readability    │  │    Efficiency    │
     │                  │  │                  │  │                  │  │                  │
     │  Does the query  │  │  Valid joins,    │  │  Formatting,     │  │  Sargable        │
     │  answer the      │  │  filters,        │  │  aliases,        │  │  predicates,     │
     │  business        │  │  grouping,       │  │  CTEs where      │  │  appropriate     │
     │  request?        │  │  NULL handling?  │  │  helpful?        │  │  joins?          │
     └──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

Each dimension has explicit scoring criteria:

| Score | Relevance | Correctness | Readability | Efficiency |
|-------|-----------|-------------|-------------|------------|
| 4 | Perfectly meets requirements | Valid SQL, correct semantics | Clean formatting, meaningful aliases | Sargable predicates, optimal plan |
| 3 | Minor deviations | Generally correct, minor issues | Generally readable | Mostly efficient |
| 2 | Moderate deviation | Noticeable semantic mistakes | Inconsistent formatting | Moderate inefficiencies |
| 1 | Significant deviations | Major errors | Poor formatting | Notable performance issues |
| 0 | Does not adhere | Invalid SQL | Unreadable | Highly inefficient |

The judge provides a score *and* reasoning for each dimension, making it easy to diagnose why a record scored low. We filtered to records scoring ≥ 3 across all four dimensions.

### Waterfall Summary

The combined pipeline rejected records at each stage:

| Stage | Records In | Records Out | Drop Rate |
|-------|-----------|-------------|-----------|
| Raw generation | 300,000 | 300,000 | --- |
| SQLFluff syntax validation | 300,000 | ~180,000 | ~40% |
| LLM Judge (all dimensions ≥ 3) | ~180,000 | 96,500 | ~46% |
| **Final dataset** | | **96,500** | **68% total rejection** |

---

## **Rich Metadata for Precision Training**

We didn't just generate text pairs --- we generated structured data. Unlike standard datasets that give you a black box of question → SQL, every single record is tagged with rich, granular metadata:

| Field | Description | Example Values |
|-------|-------------|----------------|
| `industry_sector` | Domain vertical | Healthcare, Finance, Aerospace |
| `topic` | Specific subdomain | Electronic Health Records, Fraud Detection |
| `sql_complexity` | Difficulty tier | Beginner, Intermediate, Advanced |
| `sql_concept` | Target SQL skill | Window Functions, Recursive CTEs |
| `sql_dialect` | Target database | PostgreSQL, MySQL, SQLite |
| `instruction_style` | Prompt register | imperative, interrogative, contextual |
| `relevance_score` | Judge: relevance | 0-4 |
| `correctness_score` | Judge: SQL correctness | 0-4 |
| `readability_score` | Judge: formatting | 0-4 |
| `efficiency_score` | Judge: query plan | 0-4 |

This allows researchers and engineers to "slice and dice" the training data with surgical precision. If you want to fine-tune a model specifically for Finance analytics using Window Functions in PostgreSQL, you can filter for exactly that subset.

---

## **Results**

| Metric | Value |
|--------|-------|
| Records generated | 300,000 |
| Records after Quality Waterfall | 96,500 |
| Rejection rate | 68% |
| SQL dialects | PostgreSQL, MySQL, SQLite |
| Industry coverage | 60 distinct industries |
| Topic coverage | 700 distinct subcategories |
| SQL concept coverage | 90 concepts across 3 complexity tiers |
| Syntax validation | 100% SQLFluff-verified |
| Minimum judge score | ≥ 3/4 across all four dimensions |

The high rejection rate is a feature, not a bug. By generating 3x more data than we needed and filtering aggressively, we ensured every record in the final dataset is both syntactically valid and semantically meaningful.

---

## **Key Takeaways**

1. **Conditional sampling prevents incoherent records.** `SubcategorySamplerParams` ensures "Window Functions" only appears with "Advanced" complexity, and "EHR Systems" only appears with "Healthcare". Independent samplers would produce nonsensical combinations that confuse training.

2. **Three-stage generation beats one-shot.** Separating prompt, schema, and query generation ensures the SQL actually references the tables that exist. One-shot generation frequently hallucinates tables.

3. **Dirty data must be intentional.** Explicitly prompting for anti-patterns (dates as text, currency symbols, JSON blobs) forces the model to learn defensive SQL. Clean schemas produce clean-only training data.

4. **Distractor tables teach schema linking.** Injecting semantically similar but irrelevant tables forces the model to *read* the schema instead of guessing from table names. This is the skill gap between academic benchmarks and production.

5. **Hard validators are non-negotiable for code.** LLM judges can assess quality, but they can't reliably detect syntax errors. SQLFluff catches parsing failures that the judge misses.

6. **Multi-dimension scoring enables targeted filtering.** A query that scores 4 on Relevance but 1 on Efficiency tells you the model understood the task but wrote a bad plan. You can filter differently depending on what you're training for.

7. **Chain-of-thought teaches reasoning, not just syntax.** Including reasoning traces in the training data teaches models to decompose problems, handle edge cases, and verify logic --- acting as a Data Engineer rather than a translator.

---

## **Looking Ahead: The Code Sandbox**

The current Quality Waterfall validates syntax (SQLFluff) and assesses quality (LLM judge), but it doesn't verify *semantic correctness* --- whether the query actually returns the right results. We are actively implementing Data Designer's Code Sandbox to close this gap. The sandbox would execute generated SQL against a ground-truth database and compare results, enabling:

- **Execution-based filtering:** Reject queries that parse but return wrong results.
- **End-to-end verification:** Confirm that the full chain (prompt → schema → SQL → result) is semantically coherent.
- **Harder negative mining:** Queries that execute but return incorrect results are valuable hard negatives for preference training.

---

## **A Team Effort**

This dataset builds on the foundation laid during our time at [Gretel.ai](https://gretel.ai) (creators of the [#1 trending synthetic text-to-SQL dataset on Hugging Face](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)). Today, we're proud to bring that DNA into NVIDIA, building the data infrastructure that powers the next generation of Nemotron models.

**Dataset:** [Nemotron-Text-to-SQL-Internal](#) | **Scale:** 96.5k filtered records | **Dialects:** MySQL, PostgreSQL, SQLite

Key Resources:

1. [NeMo Data Designer on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)
2. [BIRD Benchmark](https://bird-bench.github.io/)
3. [Spider 2.0 Benchmark](https://spider2-sql.github.io/)

Because this pipeline is encapsulated in Data Designer, the configuration can be shared with any team --- allowing them to fork our baseline, swap in their own schemas or industry verticals, and generate a custom, high-fidelity dataset for their specific domain in hours, not months.

---

*Want to learn more about NeMo Data Designer? Check out our [documentation](https://github.com/NVIDIA-NeMo/DataDesigner) and start building your own high-fidelity synthetic datasets today.*
