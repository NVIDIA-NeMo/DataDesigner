# Columns

Columns are the fundamental building blocks in Data Designer. Each column represents a field in your dataset and defines how to generate it—whether that's sampling from a distribution, calling an LLM, or applying a transformation.

!!! note "The Declarative Approach"
    Columns are **declarative specifications**. You describe *what* you want, and the framework handles *how* to generate it—managing execution order, batching, parallelization, and resources automatically.

## Column Types

Data Designer provides nine built-in column types, each optimized for different generation scenarios.

### 🎲 Sampler Columns

Sampler columns generate data using numerical sampling—fast, deterministic, and ideal for numerical and categorical dataset fields. They're significantly faster than LLMs and can produce data following specific distributions (Poisson for event counts, Gaussian for measurements, etc.).

Available sampler types:

- **UUID**: Unique identifiers
- **Category**: Categorical values with optional probability weights
- **Subcategory**: Hierarchical categorical data (states within countries, models within brands)
- **Uniform**: Evenly distributed numbers (integers or floats)
- **Gaussian**: Normally distributed values with configurable mean and standard deviation
- **Bernoulli**: Binary outcomes with specified success probability
- **Bernoulli Mixture**: Binary outcomes from multiple probability components
- **Binomial**: Count of successes in repeated trials
- **Poisson**: Count data and event frequencies
- **Scipy**: Access to the full scipy.stats distribution library
- **Person**: Realistic synthetic individuals with names, demographics, and attributes
- **Datetime**: Timestamps within specified ranges
- **Timedelta**: Time duration values

!!! tip "Conditional Sampling"
    Samplers support **conditional parameters** that change behavior based on other columns. Want age distributions that vary by country? Income ranges that depend on occupation? Just define conditions on existing column values.

### 📝 LLM-Text Columns

LLM-Text columns generate natural language text: product descriptions, customer reviews, narrative summaries, email threads, or anything requiring semantic understanding and creativity.

Use **Jinja2 templating** in prompts to reference other columns. Data Designer automatically manages dependencies and injects the referenced column values into the prompt.

!!! note "Reasoning Traces"
    Models that support extended thinking (chain-of-thought reasoning) can capture their reasoning process in a separate `{column_name}__reasoning_trace` column–useful for understanding *why* the model generated specific content. This column is automatically added to the dataset if the model and service provider parse and return reasoning content.

### 💻 LLM-Code Columns

LLM-Code columns generate code in specific programming languages. They handle the prompting and parsing necessary to extract clean code from the LLM's response—automatically detecting and extracting code from markdown blocks. You provide the prompt and choose the model; the column handles the extraction.

Supported languages: **Python, JavaScript, TypeScript, Java, Kotlin, Go, Rust, Ruby, Scala, Swift**, plus **SQL** dialects (SQLite, PostgreSQL, MySQL, T-SQL, BigQuery, ANSI SQL).

### 🗂️ LLM-Structured Columns

LLM-Structured columns generate JSON with a *guaranteed schema*. Define your structure using a Pydantic model or JSON schema, and Data Designer ensures the LLM output conforms—no parsing errors, no schema drift.

Use for complex nested structures: API responses, configuration files, database records with multiple related fields, or any structured data where type safety matters. Schemas can be arbitrarily complex with nested objects, arrays, enums, and validation constraints, but success depends on the model's capabilities.

!!! tip "Schema Complexity and Model Choice"
    Flat schemas with simple fields are easier and more robustly produced across models. Deeply nested schemas with complex validation constraints are more sensitive to model choice—stronger models handle complexity better. If you're experiencing schema conformance issues, try simplifying the schema or switching to a more capable model.

### ⚖️ LLM-Judge Columns

LLM-Judge columns score generated content across multiple quality dimensions using LLMs as evaluators.

Define scoring rubrics (relevance, accuracy, fluency, helpfulness) and the judge model evaluates each record. Score rubrics specify criteria and scoring options (1-5 scales, categorical grades, etc.), producing quantified quality metrics for every data point.

Use judge columns for data quality filtering (e.g., keep only 4+ rated responses), A/B testing generation strategies, and quality monitoring over time.

### 🧩 Expression Columns

Expression columns handle simple transformations using **Jinja2 templates**—concatenate first and last names, calculate numerical totals, format date strings. No LLM overhead needed.

Template capabilities:

- **Variable substitution**: Pull values from any existing column
- **String filters**: Uppercase, lowercase, strip whitespace, replace patterns
- **Conditional logic**: if/elif/else support
- **Arithmetic**: Add, subtract, multiply, divide

### 🔍 Validation Columns

Validation columns check generated content against rules and return structured pass/fail results.

Built-in validation types:

**Code validation** runs Python or SQL code through a linter to validate the code.

**Local callable validation** accepts a Python function directly when using Data Designer as a library.

**Remote validation** sends data to HTTP endpoints for validation-as-a-service. Useful for linters, security scanners, or proprietary systems.

### 🌱 Seed Dataset Columns

Seed dataset columns bootstrap generation from existing data. Provide a real dataset, and those columns become available as context for generating new synthetic data.

Typical pattern: use seed data for one part of your schema (real product names and categories), then generate synthetic fields around it (customer reviews, purchase histories, ratings). The seed data provides realism and constraints; generated columns add volume and variation.

## Shared Column Properties

Every column configuration inherits from `SingleColumnConfig` with these standard properties:

### `name`

The column's identifier—unique within your configuration, used in Jinja2 references, and becomes the column name in the output DataFrame. Choose descriptive names: `user_review` > `col_17`.

### `drop`

Boolean flag (default: `False`) controlling whether the column appears in final output. Setting `drop=True` generates the column (available as a dependency) but excludes it from final output.

**When to drop columns:**

- Intermediate calculations that feed expressions but aren't meaningful standalone
- Context columns used only for LLM prompt templates
- Validation results during development unwanted in production

Dropped columns participate fully in generation and the dependency graph—just filtered out at the end.

### `column_type`

Literal string identifying the column type: `"sampler"`, `"llm-text"`, `"expression"`, etc. Set automatically by each configuration class and serves as Pydantic's discriminator for deserialization.

You rarely set this manually—instantiating `LLMTextColumnConfig` automatically sets `column_type="llm-text"`. Serialization is reversible: save to YAML, load later, and Pydantic reconstructs the exact objects.

### `required_columns`

Computed property listing columns that must be generated before this one. The framework derives this automatically:

- For LLM/Expression columns: extracted from Jinja2 template `{{ variables }}`
- For Validation columns: explicitly listed target columns
- For Sampler columns with conditional parameters: columns referenced in conditions

You read this property for introspection but never set it—always computed from configuration details.

### `side_effect_columns`

Computed property listing columns created implicitly alongside the primary column. Currently, only LLM columns produce side effects (reasoning trace columns like `{name}__reasoning_trace` when models use extended thinking).

For detailed information on each column type, refer to the [column configuration code reference](../code_reference/column_configs.md).
