# Columns

Columns are the heart of Data Designer. If you're building a synthetic dataset, you're building with columns.

Think of columns as declarative recipes for data generation. Instead of writing procedural code to generate each value, you describe *what* you want—a UUID here, an LLM-generated story there, a validation check on some generated code—and Data Designer figures out the *how*. Each column represents a field in your dataset, but more importantly, each column encapsulates a complete generation strategy.

!!! note "Columns vs Traditional Data Generation"
    Unlike traditional data generation approaches where you write loops and functions, Data Designer columns are **declarative**. You specify the desired outcome through configuration, and the framework handles execution order, batching, parallelization, and resource management. This separation of concerns lets you focus on your data's structure and semantics rather than generation mechanics.

## What You Can Do With Columns

The column system's flexibility enables several powerful patterns:

- **Generate from scratch** using statistical samplers or LLMs—no seed data required
- **Build context chains** where one column references others, creating rich interconnected data
- **Derive features** using Jinja2 expressions for transformations without LLM overhead
- **Ensure quality** with validation columns that check generated content automatically
- **Bootstrap from reality** by using seed datasets as templates for synthetic generation
- **Extend the framework** through plugins when built-in column types don't fit your needs

The real power emerges when you combine these patterns. A typical dataset might start with sampled demographic data, use that to generate contextually appropriate LLM text, derive computed fields from the generated text, and validate everything before export.

## Column Types

Data Designer provides nine built-in column types, each optimized for different data generation scenarios. Choose the right column type and your data generation becomes simpler, faster, and more maintainable.

### Sampler Columns

When you need structured data with statistical properties but don't want the overhead of LLM calls, sampler columns are your workhorse. They generate data using numerical sampling techniques—fast, deterministic, and perfect for foundational dataset fields.

**What makes samplers powerful:** They're *blazingly fast* compared to LLM generation, they can produce data following specific distributions (need a Poisson distribution for event counts? done.), and they integrate seamlessly with Python's scientific computing ecosystem through scipy.

Available sampler types span a wide range of use cases:

- **UUID** for unique identifiers
- **Category** for categorical values with optional probability weights
- **Subcategory** for hierarchical categorical data (states within countries, models within brands)
- **Uniform** for evenly distributed numbers—integers or floats
- **Gaussian** for normally distributed values with configurable mean and standard deviation
- **Bernoulli** for binary outcomes with specified success probability
- **Bernoulli Mixture** for binary outcomes drawn from multiple probability components
- **Binomial** for counting successes in repeated trials
- **Poisson** for modeling count data and event frequencies
- **Scipy** for access to the entire scipy.stats distribution library
- **Person** for realistic synthetic individuals with names, demographics, and attributes
- **Datetime** for timestamps within specified ranges
- **Timedelta** for time duration values

!!! tip "Conditional Sampling"
    Sampler columns support **conditional parameters**—different sampling behavior triggered by conditions on other columns. Want age distributions that vary by country? Income ranges that depend on occupation? Conditional samplers make this trivial. The condition is evaluated against existing column values, enabling sophisticated interdependent generation patterns.

### LLM-Text Columns

This is where Large Language Models shine: generating creative, contextual, human-like text. LLM-Text columns produce free-form natural language—product descriptions, customer reviews, narrative summaries, email threads, or any text that benefits from semantic understanding and creativity.

The magic happens through **Jinja2 templating** in your prompts. Reference other columns by name, and Data Designer automatically manages dependencies and context injection. Generate a product review that mentions the specific product name, price, and category from other columns. Create customer support responses tailored to the customer's demographics and purchase history.

!!! note "Reasoning Traces"
    When you're using models that support extended thinking (like those with chain-of-thought reasoning), LLM-Text columns can capture the model's reasoning process in a separate column. This is invaluable for understanding *why* the model generated specific content and for debugging generation quality issues.

**Multi-modal capability:** All LLM columns support image inputs. Vision-capable models can generate text based on images, opening up use cases like image captioning, visual question answering, or generating product descriptions from product photos.

### LLM-Code Columns

Writing synthetic code datasets? LLM-Code columns are specialized for generating syntactically correct code in specific programming languages. Unlike generic text generation, code columns understand language-specific context and automatically extract clean code from markdown blocks—no manual parsing required.

Supported languages span the programming spectrum: **Python, JavaScript, TypeScript, Java, Kotlin, Go, Rust, Ruby, Scala, Swift**, plus **SQL** in multiple dialects (SQLite, PostgreSQL, MySQL, T-SQL, BigQuery, ANSI SQL). Specify the language, provide a prompt (which can reference other columns), and get properly formatted code.

Use cases include generating training data for code models, creating programming exercises, building test suites, or producing SQL queries that match schema descriptions in other columns.

### LLM-Structured Columns

When you need JSON output with a *guaranteed schema*, LLM-Structured columns deliver. Define your desired structure using a Pydantic model or JSON schema, and Data Designer ensures the LLM output conforms—no manual validation, no parsing errors, no schema drift.

This column type excels at generating complex nested structures: API responses, configuration files, database records with multiple related fields, or any structured data where type safety matters. The schema can be arbitrarily complex with nested objects, arrays, enums, and validation constraints.

!!! tip "Schema as Contract"
    Think of the schema as a contract between your generation logic and downstream consumers. The LLM-Structured column guarantees this contract is honored, making it safe to pipe generated data directly into systems expecting specific formats.

### LLM-Judge Columns

Not all generated content is created equal. LLM-Judge columns put an AI evaluator on your data pipeline, scoring generated content across multiple quality dimensions.

Define scoring rubrics—relevance, accuracy, fluency, helpfulness, whatever matters for your use case—and the judge model evaluates each record independently. Each rubric specifies clear criteria and score options (1-5 scales, categorical grades, whatever fits your needs). The result? Quantified quality metrics for every generated data point.

**Why this matters:** Judge columns enable data quality filtering (keep only 4+ rated responses), A/B testing of generation strategies (which prompt produces better output?), and quality monitoring over time. They're particularly powerful for instruction-tuning datasets, where response quality directly impacts model performance.

### Expression Columns

Sometimes you just need simple transformations—concatenate first and last names, calculate a total from quantity and price, format a date string. For these cases, spinning up an LLM is overkill. Expression columns handle derived values efficiently using **Jinja2 templates**.

The template language is remarkably expressive:

- **Variable substitution**: Pull values from any existing column
- **String filters**: Uppercase, lowercase, strip whitespace, replace patterns
- **Conditional logic**: Full if/elif/else support for branching logic
- **Arithmetic**: Add, subtract, multiply, divide numerical values
- **Type casting**: Convert results to int, float, str, or bool

Expression columns evaluate row-by-row using pandas vectorization under the hood, making them fast even for large datasets. They're perfect for feature engineering, data cleaning, and simple business logic that doesn't require AI.

### Seed Dataset Columns

Want to ground your synthetic data in reality? Seed dataset columns let you bootstrap generation from existing data. Provide a real dataset (a CSV, Parquet file, or DataFrame), and those columns become available as templates and context for generating new synthetic data.

The typical pattern: use seed data for one part of your schema (say, real product names and categories), then generate synthetic fields around it (customer reviews, purchase histories, ratings). The seed data provides realism and constraints, while generated columns add volume and variation.

**Sampling modes:** Sample with or without replacement from your seed, and configure repeating if you want to generate more rows than your seed contains. Data Designer handles the logistics—you just specify what seed columns you want available.

!!! note "Automatic Configuration"
    When you call `with_seed_dataset()` on the config builder, seed columns are created automatically. You rarely instantiate `SeedDatasetColumnConfig` directly—the builder handles it for you.

### Validation Columns

Trust but verify. Validation columns act as quality gates, checking generated content against rules and returning structured pass/fail results. This catches generation errors *during* the data creation process, not after you've deployed a broken dataset.

Three validation strategies are supported:

**Code validation** executes Python or SQL for arbitrary validation logic. Your code receives a DataFrame with the target columns and returns a DataFrame with validation results—maximum flexibility for custom checks.

**Local callable validation** lets you pass a Python function directly when using Data Designer as a library. Perfect for integrating existing validation logic or third-party validators.

**Remote validation** sends data to HTTP endpoints for validation-as-a-service scenarios. Useful for linters, security scanners, or proprietary validation systems you can't embed locally.

Validation columns process data in configurable batches (balancing throughput and memory) and can validate multiple columns simultaneously. Run a linter on generated code, check SQL syntax, verify data integrity constraints, or enforce business rules—all declaratively.

### Custom Column Types

The nine built-in column types cover most use cases, but when you need something specialized, the plugin system has you covered. Write a custom column generator, register it as a plugin, and it integrates seamlessly with the rest of Data Designer.

Custom columns participate fully in dependency resolution, batch processing, configuration serialization, and the execution DAG. From the user's perspective, they're indistinguishable from built-in columns—they just happen to implement your domain-specific generation logic.

Common plugin use cases: integrating proprietary data generation systems, implementing specialized sampling algorithms, connecting to custom APIs, or generating domain-specific structured formats.

## Column Configurations vs Generators

Data Designer maintains a clean separation between *what* you want to generate (configurations) and *how* it gets generated (generators). Understanding this distinction helps clarify the framework's architecture.

### Column Configurations

When you work with Data Designer, you interact with **configuration objects**—Pydantic models that declaratively specify your columns. Each configuration class (`SamplerColumnConfig`, `LLMTextColumnConfig`, `ExpressionColumnConfig`, etc.) captures:

- Column identity (name and type)
- Type-specific parameters (sampler distributions, LLM prompts, validation rules)
- Processing hints (batch sizes, whether to drop the column from final output)

Configurations are the user-facing API. They're serializable (save to JSON/YAML, load back), validatable (Pydantic catches errors early), and portable (share configurations across teams or environments).

!!! info "Discriminated Unions"
    The `column_type` field acts as a **discriminator** in Pydantic's discriminated union system. When you load a configuration from JSON or YAML, Pydantic reads the `column_type` value and automatically instantiates the correct configuration class. This is why you can serialize complex multi-column configurations without manual deserialization logic.

### Column Generators

Behind every configuration lives a **generator**—the engine that produces actual data. Generators are instantiated automatically from configurations and handle the execution details:

- Batch processing for efficiency
- Resource management (API rate limits, connection pooling)
- Retry logic and error handling
- Integration with the execution DAG

You rarely touch generators directly. They're internal machinery that translates your declarative specifications into concrete data. This separation keeps the user-facing API simple (you describe *what* you want) while allowing the framework to optimize *how* generation happens (parallel execution, caching, smart batching).

## How Column Generators Work

While you rarely interact with generators directly, understanding their internals helps you write more efficient configurations and debug issues when they arise.

### Generation Strategies: From-Scratch vs Context-Aware

Not all columns are created equal—some need context, others don't. This distinction fundamentally shapes how Data Designer schedules execution.

**From-scratch generators** (Sampler and Seed Dataset columns) produce data independently. They don't read other columns, so they can run immediately and in parallel. They're fast, resource-efficient, and form the foundational layer of your dataset. Think of them as your data's bedrock—everything else builds on top.

**Context-aware generators** (LLM, Expression, and Validation columns) depend on existing columns. An LLM prompt that references `{{ product_name }}` can't execute until the `product_name` column exists. Data Designer analyzes these dependencies, builds a directed acyclic graph (DAG), and schedules execution in topologically sorted order. You never specify execution order manually—the framework figures it out.

!!! tip "Performance Implications"
    Minimize dependency chains for better parallelization. If ten columns all depend on column A but not on each other, they can generate in parallel once A completes. Long dependency chains (A→B→C→D→E) force sequential execution. Design your schema to maximize independent columns.

### Batch Processing Under the Hood

Generators don't process records one-by-one (that would be slow). They batch intelligently based on column type:

- **LLM generators** batch API requests to amortize network overhead and exploit concurrent execution. The framework issues multiple requests in parallel within rate limits, dramatically improving throughput.
- **Validation generators** use configurable batch sizes. Small batches reduce memory pressure; large batches improve throughput. Tune based on your validator's complexity.
- **Expression generators** leverage pandas vectorization, evaluating Jinja templates row-by-row but using compiled numpy operations where possible.
- **Sampler generators** use vectorized numpy operations to generate entire columns at once—blazingly fast.

You control batch sizes in column configurations, but sane defaults usually work fine.

### Side Effect Columns

Some columns generate more than you asked for. When an LLM model uses chain-of-thought reasoning, Data Designer captures both the answer *and* the reasoning trace. The trace goes in a side effect column named `{column_name}__reasoning_trace`.

Side effect columns integrate fully into the dependency graph—other columns can reference them, and you can mark them for dropping if you don't need them in the final output. Custom generators can define arbitrary side effects through the plugin API.

### Resource Management

Generators handle the messy details of resource allocation:

- **Rate limiting**: LLM generators implement exponential backoff retry logic, gracefully handling API rate limits without manual intervention.
- **Parallel execution**: Independent columns run concurrently up to configurable parallelism limits.
- **Memory management**: Data flows through generators in chunks, preventing memory exhaustion on large datasets.
- **Connection pooling**: Remote validators reuse HTTP connections across batches for efficiency.

This abstraction lets you focus on *what* to generate, not *how* to manage infrastructure.

## Column Dependencies and Execution Order

One of Data Designer's key value propositions: you never manually specify execution order. The framework infers dependencies automatically and schedules execution optimally.

### How Dependency Detection Works

Data Designer scans your column configurations for references to other columns:

- **Jinja2 templates**: Any `{{ column_name }}` reference in LLM prompts, system prompts, or expressions creates a dependency
- **Validation targets**: Validation columns explicitly list target columns they check
- **Seed dataset columns**: These must exist before other columns can reference them

From these references, the framework constructs a **directed acyclic graph (DAG)** where edges represent dependencies. A topological sort of this DAG produces a valid execution order—one where every column's dependencies are satisfied before the column executes.

### Parallel Execution Opportunities

The DAG reveals parallelization opportunities. Columns at the same level (no dependencies between them) execute concurrently. Ten LLM columns that all depend only on seed data? They generate in parallel, multiplying throughput.

Long dependency chains limit parallelism. If column E depends on D, which depends on C, which depends on B, which depends on A, execution is sequential (A→B→C→D→E). When designing schemas, prefer shallow, wide dependency graphs over deep, narrow ones.

### Error Detection

Circular dependencies are caught at configuration time, not runtime. If column A depends on B, and B depends on A (directly or through intermediaries), Data Designer raises an error before generating any data. This fail-fast behavior prevents wasted computation.

!!! warning "Dependency Cycles Are Fatal"
    If you see a circular dependency error, check your Jinja2 templates and validation targets. A common mistake: column A's expression references column B, while column B's expression references column A. The fix: identify which dependency is unnecessary and remove it, or introduce an intermediate column to break the cycle.

## Column Properties

Every column configuration inherits from `SingleColumnConfig`, which provides a standard set of properties. Understanding these helps you leverage the framework's full capabilities.

### `name`

The column's identifier—unique within your configuration, used in Jinja2 references, and becomes the column name in the output DataFrame. Choose descriptive names. `user_review` is clearer than `col_17`.

### `drop`

A boolean flag (default: `False`) controlling whether the column appears in the final dataset. Setting `drop=True` generates the column (making it available as a dependency) but excludes it from the final output.

**When to drop columns:**

- Intermediate calculations that feed expressions but aren't meaningful standalone
- Context columns used only for LLM prompt templates
- Validation results during development that you don't want in production datasets

Dropped columns still participate fully in generation and the dependency graph—they're just filtered out at the end.

### `column_type`

A literal string identifying the column's type: `"sampler"`, `"llm-text"`, `"expression"`, etc. This field is set automatically by each configuration class and serves as Pydantic's discriminator for deserializing configurations from JSON/YAML.

You rarely set this manually—instantiating `LLMTextColumnConfig` automatically sets `column_type="llm-text"`. The field's presence makes serialization round-trippable: save a complex configuration to YAML, load it later, and Pydantic reconstructs the exact configuration objects.

### `required_columns`

A computed property (not a field you set) listing columns that must be generated before this one. The framework derives this automatically:

- For LLM/Expression columns: extracted from Jinja2 template `{{ variables }}`
- For Validation columns: the explicitly listed target columns
- For Sampler columns with conditional parameters: columns referenced in conditions

You read this property (useful for introspection) but never set it—it's always computed from other configuration details.

### `side_effect_columns`

Another computed property listing columns created implicitly alongside the primary column. Currently, only LLM columns produce side effects (reasoning trace columns like `{name}__reasoning_trace` when models use extended thinking).

Side effect columns integrate into the dependency graph like any other column—you can reference them in other columns' templates or mark them for dropping if unwanted.

## Best Practices

Effective column design separates good synthetic datasets from great ones. Follow these principles to maximize quality and efficiency.

### Start with the Foundation Layer

Build from the ground up. Begin with **Sampler and Seed Dataset columns**—these are fast, deterministic, and establish the structural foundation. Once these are in place, layer in LLM-generated columns that reference the base data for context. This pattern (cheap foundation, expensive elaboration) optimizes both cost and generation time.

### Match Column Types to Use Cases

Choosing the right column type is half the battle:

| Use Case | Column Type | Why |
|----------|-------------|-----|
| Structured data with known distributions | Sampler | Fast, deterministic, no API costs |
| Simple transformations and derivations | Expression | Lightweight, no LLM overhead |
| Creative or semantic text content | LLM-Text | Leverages language understanding |
| Code in specific languages | LLM-Code | Automatic parsing and language-specific context |
| JSON with guaranteed schema | LLM-Structured | Type safety and validation baked in |
| Quality scoring and evaluation | LLM-Judge | Multi-dimensional assessment |
| Checking generated content | Validation | Catches errors during generation |

When in doubt, ask: does this need AI creativity, or can a simpler approach suffice?

### Design for Parallelism

**Shallow, wide dependency graphs** beat **deep, narrow ones** for performance. Ten columns depending only on seed data execute concurrently. Ten columns in a dependency chain execute sequentially—ten times slower.

Practical tactics:

- Generate independent demographic fields in parallel (age, location, occupation) rather than making each depend on the previous
- If multiple LLM columns use similar context, have them all reference the same base columns rather than creating intermediate columns
- Use the `drop` flag liberally for intermediate columns, but minimize their necessity

### Tune Batch Sizes and Resource Usage

Default batch sizes work well, but tuning can yield significant speedups:

- **Large batches** for simple operations (sampler columns, lightweight expressions)
- **Medium batches** for LLM generation (balancing API overhead and parallelism)
- **Small batches** for expensive validators or complex structured output

Monitor memory usage and API rate limits. If you're hitting limits, reduce batch sizes or parallelism settings.

### Build Quality Gates Into Generation

Don't wait until after generation to discover quality issues. **Validation columns** catch problems early:

- Lint generated code immediately (syntax errors, style violations)
- Validate structured outputs against schemas (business rules, data integrity)
- Check text formats (email addresses, URLs, date formats)
- Run semantic checks (content moderation, factuality)

Pair Validation columns with LLM-Judge columns for comprehensive quality control—validators check correctness, judges assess quality.

### Make Configurations Readable and Maintainable

Future you (or your teammates) will thank present you for clear, well-organized configurations:

- **Descriptive column names**: `customer_support_response` > `llm_output_3`
- **Well-structured prompts**: Use clear instructions, separate concerns, format for readability
- **Comments in YAML**: Document non-obvious choices, explain business logic, note constraints
- **Modular design**: Break complex schemas into logical sections (demographics, transactions, text content)

### Iterate with Small Samples

Generate 10-100 rows before scaling to thousands. Small samples let you:

- Verify column types and dependencies work correctly
- Test prompt effectiveness and tune as needed
- Check validation logic catches expected errors
- Estimate costs and generation time for full datasets

Use the `drop` flag to experiment with different generation strategies without polluting output.

## Summary

Columns are Data Designer's fundamental abstraction—the declarative building blocks for synthetic data. Whether you're generating simple categorical values with samplers or complex nested JSON with LLMs, columns provide a unified, configuration-driven interface.

The framework handles the hard parts: dependency resolution, parallel execution, batch optimization, resource management. You focus on the data's structure and semantics. This separation of concerns scales from simple datasets to complex multi-stage generation pipelines.

With nine built-in column types and a plugin system for custom extensions, Data Designer adapts to virtually any synthetic data generation scenario. Start simple, leverage the right column types for each use case, and let the framework optimize execution.
