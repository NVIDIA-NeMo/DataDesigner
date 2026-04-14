# Sampling

The sampling subsystem generates statistically distributed data without LLM calls. It handles built-in sampler types (UUID, Category, Gaussian, Person, DateTime, etc.), constraint-based rejection sampling, and locale-aware person/entity generation.

Source: `packages/data-designer-engine/src/data_designer/engine/sampling_gen/`

## Overview

Sampling is used for columns that don't need LLM generation — identifiers, categories, numerical distributions, timestamps, and person data. The subsystem builds a schema DAG from sampler configs, validates acyclicity, and generates data column-by-column with optional inter-column constraints.

## Key Components

### DatasetGenerator

The main entry point for sampler-based generation. Given a `DataSchema` (or `SamplerMultiColumnConfig`):

1. Builds a NetworkX DAG from the schema's column dependencies
2. Topologically sorts columns
3. Generates each column with rejection sampling when constraints are present
4. Shared kwargs include `people_gen_resource` for person-type samplers

### DataSchema and DAG

`DataSchema` defines the sampler columns and their relationships. `Dag` validation ensures acyclicity. Edges come from:
- Conditional parameters (column A's distribution depends on column B's value)
- Required columns (explicit dependencies)
- Constraints (inter-column relationships like "start_date < end_date")

### Constraint System

`ConstraintChecker` enforces inter-column constraints during generation:
- **`ScalarInequalityConstraint`** — column value vs. a constant
- **`ColumnInequalityConstraint`** — column value vs. another column's value

Rejection sampling retries generation when constraints are violated, up to a configurable limit.

### Person/Entity Generation

`PeopleGen` (abstract) → `PeopleGenFaker` (Faker-based) provides locale-aware person data:

- **Faker integration** — generates names, addresses, and base attributes by locale
- **Managed datasets** — for locales in `LOCALES_WITH_MANAGED_DATASETS`, uses pre-built datasets via `ManagedDatasetGenerator` for higher quality and consistency
- **Derived fields** — `Person` entity computes birth dates, emails, phone numbers, national IDs with locale-specific behavior (e.g., US-only SSN format)

`PersonReader` on `ResourceProvider` loads managed person datasets when person samplers are used.

### SamplerColumnGenerator

The engine-side generator for sampler columns. Extends `FromScratchColumnGenerator` with `FULL_COLUMN` strategy. Uses `DatasetGenerator` internally, passing the appropriate `PeopleGen` resource.

## Data Flow

1. `SamplerColumnConfig` declares `sampler_type` and `params` (discriminated union)
2. `SamplerColumnGenerator` creates a `DatasetGenerator` with the schema
3. `DatasetGenerator` topologically sorts columns, then for each:
   - Samples values from the configured distribution
   - Applies constraint checking via rejection sampling
   - For person columns, delegates to `PeopleGen` with the configured locale
4. Returns a DataFrame with all sampler columns populated

## Design Decisions

- **Rejection sampling over constraint propagation** keeps the implementation simple and general. Most constraints are satisfied quickly; the retry limit prevents infinite loops on unsatisfiable constraints.
- **Managed datasets for person data** provide realistic, locale-consistent person records that Faker alone cannot guarantee (e.g., matching name ethnicity to locale, consistent address formatting).
- **Separate DAG from the main execution DAG** — sampler columns have their own dependency graph within the `DatasetGenerator`, independent of the broader column execution DAG in `DatasetBuilder`. This is because sampler columns are generated as a batch before LLM columns.
- **Discriminated union for sampler params** mirrors the column config pattern — each sampler type has its own params class with a `Literal` discriminator, enabling type-safe deserialization and validation.

## Cross-References

- [System Architecture](overview.md) — where sampling fits in the data flow
- [Engine Layer](engine.md) — `SamplerColumnGenerator` in the generator hierarchy
- [Config Layer](config.md) — `SamplerColumnConfig`, `SamplerParamsT`, constraints
- [Dataset Builders](dataset-builders.md) — how sampler generators are orchestrated
