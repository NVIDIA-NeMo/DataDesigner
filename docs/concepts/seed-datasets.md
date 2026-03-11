# Seed Datasets

Seed datasets let you bootstrap synthetic data generation from existing data. Instead of generating everything from scratch, you provide a dataset whose columns become available as context in your prompts and expressions—grounding your synthetic data in real-world examples.

!!! tip "When to Use Seed Datasets"
    Seed datasets shine when you have **real data you want to build on**:

    - Product catalogs → generate customer reviews
    - Medical diagnoses → generate physician notes
    - Code snippets → generate documentation
    - Company profiles → generate financial reports

    The seed data provides realism and domain specificity; Data Designer adds volume and variation.

## The Basic Pattern

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

# Define your model configuration
model_configs = [
    dd.ModelConfig(
        alias="my-model",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
    )
]

config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# 1. Attach a seed dataset
seed_source = dd.LocalFileSeedSource(path="products.csv")
config_builder.with_seed_dataset(seed_source)

# 2. Reference seed columns in your prompts
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="review",
        model_alias="my-model",
        prompt="""\
Write a customer review for {{ product_name }}.
Category: {{ category }}
Price: ${{ price }}
""",
    )
)
```

Every column in your seed dataset becomes available as a Jinja2 variable in prompts and expressions. Data Designer automatically:

- Reads rows from the seed dataset
- Injects seed column values into templates

## Seed Sources

Data Designer supports four ways to provide seed data:

### 📁 LocalFileSeedSource

Load from a local file—CSV, Parquet, or JSON.

```python
# Single file
seed_source = dd.LocalFileSeedSource(path="data/products.csv")

# Parquet files with wildcard
seed_source = dd.LocalFileSeedSource(path="data/products/*.parquet")
```

!!! note "Supported Formats"
    - CSV (`.csv`)
    - Parquet (`.parquet`)
    - JSON (`.json`, `.jsonl`)

### 🤗 HuggingFaceSeedSource

Load directly from HuggingFace datasets without downloading manually.

```python
seed_source = dd.HuggingFaceSeedSource(
    path="datasets/gretelai/symptom_to_diagnosis/data/train.parquet",
    token="hf_...",  # Optional, for private datasets
)
```

### 🐼 DataFrameSeedSource

Use an in-memory pandas DataFrame—great for preprocessing or combining multiple sources.

```python
import pandas as pd

df = pd.read_csv("raw_data.csv")
df = df[df["quality_score"] > 0.8]  # Filter to high-quality rows

seed_source = dd.DataFrameSeedSource(df=df)
```

!!! warning "Serialization"
    `DataFrameSeedSource` can't be serialized to YAML/JSON configs. Use `LocalFileSeedSource` if you need to save and share configurations.

### 📂 DirectorySeedSource

Read a directory of files as a seed dataset, optionally applying a full-batch transform before normal seeding begins.

Without a transform, each matched file becomes one row with file metadata such as `source_path`, `relative_path`, and `file_name`.

```python
seed_source = dd.DirectorySeedSource(
    path="seed-files",
    glob="**/*.txt",
)
```

You can also attach built-in trace normalizers to turn directories of agent traces into a first-class normalized seed dataset:

```python
seed_source = dd.DirectorySeedSource(
    path="trace-data/codex",
    glob="**/*.jsonl",
    transform=dd.CodexTraceNormalizer(),
)
```

Built-in directory transforms in this PR:

- `dd.ClaudeCodeTraceNormalizer()`
- `dd.CodexTraceNormalizer()`
- `dd.ChatCompletionJsonlNormalizer()`

These transforms expose normalized trace rows with common metadata such as `trace_id`, `source_kind`, and `final_assistant_message`, plus a `messages` column containing the full conversation in the same message-list shape used by Data Designer traces.

For the common trace cases, Data Designer also ships convenience wrappers:

- `dd.ClaudeCodeTraceSeedSource(path=...)`
- `dd.CodexTraceSeedSource(path=...)`
- `dd.ChatCompletionJsonlSeedSource(path=...)`

`ClaudeCodeTraceSeedSource()` defaults to `~/.claude/projects`, and `CodexTraceSeedSource()` defaults to `~/.codex/sessions`.

## Sampling Strategies

Control how rows are read from the seed dataset.

### Ordered (Default)

Rows are read sequentially in their original order. Each generated record corresponds to the next row in the seed dataset. If you generate more records than exist in the seed dataset, it will cycle in order until completion.

```python
config_builder.with_seed_dataset(
    seed_source,
    sampling_strategy=dd.SamplingStrategy.ORDERED,
)
```

### Shuffle

Rows are randomly shuffled before sampling. Useful when your seed data has some ordering you want to break.

```python
config_builder.with_seed_dataset(
    seed_source,
    sampling_strategy=dd.SamplingStrategy.SHUFFLE,
)
```

## Selection Strategies

Select a subset of your seed dataset—useful for large datasets or parallel processing.

### IndexRange

Select a specific range of row indices.

```python
# Use only rows 100-199 (100 rows total)
config_builder.with_seed_dataset(
    seed_source,
    selection_strategy=dd.IndexRange(start=100, end=199),
)
```

### PartitionBlock

Split the dataset into N equal partitions and select one. Perfect for distributing work across multiple jobs.

```python
# Split into 5 partitions, use the 3rd one (index=2, zero-based)
config_builder.with_seed_dataset(
    seed_source,
    selection_strategy=dd.PartitionBlock(index=2, num_partitions=5),
)
```

!!! tip "Parallel Processing"
    Run 5 parallel jobs, each with a different partition index, to process a large seed dataset in parallel:

    ```python
    # Job 0: PartitionBlock(index=0, num_partitions=5)
    # Job 1: PartitionBlock(index=1, num_partitions=5)
    # Job 2: PartitionBlock(index=2, num_partitions=5)
    # ...
    ```

### Combining Strategies

Sampling and selection strategies work together. For example, shuffle rows *within* a specific partition:

```python
config_builder.with_seed_dataset(
    seed_source,
    sampling_strategy=dd.SamplingStrategy.SHUFFLE,
    selection_strategy=dd.PartitionBlock(index=0, num_partitions=10),
)
```

## Complete Example

Here's a complete example generating physician notes from a symptom-to-diagnosis seed dataset:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()

model_configs = [
    dd.ModelConfig(
        alias="medical-notes",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
    )
]

config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# Attach seed dataset (has 'diagnosis' and 'symptoms' columns)
seed_source = dd.LocalFileSeedSource(path="symptom_to_diagnosis.csv")
config_builder.with_seed_dataset(seed_source)

# Generate patient info
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="patient",
        sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
        params=dd.PersonFromFakerSamplerParams(),
    )
)

config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="patient_name",
        expr="{{ patient.first_name }} {{ patient.last_name }}",
    )
)

# Generate notes grounded in seed data
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="physician_notes",
        model_alias="medical-notes",
        prompt="""\
You are a physician writing notes after a patient visit.

Patient: {{ patient_name }}
Diagnosis: {{ diagnosis }}
Reported Symptoms: {{ symptoms }}

Write detailed clinical notes for this visit.
""",
    )
)

# Preview
preview = designer.preview(config_builder, num_records=5)
preview.display_sample_record()
```

## Best Practices

### Keep Seed Data Clean

Garbage in, garbage out. Clean your seed data before using it:

- Remove duplicates
- Fix encoding issues
- Filter out low-quality rows
- Standardize column names

### Match Generation Volume to Seed Size

If your seed dataset has 1,000 rows and you generate 10,000 records, each seed row will be used ~10 times. Consider whether that's appropriate for your use case.

### Use Seed Data for Diversity Control

Seed datasets are excellent for controlling the distribution of your synthetic data. Want 30% electronics, 50% clothing, 20% home goods? Curate your seed dataset to match.
