# Data Designer API Reference

Complete API reference for `data-designer` v0.5.x. All classes imported via `import data_designer.config as dd` unless noted.

---

## Table of Contents

1. [Core Classes](#1-core-classes)
2. [Column Types](#2-column-types)
3. [Sampler Types](#3-sampler-types)
4. [Model Configuration](#4-model-configuration)
5. [Seed Datasets](#5-seed-datasets)
6. [Constraints](#6-constraints)
7. [Processors](#7-processors)
8. [Validators](#8-validators)
9. [MCP / Tool Configuration](#9-mcp--tool-configuration)
10. [RunConfig](#10-runconfig)
11. [Profilers](#11-profilers)
12. [Results](#12-results)
13. [Default Model Aliases](#13-default-model-aliases)

---

## 1. Core Classes

### DataDesigner

```python
from data_designer.interface import DataDesigner

DataDesigner(
    artifact_path: Path | str | None = None,          # default: ./artifacts
    model_providers: list[ModelProvider] | None = None, # default: nvidia + openai + openrouter
    secret_resolver: SecretResolver | None = None,
    seed_readers: list[SeedReader] | None = None,
    managed_assets_path: Path | str | None = None,     # default: ~/.data-designer/managed-assets
    mcp_providers: list[MCPProviderT] | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `preview(config_builder, *, num_records=10)` | `PreviewResults` | In-memory preview |
| `create(config_builder, *, num_records=10, dataset_name="dataset")` | `DatasetCreationResults` | Full generation + save |
| `validate(config_builder)` | `None` | Validate without generating |
| `set_run_config(run_config)` | `None` | Set runtime parameters |
| `get_models(model_aliases)` | `dict[str, ModelFacade]` | Get model facades for custom columns |
| `get_default_model_configs()` | `list[ModelConfig]` | List default model configs |
| `get_default_model_providers()` | `list[ModelProvider]` | List default providers |
| `info` | `InterfaceInfo` | Property: `dd.info.display("model_providers")` |

### DataDesignerConfigBuilder

```python
dd.DataDesignerConfigBuilder(
    model_configs: list[ModelConfig] | str | Path | None = None,  # None = use defaults
    tool_configs: list[ToolConfig] | None = None,
)
```

| Method | Description |
|--------|-------------|
| `add_column(column_config)` | Add a column config |
| `delete_column(column_name)` | Remove a column |
| `get_column_config(name)` | Get column by name |
| `get_column_configs()` | Get all columns |
| `get_columns_of_type(column_type)` | Filter by type |
| `add_model_config(model_config)` | Add model config |
| `delete_model_config(alias)` | Remove model config |
| `add_tool_config(tool_config)` | Add MCP tool config |
| `delete_tool_config(alias)` | Remove tool config |
| `add_constraint(constraint)` | Add sampler constraint |
| `delete_constraints(target_column)` | Remove constraints |
| `add_processor(processor_config)` | Add processor |
| `add_profiler(profiler_config)` | Add profiler |
| `with_seed_dataset(seed_source, *, sampling_strategy=ORDERED, selection_strategy=None)` | Set seed data |
| `build()` | Build final config |
| `write_config(path, indent=2)` | Serialize to YAML/JSON |
| `from_config(config)` | Class method: load from file/dict |
| `info` | Property: `builder.info.display("samplers")` |
| `allowed_references` | Property: list of referenceable column names |

---

## 2. Column Types

### SamplerColumnConfig

```python
dd.SamplerColumnConfig(
    name: str,
    sampler_type: dd.SamplerType,
    params: SamplerParamsT,
    conditional_params: dict[str, SamplerParamsT] = {},  # condition -> override params
    convert_to: str | None = None,  # "int", "float", "str"
    drop: bool = False,
)
```

### LLMTextColumnConfig

```python
dd.LLMTextColumnConfig(
    name: str,
    prompt: str,                                        # Jinja2 template
    model_alias: str,
    system_prompt: str | None = None,
    tool_alias: str | None = None,                      # MCP tool reference
    with_trace: dd.TraceType = TraceType.NONE,          # NONE, LAST_MESSAGE, ALL_MESSAGES
    extract_reasoning_content: bool = False,            # -> {name}__reasoning_content
    multi_modal_context: list[dd.ImageContext] | None = None,
    drop: bool = False,
)
```

### LLMCodeColumnConfig

Extends LLMTextColumnConfig with:
```python
dd.LLMCodeColumnConfig(
    ...,  # all LLMTextColumnConfig fields
    code_lang: dd.CodeLang,  # PYTHON, JAVASCRIPT, SQL_POSTGRES, etc.
)
```

**CodeLang values**: BASH, C, COBOL, CPP, CSHARP, GO, JAVA, JAVASCRIPT, KOTLIN, PYTHON, RUBY, RUST, SCALA, SWIFT, TYPESCRIPT, SQL_SQLITE, SQL_TSQL, SQL_BIGQUERY, SQL_MYSQL, SQL_POSTGRES, SQL_ANSI

### LLMStructuredColumnConfig

```python
dd.LLMStructuredColumnConfig(
    ...,  # all LLMTextColumnConfig fields
    output_format: dict | type[BaseModel],  # Pydantic model or JSON schema dict
)
```

Access nested fields in downstream prompts: `{{ column_name.field_name }}`

### LLMJudgeColumnConfig

```python
dd.LLMJudgeColumnConfig(
    ...,  # all LLMTextColumnConfig fields
    scores: list[dd.Score],
)

dd.Score(
    name: str,
    description: str,
    options: dict[int | str, str],  # score_value -> description
)
```

### ExpressionColumnConfig

```python
dd.ExpressionColumnConfig(
    name: str,
    expr: str,                     # Jinja2 expression
    dtype: str = "str",            # "int", "float", "str", "bool"
    drop: bool = False,
)
```

### EmbeddingColumnConfig

```python
dd.EmbeddingColumnConfig(
    name: str,
    target_column: str,            # column to embed (text or JSON list of texts)
    model_alias: str,              # embedding model alias
    drop: bool = False,
)
```

### ValidationColumnConfig

```python
dd.ValidationColumnConfig(
    name: str,
    target_columns: list[str],
    validator_type: dd.ValidatorType,   # CODE, LOCAL_CALLABLE, REMOTE
    validator_params: ValidatorParamsT,
    batch_size: int = 10,
    drop: bool = False,
)
```

### CustomColumnConfig

```python
dd.CustomColumnConfig(
    name: str,
    generator_function: Any,       # decorated with @custom_column_generator
    generation_strategy: dd.GenerationStrategy = GenerationStrategy.CELL_BY_CELL,
    generator_params: BaseModel | None = None,
    drop: bool = False,
)
```

---

## 3. Sampler Types

| SamplerType | Params Class | Key Params |
|-------------|--------------|------------|
| `CATEGORY` | `CategorySamplerParams` | `values: list`, `weights: list[float] | None` |
| `SUBCATEGORY` | `SubcategorySamplerParams` | `category: str` (parent column), `values: dict[str, list]` |
| `PERSON` | `PersonSamplerParams` | `locale`, `sex`, `city`, `age_range`, `with_synthetic_personas`, `select_field_values` |
| `PERSON_FROM_FAKER` | `PersonFromFakerSamplerParams` | `locale`, `sex`, `city`, `age_range` |
| `UUID` | `UUIDSamplerParams` | `prefix`, `short_form`, `uppercase` |
| `DATETIME` | `DatetimeSamplerParams` | `start`, `end`, `unit` (Y/M/D/h/m/s) |
| `TIMEDELTA` | `TimeDeltaSamplerParams` | `reference_column_name`, `dt_min`, `dt_max`, `unit` |
| `UNIFORM` | `UniformSamplerParams` | `low`, `high`, `decimal_places` |
| `GAUSSIAN` | `GaussianSamplerParams` | `mean`, `stddev`, `decimal_places` |
| `POISSON` | `PoissonSamplerParams` | `mean` |
| `BINOMIAL` | `BinomialSamplerParams` | `n`, `p` |
| `BERNOULLI` | `BernoulliSamplerParams` | `p` |
| `BERNOULLI_MIXTURE` | `BernoulliMixtureSamplerParams` | `p`, `dist_name`, `dist_params` |
| `SCIPY` | `ScipySamplerParams` | `dist_name`, `dist_params`, `decimal_places` |

### Person Object Fields

Access via `{{ person.field_name }}`:
- `first_name`, `last_name`, `full_name`
- `age`, `birth_date`, `sex`
- `email`, `phone`
- `city`, `state`, `country`, `address`

**Nemotron Personas** (SamplerType.PERSON only):
- Supported locales: `en_US`, `en_IN`, `en_SG`, `hi_Deva_IN`, `hi_Latn_IN`, `ja_JP`, `pt_BR`
- Download: `data-designer download personas --locale en_US`
- Extra fields: Big Five personality traits, cultural backgrounds, domain-specific personas
- Filtering: `select_field_values={"state": ["NY", "CA"]}`

---

## 4. Model Configuration

### ModelConfig

```python
dd.ModelConfig(
    alias: str,                    # reference name for columns
    model: str,                    # model identifier
    provider: str | None = None,
    inference_parameters: ChatCompletionInferenceParams | EmbeddingInferenceParams,
    skip_health_check: bool = False,
)
```

### ChatCompletionInferenceParams

```python
dd.ChatCompletionInferenceParams(
    temperature: float | DistributionT | None,
    top_p: float | DistributionT | None,
    max_tokens: int | None,
    max_parallel_requests: int = 4,
    timeout: int | None,
    extra_body: dict | None,
)
```

### Temperature/Top-p Distributions

For diversity, use distributions instead of fixed values:

```python
# Uniform
dd.UniformDistribution(params=dd.UniformDistributionParams(low=0.5, high=1.0))

# Manual (discrete)
dd.ManualDistribution(params=dd.ManualDistributionParams(
    values=[0.8, 0.9, 1.0], weights=[0.2, 0.5, 0.3]
))
```

### EmbeddingInferenceParams

```python
dd.EmbeddingInferenceParams(
    encoding_format: str = "float",  # "float" or "base64"
    dimensions: int | None = None,
    max_parallel_requests: int = 4,
)
```

### ModelProvider

```python
dd.ModelProvider(
    name: str,
    endpoint: str,
    provider_type: str = "openai",  # API format
    api_key: str | None = None,
    extra_body: dict | None = None,
    extra_headers: dict | None = None,
)
```

---

## 5. Seed Datasets

### Sources

```python
# Local file (CSV, Parquet, JSON/JSONL, XLSX; supports wildcards)
dd.LocalFileSeedSource(path="data/*.parquet")
dd.LocalFileSeedSource.from_dataframe(df, path="saved.parquet")

# HuggingFace
dd.HuggingFaceSeedSource(path="datasets/user/name/data/*.parquet", token="hf_xxx")

# In-memory DataFrame (not serializable to YAML/JSON)
dd.DataFrameSeedSource(df=my_dataframe)
```

### Sampling & Selection Strategies

```python
config_builder.with_seed_dataset(
    seed_source,
    sampling_strategy=dd.SamplingStrategy.SHUFFLE,     # ORDERED (default) or SHUFFLE
    selection_strategy=dd.IndexRange(start=0, end=99),  # or PartitionBlock(index=0, num_partitions=10)
)
```

---

## 6. Constraints

```python
# Column vs scalar
dd.ScalarInequalityConstraint(target_column="age", operator=dd.InequalityOperator.GE, rhs=18)

# Column vs column
dd.ColumnInequalityConstraint(target_column="end_date", operator=dd.InequalityOperator.GT, rhs="start_date")
```

Operators: `LT`, `LE`, `GT`, `GE`

Only for numerical sampler columns.

---

## 7. Processors

Run at `BuildStage.POST_BATCH` after column generation.

### DropColumnsProcessorConfig

```python
dd.DropColumnsProcessorConfig(
    name: str,
    column_names: list[str],
)
```

Dropped columns saved separately in `dropped-columns/` directory.

### SchemaTransformProcessorConfig

```python
dd.SchemaTransformProcessorConfig(
    name: str,
    template: dict[str, Any],  # keys = new column names, values = Jinja2 templates
)
```

Creates an **additional** dataset alongside the original. Output in `processors-outputs/{name}/`.

Example:
```python
dd.SchemaTransformProcessorConfig(
    name="chat_format",
    template={
        "messages": [
            {"role": "user", "content": "{{ question }}"},
            {"role": "assistant", "content": "{{ answer }}"},
        ],
        "metadata": {"category": "{{ category | upper }}"},
    },
)
```

---

## 8. Validators

### CodeValidatorParams

```python
dd.CodeValidatorParams(code_lang=dd.CodeLang.PYTHON)
```

Python: uses Ruff. Returns `is_valid`, `python_linter_score` (0-10), `python_linter_severity`, `python_linter_messages`.

SQL: uses SQLFluff. Dialects: SQL_POSTGRES, SQL_ANSI, SQL_MYSQL, SQL_SQLITE, SQL_TSQL, SQL_BIGQUERY. Returns `is_valid`, `error_messages`.

### LocalCallableValidatorParams

```python
dd.LocalCallableValidatorParams(
    validation_function: Callable[[pd.DataFrame], pd.DataFrame],  # must return df with is_valid column
    output_schema: dict | None = None,
)
```

### RemoteValidatorParams

```python
dd.RemoteValidatorParams(
    endpoint_url: str,
    output_schema: dict | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
    max_parallel_requests: int = 4,
)
```

### Batch Size Recommendations

- Code validators: 5-20
- Local callable: 10-50
- Remote validators: 1-10

---

## 9. MCP / Tool Configuration

### Providers

```python
# Local subprocess (stdio transport)
dd.LocalStdioMCPProvider(
    name: str,
    command: str,               # e.g., "python"
    args: list[str] | None,     # e.g., ["-m", "my_mcp_server"]
    env: dict[str, str] | None,
)

# Remote SSE
dd.MCPProvider(
    name: str,
    endpoint: str,              # e.g., "http://localhost:8080/sse"
    api_key: str | None,
)
```

### ToolConfig

```python
dd.ToolConfig(
    tool_alias: str,
    providers: list[str],              # MCP provider names
    allow_tools: list[str] | None,     # allowlist (None = all tools)
    max_tool_call_turns: int = 5,
    timeout_sec: float | None = None,
)
```

### Usage

```python
data_designer = DataDesigner(mcp_providers=[mcp_provider])
config_builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

config_builder.add_column(dd.LLMTextColumnConfig(
    name="answer",
    prompt="Use tools to answer: {{ question }}",
    model_alias="nvidia-text",
    tool_alias="my-tools",
))
```

---

## 10. RunConfig

```python
dd.RunConfig(
    buffer_size: int = 1000,                        # records per batch
    disable_early_shutdown: bool = False,
    shutdown_error_rate: float = 0.5,               # 0.0-1.0
    shutdown_error_window: int = 10,                # min tasks before monitoring
    non_inference_max_parallel_workers: int = 4,
    max_conversation_restarts: int = 5,             # full restarts on failure
    max_conversation_correction_steps: int = 0,     # in-conversation corrections
)
```

Apply: `data_designer.set_run_config(RunConfig(...))`

---

## 11. Profilers

```python
dd.JudgeScoreProfilerConfig(
    model_alias: str,
    summary_score_sample_size: int | None = 20,
)
```

Add: `config_builder.add_profiler(dd.JudgeScoreProfilerConfig(model_alias="nvidia-text"))`

---

## 12. Results

### PreviewResults

```python
preview.dataset                    # pd.DataFrame | None
preview.analysis                   # DatasetProfilerResults | None
preview.processor_artifacts        # dict | None
preview.display_sample_record()
```

### DatasetCreationResults

```python
results.load_dataset()                           # pd.DataFrame
results.load_analysis()                          # DatasetProfilerResults
results.load_processor_dataset(processor_name)   # pd.DataFrame
results.get_path_to_processor_artifacts(name)    # Path
```

### DatasetProfilerResults

```python
analysis.num_records
analysis.percent_complete
analysis.column_statistics          # list of column stats
analysis.to_report(save_path=None)  # Rich console report (HTML/SVG save)
```

---

## 13. Default Model Aliases

Auto-configured per provider. No manual `ModelConfig` needed.

| Alias | nvidia model | openai model |
|-------|-------------|--------------|
| `{provider}-text` | nemotron-3-nano-30b-a3b | gpt-4.1 |
| `{provider}-reasoning` | gpt-oss-20b | gpt-5 |
| `{provider}-vision` | nemotron-nano-12b-v2-vl | gpt-5 |
| `{provider}-embedding` | llama-3.2-nv-embedqa-1b-v2 | text-embedding-3-large |

Providers: `nvidia`, `openai`, `openrouter`

Environment variables: `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
