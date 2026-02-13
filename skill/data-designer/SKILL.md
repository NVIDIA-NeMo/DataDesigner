---
name: data-designer
description: >-
  Generate synthetic datasets using NVIDIA NeMo Data Designer.
  Use when the user wants to create, design, or generate synthetic data,
  build training/evaluation datasets, generate text/code/structured data
  with LLMs, score data quality with LLM judges, validate generated code,
  or work with the data-designer Python library in any capacity.
argument-hint: [describe the dataset you want to generate]
disable-model-invocation: true
hooks:
  SessionStart:
    - matcher: startup
      hooks:
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/skills/data-designer/hooks/check_data_designer.sh"
          once: true
  PostToolUse:
    - matcher: Write|Edit
      hooks:
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/skills/data-designer/hooks/ruff_lint.sh" $filePath
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/skills/data-designer/hooks/ty_check.sh" $filePath
---

# Data Designer Synthetic Dataset Generator

Generate synthetic datasets using NVIDIA NeMo Data Designer.

---

## 1. Before You Start

**Pre-flight check** runs automatically on session start (Claude Code hook).
For Cursor, run manually: `.claude/skills/data-designer/hooks/check_data_designer.sh`

**Clarify with the user:** purpose (training/eval/fine-tuning), record count, schema (columns/fields), seed data, quality needs (validation/judging), and model provider.

**Verify environment and discover model aliases:** Run `uv run data-designer config list` to confirm API keys are set and to see the available providers, model aliases, and backing models. Use the aliases from this output (e.g., `nvidia-text`, `openai-reasoning`) as the `model_alias` argument in column configs.

---

## 2. Schema Design

Run discovery scripts to see available types:
```bash
uv run .claude/skills/data-designer/scripts/get_column_info.py --list
uv run .claude/skills/data-designer/scripts/get_sampler_info.py --list
uv run .claude/skills/data-designer/scripts/get_processor_info.py --list
uv run .claude/skills/data-designer/scripts/get_validator_info.py --list
```

### Column Type Decision Tree

```
Need data?
+-- Statistical/random values --> SamplerColumnConfig
|   +-- Categorical --> SamplerType.CATEGORY
|   +-- Hierarchical --> SamplerType.SUBCATEGORY
|   +-- Person data --> SamplerType.PERSON (Nemotron Personas) / PERSON_FROM_FAKER
|   +-- Dates --> SamplerType.DATETIME / TIMEDELTA
|   +-- IDs --> SamplerType.UUID
|   +-- Numeric --> UNIFORM, GAUSSIAN, POISSON, BINOMIAL, BERNOULLI, SCIPY
+-- LLM-generated content
|   +-- Free-form text --> LLMTextColumnConfig
|   +-- Code output --> LLMCodeColumnConfig (20+ languages)
|   +-- Multiple related fields that must be internally consistent
|       --> LLMStructuredColumnConfig (Pydantic or JSON schema)
|       Use when: a single entity has 2+ fields that should cohere
|       (e.g., a customer with name + title + department, a product
|       with SKU + regulatory class + clearance type). One LLM call
|       ensures consistency; separate LLM text columns would not.
|   +-- Quality scoring --> LLMJudgeColumnConfig (Score rubrics)
+-- Derived/computed --> ExpressionColumnConfig (Jinja2, no LLM)
+-- Custom logic --> CustomColumnConfig (@custom_column_generator)
+-- From seed data --> with_seed_dataset() (auto-creates SeedDatasetColumnConfig)
+-- Embeddings --> EmbeddingColumnConfig
+-- Validation gates --> ValidationColumnConfig (code lint, callable, remote)
```

Before writing any column config, run the relevant info script for exact field details:
```bash
uv run .claude/skills/data-designer/scripts/get_column_info.py <column-type>
uv run .claude/skills/data-designer/scripts/get_sampler_info.py <sampler-type>
uv run .claude/skills/data-designer/scripts/get_processor_info.py <processor-type>
uv run .claude/skills/data-designer/scripts/get_validator_info.py <validator-type>
```

---

## 3. Workflow

1. **Initialize** `DataDesigner()` and `DataDesignerConfigBuilder()`
2. **Add columns** in any order (DAG auto-resolves dependencies from Jinja2 templates)
3. **Add optional features**: constraints, processors, validators, seed data, profilers
4. **Validate** with `data_designer.validate(config_builder)`
5. **Preview** with `data_designer.preview(config_builder, num_records=5)`
6. **Iterate** on prompts/params based on preview
7. **Create** full dataset with `data_designer.create(config_builder, num_records=N, dataset_name="name")`
8. **Load and inspect** results

### Default Model Aliases

No manual `ModelConfig` needed — default aliases load automatically per provider.
Alias pattern: `{provider}-text`, `{provider}-reasoning`, `{provider}-vision`, `{provider}-embedding`.
**Always run `uv run data-designer config list` before writing column configs** to discover which aliases are available for the user's provider. Use those aliases as the `model_alias` argument — do not guess or hardcode aliases without checking.

### Minimal Skeleton

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
config_builder = dd.DataDesignerConfigBuilder()

# Add columns here...

data_designer.validate(config_builder)
preview = data_designer.preview(config_builder, num_records=5)
preview.display_sample_record()

results = data_designer.create(config_builder, num_records=100, dataset_name="my-dataset")
dataset = results.load_dataset()
analysis = results.load_analysis()
```

---

## 4. Examples

Read these before writing any Data Designer script. Each demonstrates **patterns and API usage**, not domain-specific content — adapt the structure to the user's requirements, don't copy example-specific values (categories, prompts, schemas, etc.).

- **`examples/basic_text_generation.py`** -- Product reviews: CATEGORY sampler with weights, PERSON_FROM_FAKER with drop, UNIFORM with convert_to, ExpressionColumnConfig, LLMTextColumnConfig.
- **`examples/structured_and_code.py`** -- Programming tasks: Pydantic BaseModel as output_format, LLMStructuredColumnConfig with nested field access, LLMCodeColumnConfig.
- **`examples/seed_dataset_with_judge.py`** -- Clinical notes: LocalFileSeedSource with SHUFFLE, UUID/DATETIME samplers, LLMJudgeColumnConfig with Score rubrics.
- **`examples/custom_column_with_llm.py`** -- Writer-editor pattern: @custom_column_generator with model access, side_effect_columns, multi-model orchestration.
- **`examples/mcp_tool_use.py`** -- MCP tool calling: LocalStdioMCPProvider, ToolConfig, tool_alias on LLM columns, trace capture.

---

## 5. Key Patterns

### Jinja2 Templating

All prompts use Jinja2 for column references. Dependencies are auto-resolved.

```python
# Basic reference
"Write a review for {{ product_name }}"

# Nested object access (Person sampler, structured output)
"{{ customer.first_name }} from {{ customer.city }}"
"{{ product_info.price }}"

# Conditional logic
"{% if rating >= 4 %}positive{% else %}negative{% endif %}"

# Filters
"{{ name | upper }}"
"{{ price | round(2) }}"
```

### Drop Intermediate Columns

Generate a column for dependency use but exclude from final output:
```python
dd.SamplerColumnConfig(name="person", ..., drop=True)
dd.ExpressionColumnConfig(name="name", expr="{{ person.full_name }}")
```

### Structured Entity with Extracted Fields

Generate a coherent multi-field entity with LLMStructuredColumnConfig,
then extract individual fields into top-level columns with ExpressionColumnConfig:

```python
class CustomerProfile(BaseModel):
    facility_name: str = Field(description="Healthcare facility name")
    bed_count: int = Field(description="Number of beds (50-2000)")
    department: str = Field(description="Primary department")

config_builder.add_column(dd.LLMStructuredColumnConfig(
    name="customer_profile",
    prompt="...",
    output_format=CustomerProfile,
    model_alias="<model-alias>",
))
# Extract individual fields for a flat output schema
config_builder.add_column(dd.ExpressionColumnConfig(
    name="facility_name", expr="{{ customer_profile.facility_name }}"))
config_builder.add_column(dd.ExpressionColumnConfig(
    name="department", expr="{{ customer_profile.department }}"))
```

This ensures facility_name and department are contextually consistent
(generated together) while still appearing as flat columns in the output.

### Seed Datasets

Bootstrap from existing data (CSV, Parquet, HuggingFace, DataFrame):
```python
seed_source = dd.LocalFileSeedSource(path="data.csv")
config_builder.with_seed_dataset(seed_source, sampling_strategy=dd.SamplingStrategy.SHUFFLE)
# Seed columns are auto-available in Jinja2 templates: {{ column_from_seed }}
```

### Processors (Post-Generation Transforms)

Transform output schema after generation:
```python
config_builder.add_processor(dd.SchemaTransformProcessorConfig(
    name="chat_format",
    template={
        "messages": [
            {"role": "user", "content": "{{ question }}"},
            {"role": "assistant", "content": "{{ answer }}"},
        ]
    },
))
# Access: results.load_processor_dataset("chat_format")
```

### Validators (Quality Gates)

Validate generated code or data:
```python
config_builder.add_column(dd.ValidationColumnConfig(
    name="code_check",
    target_columns=["solution"],
    validator_type=dd.ValidatorType.CODE,
    validator_params=dd.CodeValidatorParams(code_lang=dd.CodeLang.PYTHON),
    batch_size=20,
))
```

### Constraints (Sampler Bounds)

```python
config_builder.add_constraint(dd.ScalarInequalityConstraint(
    target_column="age", operator=dd.InequalityOperator.GE, rhs=18,
))
config_builder.add_constraint(dd.ColumnInequalityConstraint(
    target_column="end_date", operator=dd.InequalityOperator.GT, rhs="start_date",
))
```

### Trace & Reasoning Capture

```python
dd.LLMTextColumnConfig(
    name="answer",
    prompt="...",
    model_alias="nvidia-reasoning",
    with_trace=dd.TraceType.ALL_MESSAGES,       # -> answer__trace column
    extract_reasoning_content=True,              # -> answer__reasoning_content column
)
```

### Performance Tuning (RunConfig)

```python
from data_designer.config import RunConfig
data_designer.set_run_config(RunConfig(
    buffer_size=500,              # records per batch (default: 1000)
    disable_early_shutdown=True,  # don't stop on high error rate
    max_conversation_restarts=7,  # retries for strict schemas
    max_conversation_correction_steps=2,  # in-conversation corrections
))
```

---

## 6. Best Practices

- Always preview (3-5 records) before full generation
- Use samplers for diversity control (not LLMs)
- Keep prompts deterministic and scoped
- Add validators/judges when quality matters
- Use temperature distributions for output diversity
- Drop intermediate columns to keep final dataset clean
- Use `ExpressionColumnConfig` for derived fields (no LLM cost)
- Use `SchemaTransformProcessorConfig` to reshape for training formats (e.g., chat messages)
- Use `LLMStructuredColumnConfig` when generating 2+ related fields that must
  be internally consistent (e.g., a person's name + title, a product's SKU +
  regulatory class). This is more coherent than separate `LLMTextColumnConfig`
  calls and more realistic than hardcoded `CATEGORY` samplers. Extract
  individual fields with `ExpressionColumnConfig` for a flat output schema.
- Always call `validate()` before `preview()`/`create()`

---

## 7. Reference Material

**Start here:** Read the example scripts in `examples/` (Section 4).

Then consult these references as needed:

- **`references/api_reference.md`** -- Complete API: all column types, sampler types, model config, constraints, processors, validators, seed sources, MCP, RunConfig, profilers, results.
- **`references/advanced_patterns.md`** -- Custom columns with `@custom_column_generator`, MCP tool integration, multimodal inputs, schema transforms, performance tuning, multi-stage refinement, conditional sampling, Nemotron Personas, config serialization.

---

## 8. Output Expectations

- Produce a single runnable Python script
- Clear inputs: `num_records`, `model_alias`, `dataset_name`
- Clear outputs: dataset path, analysis report
- Include error handling for common issues
