# Processors

Processors are transformations that modify your dataset before or after columns are generated. They run at different stages and can reshape, filter, or augment the data.

!!! tip "When to Use Processors"
    Processors handle transformations that don't fit the "column" model: restructuring the schema for a specific output format, dropping intermediate columns in bulk, or applying batch-wide operations.

## Overview

Each processor:

- Receives the complete batch DataFrame
- Applies its transformation
- Passes the result to the next processor (or to output)

Currently, processors run only at the `POST_BATCH` stage, i.e., after column generation completes for each batch.

## Processor Types

### 🗑️ Drop Columns Processor

Removes specified columns from the output dataset. Dropped columns are saved separately in the `dropped-columns` directory for reference.

!!! tip "Dropping Columns is More Easily Achieved via `drop = True`"
    The Drop Columns Processor is different from others in the sense that it does not need to be explicitly added: setting `drop = True` when configuring a column will accomplish the same.

**Configuration:**

```python
from data_designer.essentials import DropColumnsProcessorConfig

processor = DropColumnsProcessorConfig(
    name="remove_intermediate",
    column_names=["temp_calculation", "raw_input", "debug_info"],
)
```

**Behavior:**

- Columns specified in `column_names` are removed from the output
- Original values are preserved in a separate parquet file
- Missing columns produce a warning but don't fail the build
- Column configs are automatically marked with `drop=True` when this processor is added

**Use Cases:**

- Removing intermediate columns used only for LLM context
- Cleaning up debug or validation columns before final output
- Separating sensitive data from the main dataset

### 🔄 Schema Transform Processor

Creates an additional dataset with a transformed schema using Jinja2 templates. The output is written to a separate directory alongside the main dataset.

**Configuration:**

```python
from data_designer.essentials import SchemaTransformProcessorConfig

processor = SchemaTransformProcessorConfig(
    name="chat_format",
    template={
        "messages": [
            {"role": "user", "content": "{{ question }}"},
            {"role": "assistant", "content": "{{ answer }}"},
        ],
        "metadata": "{{ category | upper }}",
    },
)
```

**Behavior:**

- Each key in `template` becomes a column in the transformed dataset
- Values are Jinja2 templates with access to all columns in the batch
- Complex structures (lists, nested dicts) are supported
- Output is saved to the `processors-outputs/{name}/` directory
- The original dataset passes through unchanged

**Template Capabilities:**

- **Variable substitution**: `{{ column_name }}`
- **Filters**: `{{ text | upper }}`, `{{ text | lower }}`, `{{ text | trim }}`
- **Nested structures**: Arbitrarily deep JSON structures
- **Lists**: `["{{ col1 }}", "{{ col2 }}"]`

**Use Cases:**

- Converting flat columns to chat message format
- Restructuring data for specific model training formats
- Creating derived views without modifying the source dataset

## Using Processors

Add processors to your configuration using the builder's `add_processor` method:

```python
from data_designer.essentials import (
    DataDesignerConfigBuilder,
    DropColumnsProcessorConfig,
    SchemaTransformProcessorConfig,
)

builder = DataDesignerConfigBuilder()

# ... add columns ...

# Drop intermediate columns
builder.add_processor(
    DropColumnsProcessorConfig(
        name="cleanup",
        column_names=["scratch_work", "raw_context"],
    )
)

# Transform to chat format
builder.add_processor(
    SchemaTransformProcessorConfig(
        name="chat_format",
        template={
            "messages": [
                {"role": "user", "content": "{{ question }}"},
                {"role": "assistant", "content": "{{ answer }}"},
            ],
        },
    )
)
```

### Execution Order

Processors execute in the order they're added. Plan accordingly when one processor's output affects another.

## Configuration Parameters

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Identifier for the processor, used in output directory names |
| `build_stage` | BuildStage | When to run (default: `POST_BATCH`) |

### DropColumnsProcessorConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_names` | list[str] | Columns to remove from output |

### SchemaTransformProcessorConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `template` | dict[str, Any] | Jinja2 template defining the output schema. Must be JSON-serializable. |
