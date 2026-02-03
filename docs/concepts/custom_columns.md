# Custom Columns

Custom columns let you implement your own generation logic using Python functions. Use them for multi-step LLM workflows, external API integration, or any scenario requiring full programmatic control. For reusable, distributable components, see [Plugins](../plugins/overview.md) instead.

## Quick Start

```python
import data_designer.config as dd

@dd.custom_column_generator(required_columns=["name"])
def create_greeting(row: dict) -> dict:
    row["greeting"] = f"Hello, {row['name']}!"
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="greeting",
        generator_function=create_greeting,
    )
)
```

## Function Signatures

Three signatures are supported, detected automatically by argument count:

| Args | Signature | Use Case |
|------|-----------|----------|
| 1 | `fn(row) -> dict` | Simple transforms |
| 2 | `fn(row, params) -> dict` | With typed params |
| 3 | `fn(row, params, ctx) -> dict` | LLM access via context |

For LLM access without params, use `params: None`:

```python
@dd.custom_column_generator(required_columns=["name"], model_aliases=["my-model"])
def generate_message(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    row[ctx.column_name] = ctx.generate_text(model_alias="my-model", prompt=f"Greet {row['name']}")
    return row
```

## Generation Strategies

| Strategy | Input | Parallelization |
|----------|-------|-----------------|
| `cell_by_cell` (default) | `row: dict` | Framework handles |
| `full_column` | `df: DataFrame` | Use `generate_text_batch()` |

For `full_column`, set `generation_strategy=dd.GenerationStrategy.FULL_COLUMN`.

## The Decorator

```python
@dd.custom_column_generator(
    required_columns=["col1"],        # DAG ordering
    side_effect_columns=["extra"],    # Additional columns created
    model_aliases=["model1"],         # For health checks
)
```

## CustomColumnContext

The context provides convenience methods for LLM access:

| Method | Description |
|--------|-------------|
| `generate_text(model_alias, prompt, ...)` | Single prompt |
| `generate_text_batch(model_alias, prompts, ...)` | Parallel prompts |
| `get_model(model_alias)` | Full `ModelFacade` access |

**The helpers are just convenience wrappers.** For full control, use `get_model()`:

```python
model = ctx.get_model("my-model")
response, trace = model.generate(
    prompt="...",
    parser=my_custom_parser,
    system_prompt="...",
    max_correction_steps=3,
)
```

This gives you access to all `ModelFacade` capabilities: custom parsers, correction loops, structured output, etc.

## Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Column name |
| `generator_function` | Callable | Yes | Decorated function |
| `generation_strategy` | GenerationStrategy | No | `CELL_BY_CELL` or `FULL_COLUMN` |
| `generator_params` | BaseModel | No | Typed params passed to function |

## Multi-Turn Example

```python
@dd.custom_column_generator(
    required_columns=["topic"],
    side_effect_columns=["draft", "critique"],
    model_aliases=["writer", "editor"],
)
def writer_editor(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    draft = ctx.generate_text("writer", f"Write about '{row['topic']}'")
    critique = ctx.generate_text("editor", f"Critique: {draft}")
    revised = ctx.generate_text("writer", f"Revise based on: {critique}\n\nOriginal: {draft}")

    row[ctx.column_name] = revised
    row["draft"] = draft
    row["critique"] = critique
    return row
```

## Development Testing

Test generators with real LLM calls without running the full pipeline:

```python
data_designer = DataDesigner()
ctx = dd.CustomColumnContext.from_data_designer(data_designer)
result = my_generator({"name": "Alice"}, None, ctx)
```

## See Also

- [Column Configs Reference](../code_reference/column_configs.md)
- [Plugins Overview](../plugins/overview.md)
