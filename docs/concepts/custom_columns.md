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

Three signatures are supported. **Parameter names are validated**:

| Args | Signature | Use Case |
|------|-----------|----------|
| 1 | `fn(row) -> dict` | Simple transforms |
| 2 | `fn(row, generator_params) -> dict` | With typed params |
| 3 | `fn(row, generator_params, models) -> dict` | LLM access via models dict |

For `full_column` strategy, use `df` instead of `row`.

For LLM access without params, use `generator_params: None`:

```python
@dd.custom_column_generator(required_columns=["name"], model_aliases=["my-model"])
def generate_message(row: dict, generator_params: None, models: dict) -> dict:
    response, _ = models["my-model"].generate(prompt=f"Greet {row['name']}")
    row["greeting"] = response
    return row
```

Model aliases are validated before generation starts. If an alias doesn't exist in your config, an error is raised during the health check.

## Generation Strategies

| Strategy | Input | Use Case |
|----------|-------|----------|
| `cell_by_cell` (default) | `row: dict` | LLM calls, row-by-row logic |
| `full_column` | `df: DataFrame` | Vectorized DataFrame operations |

**Recommendation:** Use `cell_by_cell` for LLM calls. The framework handles parallelization automatically. Use `full_column` only for vectorized operations that don't involve LLM calls.

For `full_column`, set `generation_strategy=dd.GenerationStrategy.FULL_COLUMN`.

## The Decorator

```python
@dd.custom_column_generator(
    required_columns=["col1"],        # DAG ordering
    side_effect_columns=["extra"],    # Additional columns created
    model_aliases=["model1"],         # Required for LLM access
)
```

## Models Dict

The third argument is a dict of `ModelFacade` instances, keyed by alias. **You must declare all models required in your custom column generator in `model_aliases`** - this populates the `models` dict and enables health checks before generation starts.

```python
@dd.custom_column_generator(model_aliases=["my-model"])
def my_generator(row: dict, generator_params: None, models: dict) -> dict:
    model = models["my-model"]
    response, trace = model.generate(
        prompt="...",
        parser=my_custom_parser,  # optional, defaults to identity
        system_prompt="...",
        max_correction_steps=3,
    )
    row["result"] = response
    return row
```

This gives you direct access to all `ModelFacade` capabilities: custom parsers, correction loops, structured output, tool use, etc.

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
def writer_editor(row: dict, generator_params: None, models: dict) -> dict:
    draft, _ = models["writer"].generate(prompt=f"Write about '{row['topic']}'")
    critique, _ = models["editor"].generate(prompt=f"Critique: {draft}")
    revised, _ = models["writer"].generate(prompt=f"Revise based on: {critique}\n\nOriginal: {draft}")

    row["final_text"] = revised
    row["draft"] = draft
    row["critique"] = critique
    return row
```

## Development Testing

Test generators with real LLM calls without running the full pipeline:

```python
data_designer = DataDesigner()
models = data_designer.get_models(["my-model"])
result = my_generator({"name": "Alice"}, None, models)
```

## See Also

- [Column Configs Reference](../code_reference/column_configs.md)
- [Plugins Overview](../plugins/overview.md)
