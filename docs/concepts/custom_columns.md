# Custom Columns

Custom columns let you implement your own generation logic using Python functions. Use them for multi-step LLM workflows, external API integration, or any scenario requiring full programmatic control.

## Custom Columns vs Plugins

Both custom columns and [plugins](../plugins/overview.md) allow custom generation logic, but serve different purposes:

| | Custom Columns | Plugins |
|---|---|---|
| **Setup** | Inline function in your code | Separate Python package |
| **Sharing** | Project-specific | Distributable via PyPI |
| **LLM Access** | Via `CustomColumnContext` | Via `ResourceProvider` |
| **Best for** | One-off logic, prototyping | Reusable components |

!!! tip "When to use which"
    Start with custom columns for quick iteration. Convert to a plugin when you need to share the logic across projects or with other users.

## Quick Start

Define a custom column generator using the `@custom_column_generator` decorator:

```python
import data_designer.config as dd

@dd.custom_column_generator(
    required_columns=["name"],
)
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

The decorator declares the column's dependencies, which the framework uses for DAG ordering and validation.

## Function Signatures

Custom columns support three function signatures, detected automatically:

| Args | Signature | Use Case |
|------|-----------|----------|
| 1 | `fn(row: dict) -> dict` | Simple transforms, no LLM |
| 2 | `fn(row: dict, params: BaseModel) -> dict` | Transforms with typed params |
| 3 | `fn(row: dict, params: BaseModel, ctx: CustomColumnContext) -> dict` | Full LLM access |

### Simple Transform (1 arg)

```python
@dd.custom_column_generator(required_columns=["name"])
def simple_greeting(row: dict) -> dict:
    row["greeting"] = f"Hello, {row['name']}!"
    return row
```

### With Typed Params (2 args)

```python
from pydantic import BaseModel

class GreetingParams(BaseModel):
    prefix: str = "Hello"

@dd.custom_column_generator(required_columns=["name"])
def greeting_with_params(row: dict, params: GreetingParams) -> dict:
    row["greeting"] = f"{params.prefix}, {row['name']}!"
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="greeting",
        generator_function=greeting_with_params,
        generator_params=GreetingParams(prefix="Welcome"),
    )
)
```

### With LLM Access (3 args)

```python
@dd.custom_column_generator(
    required_columns=["name"],
    model_aliases=["my-model"],
)
def generate_message(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    response = ctx.generate_text(
        model_alias="my-model",
        prompt=f"Write a message for {row['name']}.",
        system_prompt="Be concise.",
    )
    row[ctx.column_name] = response
    return row
```

## Generation Strategies

Custom columns support two strategies:

| Strategy | Function Signature | Parallelization | Use Case |
|----------|-------------------|-----------------|----------|
| `cell_by_cell` (default) | `(row: dict, ...) -> dict` | Framework handles it | LLM calls, I/O-bound work |
| `full_column` | `(df: DataFrame, ...) -> DataFrame` | User handles via `generate_text_batch()` | Vectorized ops, cross-row access |

### cell_by_cell (default)

The framework calls your function once per row and parallelizes execution automatically:

```python
@dd.custom_column_generator(
    required_columns=["input"],
    model_aliases=["my-model"],
)
def my_generator(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    row["result"] = ctx.generate_text(model_alias="my-model", prompt="...")
    return row

dd.CustomColumnConfig(
    name="result",
    generator_function=my_generator,
    # generation_strategy="cell_by_cell" is the default
)
```

### full_column

The framework calls your function once with the entire DataFrame. Use `generate_text_batch()` to parallelize LLM calls:

```python
import pandas as pd

@dd.custom_column_generator(
    required_columns=["input"],
    model_aliases=["my-model"],
)
def batch_generator(
    df: pd.DataFrame, params: None, ctx: dd.CustomColumnContext
) -> pd.DataFrame:
    prompts = [f"Process: {val}" for val in df["input"]]
    results = ctx.generate_text_batch(model_alias="my-model", prompts=prompts)
    df["result"] = results
    return df

dd.CustomColumnConfig(
    name="result",
    generator_function=batch_generator,
    generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
)
```

## The @custom_column_generator Decorator

The decorator declares metadata that the framework uses for DAG ordering and validation:

```python
@dd.custom_column_generator(
    required_columns=["col1", "col2"],      # Columns that must exist before this runs
    side_effect_columns=["extra_col"],       # Additional columns your function creates
    model_aliases=["model1", "model2"],      # Models used (enables health checks)
)
def my_generator(row: dict, params: MyParams, ctx: dd.CustomColumnContext) -> dict:
    ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `required_columns` | list[str] | Columns that must exist before this column runs (determines DAG order) |
| `side_effect_columns` | list[str] | Additional columns created by the function |
| `model_aliases` | list[str] | Model aliases used (optional, enables health checks) |

## CustomColumnContext API

The `ctx` argument provides access to LLM models.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `column_name` | str | Name of the column being generated |
| `model_registry` | ModelRegistry | Access to all configured models |

### Methods

**`generate_text(model_alias, prompt, system_prompt=None, return_trace=False)`** — Generate text for a single prompt. Returns a string, or a `(string, trace)` tuple if `return_trace=True`.

**`generate_text_batch(model_alias, prompts, system_prompt=None, max_workers=8, return_trace=False)`** — Generate text for multiple prompts in parallel. Returns a list of strings, or a list of `(string, trace)` tuples if `return_trace=True`. Use this in `full_column` strategy.

**`get_model(model_alias)`** — Returns a `ModelFacade` for advanced control:

```python
model = ctx.get_model("my-model")
response, metadata = model.generate(
    prompt="...",
    parser=lambda x: x,
    system_prompt="...",
)
```

## Configuration Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Primary column name |
| `generator_function` | Callable | Yes | Generator function decorated with `@custom_column_generator` |
| `generation_strategy` | GenerationStrategy | No | `GenerationStrategy.CELL_BY_CELL` (default) or `GenerationStrategy.FULL_COLUMN` |
| `generator_params` | BaseModel | No | Typed configuration object passed as second argument to the function |

## Multiple Output Columns

Declare additional columns with `side_effect_columns` in the decorator:

```python
@dd.custom_column_generator(
    required_columns=["topic"],
    side_effect_columns=["prompt_used"],
    model_aliases=["my-model"],
)
def generate_with_trace(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    prompt = f"Write about {row['topic']}."
    response = ctx.generate_text(model_alias="my-model", prompt=prompt)

    row[ctx.column_name] = response
    row["prompt_used"] = prompt
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="content",
        generator_function=generate_with_trace,
    )
)
```

!!! warning "Undeclared columns are removed"
    Columns not declared in `name` or `side_effect_columns` will be removed with a warning. This ensures an explicit contract between your function and the framework.

!!! danger "Don't remove existing columns"
    Your generation function must not remove any pre-existing columns from the row/DataFrame. The framework validates this and will raise an error.

## Typed Configuration

Pass typed configuration via `generator_params`. The params are passed as the second argument to your function:

```python
from pydantic import BaseModel

class ContentParams(BaseModel):
    tone: str = "neutral"
    max_length: int = 100

@dd.custom_column_generator(
    required_columns=["topic"],
    model_aliases=["my-model"],
)
def configurable_generator(
    row: dict, params: ContentParams, ctx: dd.CustomColumnContext
) -> dict:
    prompt = f"Write a {params.tone} message about {row['topic']} in under {params.max_length} words."
    row[ctx.column_name] = ctx.generate_text(model_alias="my-model", prompt=prompt)
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="styled_content",
        generator_function=configurable_generator,
        generator_params=ContentParams(tone="professional", max_length=50),
    )
)
```

This pattern provides:

- **Type safety**: Pydantic validates your configuration at construction time
- **IDE support**: Full autocomplete and type hints for your config fields
- **Reusability**: Same function with different typed configurations

## Capturing Conversation Traces

Use `return_trace=True` to capture the full conversation history:

```python
@dd.custom_column_generator(
    required_columns=["topic"],
    side_effect_columns=["conversation_trace"],
    model_aliases=["my-model"],
)
def generate_with_trace(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    response, trace = ctx.generate_text(
        model_alias="my-model",
        prompt=f"Write about {row['topic']}.",
        return_trace=True,
    )

    row[ctx.column_name] = response
    row["conversation_trace"] = [msg.to_dict() for msg in trace]
    return row
```

## Multi-Turn Workflows

Custom columns excel at multi-step LLM workflows:

```python
@dd.custom_column_generator(
    required_columns=["topic"],
    side_effect_columns=["draft", "critique"],
    model_aliases=["writer", "editor"],
)
def writer_editor_workflow(
    row: dict, params: None, ctx: dd.CustomColumnContext
) -> dict:
    topic = row["topic"]

    draft = ctx.generate_text(
        model_alias="writer",
        prompt=f"Write a hook about '{topic}'.",
    )

    critique = ctx.generate_text(
        model_alias="editor",
        prompt=f"Critique this: {draft}",
    )

    revised = ctx.generate_text(
        model_alias="writer",
        prompt=f"Revise based on: {critique}\n\nOriginal: {draft}",
    )

    row[ctx.column_name] = revised
    row["draft"] = draft
    row["critique"] = critique
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="refined_content",
        generator_function=writer_editor_workflow,
    )
)
```

## Developing Generators

Use `CustomColumnContext.from_data_designer()` to create a real context for testing your generators with actual LLM calls:

```python
from pydantic import BaseModel
import data_designer.config as dd
from data_designer.interface import DataDesigner

class MyParams(BaseModel):
    tone: str = "friendly"

@dd.custom_column_generator(
    required_columns=["name"],
    model_aliases=["my-model"],
)
def my_generator(row: dict, params: MyParams, ctx: dd.CustomColumnContext) -> dict:
    prompt = f"Write a {params.tone} message for {row['name']}"
    row["message"] = ctx.generate_text(model_alias="my-model", prompt=prompt)
    return row

# Initialize DataDesigner with your model configs
data_designer = DataDesigner()

# Create a real context for development
ctx = dd.CustomColumnContext.from_data_designer(data_designer, column_name="message")

# Test your generator with real LLM calls
params = MyParams(tone="professional")
result = my_generator({"name": "Alice"}, params, ctx)
print(result)
```

This approach allows you to:

- Test your generator logic with actual LLM responses
- Iterate quickly without running the full pipeline
- Debug issues in isolation

## See Also

- [Column Configs Reference](../code_reference/column_configs.md)
- [Model Configs](models/model-configs.md)
- [Plugins Overview](../plugins/overview.md)
