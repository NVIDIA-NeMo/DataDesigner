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

## Generation Strategies

Custom columns support two strategies:

| Strategy | Function Signature | Parallelization | Use Case |
|----------|-------------------|-----------------|----------|
| `cell_by_cell` (default) | `(row: dict) -> dict` | Framework handles it | LLM calls, I/O-bound work |
| `full_column` | `(df: DataFrame) -> DataFrame` | User handles via `generate_text_batch()` | Vectorized ops, cross-row access |

### cell_by_cell (default)

The framework calls your function once per row and parallelizes execution automatically:

```python
def my_generator(row: dict, ctx: dd.CustomColumnContext) -> dict:
    row["result"] = ctx.generate_text(model_alias="my-model", prompt="...")
    return row

dd.CustomColumnConfig(
    name="result",
    generator_function=my_generator,
    input_columns=["input"],
    model_aliases=["my-model"],
    # generation_strategy="cell_by_cell" is the default
)
```

### full_column

The framework calls your function once with the entire DataFrame. Use `generate_text_batch()` to parallelize LLM calls:

```python
import pandas as pd

def batch_generator(df: pd.DataFrame, ctx: dd.CustomColumnContext) -> pd.DataFrame:
    prompts = [f"Process: {val}" for val in df["input"]]
    results = ctx.generate_text_batch(model_alias="my-model", prompts=prompts)
    df["result"] = results
    return df

dd.CustomColumnConfig(
    name="result",
    generator_function=batch_generator,
    input_columns=["input"],
    model_aliases=["my-model"],
    generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
)
```

## Basic Example

```python
import data_designer.config as dd

def create_greeting(row: dict) -> dict:
    row["greeting"] = f"Hello, {row['name']}!"
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="greeting",
        generator_function=create_greeting,
        input_columns=["name"],
    )
)
```

## LLM Generation

```python
import data_designer.config as dd

def generate_message(row: dict, ctx: dd.CustomColumnContext) -> dict:
    response = ctx.generate_text(
        model_alias="my-model",
        prompt=f"Write a message for {row['name']}.",
        system_prompt="Be concise.",
    )
    row[ctx.column_name] = response
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="message",
        generator_function=generate_message,
        input_columns=["name"],
        model_aliases=["my-model"],
    )
)
```

## CustomColumnContext API

The `ctx` argument provides access to LLM models and custom parameters.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `column_name` | str | Name of the column being generated |
| `generator_config` | BaseModel \| None | Typed configuration object from the config |
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
| `generator_function` | Callable | Yes | Generator function |
| `generation_strategy` | GenerationStrategy | No | `GenerationStrategy.CELL_BY_CELL` (default) or `GenerationStrategy.FULL_COLUMN`. String values `"cell_by_cell"` and `"full_column"` are also accepted. |
| `input_columns` | list[str] | No | Columns that must exist before this column runs (determines DAG order) |
| `output_columns` | list[str] | No | Additional columns created by the function |
| `model_aliases` | list[str] | No | Model aliases used by the function (enables health checks) |
| `generator_config` | BaseModel | No | Typed configuration object accessible via `ctx.generator_config` |

## Multiple Output Columns

Declare additional columns with `output_columns`:

```python
def generate_with_trace(row: dict, ctx: dd.CustomColumnContext) -> dict:
    prompt = f"Write about {row['topic']}."
    response = ctx.generate_text(model_alias="my-model", prompt=prompt)

    row[ctx.column_name] = response
    row["prompt_used"] = prompt
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="content",
        generator_function=generate_with_trace,
        input_columns=["topic"],
        output_columns=["prompt_used"],
        model_aliases=["my-model"],
    )
)
```

!!! warning "Undeclared columns are removed"
    Columns not declared in `name` or `output_columns` will be removed with a warning. This ensures an explicit contract between your function and the framework, preventing accidental columns from polluting the dataset.

!!! danger "Don't remove existing columns"
    Your generation function must not remove any pre-existing columns from the row/DataFrame. The framework validates this and will raise an error if columns are removed.

## Typed Configuration

Pass typed configuration to your function via the `generator_config` parameter. Define a Pydantic model for type safety and validation:

```python
from pydantic import BaseModel

class ContentGeneratorConfig(BaseModel):
    tone: str = "neutral"
    max_length: int = 100

def configurable_generator(row: dict, ctx: dd.CustomColumnContext) -> dict:
    # Access typed config - IDE autocomplete works!
    config = ctx.generator_config
    prompt = f"Write a {config.tone} message about {row['topic']} in under {config.max_length} words."
    row[ctx.column_name] = ctx.generate_text(model_alias="my-model", prompt=prompt)
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="styled_content",
        generator_function=configurable_generator,
        input_columns=["topic"],
        model_aliases=["my-model"],
        generator_config=ContentGeneratorConfig(tone="professional", max_length=50),
    )
)
```

This pattern provides:

- **Type safety**: Pydantic validates your configuration at construction time
- **IDE support**: Full autocomplete and type hints for your config fields
- **Reusability**: Same function with different typed configurations

## Capturing Conversation Traces

Use `return_trace=True` to capture the full conversation history, including any corrections or tool calls:

```python
def generate_with_trace(row: dict, ctx: dd.CustomColumnContext) -> dict:
    response, trace = ctx.generate_text(
        model_alias="my-model",
        prompt=f"Write about {row['topic']}.",
        return_trace=True,
    )

    row[ctx.column_name] = response
    # Store trace as JSON for debugging or analysis
    row["conversation_trace"] = [msg.to_dict() for msg in trace]
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="content",
        generator_function=generate_with_trace,
        input_columns=["topic"],
        output_columns=["conversation_trace"],
        model_aliases=["my-model"],
    )
)
```

The trace is a list of `ChatMessage` objects representing the full conversation, useful for debugging, auditing, or analyzing model behavior.

## Multi-Turn Workflows

Custom columns excel at multi-step LLM workflows where intermediate results feed into subsequent calls:

```python
def writer_editor_workflow(row: dict, ctx: dd.CustomColumnContext) -> dict:
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
        input_columns=["topic"],
        output_columns=["draft", "critique"],
        model_aliases=["writer", "editor"],
    )
)
```

## See Also

- [Column Configs Reference](../code_reference/column_configs.md)
- [Model Configs](models/model-configs.md)
- [Plugins Overview](../plugins/overview.md)
