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

## Function Signatures

Two signatures are supported:

```python
# Simple: no LLM access
def my_generator(row: dict) -> dict:
    row["new_column"] = transform(row["input"])
    return row

# With context: LLM and resource access
def my_generator(row: dict, ctx: dd.CustomColumnContext) -> dict:
    row["new_column"] = ctx.generate_text(model_alias="my-model", prompt="...")
    return row
```

The `ctx` argument is a `CustomColumnContext` object that provides access to configured LLM models and custom parameters—add it when your function needs to call LLMs or access `kwargs`. The framework parallelizes execution across rows automatically.

## Basic Example

```python
import data_designer.config as dd

def create_greeting(row: dict) -> dict:
    row["greeting"] = f"Hello, {row['name']}!"
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="greeting",
        generate_fn=create_greeting,
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
        generate_fn=generate_message,
        input_columns=["name"],
    )
)
```

## CustomColumnContext API

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `column_name` | str | Name of the column being generated |
| `kwargs` | dict | Custom parameters from configuration |
| `model_registry` | ModelRegistry | Access to all configured models |

### Methods

**`generate_text(model_alias, prompt, system_prompt=None)`** — Returns generated text as a string.

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
| `generate_fn` | Callable | Yes | Generator function |
| `input_columns` | list[str] | No | Required input columns |
| `output_columns` | list[str] | No | Additional columns created |
| `kwargs` | dict | No | Custom parameters for `ctx.kwargs` |

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
        generate_fn=generate_with_trace,
        input_columns=["topic"],
        output_columns=["prompt_used"],
    )
)
```

!!! warning "Undeclared columns are removed"
    Columns not declared in `name` or `output_columns` will be removed with a warning.

## Custom Parameters

Pass configuration via `kwargs`:

```python
def configurable_generator(row: dict, ctx: dd.CustomColumnContext) -> dict:
    tone = ctx.kwargs.get("tone", "neutral")
    prompt = f"Write a {tone} message about {row['topic']}."
    row[ctx.column_name] = ctx.generate_text(model_alias="my-model", prompt=prompt)
    return row

config_builder.add_column(
    dd.CustomColumnConfig(
        name="styled_content",
        generate_fn=configurable_generator,
        input_columns=["topic"],
        kwargs={"tone": "professional"},
    )
)
```

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
        generate_fn=writer_editor_workflow,
        input_columns=["topic"],
        output_columns=["draft", "critique"],
    )
)
```

## See Also

- [Column Configs Reference](../code_reference/column_configs.md)
- [Model Configs](models/model-configs.md)
- [Plugins Overview](../plugins/overview.md)
