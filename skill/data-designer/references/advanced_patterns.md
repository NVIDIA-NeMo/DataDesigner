# Data Designer Advanced Patterns

Advanced usage patterns for `data-designer`. See `references/api_reference.md` for the complete API.

---

## Table of Contents

1. [Custom Column Generators](#1-custom-column-generators)
2. [MCP Tool Integration](#2-mcp-tool-integration)
3. [Multimodal Inputs](#3-multimodal-inputs)
4. [Schema Transform Processors](#4-schema-transform-processors)
5. [Performance Tuning](#5-performance-tuning)
6. [Multi-Stage Refinement](#6-multi-stage-refinement)
7. [Conditional Sampling](#7-conditional-sampling)
8. [Trace & Reasoning Extraction](#8-trace--reasoning-extraction)
9. [Nemotron Personas](#9-nemotron-personas)
10. [Configuration Serialization](#10-configuration-serialization)

---

## 1. Custom Column Generators

Use `@custom_column_generator` for logic that can't be expressed with built-in column types.

### Decorator Signature

```python
from data_designer.config import custom_column_generator, GenerationStrategy

@custom_column_generator(
    required_columns: list[str] | None = None,      # columns this generator reads
    side_effect_columns: list[str] | None = None,    # extra columns this generator creates
    model_aliases: list[str] | None = None,          # LLM models needed
)
```

### Function Signatures (1-3 args, names matter)

```python
# 1-arg: row-only (cell_by_cell) or df-only (full_column)
def my_gen(row): ...
def my_gen(df): ...

# 2-arg: with typed parameters
def my_gen(row, generator_params): ...

# 3-arg: with LLM access
def my_gen(row, generator_params, models): ...
```

First param name determines strategy: `row` -> `CELL_BY_CELL`, `df` -> `FULL_COLUMN`.

### Writer-Editor Pattern (Multi-Model)

```python
@custom_column_generator(
    required_columns=["topic"],
    side_effect_columns=["draft", "critique"],
    model_aliases=["writer", "editor"],
)
def writer_editor(row: dict, generator_params: None, models: dict) -> dict:
    draft, _ = models["writer"].generate(prompt=f"Write about '{row['topic']}'")
    critique, _ = models["editor"].generate(prompt=f"Critique: {draft}")
    revised, _ = models["writer"].generate(
        prompt=f"Revise based on: {critique}\n\nOriginal: {draft}"
    )
    row["final_text"] = revised
    row["draft"] = draft
    row["critique"] = critique
    return row

config_builder.add_column(dd.CustomColumnConfig(
    name="final_text",
    generator_function=writer_editor,
    generation_strategy=dd.GenerationStrategy.CELL_BY_CELL,
))
```

### With Typed Parameters

```python
from pydantic import BaseModel

class FormatConfig(BaseModel):
    prefix: str = "Dr."
    uppercase: bool = True

@custom_column_generator(required_columns=["name"])
def format_name(row, generator_params):
    name = f"{generator_params.prefix} {row['name']}"
    return name.upper() if generator_params.uppercase else name

config_builder.add_column(dd.CustomColumnConfig(
    name="formal_name",
    generator_function=format_name,
    generator_params=FormatConfig(prefix="Prof."),
))
```

### Full-Column Strategy (Vectorized)

```python
@custom_column_generator(required_columns=["score"])
def normalize_scores(df):
    return (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())

config_builder.add_column(dd.CustomColumnConfig(
    name="normalized_score",
    generator_function=normalize_scores,
    generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
))
```

### Testing Custom Generators

Test generators outside the full pipeline:

```python
data_designer = DataDesigner()
models = data_designer.get_models(["nvidia-text"])
result = my_generator({"topic": "AI"}, None, models)
```

---

## 2. MCP Tool Integration

### Self-Contained Server + Client Pattern

A common pattern is a single script that acts as both MCP server and Data Designer client:

```python
import sys
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP("my-tools")

@mcp_server.tool()
def search_docs(query: str) -> str:
    """Search documents by keyword."""
    # ... implementation
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        mcp_server.run()
    else:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner

        provider = dd.LocalStdioMCPProvider(
            name="my-tools",
            command=sys.executable,
            args=[__file__, "serve"],
        )

        tool_config = dd.ToolConfig(
            tool_alias="search",
            providers=["my-tools"],
            allow_tools=["search_docs"],
            max_tool_call_turns=10,
            timeout_sec=30.0,
        )

        data_designer = DataDesigner(mcp_providers=[provider])
        config_builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

        config_builder.add_column(dd.LLMTextColumnConfig(
            name="answer",
            prompt="Use tools to find and answer: {{ question }}",
            model_alias="nvidia-text",
            tool_alias="search",
            system_prompt="You have access to search tools. Use them to find information.",
            with_trace=dd.TraceType.ALL_MESSAGES,
        ))
```

### Remote MCP Provider

```python
mcp_provider = dd.MCPProvider(
    name="remote-tools",
    endpoint="https://mcp.example.com/sse",
    api_key="REMOTE_API_KEY",
)
```

---

## 3. Multimodal Inputs

### Image Context for Vision Models

```python
# From URL column
dd.LLMTextColumnConfig(
    name="description",
    prompt="Describe this image in detail.",
    model_alias="nvidia-vision",
    multi_modal_context=[
        dd.ImageContext(
            column_name="image_url",
            data_type=dd.ModalityDataType.URL,
        )
    ],
)

# From Base64 column
dd.LLMTextColumnConfig(
    name="summary",
    prompt="Summarize this document.",
    model_alias="nvidia-vision",
    multi_modal_context=[
        dd.ImageContext(
            column_name="base64_image",
            data_type=dd.ModalityDataType.BASE64,
            image_format=dd.ImageFormat.PNG,  # required for base64
        )
    ],
)
```

Image formats: `PNG`, `JPG`/`JPEG`, `GIF`, `WEBP`

Multiple images: column can contain a JSON array of URLs.

---

## 4. Schema Transform Processors

Create additional output datasets with transformed schemas. The original dataset passes through unchanged.

### Chat Format Transform

```python
config_builder.add_processor(dd.SchemaTransformProcessorConfig(
    name="openai_chat",
    template={
        "messages": [
            {"role": "system", "content": "{{ system_prompt }}"},
            {"role": "user", "content": "{{ question }}"},
            {"role": "assistant", "content": "{{ answer }}"},
        ],
        "metadata": {
            "category": "{{ category | upper }}",
            "difficulty": "{{ difficulty }}",
        },
    },
))

# Load transformed data
transformed = results.load_processor_dataset("openai_chat")
```

### Multi-Turn Chat Transform

```python
config_builder.add_processor(dd.SchemaTransformProcessorConfig(
    name="multi_turn",
    template={
        "conversations": [
            {"from": "human", "value": "{{ turn_1_user }}"},
            {"from": "gpt", "value": "{{ turn_1_assistant }}"},
            {"from": "human", "value": "{{ turn_2_user }}"},
            {"from": "gpt", "value": "{{ turn_2_assistant }}"},
        ],
    },
))
```

---

## 5. Performance Tuning

### Key Parameters

| Parameter | Default | Tune When |
|-----------|---------|-----------|
| `buffer_size` | 1000 | Memory issues -> lower; faster batch cycling -> lower |
| `max_parallel_requests` | 4 | Low GPU util -> increase (self-hosted: try 256-1024) |
| `max_conversation_restarts` | 5 | Strict schemas -> increase to 7+ |
| `max_conversation_correction_steps` | 0 | Schema conformance issues -> set to 2-3 |
| `disable_early_shutdown` | False | Debugging -> set True |
| `non_inference_max_parallel_workers` | 4 | Many non-LLM columns -> increase |

### Concurrency Formula

```
concurrent_requests = min(buffer_size, max_parallel_requests, remaining_cells)
```

### Execution Model

1. Dataset split into batches of `buffer_size`
2. Within a batch, columns processed **sequentially** (DAG order)
3. Within a column, cells processed **in parallel** (up to limit)

### Benchmarking

Run 100 records with increasing `max_parallel_requests` (4 -> 8 -> 16 -> 32 -> ...). Stop when runtime plateaus.

```python
from data_designer.config import RunConfig

data_designer.set_run_config(RunConfig(
    buffer_size=500,
    max_conversation_restarts=7,
    max_conversation_correction_steps=2,
))
```

---

## 6. Multi-Stage Refinement

Chain LLM columns for iterative improvement:

```python
# Stage 1: Generate draft
config_builder.add_column(dd.LLMTextColumnConfig(
    name="draft", prompt="Write an article about {{ topic }}.", model_alias="nvidia-text",
))

# Stage 2: Critique
config_builder.add_column(dd.LLMStructuredColumnConfig(
    name="critique",
    prompt="Identify 3 improvements for:\n\n{{ draft }}",
    output_format=CritiqueSchema,
    model_alias="nvidia-reasoning",
))

# Stage 3: Refine
config_builder.add_column(dd.LLMTextColumnConfig(
    name="final_article",
    prompt="Improve based on feedback:\n\nDraft: {{ draft }}\nFeedback: {{ critique.suggestions }}",
    model_alias="nvidia-text",
))

# Stage 4: Judge
config_builder.add_column(dd.LLMJudgeColumnConfig(
    name="quality",
    prompt="Rate this article:\n\n{{ final_article }}",
    scores=[dd.Score(name="Quality", description="...", options={1: "Poor", ..., 5: "Excellent"})],
    model_alias="nvidia-text",
))
```

---

## 7. Conditional Sampling

Override sampler params based on other column values:

```python
dd.SamplerColumnConfig(
    name="review_style",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(
        values=["brief", "detailed", "rambling"],
        weights=[0.4, 0.4, 0.2],
    ),
    conditional_params={
        "age_group == '18-25'": dd.CategorySamplerParams(values=["rambling"]),
        "age_group == '65+'": dd.CategorySamplerParams(values=["detailed"]),
    },
)
```

---

## 8. Trace & Reasoning Extraction

### Trace Types

- `TraceType.NONE` (default): No trace
- `TraceType.LAST_MESSAGE`: Only final response -> `{name}__trace`
- `TraceType.ALL_MESSAGES`: Full conversation -> `{name}__trace`

### Reasoning Content

`extract_reasoning_content=True` creates `{name}__reasoning_content` with chain-of-thought.

Available on all LLM column types.

### Use Cases

- **Debugging**: `ALL_MESSAGES` to see full conversation including tool calls
- **Fine-tuning data**: `extract_reasoning_content=True` for clean reasoning extraction
- **Tool-use training**: `ALL_MESSAGES` to capture tool call patterns

---

## 9. Nemotron Personas

Census-grounded synthetic person data with rich personality traits.

### Setup

```bash
data-designer download personas --locale en_US
```

### Usage

```python
dd.SamplerColumnConfig(
    name="customer",
    sampler_type=dd.SamplerType.PERSON,
    params=dd.PersonSamplerParams(
        locale="en_US",
        sex="Female",
        age_range=[25, 45],
        with_synthetic_personas=True,  # personality traits, cultural backgrounds
        select_field_values={"state": ["NY", "CA"]},
    ),
)
```

### Available Locales

`en_US`, `en_IN`, `en_SG`, `hi_Deva_IN`, `hi_Latn_IN`, `ja_JP`, `pt_BR`

### Persona Fields

Big Five personality traits, cultural backgrounds, skills, hobbies, career goals, plus domain-specific personas (professional, financial, healthcare, sports, arts, travel, culinary).

---

## 10. Configuration Serialization

### Save

```python
config_builder.write_config("my_config.yaml")
config_builder.write_config("my_config.json", indent=2)
```

Note: `DataFrameSeedSource` is not serializable. Use `LocalFileSeedSource` for shareable configs.

### Load

```python
loaded = dd.DataDesignerConfigBuilder.from_config("my_config.yaml")
loaded = dd.DataDesignerConfigBuilder.from_config({"columns": [...]})
```
