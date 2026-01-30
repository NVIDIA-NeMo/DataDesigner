# Message Traces

Traces capture the full conversation history during LLM generation, including system prompts, user prompts, model reasoning, and the final response. This visibility is essential for understanding model behavior, debugging generation issues, and iterating on prompts.

## Overview

When generating content with LLM columns, you often need to understand what happened during generation:

- What system prompt was used?
- What did the rendered user prompt look like?
- Did the model provide any reasoning content?
- Did the model retry after failures?
- How did the model arrive at the final answer?

Traces provide this visibility by capturing the ordered message history for each generation, including any multi-turn conversations that occur during retry scenarios.

## Enabling Traces

### Per-Column (Recommended)

Enable `with_trace=True` on specific LLM columns:

```python
import data_designer.config as dd

builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Answer: {{ question }}",
        model_alias="nvidia-text",
        with_trace=True,  # Enable trace for this column
    )
)
```

### Global Debug Override

Enable traces for ALL LLM columns (useful during development):

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
data_designer.set_run_config(
    dd.RunConfig(debug_override_save_all_column_traces=True)
)
```

## Trace Column Naming

When enabled, LLM columns produce an additional side-effect column:

- `{column_name}__trace`

For example, if your column is named `"answer"`, the trace column will be `"answer__trace"`.

## Trace Data Structure

Each trace is a `list[dict]` where each dict represents a message in the conversation.

### Message Fields by Role

| Role | Fields | Description |
|------|--------|-------------|
| `system` | `role`, `content` | System prompt setting model behavior |
| `user` | `role`, `content` | User prompt (rendered from template) |
| `assistant` | `role`, `content`, `reasoning_content` | Model response; may include reasoning from extended thinking models |

### Example Trace (Simple Generation)

A basic trace without retries:

```python
[
    # System message (if configured)
    {
        "role": "system",
        "content": "You are a helpful assistant that provides clear, concise answers."
    },
    # User message (the rendered prompt)
    {
        "role": "user",
        "content": "What is the capital of France?"
    },
    # Final assistant response
    {
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "reasoning_content": None  # May contain reasoning if model supports it
    }
]
```

### Example Trace (With Correction Retry)

When `max_correction_steps > 0` and parsing fails, traces capture the retry conversation:

```python
[
    # System message
    {
        "role": "system",
        "content": "Return only valid JSON."
    },
    # User message
    {
        "role": "user",
        "content": "Generate a person object with name and age."
    },
    # First attempt (invalid)
    {
        "role": "assistant",
        "content": "Here's a person: {name: 'John', age: 30}"  # Invalid JSON
    },
    # Error feedback
    {
        "role": "user",
        "content": "JSONDecodeError: Expecting property name enclosed in double quotes"
    },
    # Corrected response
    {
        "role": "assistant",
        "content": "{\"name\": \"John\", \"age\": 30}"
    }
]
```

## See Also

- **[Run Config](../code_reference/run_config.md)**: Runtime options including `debug_override_save_all_column_traces`
