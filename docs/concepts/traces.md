# Message Traces

Traces capture the full conversation history during LLM generation, including system prompts, user prompts, model reasoning, tool calls, tool results, and the final response. This visibility is essential for understanding model behavior, debugging generation issues, and iterating on prompts.

Traces are also useful in certain scenarios as the target output of the workflow, e.g. producing an SFT dataset for fine-tuning tool-use capability, for instance.

## Overview

When generating content with LLM columns, you often need to understand what happened during generation:

- What system prompt was used?
- What did the rendered user prompt look like?
- Did the model provide any reasoning content?
- Which tools were called (if tool use is enabled)?
- What arguments were passed to tools?
- What did tools return?
- Did the model retry after failures?
- How did the model arrive at the final answer?

Traces provide this visibility by capturing the ordered message history for each generation, including any multi-turn conversations that occur during tool use or retry scenarios.

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
| `system` | `role`, `content` | System prompt setting model behavior. `content` is a list of blocks in ChatML format. |
| `user` | `role`, `content` | User prompt (rendered from template). `content` is a list of blocks (text + multimodal). |
| `assistant` | `role`, `content`, `tool_calls`, `reasoning_content` | Model response; `content` may be empty if only requesting tools. |
| `tool` | `role`, `content`, `tool_call_id` | Tool execution result; `tool_call_id` links to the request. |

### Example Trace (Simple Generation)

A basic trace without tool use:

```python
[
    # System message (if configured)
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant that provides clear, concise answers."}]
    },
    # User message (the rendered prompt)
    {
        "role": "user",
        "content": [{"type": "text", "text": "What is the capital of France?"}]
    },
    # Final assistant response
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "The capital of France is Paris."}],
        "reasoning_content": None  # May contain reasoning if model supports it
    }
]
```

### Example Trace (With Tool Use)

When tool use is enabled, traces capture the full conversation including tool calls:

```python
[
    # System message
    {
        "role": "system",
        "content": [{"type": "text", "text": "You must call tools before answering. Only use tool results."}]
    },
    # User message (the rendered prompt)
    {
        "role": "user",
        "content": [{"type": "text", "text": "What documents are in the knowledge base about machine learning?"}]
    },
    # Assistant requests tool calls
    {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "list_docs",
                    "arguments": "{\"query\": \"machine learning\"}"
                }
            }
        ]
    },
    # Tool response (linked by tool_call_id)
    {
        "role": "tool",
        "content": [{"type": "text", "text": "Found 3 documents: intro_ml.pdf, neural_networks.pdf, transformers.pdf"}],
        "tool_call_id": "call_abc123"
    },
    # Final assistant response
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "The knowledge base contains three documents about machine learning: ..."}]
    }
]
```

### The tool_calls Structure

When an assistant message includes tool calls:

```python
{
    "id": "call_abc123",           # Unique ID linking to tool response
    "type": "function",            # Always "function" for MCP tools
    "function": {
        "name": "search_docs",     # Tool name
        "arguments": "{...}"       # JSON string of tool arguments
    }
}
```

## See Also

- **[Safety and Limits](mcp/safety-and-limits.md)**: Understand turn limits and timeout behavior
- **[Run Config](../code_reference/run_config.md)**: Runtime options including `debug_override_save_all_column_traces`
