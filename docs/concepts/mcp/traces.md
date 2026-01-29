# Message Traces

Traces capture the full conversation history during LLM generation, including prompts, tool calls, tool results, and the final answer. This is especially useful for understanding and debugging tool use behavior.

## Overview

When tool use is enabled, you often need to inspect what happened during generation:

- Which tools did the model call?
- What arguments were passed?
- What did the tools return?
- How did the model use the results?

Traces provide this visibility by capturing the ordered message history for each generation.

## Enabling Traces

### Per-Column (Recommended)

Enable `with_trace=True` on specific LLM columns:

```python
import data_designer.config as dd

builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Use tools to answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",
        with_trace=True,  # Enable trace for this column
    )
)
```

### Global Debug Override

Enable traces for ALL LLM columns (useful during development):

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner(mcp_providers=[...])
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
| `assistant` | `role`, `content`, `tool_calls`, `reasoning_content` | Model response; `content` may be `None` if only requesting tools |
| `tool` | `role`, `content`, `tool_call_id` | Tool execution result; `tool_call_id` links to the request |

### The tool_calls Structure

When an assistant message includes tool calls:

```python
{
    "id": "call_abc123",           # Unique ID linking to tool response
    "type": "function",            # Always "function" for MCP tools
    "function": {
        "name": "search_docs",     # Tool name from MCP server
        "arguments": "{...}"       # JSON string of tool arguments
    }
}
```

### Example Trace

Here's a complete trace showing a tool call flow:

```python
[
    # System message (if configured)
    {
        "role": "system",
        "content": "You must call tools before answering. Only use tool results."
    },
    # User message (the rendered prompt)
    {
        "role": "user",
        "content": "What documents are in the knowledge base about machine learning?"
    },
    # Assistant requests tool calls
    {
        "role": "assistant",
        "content": None,
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
        "content": "Found 3 documents: intro_ml.pdf, neural_networks.pdf, transformers.pdf",
        "tool_call_id": "call_abc123"
    },
    # Final assistant response
    {
        "role": "assistant",
        "content": "The knowledge base contains three documents about machine learning: ..."
    }
]
```

## Working with Traces

Access traces from `PreviewResults` or generated datasets:

```python
# After preview
results = data_designer.preview(builder, num_records=5)
df = results.dataset.to_pandas()

# Access trace for a specific row
trace = df["answer__trace"].iloc[0]
```

### Count Tool Calls

```python
tool_messages = [msg for msg in trace if msg.get("role") == "tool"]
print(f"This generation made {len(tool_messages)} tool calls")
```

### Extract Tool Names Used

```python
tool_calls = []
for msg in trace:
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            tool_calls.append(tc["function"]["name"])
print(f"Tools used: {tool_calls}")
```

### Find the Final Answer

```python
final_answer = next(
    (msg["content"] for msg in reversed(trace)
     if msg.get("role") == "assistant" and msg.get("content")),
    None
)
```

### Check for Errors

```python
for msg in trace:
    if msg.get("role") == "tool" and "Error" in msg.get("content", ""):
        print(f"Tool error: {msg['content']}")
```

### Inspect Tool Arguments

```python
for msg in trace:
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            print(f"Tool: {tc['function']['name']}")
            print(f"Args: {tc['function']['arguments']}")
```

## When to Use Traces

| Use Case | Benefit |
|----------|---------|
| **Debugging tool call issues** | See exactly what happened during generation |
| **Understanding model behavior** | Learn how the model uses available tools |
| **Quality analysis** | Verify the model is using tools correctly |
| **Pipeline validation** | Ensure deterministic tool behavior in production |
| **Prompt engineering** | Iterate on prompts based on actual tool usage |

## See Also

- **[Debugging](debugging.md)**: Troubleshoot tool call issues using traces
- **[Safety and Limits](safety-and-limits.md)**: Understand turn limits and timeout behavior
- **[Run Config](../../code_reference/run_config.md)**: Runtime options including `debug_override_save_all_column_traces`
