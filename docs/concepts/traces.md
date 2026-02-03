# Message Traces

Traces capture the conversation history during LLM generation, including system prompts, user prompts, model reasoning, tool calls, tool results, and the final response. This visibility is essential for understanding model behavior, debugging generation issues, and iterating on prompts.

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

## Trace Types

Data Designer supports three trace modes via the `TraceType` enum:

| TraceType | Description |
|-----------|-------------|
| `TraceType.NONE` | No trace captured (default) |
| `TraceType.LAST_MESSAGE` | Only the final assistant message is captured |
| `TraceType.ALL_MESSAGES` | Full conversation history (system/user/assistant/tool) |

## Enabling Traces

### Per-Column (Recommended)

Set `with_trace` on specific LLM columns:

```python
import data_designer.config as dd

# Capture full conversation history
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Answer: {{ question }}",
        model_alias="nvidia-text",
        with_trace=dd.TraceType.ALL_MESSAGES,  # Full trace
    )
)

# Capture only the final assistant response
builder.add_column(
    dd.LLMTextColumnConfig(
        name="summary",
        prompt="Summarize: {{ text }}",
        model_alias="nvidia-text",
        with_trace=dd.TraceType.LAST_MESSAGE,  # Just the final response
    )
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

## Extracting Reasoning Content

Some models (particularly those with extended thinking or chain-of-thought capabilities) expose their reasoning process separately via the `reasoning_content` field in assistant messages. While this is included in full traces, you may want to capture it separately without the overhead of storing the entire conversation history.

### Dedicated Reasoning Column

Set `extract_reasoning_content=True` on any LLM column to create a `{column_name}__reasoning_content` side-effect column:

```python
import data_designer.config as dd

builder.add_column(
    dd.LLMTextColumnConfig(
        name="solution",
        prompt="Solve this math problem step by step: {{ problem }}",
        model_alias="reasoning-model",
        extract_reasoning_content=True,  # Creates solution__reasoning_content
    )
)
```

The extracted reasoning content:

- Contains only the `reasoning_content` from the **final** assistant message in the trace
- Is stripped of leading/trailing whitespace
- Is `None` if the model didn't provide reasoning content or if it was whitespace-only

### When to Use Each Approach

| Need | Approach |
|------|----------|
| Full conversation history for debugging | `with_trace=True` |
| Just the model's reasoning/thinking | `extract_reasoning_content=True` |
| Both conversation history and separate reasoning | Use both options |
| Fine-tuning data with reasoning | `extract_reasoning_content=True` for clean extraction |

### Availability

The `extract_reasoning_content` option is available on all LLM column types:

- `LLMTextColumnConfig`
- `LLMCodeColumnConfig`
- `LLMStructuredColumnConfig`
- `LLMJudgeColumnConfig`

## See Also

- **[Safety and Limits](mcp/safety-and-limits.md)**: Understand turn limits and timeout behavior
