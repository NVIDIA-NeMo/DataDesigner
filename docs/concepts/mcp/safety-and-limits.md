# Safety and Limits

This guide covers the safety controls available for tool use, including allowlists, turn budgets, and timeouts. These controls help prevent runaway loops and ensure predictable generation behavior.

## Overview

When LLM columns use tools, the model can make multiple tool calls in a loop until it produces a final answer. Without limits, this could lead to:

- Excessive API calls and costs
- Long generation times
- Infinite loops if the model keeps requesting tools

Data Designer provides three types of controls:

| Control | Purpose |
|---------|---------|
| **Tool allowlists** | Restrict which tools can be called |
| **Turn budgets** | Limit iterations of tool-calling |
| **Timeouts** | Cap individual tool call latency |

## Tool Allowlists

Restrict which tools are available using `allow_tools`:

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="restricted-tools",
    providers=["demo-mcp"],
    allow_tools=["search_docs", "list_docs"],  # Only these tools
)
```

### Behavior

| Setting | Behavior |
|---------|----------|
| `allow_tools=None` (default) | All tools from the providers are available |
| `allow_tools=["tool1", "tool2"]` | Only the specified tools are available |

Tools not in the allowlist won't be included in the schemas sent to the model, so the model won't know they exist.

!!! tip "Use allowlists for untrusted tools"
    If your MCP providers expose tools that could be dangerous or expensive, use allowlists to restrict access to only the tools you need.

## Turn Budgets

Limit the number of tool-calling iterations using `max_tool_call_turns`:

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="limited-tools",
    providers=["demo-mcp"],
    max_tool_call_turns=5,  # Maximum 5 iterations (default)
)
```

### Understanding Turns

A **turn** is one iteration where the LLM requests tool calls. With parallel tool calling, a single turn may execute multiple tools simultaneously.

| Scenario | Turn Count |
|----------|------------|
| Model requests 1 tool | 1 turn |
| Model requests 3 tools in parallel | 1 turn |
| Model requests 1 tool, then 2 more, then 1 more | 3 turns |

This approach gives models flexibility to use parallel calling efficiently while still bounding total iterations.

### Graceful Budget Exhaustion

When the turn limit is reached, Data Designer doesn't abruptly stop generation. Instead:

1. The model's tool call request is recorded in the conversation
2. Tool "results" are returned with a refusal message explaining the limit was reached
3. The model receives this feedback and can produce a final response

This ensures the model can still provide a useful answer based on the tools it already called, rather than failing silently.

## Timeouts

Limit how long each tool call can take using `timeout_sec`:

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="fast-tools",
    providers=["demo-mcp"],
    timeout_sec=30.0,  # 30 seconds per tool call
)
```

### Timeout Behavior

When a timeout occurs:

1. The tool call is terminated
2. An error message is returned to the model
3. The model can attempt recovery (retry, skip, or answer without the result)

```python
# Example error in trace when timeout occurs
{
    "role": "tool",
    "content": "Error: Tool 'search_docs' failed: Connection timeout after 30s",
    "tool_call_id": "call_abc123"
}
```

### Default Timeout

The default timeout is 60 seconds. Adjust based on your tools:

| Tool Type | Recommended Timeout |
|-----------|---------------------|
| Fast lookups | 5-10 seconds |
| Database queries | 15-30 seconds |
| External API calls | 30-60 seconds |
| Complex computations | 60+ seconds |

## Combining Controls

You can use all controls together for defense in depth:

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="secure-tools",
    providers=["demo-mcp"],
    allow_tools=["search_docs", "get_fact"],  # Restricted tools
    max_tool_call_turns=3,                     # Limited iterations
    timeout_sec=15.0,                          # Fast timeout
)
```

## Best Practices

### Start Restrictive

Begin with conservative limits and relax them as needed:

```python
# Start with restrictive settings
tool_config = dd.ToolConfig(
    tool_alias="cautious-tools",
    providers=["demo-mcp"],
    allow_tools=["safe_tool_1"],
    max_tool_call_turns=3,
    timeout_sec=10.0,
)
```

### Use Allowlists for Untrusted Sources

If tools come from external or untrusted MCP servers, always specify an allowlist:

```python
# External server - restrict access
tool_config = dd.ToolConfig(
    tool_alias="external-tools",
    providers=["third-party-mcp"],
    allow_tools=["public_api_1", "public_api_2"],  # Only known-safe tools
)
```

### Monitor with Traces

Use traces to understand actual tool usage and tune limits:

```python
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="...",
        model_alias="nvidia-text",
        tool_alias="my-tools",
        with_trace=True,  # Monitor tool usage
    )
)

# After generation, analyze traces
trace = df["answer__trace"].iloc[0]
tool_count = sum(1 for msg in trace if msg.get("role") == "tool")
print(f"Tool calls made: {tool_count}")
```

### Adjust Based on Task Complexity

Different tasks may need different limits:

| Task Type | Suggested Settings |
|-----------|-------------------|
| Simple lookups | `max_tool_call_turns=2`, `timeout_sec=10` |
| Multi-step research | `max_tool_call_turns=10`, `timeout_sec=30` |
| Complex workflows | `max_tool_call_turns=20`, `timeout_sec=60` |

## See Also

- **[Tool Configurations](tool-configs.md)**: Complete ToolConfig reference
- **[Debugging](debugging.md)**: Troubleshoot issues with limits
- **[Traces](traces.md)**: Monitor tool usage patterns
