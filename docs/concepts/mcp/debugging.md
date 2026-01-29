# Debugging Tool Use

This guide helps you diagnose and fix common issues with tool use in Data Designer. It covers common problems, debugging techniques, and solutions.

## Common Issues

### Tools Aren't Being Called

**Symptoms:** The model produces output without making any tool calls, even when tools should be useful.

**Possible causes and solutions:**

| Cause | Solution |
|-------|----------|
| `tool_alias` doesn't reference a valid `ToolConfig` | Verify the alias matches exactly |
| Prompt doesn't encourage tool use | Add explicit instructions to use tools |
| MCP provider isn't accessible | Check provider configuration and connectivity |
| Tool schemas aren't being loaded | Verify provider is returning tools |

**Debugging steps:**

```python
# 1. Verify ToolConfig exists
builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])
# tool_config.tool_alias must match tool_alias in column

# 2. Check prompt encourages tool use
prompt = """You MUST use the search_docs tool before answering.

Question: {{ question }}

Instructions:
1. Call search_docs to find information
2. Base your answer only on tool results"""

# 3. Enable traces to see what happens
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt=prompt,
        model_alias="nvidia-text",
        tool_alias="my-tools",
        with_trace=True,
    )
)
```

### Tool Calls Are Failing

**Symptoms:** Traces show tool errors, or the model reports it couldn't use tools.

**Common error patterns:**

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `"Tool not found"` | Tool not in allowlist or provider | Check `allow_tools`, verify provider has the tool |
| `"Connection timeout"` | Provider unreachable or slow | Check provider is running, increase `timeout_sec` |
| `"Invalid arguments"` | Model passing wrong parameters | Improve prompt clarity about tool usage |
| `"Server error"` | MCP server issue | Check MCP server logs |

**Debugging with traces:**

```python
# Enable traces
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="...",
        model_alias="nvidia-text",
        tool_alias="my-tools",
        with_trace=True,
    )
)

# After generation, check for errors
results = data_designer.preview(builder, num_records=1)
df = results.dataset.to_pandas()
trace = df["answer__trace"].iloc[0]

# Find tool errors
for msg in trace:
    if msg.get("role") == "tool":
        content = msg.get("content", "")
        if "Error" in content or "error" in content:
            print(f"Tool error: {content}")
            print(f"Tool call ID: {msg.get('tool_call_id')}")
```

### Model Entering Loops

**Symptoms:** Generation takes a long time, trace shows many repeated tool calls, or max turns is reached.

**Possible causes and solutions:**

| Cause | Solution |
|-------|----------|
| Model keeps retrying failed tools | Improve error handling in prompt |
| Model doesn't know when to stop | Add explicit stopping criteria in prompt |
| Tools returning unhelpful results | Check tool implementations |
| Max turns too high | Reduce `max_tool_call_turns` |

**Debugging:**

```python
# Count tool calls in trace
trace = df["answer__trace"].iloc[0]
tool_count = sum(1 for msg in trace if msg.get("role") == "tool")
print(f"Total tool calls: {tool_count}")

# Check for repeated tool calls
tool_calls = []
for msg in trace:
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            tool_calls.append(tc["function"]["name"])

from collections import Counter
print(f"Tool call frequency: {Counter(tool_calls)}")
```

**Solution - reduce max turns and improve prompt:**

```python
tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
    max_tool_call_turns=3,  # Reduce from default 5
)

prompt = """Answer the question using available tools.

Important:
- Make at most 2 tool calls
- If the first tool call doesn't help, provide your best answer
- Do not retry the same tool with the same arguments

Question: {{ question }}"""
```

## Debugging Techniques

### Enable Global Traces

During development, enable traces for all columns:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner(mcp_providers=[...])
data_designer.set_run_config(
    dd.RunConfig(debug_override_save_all_column_traces=True)
)
```

### Inspect Full Traces

Print the complete trace for detailed analysis:

```python
import json

trace = df["answer__trace"].iloc[0]
print(json.dumps(trace, indent=2))
```

### Check Tool Arguments

Verify the model is passing correct arguments:

```python
for msg in trace:
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            print(f"\nTool: {tc['function']['name']}")
            print(f"Arguments: {tc['function']['arguments']}")
```

### Verify Provider Connectivity

Test your MCP provider separately:

```python
# For LocalStdioMCPProvider, verify the command works
import subprocess
result = subprocess.run(
    ["python", "-m", "my_mcp_server", "--help"],
    capture_output=True,
    text=True
)
print(result.stdout)
print(result.stderr)
```

### Check Allowlist Configuration

Verify tools are available:

```python
# If using allow_tools, ensure the tool is in the list
tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
    allow_tools=["search_docs", "get_fact"],  # Tool must be here
)
```

## Troubleshooting Checklist

When tool use isn't working as expected, work through this checklist:

- [ ] **Provider configured correctly?**
    - MCP provider in `DataDesigner(mcp_providers=[...])`
    - Provider name matches in `ToolConfig.providers`

- [ ] **ToolConfig set up correctly?**
    - `tool_alias` matches column's `tool_alias`
    - `providers` list includes the correct provider(s)
    - `allow_tools` includes the tools you need (or is `None`)

- [ ] **Column configured correctly?**
    - `tool_alias` references a valid `ToolConfig`
    - Prompt encourages tool use

- [ ] **Provider accessible?**
    - For SSE: endpoint reachable, API key valid
    - For stdio: command executable, module exists

- [ ] **Traces enabled for debugging?**
    - `with_trace=True` on column, or
    - `debug_override_save_all_column_traces=True` in RunConfig

## See Also

- **[Traces](traces.md)**: Detailed guide on capturing and analyzing traces
- **[Safety and Limits](safety-and-limits.md)**: Configure turn limits and timeouts
- **[Tool Configurations](tool-configs.md)**: Complete ToolConfig reference
- **[Enabling Tools on Columns](enabling-tools.md)**: Set up tool use on columns
