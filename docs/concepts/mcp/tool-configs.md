# Tool Configurations

Tool configurations define how LLM columns access external tools during generation. Each `ToolConfig` specifies which MCP providers to use, which tools are allowed, and operational limits.

## Overview

A `ToolConfig` connects LLM columns to MCP providers. When you create column configurations (like `LLMTextColumnConfig` or `LLMCodeColumnConfig`), you reference a tool configuration by its alias. Data Designer uses the tool configuration to determine which tools are available and how to manage tool calls.

## ToolConfig Structure

The `ToolConfig` class has the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_alias` | `str` | Yes | Unique identifier for this tool configuration (referenced by columns) |
| `providers` | `list[str]` | Yes | List of MCP provider names to use (can reference multiple providers) |
| `allow_tools` | `list[str]` | No | Restrict to specific tools (`None` = allow all tools from providers) |
| `max_tool_call_turns` | `int` | No | Maximum tool-calling iterations (default: 5) |
| `timeout_sec` | `float` | No | Per-call timeout in seconds (default: 60.0) |

## Examples

### Basic Tool Configuration

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
)
```

### Restricting Allowed Tools

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="search-only",
    providers=["demo-mcp"],
    allow_tools=["search_docs", "list_docs"],  # Only these tools allowed
)
```

### Using Multiple Providers

A single `ToolConfig` can reference multiple MCP providers, allowing tools to be drawn from different sources:

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="multi-search",
    providers=["doc-search-mcp", "web-search-mcp"],
    allow_tools=["search_docs", "search_web", "list_docs"],
    max_tool_call_turns=10,
)
```

When the model requests a tool call, Data Designer automatically finds which provider hosts that tool and routes the call appropriately.

### Setting Operational Limits

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="limited-tools",
    providers=["demo-mcp"],
    max_tool_call_turns=3,   # Maximum 3 tool-calling iterations
    timeout_sec=30.0,        # 30 seconds per tool call
)
```

## Adding to Config Builder

Tool configurations can be added to the config builder in two ways:

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
)

# Method 1: Pass at initialization
builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

# Method 2: Add later
builder = dd.DataDesignerConfigBuilder()
builder.add_tool_config(tool_config)
```

## Understanding Turn-Based Limiting

The `max_tool_call_turns` parameter limits how many tool-calling iterations (turns) are permitted, not the total number of individual tool calls.

!!! note "Turn-based vs call-based counting"
    A **turn** is one iteration where the LLM requests tool calls. With parallel tool calling, a single turn may execute multiple tools simultaneously.

    For example, if the model requests 3 tools in parallel, that counts as 1 turn, not 3. This gives models flexibility to use parallel calling efficiently while still bounding total iterations.

When the turn limit is reached, Data Designer gracefully refuses additional tool calls rather than failing abruptly. The model receives feedback explaining the limit was reached and can produce a final response based on the tools it already called.

## See Also

- **[MCP Providers](mcp-providers.md)**: Configure connections to MCP servers
- **[Enabling Tools on Columns](enabling-tools.md)**: Reference tool configs from LLM columns
- **[Safety and Limits](safety-and-limits.md)**: Detailed guide on tool safety controls
- **[Configure MCP with the CLI](configure-mcp-cli.md)**: Use the CLI to manage tool configurations
