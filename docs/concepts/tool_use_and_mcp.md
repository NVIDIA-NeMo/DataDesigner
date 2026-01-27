# Tool Use & MCP

Tool use lets LLM columns call external tools during generation (e.g., lookups, calculations, retrieval, domain services). Data Designer supports tool use via the **Model Context Protocol (MCP)**, which standardizes how tools are discovered and invoked.

!!! note "Two deployment modes"
    MCP servers can be configured in two ways:

    - **Local stdio**: run a server as a subprocess (configured via `command` + `args`)
    - **Remote SSE**: connect to a server over HTTP Server-Sent Events (configured via `url`)

## Overview

At a high level:

- You configure one or more MCP servers on the `DataDesigner` instance.
- You enable tools per LLM column via `MCPToolConfig`.
- During generation, the model can request tool calls; Data Designer executes them and feeds tool outputs back to the model until it produces a final answer.

## Configuring MCP servers

Use `MCPServerConfig` to define how to connect to each server.

### Local stdio (subprocess) server

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

mcp_server = dd.MCPServerConfig(
    name="demo-mcp",
    command="python",
    args=["-m", "my_mcp_server_module"],
    env={"MY_SERVICE_TOKEN": "..."},  # Optional
)

data_designer = DataDesigner(mcp_servers=[mcp_server])
```

### Remote SSE server

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

mcp_server = dd.MCPServerConfig(
    name="remote-mcp",
    url="http://localhost:8080/sse",
)

data_designer = DataDesigner(mcp_servers=[mcp_server])
```

## Enabling tools on an LLM column

Tool permissions are configured per column using `MCPToolConfig`.

```python
import data_designer.config as dd

tool_config = dd.MCPToolConfig(
    server_name="demo-mcp",
    tool_names=["get_fact", "add_numbers"],  # None = allow all tools on that server
    max_tool_calls=5,
    timeout_sec=45.0,
)
```

Then attach it to an LLM column:

```python
import data_designer.config as dd

builder = dd.DataDesignerConfigBuilder()

builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Use tools as needed to answer: {{ question }}",
        model_alias="nvidia-text",
        tool_config=tool_config,
    )
)
```

!!! tip "Make tool usage explicit in prompts"
    If you want deterministic tool behavior (for testing or pipelines), explicitly instruct the model which tools to call and in what order.

## Full message traces (`*__trace`)

When tool use is enabled, many users want to inspect the full interaction history: prompts, tool calls, tool results, and the final answer. Data Designer supports this via **optional trace capture**.

### Enabling trace capture

Set `RunConfig(include_full_traces=True)` on your `DataDesigner` instance:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner(mcp_servers=[...])
data_designer.set_run_config(dd.RunConfig(include_full_traces=True))
```

### What is stored

When enabled, LLM columns produce an additional side-effect column named:

- `{column_name}__trace`

This value is a `list[dict]` containing the ordered message history for the **final generation attempt**, including:

- system + user messages (the rendered prompts)
- assistant messages that include `tool_calls`
- tool messages (`role="tool"`, `tool_call_id`, `content`)
- the final assistant message

If the provider exposes it, assistant messages may also include a `reasoning_content` field.

!!! note "Replacement for `__reasoning_trace`"
    The previous `{column_name}__reasoning_trace` side-effect column has been replaced by `{column_name}__trace`, which captures the entire conversation rather than only reasoning.

## Safety and limits

- **Tool allowlists**: Restrict which tools can be used via `MCPToolConfig(tool_names=[...])`.
- **Tool call budgets**: Use `MCPToolConfig(max_tool_calls=...)` to prevent runaway loops.
- **Tool timeouts**: Use `MCPToolConfig(timeout_sec=...)` to cap MCP call latency.
- **Provider support**: Tool calling behavior depends on model/provider capability and prompt design.

## See Also

- [Columns](columns.md): Overview of LLM columns and side effect columns
- [Run Config](../code_reference/run_config.md): Runtime options including `include_full_traces`
