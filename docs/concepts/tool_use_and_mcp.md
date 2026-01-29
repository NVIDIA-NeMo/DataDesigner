# Tool Use & MCP

Tool use lets LLM columns call external tools during generation (e.g., lookups, calculations, retrieval, domain services). Data Designer supports tool use via the **Model Context Protocol (MCP)**, which standardizes how tools are discovered and invoked.

!!! note "Two provider types"
    MCP providers can be configured in two ways:

    - **LocalStdioMCPProvider**: run a server as a subprocess (configured via `command` + `args`)
    - **MCPProvider**: connect to a pre-existing server over HTTP Server-Sent Events (configured via `endpoint`)

## Architecture Overview

The MCP layer mirrors the Model layer's registry/facade pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA DESIGNER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐         │
│  │       MODEL LAYER           │    │        MCP LAYER            │         │
│  ├─────────────────────────────┤    ├─────────────────────────────┤         │
│  │                             │    │                             │         │
│  │  ModelProviderRegistry      │    │  MCPProviderRegistry        │         │
│  │    └─ ModelProvider configs │    │    └─ MCPProvider configs   │         │
│  │                             │    │    └─ LocalStdioMCPProvider │         │
│  │          │                  │    │          │                  │         │
│  │          ▼                  │    │          ▼                  │         │
│  │  ModelRegistry              │    │  MCPRegistry                │         │
│  │    └─ ModelConfigs by alias │    │    └─ ToolConfigs by alias  │         │
│  │    └─ Lazy ModelFacade      │    │    └─ Lazy MCPFacade        │         │
│  │       creation              │    │       creation              │         │
│  │          │                  │    │          │                  │         │
│  │          ▼                  │    │          ▼                  │         │
│  │  ModelFacade                │    │  mcp/io.py                  │         │
│  │    └─ Per ModelConfig       │    │    └─ Session pool          │         │
│  │    └─ generate()            │    │    └─ Request coalescing    │         │
│  │    └─ completion()          │    │    └─ Background loop       │         │
│  │                             │    │          │                  │         │
│  │                             │    │          ▼                  │         │
│  │                             │    │  MCPFacade                  │         │
│  │                             │    │    └─ Per ToolConfig        │         │
│  │                             │    │    └─ process_completion()  │         │
│  │                             │    │    └─ refuse_completion()   │         │
│  │                             │    │    └─ get_tool_schemas()    │         │
│  │                             │    │                             │         │
│  └─────────────────────────────┘    └─────────────────────────────┘         │
│                                                                              │
│                    ModelFacade.generate(tool_alias=...)                      │
│                              │                                               │
│                              ▼                                               │
│                    MCPRegistry.get_mcp(tool_alias)                           │
│                              │                                               │
│                              ▼                                               │
│                    MCPFacade.process_completion_response()                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Parallel structure:**

| Model Layer | MCP Layer | Purpose |
|-------------|-----------|---------|
| `ModelProviderRegistry` | `MCPProviderRegistry` | Holds provider configurations |
| `ModelRegistry` | `MCPRegistry` | Manages configs by alias, lazy facade creation |
| `ModelFacade` | `MCPFacade` | Lightweight facade scoped to specific config |
| `ModelConfig.alias` | `ToolConfig.tool_alias` | Alias for referencing in column configs |

**Data flow:**

1. Column config specifies `tool_alias`
2. Column generator calls `model.generate(tool_alias=...)`
3. ModelFacade looks up MCPFacade from MCPRegistry via `tool_alias`
4. MCPFacade provides tool schemas and handles tool execution
5. Tool responses are returned to ModelFacade for the conversation loop

## Overview

At a high level:

- You configure one or more MCP providers on the `DataDesigner` instance.
- You define tool configurations with `ToolConfig` and add them to the config builder.
- You enable tools per LLM column via `tool_alias`, which references a `ToolConfig`.
- During generation, the model can request tool calls; Data Designer executes them and feeds tool outputs back to the model until it produces a final answer.

## Configuring MCP providers

### Local stdio (subprocess) provider

Use `LocalStdioMCPProvider` to launch an MCP server as a subprocess:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

mcp_provider = dd.LocalStdioMCPProvider(
    name="demo-mcp",
    command="python",
    args=["-m", "my_mcp_server_module"],
    env={"MY_SERVICE_TOKEN": "..."},  # Optional
)

data_designer = DataDesigner(mcp_providers=[mcp_provider])
```

### Remote SSE provider

Use `MCPProvider` to connect to a pre-existing MCP server via SSE:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

mcp_provider = dd.MCPProvider(
    name="remote-mcp",
    endpoint="http://localhost:8080/sse",
    api_key="your-api-key",  # Optional
)

data_designer = DataDesigner(mcp_providers=[mcp_provider])
```

### Provider types and YAML configuration

Both provider types use a `provider_type` discriminator field that distinguishes them in YAML configurations:

| Provider Class | `provider_type` | Connection Method |
|---------------|-----------------|-------------------|
| `LocalStdioMCPProvider` | `"stdio"` | Subprocess via stdin/stdout |
| `MCPProvider` | `"sse"` | HTTP Server-Sent Events |

When writing YAML configs manually (e.g., in `~/.data-designer/mcp_providers.yaml`), include the discriminator:

```yaml
providers:
  - name: doc-search
    provider_type: sse
    endpoint: http://localhost:8080/sse
    api_key: ${MCP_API_KEY}
```

### CLI configuration

You can configure MCP providers interactively via the CLI:

```bash
data-designer config mcp
```

This launches an interactive wizard that lets you:

1. Choose provider type (Remote SSE or Local stdio subprocess)
2. Configure provider-specific settings
3. Add, update, or delete MCP provider configurations

The configurations are stored in `~/.data-designer/mcp_providers.yaml`.

## Enabling tools on an LLM column

Tool permissions are configured using `ToolConfig`, which is added to the config builder at initialization.

### Step 1: Define tool configuration

```python
import data_designer.config as dd

tool_config = dd.ToolConfig(
    tool_alias="my-tools",              # Reference name for columns
    providers=["demo-mcp"],             # List of MCP provider names (can use multiple)
    allow_tools=["get_fact", "add_numbers"],  # None = allow all tools
    max_tool_call_turns=5,
    timeout_sec=45.0,
)
```

### Step 2: Add to config builder

```python
import data_designer.config as dd

# Pass tool_configs at initialization
builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

# Or add later
builder.add_tool_config(tool_config)
```

Alternatively, configure tool configs interactively via the CLI:

```bash
data-designer config tools
```

This launches an interactive wizard where you can:

- Select one or more MCP providers (checkbox selection)
- Optionally restrict allowed tools
- Set max tool calls and timeout limits

Tool configurations are stored in `~/.data-designer/tool_configs.yaml`.

### Using multiple providers

A single `ToolConfig` can reference multiple MCP providers, allowing tools to be drawn from different sources:

```python
tool_config = dd.ToolConfig(
    tool_alias="multi-search",
    providers=["doc-search-mcp", "web-search-mcp"],  # Multiple providers!
    allow_tools=["search_docs", "search_web", "list_docs"],
    max_tool_call_turns=20,
)
```

When the model requests a tool call, Data Designer automatically finds which provider hosts that tool and routes the call appropriately. The `allow_tools` list can restrict tools from any of the configured providers.

### Step 3: Reference in LLM columns

```python
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Use tools as needed to answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",  # References the ToolConfig by alias
    )
)
```

!!! tip "Make tool usage explicit in prompts"
    If you want deterministic tool behavior (for testing or pipelines), explicitly instruct the model which tools to call and in what order.

## Full message traces (`*__trace`)

When tool use is enabled, many users want to inspect the full interaction history: prompts, tool calls, tool results, and the final answer. Data Designer supports this via **optional trace capture**.

### Enabling trace capture

There are two ways to enable trace capture:

**Per-column (recommended)**: Enable `with_trace=True` on specific LLM columns:

```python
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Use tools as needed to answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",
        with_trace=True,  # Enable trace for this column
    )
)
```

**Global debug override**: Enable traces for ALL LLM columns (useful for debugging):

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner(mcp_providers=[...])
data_designer.set_run_config(dd.RunConfig(debug_override_save_all_column_traces=True))
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

### Trace data structure

Each trace is a `list[dict]` where each dict represents a message in the conversation. Here's a sample trace from an LLM column with tool use:

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

**Message fields by role:**

| Role | Fields | Description |
|------|--------|-------------|
| `system` | `role`, `content` | System prompt setting model behavior |
| `user` | `role`, `content` | User prompt (rendered from template) |
| `assistant` | `role`, `content`, `tool_calls`, `reasoning_content` | Model response; `content` may be `None` if only requesting tools |
| `tool` | `role`, `content`, `tool_call_id` | Tool execution result; `tool_call_id` links to the request |

**The `tool_calls` structure:**

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

### Working with traces

Access traces from `PreviewResults` or generated datasets:

```python
# After preview
results = data_designer.preview(config_builder, num_records=5)
df = results.dataset.to_pandas()

# Access trace for a specific row
trace = df["answer__trace"].iloc[0]

# Count tool calls in the conversation
tool_messages = [msg for msg in trace if msg.get("role") == "tool"]
print(f"This generation made {len(tool_messages)} tool calls")

# Extract all tool names used
tool_calls = []
for msg in trace:
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            tool_calls.append(tc["function"]["name"])
print(f"Tools used: {tool_calls}")

# Find the final answer
final_answer = next(
    (msg["content"] for msg in reversed(trace)
     if msg.get("role") == "assistant" and msg.get("content")),
    None
)
```

## Safety and limits

- **Tool allowlists**: Restrict which tools can be used via `ToolConfig(allow_tools=[...])`.
- **Tool call budgets**: Use `ToolConfig(max_tool_call_turns=...)` to prevent runaway loops.
- **Tool timeouts**: Use `ToolConfig(timeout_sec=...)` to cap MCP call latency.
- **Provider support**: Tool calling behavior depends on model/provider capability and prompt design.

## Error handling and debugging

### Tool call failures

When a tool call fails (e.g., MCP server error, invalid arguments), the error is captured and returned to the model as a tool response with the error message. The model can then decide how to proceed—retry with different arguments, try a different tool, or provide a response based on available information.

```python
# Example error in trace
{
    "role": "tool",
    "content": "Error: Tool 'search_docs' failed: Connection timeout after 30s",
    "tool_call_id": "call_abc123"
}
```

### Timeout behavior

The `timeout_sec` parameter in `ToolConfig` controls how long Data Designer waits for each individual tool call:

```python
tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
    timeout_sec=30.0,  # 30 seconds per tool call (default: 60.0)
)
```

When a timeout occurs:

1. The tool call is terminated
2. An error message is returned to the model
3. The model can attempt recovery (retry, skip, or answer without the result)

### Max tool call turns

The `max_tool_call_turns` parameter limits how many tool-calling iterations (turns) are permitted:

```python
tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
    max_tool_call_turns=5,  # Maximum 5 tool-calling turns (default: 5)
)
```

!!! note "Turn-based limiting vs individual call counting"
    A **turn** is one iteration where the LLM requests tool calls. With parallel tool calling, a single turn may execute multiple tools simultaneously. For example, if the model requests 3 tools in parallel, that counts as 1 turn, not 3.

    This approach gives models flexibility to use parallel calling efficiently while still bounding total iterations.

### Graceful budget exhaustion

When the turn limit is reached, Data Designer doesn't abruptly stop generation. Instead, it uses `refuse_completion_response()` to gracefully refuse additional tool calls:

1. The model's tool call request is recorded in the conversation
2. Tool "results" are returned with a refusal message explaining the limit was reached
3. The model receives this feedback and can produce a final response

This ensures the model can still provide a useful answer based on the tools it already called, rather than failing silently.

### Debugging with traces

Enable traces to diagnose tool call issues:

```python
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Search and answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",
        with_trace=True,  # Enable trace capture
    )
)
```

**Common debugging patterns:**

```python
# Check if any tool calls failed
trace = df["answer__trace"].iloc[0]
for msg in trace:
    if msg.get("role") == "tool" and "Error" in msg.get("content", ""):
        print(f"Tool error: {msg['content']}")

# Count tool calls to check for loops
tool_call_count = sum(1 for msg in trace if msg.get("role") == "tool")
print(f"Total tool calls: {tool_call_count}")

# Inspect tool arguments for debugging
for msg in trace:
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            print(f"Tool: {tc['function']['name']}")
            print(f"Args: {tc['function']['arguments']}")
```

**Tips for debugging:**

- If tools aren't being called, check that `tool_alias` references a valid `ToolConfig` and the prompt encourages tool use
- If tool calls fail, verify the MCP provider is running and accessible
- If the model enters a loop, reduce `max_tool_call_turns` or improve the prompt to guide better tool selection
- Use `debug_override_save_all_column_traces=True` in `RunConfig` to capture traces for all columns during development

## Internal Architecture

For developers working with Data Designer internals, this section explains the MCP components in detail.

### MCPProviderRegistry

Holds MCP provider configurations. Can be empty (MCP is optional). Created first during resource initialization.

### MCPRegistry

The central registry for tool configurations:

- Holds `ToolConfig` instances by `tool_alias`
- Lazily creates `MCPFacade` instances via `get_mcp(tool_alias)`
- Manages shared connection pool and tool cache across all facades
- Validates that tool configs reference valid providers

### MCPFacade

A lightweight facade scoped to a specific `ToolConfig`. Key methods:

| Method | Description |
|--------|-------------|
| `tool_call_count(response)` | Count tool calls in a completion response |
| `has_tool_calls(response)` | Check if response contains tool calls |
| `get_tool_schemas()` | Get OpenAI-format tool schemas for this config |
| `process_completion_response(response)` | Execute tool calls and return messages |
| `refuse_completion_response(response)` | Refuse tool calls gracefully (budget exhaustion) |

Properties: `tool_alias`, `providers`, `max_tool_call_turns`, `allow_tools`, `timeout_sec`

### I/O Layer (mcp/io.py)

The `io.py` module provides low-level MCP communication with performance optimizations:

**Single event loop architecture:**
All MCP operations funnel through a dedicated background daemon thread running an asyncio event loop. This allows:

- Efficient concurrent I/O without per-thread event loop overhead
- Natural session sharing across all worker threads
- Clean async implementation for parallel tool calls

**Session pooling:**
MCP sessions are created lazily and kept alive for the program's duration:

- One session per provider (keyed by serialized config)
- No per-call connection/handshake overhead
- Graceful cleanup on program exit via `atexit` handler

**Request coalescing:**
The `list_tools` operation uses request coalescing to prevent thundering herd:

- When multiple workers request tools from the same provider simultaneously
- Only one request is made; others wait for the cached result
- Uses asyncio.Lock per provider key

**Parallel tool execution:**
The `call_tools_parallel()` function executes multiple tool calls concurrently via `asyncio.gather()`. This is used by MCPFacade when the model returns parallel tool calls in a single response.

### Integration with ModelFacade.generate()

The `ModelFacade.generate()` method accepts an optional `tool_alias` parameter:

```python
output, messages = model_facade.generate(
    prompt="Search and answer...",
    parser=my_parser,
    tool_alias="my-tools",  # Enables tool calling for this generation
)
```

When `tool_alias` is provided:

1. `ModelFacade` looks up the `MCPFacade` from `MCPRegistry`
2. Tool schemas are fetched and passed to the LLM
3. After each completion, `MCPFacade` processes tool calls
4. Turn counting tracks iterations; refusal kicks in when budget exhausted
5. Messages (including tool results) are returned for trace capture

## See Also

- [Columns](columns.md): Overview of LLM columns and side effect columns
- [Run Config](../code_reference/run_config.md): Runtime options including `debug_override_save_all_column_traces`
