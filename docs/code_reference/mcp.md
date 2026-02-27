# MCP (Model Context Protocol)

The `mcp` module defines configuration and execution classes for tool use via MCP (Model Context Protocol).

## Configuration Classes

[MCPProvider](#data_designer.config.mcp.MCPProvider) configures remote MCP servers via SSE or Streamable HTTP transport. [LocalStdioMCPProvider](#data_designer.config.mcp.LocalStdioMCPProvider) configures local MCP servers as subprocesses via stdio transport. [ToolConfig](#data_designer.config.mcp.ToolConfig) defines which tools are available for LLM columns and how they are constrained.

For user-facing guides, see:

- **[MCP Providers](../concepts/mcp/mcp-providers.md)** - Configure local or remote MCP providers
- **[Tool Configs](../concepts/mcp/tool-configs.md)** - Define tool permissions and limits
- **[Enabling Tools](../concepts/mcp/enabling-tools.md)** - Use tools in LLM columns
- **[Traces](../concepts/traces.md)** - Capture full conversation history

## Internal Architecture

### Parallel Structure

| Model Layer | MCP Layer | Purpose |
|-------------|-----------|---------|
| `ModelProviderRegistry` | `MCPProviderRegistry` | Holds provider configurations |
| `ModelRegistry` | `MCPRegistry` | Manages configs by alias, lazy facade creation |
| `ModelFacade` | `MCPFacade` | Lightweight facade scoped to specific config |
| `ModelConfig.alias` | `ToolConfig.tool_alias` | Alias for referencing in column configs |

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

## Config Module

::: data_designer.config.mcp
