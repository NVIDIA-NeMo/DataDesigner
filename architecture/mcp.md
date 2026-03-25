# MCP

The MCP (Model Context Protocol) subsystem enables tool-augmented LLM generation. It manages tool discovery, session pooling, and parallel tool execution for column generators that use external tools.

Source: `packages/data-designer-engine/src/data_designer/engine/mcp/`

## Overview

MCP integration allows column generators to augment LLM completions with tool calls. When a column config specifies a `tool_alias`, the model facade routes tool calls through the MCP subsystem, which handles session management, tool schema discovery, and parallel execution.

The subsystem has three layers:
- **`MCPIOService`** — low-level I/O: session pooling, tool listing, tool execution on a background async loop
- **`MCPFacade`** — scoped to a single tool config: schema formatting, completion response processing, tool call execution
- **`MCPRegistry`** — maps tool aliases to configs, lazy facade construction, health checks

## Key Components

### MCPIOService

Singleton module-level service that manages the async I/O layer:

- **Background async loop** — runs on a daemon thread; sync callers use `asyncio.run_coroutine_threadsafe` to bridge
- **Session pool** — `_sessions` keyed by provider cache key (JSON of provider config); `_get_or_create_session` with in-flight deduplication prevents redundant connections
- **Tool listing** — cached per session; coalescing for concurrent list requests via `_inflight_tools` prevents duplicate discovery calls
- **Tool execution** — parallel tool calls within a single completion response

Module-level functions (`list_tools`, `call_tools`, `clear_session_pool`) delegate to the singleton instance.

### MCPFacade

Scoped to one `ToolConfig`. Provides the interface that `ModelFacade` uses:

- **`get_tool_schemas()`** — returns tool schemas in OpenAI function-calling format
- **`process_completion_response`** — extracts tool calls from a completion, executes them in parallel via `MCPIOService`, returns `ChatMessage` list with results
- **`refuse_completion_response`** — handles tool-call turn limits (prevents infinite tool loops)

### MCPRegistry

Maps `tool_alias` → `ToolConfig`. Lazy `MCPFacade` construction mirrors `ModelRegistry`. Provides health checks for configured tools.

## Data Flow

1. Column config declares `tool_alias` referencing a configured MCP tool
2. Generator's `ModelFacade` includes tool schemas in the completion request
3. LLM returns a completion with tool calls
4. `ModelFacade` delegates to `MCPFacade.process_completion_response`
5. `MCPFacade` extracts tool calls, executes them in parallel via `MCPIOService`
6. Tool results are formatted as `ChatMessage`s and fed back to the LLM for another completion round
7. Process repeats until the LLM produces a final response or the turn limit is reached

## Design Decisions

- **Single background async loop** avoids creating event loops per request. All MCP I/O funnels through one loop on a daemon thread, with sync callers bridging via `run_coroutine_threadsafe`.
- **Session pooling with in-flight deduplication** prevents redundant connections when multiple generators discover tools from the same provider concurrently.
- **Tool schema coalescing** — concurrent `list_tools` calls for the same session share a single in-flight request, reducing startup latency when many columns use the same tool.
- **Turn limits on tool loops** prevent runaway tool-call chains. `refuse_completion_response` gracefully terminates when the limit is reached.

## Cross-References

- [System Architecture](overview.md) — where MCP fits in the stack
- [Models](models.md) — how `ModelFacade` integrates MCP tool loops
- [Engine Layer](engine.md) — `ResourceProvider` provides `MCPRegistry` to generators
- [Config Layer](config.md) — `ToolConfig` definition
