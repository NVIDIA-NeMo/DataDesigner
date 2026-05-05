# Engine MCP

The `data_designer.engine.mcp` modules implement execution-time MCP behavior. MCP config objects live in [`data_designer.config.mcp`](../config/mcp.md); the runtime maps those configs into registries, facades, sessions, schema discovery, and tool calls.

!!! note "Engine"
    This page is for Data Designer developers working on MCP execution. For the user-facing MCP provider and tool config objects, see [MCP configuration](../config/mcp.md).

## Parallel Structure

| Model layer | MCP layer | Purpose |
|-------------|-----------|---------|
| `ModelProviderRegistry` | `MCPProviderRegistry` | Holds provider configurations. |
| `ModelRegistry` | `MCPRegistry` | Manages configs by alias and lazily creates facades. |
| `ModelFacade` | `MCPFacade` | Provides a lightweight runtime facade scoped to one config. |
| `ModelConfig.alias` | `ToolConfig.tool_alias` | Alias referenced by column configs. |

## Registry

### `MCPToolDefinition` {#data_designer.engine.mcp.registry.MCPToolDefinition}

::: data_designer.engine.mcp.registry.MCPToolDefinition
    options:
      show_root_toc_entry: false

### `MCPToolResult` {#data_designer.engine.mcp.registry.MCPToolResult}

::: data_designer.engine.mcp.registry.MCPToolResult
    options:
      show_root_toc_entry: false

### `MCPRegistry` {#data_designer.engine.mcp.registry.MCPRegistry}

::: data_designer.engine.mcp.registry.MCPRegistry
    options:
      show_root_toc_entry: false

### `create_mcp_registry` {#data_designer.engine.mcp.factory.create_mcp_registry}

::: data_designer.engine.mcp.factory.create_mcp_registry
    options:
      show_root_toc_entry: false

## Facade

`ModelFacade.generate()` accepts a `tool_alias` parameter. When it is provided, `ModelFacade` looks up the matching `MCPFacade` from `MCPRegistry`, fetches tool schemas, passes them to the model, processes tool calls after each completion, tracks tool-call turns, and returns messages that include tool results for trace capture.

### `MCPFacade` {#data_designer.engine.mcp.facade.MCPFacade}

::: data_designer.engine.mcp.facade.MCPFacade
    options:
      show_root_toc_entry: false

## I/O Service

The I/O service owns a background event loop, pools MCP sessions by provider config, coalesces concurrent tool schema lookups, and executes parallel tool calls.

### `MCPIOService` {#data_designer.engine.mcp.io.MCPIOService}

::: data_designer.engine.mcp.io.MCPIOService
    options:
      show_root_toc_entry: false

### Runtime Helpers

::: data_designer.engine.mcp.io.list_tools
    options:
      show_root_toc_entry: false

::: data_designer.engine.mcp.io.list_tool_names
    options:
      show_root_toc_entry: false

::: data_designer.engine.mcp.io.call_tools
    options:
      show_root_toc_entry: false

::: data_designer.engine.mcp.io.clear_provider_caches
    options:
      show_root_toc_entry: false

::: data_designer.engine.mcp.io.clear_tools_cache
    options:
      show_root_toc_entry: false

::: data_designer.engine.mcp.io.get_cache_info
    options:
      show_root_toc_entry: false

::: data_designer.engine.mcp.io.clear_session_pool
    options:
      show_root_toc_entry: false

::: data_designer.engine.mcp.io.get_session_pool_info
    options:
      show_root_toc_entry: false
