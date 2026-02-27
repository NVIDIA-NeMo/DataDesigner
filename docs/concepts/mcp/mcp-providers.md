# MCP Providers

MCP providers are external services that host and serve tools via the Model Context Protocol (MCP). Data Designer uses provider configurations to establish connections to these services.

## Overview

An MCP provider defines how Data Designer connects to a tool server. Data Designer supports two provider types:

| Provider Class | Connection Method | Use Case |
|---------------|-------------------|----------|
| `MCPProvider` | SSE or Streamable HTTP | Connect to a pre-existing MCP server |
| `LocalStdioMCPProvider` | Subprocess via stdin/stdout | Launch an MCP server as a subprocess |

When you create a `ToolConfig`, you reference providers by name, and Data Designer uses those provider settings to communicate with the appropriate MCP servers.

## MCPProvider (Remote)

Use `MCPProvider` to connect to a pre-existing MCP server. Both SSE (Server-Sent Events) and Streamable HTTP transports are supported:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

# SSE transport (default)
sse_provider = dd.MCPProvider(
    name="remote-mcp",
    endpoint="http://localhost:8080/sse",
    api_key="MCP_API_KEY",  # Environment variable name
)

# Streamable HTTP transport
http_provider = dd.MCPProvider(
    name="remote-tools",
    endpoint="https://mcp.example.com/mcp",
    api_key="MCP_API_KEY",
    provider_type="streamable_http",
)

data_designer = DataDesigner(mcp_providers=[sse_provider, http_provider])
```

### MCPProvider Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the provider |
| `endpoint` | `str` | Yes | Endpoint URL for the remote MCP server |
| `api_key` | `str` | No | API key or environment variable name |
| `provider_type` | `str` | No | Transport type: `"sse"` (default) or `"streamable_http"` |

## LocalStdioMCPProvider (Subprocess)

Use `LocalStdioMCPProvider` to launch an MCP server as a subprocess:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

mcp_provider = dd.LocalStdioMCPProvider(
    name="demo-mcp",
    command="python",
    args=["-m", "my_mcp_server_module"],
    env={"MY_SERVICE_TOKEN": "..."},
)

data_designer = DataDesigner(mcp_providers=[mcp_provider])
```

### LocalStdioMCPProvider Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the provider |
| `command` | `str` | Yes | Executable to run (e.g., `"python"`, `"node"`) |
| `args` | `list[str]` | No | Command-line arguments |
| `env` | `dict[str, str]` | No | Environment variables for the subprocess |
| `provider_type` | `str` | No | Always `"stdio"` (set automatically) |

## API Key Configuration

The `api_key` field can be specified in two ways:

1. **Environment variable name** (recommended): Set `api_key` to the name of an environment variable (e.g., `"MCP_API_KEY"`). Data Designer will resolve it at runtime.

2. **Plain-text value**: Set `api_key` to the actual API key string. This is less secure and not recommended for production.

```python
# Method 1: Environment variable (recommended)
provider = dd.MCPProvider(
    name="secure-mcp",
    endpoint="https://mcp.example.com/sse",
    api_key="MCP_API_KEY",  # Will be resolved from environment
)

# Method 2: Direct value (not recommended)
provider = dd.MCPProvider(
    name="secure-mcp",
    endpoint="https://mcp.example.com/sse",
    api_key="actual-api-key-value",
)
```

## YAML Configuration

Both provider types use a `provider_type` discriminator field in YAML configurations. When writing YAML configs manually (e.g., in `~/.data-designer/mcp_providers.yaml`), include the discriminator:

```yaml
providers:
  # Remote SSE provider
  - name: doc-search
    provider_type: sse
    endpoint: http://localhost:8080/sse
    api_key: ${MCP_API_KEY}

  # Remote Streamable HTTP provider
  - name: remote-tools
    provider_type: streamable_http
    endpoint: https://mcp.example.com/mcp
    api_key: ${MCP_API_KEY}

  # Local stdio provider
  - name: local-tools
    provider_type: stdio
    command: python
    args:
      - -m
      - my_mcp_server
    env:
      DEBUG: "true"
```

## Using Multiple Providers

You can configure multiple MCP providers and use them together in a single `ToolConfig`:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

providers = [
    dd.MCPProvider(
        name="doc-search-mcp",
        endpoint="http://localhost:8080/sse",
    ),
    dd.LocalStdioMCPProvider(
        name="calculator-mcp",
        command="python",
        args=["-m", "calculator_mcp"],
    ),
]

data_designer = DataDesigner(mcp_providers=providers)
```

## See Also

- **[Tool Configurations](tool-configs.md)**: Configure tool access with ToolConfig
- **[Configure MCP with the CLI](configure-mcp-cli.md)**: Use the CLI to manage MCP providers
- **[Enabling Tools on Columns](enabling-tools.md)**: Use tools in LLM columns
