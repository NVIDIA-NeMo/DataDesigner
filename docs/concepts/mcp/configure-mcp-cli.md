# Configuring MCP Using the CLI

The Data Designer CLI provides an interactive interface for creating and managing MCP providers and tool configurations stored in your Data Designer home directory (default: `~/.data-designer/`).

## Configuration Files

The CLI manages two YAML configuration files for MCP:

- **`mcp_providers.yaml`**: MCP provider configurations
- **`tool_configs.yaml`**: Tool configurations

!!! note "Custom Directory"
    You can customize the configuration directory location with the `DATA_DESIGNER_HOME` environment variable:
    ```bash
    export DATA_DESIGNER_HOME="/path/to/your/custom/directory"
    ```

## CLI Commands

The Data Designer CLI provides commands for MCP configuration:

```bash
# Configure MCP providers
data-designer config mcp

# Configure tool configs
data-designer config tools

# List all configurations (including MCP)
data-designer config list
```

!!! tip "Getting help"
    See available commands:
    ```bash
    data-designer config --help
    ```

## Configuring MCP Providers

Run the interactive MCP provider configuration command:

```bash
data-designer config mcp
```

### Provider Type Selection

The wizard first asks you to choose a provider type:

1. **Remote SSE**: Connect to a pre-existing MCP server via HTTP Server-Sent Events
2. **Local stdio subprocess**: Launch an MCP server as a subprocess

### Remote SSE Configuration

When configuring a Remote SSE provider, you'll be prompted for:

- **Name**: Unique identifier (e.g., `"doc-search"`)
- **Endpoint**: SSE endpoint URL (e.g., `"http://localhost:8080/sse"`)
- **API Key**: Optional API key or environment variable name

### Local Stdio Configuration

When configuring a Local stdio provider, you'll be prompted for:

- **Name**: Unique identifier (e.g., `"local-tools"`)
- **Command**: Executable to run (e.g., `"python"`)
- **Arguments**: Command-line arguments (e.g., `"-m my_mcp_server"`)
- **Environment Variables**: Optional environment variables for the subprocess

### Available Operations

- **Add a new provider**: Define a new MCP provider
- **Update an existing provider**: Modify provider settings
- **Delete a provider**: Remove a provider
- **Delete all providers**: Remove all MCP providers

## Configuring Tool Configs

Run the interactive tool configuration command:

```bash
data-designer config tools
```

!!! info "Provider Required"
    You need at least one MCP provider configured before adding tool configs. Run `data-designer config mcp` first if none exist.

### Configuration Options

When creating a tool config, you'll be prompted for:

- **Tool Alias**: Unique name for referencing in columns (e.g., `"my-tools"`)
- **Providers**: Select one or more MCP providers (checkbox selection)
- **Allowed Tools**: Optionally restrict to specific tools (leave empty for all)
- **Max Tool Call Turns**: Maximum tool-calling iterations (default: 5)
- **Timeout**: Per-call timeout in seconds (default: 60.0)

### Available Operations

- **Add a new tool config**: Define a new tool configuration
- **Update an existing tool config**: Modify settings
- **Delete a tool config**: Remove a tool configuration
- **Delete all tool configs**: Remove all tool configurations

## Listing Configurations

View all current configurations:

```bash
data-designer config list
```

This command displays:

- **Model Providers**: All configured model providers
- **Model Configurations**: All configured models
- **MCP Providers**: All configured MCP providers with their endpoints
- **Tool Configurations**: All configured tool configs with their settings

## YAML File Format

### MCP Providers (`mcp_providers.yaml`)

```yaml
providers:
  # Remote SSE provider
  - name: doc-search
    provider_type: sse
    endpoint: http://localhost:8080/sse
    api_key: ${MCP_API_KEY}

  # Local stdio provider
  - name: calculator
    provider_type: stdio
    command: python
    args:
      - -m
      - calculator_mcp
    env:
      DEBUG: "true"
```

### Tool Configurations (`tool_configs.yaml`)

```yaml
tool_configs:
  - tool_alias: my-tools
    providers:
      - doc-search
      - calculator
    allow_tools:
      - search_docs
      - calculate
    max_tool_call_turns: 5
    timeout_sec: 60.0

  - tool_alias: search-only
    providers:
      - doc-search
    allow_tools:
      - search_docs
      - list_docs
    max_tool_call_turns: 10
    timeout_sec: 30.0
```

## Manual Editing

You can also edit the YAML files directly for advanced configurations. The files are located at:

- `~/.data-designer/mcp_providers.yaml`
- `~/.data-designer/tool_configs.yaml`

After manual edits, the changes take effect the next time you initialize `DataDesigner`.

## See Also

- **[MCP Providers](mcp-providers.md)**: Learn about provider configuration options
- **[Tool Configurations](tool-configs.md)**: Learn about tool config options
- **[Configure Model Settings with the CLI](../models/configure-model-settings-with-the-cli.md)**: CLI guide for model configuration
