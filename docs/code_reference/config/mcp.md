# MCP Configuration

The `data_designer.config.mcp` module defines declarative configuration classes for tool use via MCP (Model Context Protocol). These configs tell Data Designer which MCP providers exist and which tools an LLM column may use.

!!! note "Config"
    This page documents the MCP config objects users add to a configuration. If you are changing MCP execution or debugging tool-call runtime behavior, see [Engine MCP](../engine/mcp.md).

[MCPProvider](#data_designer.config.mcp.MCPProvider) configures remote MCP servers via SSE or Streamable HTTP transport. [LocalStdioMCPProvider](#data_designer.config.mcp.LocalStdioMCPProvider) configures local MCP servers as subprocesses via stdio transport. [ToolConfig](#data_designer.config.mcp.ToolConfig) defines which tools are available for LLM columns and how they are constrained.

For user-facing guides, see:

- **[MCP Providers](../../concepts/mcp/mcp-providers.md)** - Configure local or remote MCP providers
- **[Tool Configs](../../concepts/mcp/tool-configs.md)** - Define tool permissions and limits
- **[Enabling Tools](../../concepts/mcp/enabling-tools.md)** - Use tools in LLM columns
- **[Traces](../../concepts/traces.md)** - Capture full conversation history

## API Reference

::: data_designer.config.mcp
