# MCP Configuration

MCP config objects tell Data Designer which Model Context Protocol providers exist and which tools an LLM column may use.

[MCPProvider](#data_designer.config.mcp.MCPProvider) configures remote MCP servers via SSE or Streamable HTTP transport. [LocalStdioMCPProvider](#data_designer.config.mcp.LocalStdioMCPProvider) configures local MCP servers as subprocesses via stdio transport. [ToolConfig](#data_designer.config.mcp.ToolConfig) sets which tools are available for LLM columns and how they are constrained.

For MCP execution internals, see [Engine MCP](../engine/mcp.md). Related guides:

- **[MCP Providers](../../concepts/mcp/mcp-providers.md)** - Configure local or remote MCP providers
- **[Tool Configs](../../concepts/mcp/tool-configs.md)** - Define tool permissions and limits
- **[Enabling Tools](../../concepts/mcp/enabling-tools.md)** - Use tools in LLM columns
- **[Traces](../../concepts/traces.md)** - Capture full conversation history

## API Reference

::: data_designer.config.mcp
