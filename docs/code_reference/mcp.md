# MCP (Model Context Protocol)

The `mcp` module defines configuration and execution classes for tool use via MCP (Model Context Protocol).

## Configuration Classes

[MCPProvider](#data_designer.config.mcp.MCPProvider) configures remote MCP servers via SSE transport. [LocalStdioMCPProvider](#data_designer.config.mcp.LocalStdioMCPProvider) configures local MCP servers as subprocesses via stdio transport. [ToolConfig](#data_designer.config.mcp.ToolConfig) defines which tools are available for LLM columns and how they are constrained.

For more information on usage, see:

- **[Tool Use & MCP](../concepts/tool_use_and_mcp.md)**

## Config Module

::: data_designer.config.mcp
