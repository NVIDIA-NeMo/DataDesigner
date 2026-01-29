# Tool Use & MCP

Tool use lets LLM columns call external tools during generation (e.g., lookups, calculations, retrieval, domain services). Data Designer supports tool use via the **Model Context Protocol (MCP)**, which standardizes how tools are discovered and invoked.

## Quick Start

1. Configure an MCP provider ([Local](mcp/mcp-providers.md#localstdiomcpprovider-subprocess) or [Remote](mcp/mcp-providers.md#mcpprovider-remote-sse))
2. Create a [ToolConfig](mcp/tool-configs.md) referencing your provider
3. Add `tool_alias` to your [LLM column](mcp/enabling-tools.md)

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

# 1. Configure provider
mcp_provider = dd.LocalStdioMCPProvider(
    name="demo-mcp",
    command="python",
    args=["-m", "my_mcp_server"],
)

data_designer = DataDesigner(mcp_providers=[mcp_provider])

# 2. Create tool config
tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
)

builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

# 3. Use tools in column
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Use tools to answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",
    )
)
```

## Guides

| Guide | Description |
|-------|-------------|
| **[MCP Providers](mcp/mcp-providers.md)** | Configure local subprocess or remote SSE providers |
| **[Tool Configs](mcp/tool-configs.md)** | Define tool permissions and limits |
| **[Enabling Tools on Columns](mcp/enabling-tools.md)** | Use tools in LLM generation |
| **[Configure via CLI](mcp/configure-mcp-cli.md)** | Interactive CLI configuration |
| **[Traces](mcp/traces.md)** | Capture full conversation history |
| **[Safety & Limits](mcp/safety-and-limits.md)** | Allowlists, budgets, timeouts |
| **[Debugging](mcp/debugging.md)** | Troubleshoot tool call issues |

## Example

See the [PDF Q&A Recipe](../recipes/mcp_and_tooluse/pdf_qa.md) for a complete working example.

## Code Reference

For internal architecture and API documentation, see [MCP Code Reference](../code_reference/mcp.md).
