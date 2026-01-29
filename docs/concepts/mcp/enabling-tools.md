# Enabling Tools on Columns

This guide explains how to enable tool use on LLM columns by connecting them to tool configurations via the `tool_alias` parameter.

## Overview

To enable tool use on an LLM column, you reference a `ToolConfig` by its alias. During generation, the model can then request tool calls, and Data Designer executes them and feeds the results back to the model.

## Using tool_alias

Add the `tool_alias` parameter to any supported LLM column configuration:

```python
import data_designer.config as dd

builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Use tools as needed to answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",  # References a ToolConfig
    )
)
```

## Supported Column Types

Tool use is supported on these column configuration types:

| Column Type | Description |
|------------|-------------|
| `LLMTextColumnConfig` | Text generation with tool access |
| `LLMCodeColumnConfig` | Code generation with tool access |
| `LLMStructuredColumnConfig` | Structured JSON generation with tool access |
| `LLMJudgeColumnConfig` | Judge/scoring with tool access |

## How It Works

When `tool_alias` is specified:

1. **Tool schemas are fetched** from the referenced MCP providers
2. **Model receives tool schemas** with the prompt
3. **Model can request tool calls** in its response
4. **Data Designer executes calls** and returns results to the model
5. **Iteration continues** until the model produces a final answer (or limits are reached)

## Complete Example

Here's a complete workflow showing provider → ToolConfig → column:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

# 1. Configure MCP provider
mcp_provider = dd.LocalStdioMCPProvider(
    name="demo-mcp",
    command="python",
    args=["-m", "my_mcp_server"],
)

# 2. Create DataDesigner instance with provider
data_designer = DataDesigner(mcp_providers=[mcp_provider])

# 3. Define tool configuration
tool_config = dd.ToolConfig(
    tool_alias="my-tools",
    providers=["demo-mcp"],
    allow_tools=["search_docs", "get_fact"],
    max_tool_call_turns=5,
)

# 4. Create config builder with tool config
builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

# 5. Add columns that use tools
builder.add_column(
    dd.SamplerColumnConfig(
        name="question",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["What is machine learning?", "Explain neural networks"]
        ),
    )
)

builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Use the available tools to research and answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",  # Enable tools
        with_trace=True,        # Capture tool call history
    )
)

# 6. Generate data
results = data_designer.preview(builder, num_records=5)
```

## Prompt Design Tips

Tool use behavior depends heavily on how you craft your prompts.

### Making Tool Use Explicit

If you want deterministic tool behavior, explicitly instruct the model:

```python
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="""You must call the search_docs tool before answering.

Question: {{ question }}

Instructions:
1. First, call search_docs to find relevant information
2. Then, provide your answer based on the tool results
3. Only use information from tool results, not prior knowledge""",
        model_alias="nvidia-text",
        tool_alias="my-tools",
    )
)
```

### Guiding Tool Selection

When multiple tools are available, guide the model on which to use:

```python
prompt = """Answer the following question using the available tools.

Available tools:
- search_docs: Search the knowledge base for documents
- get_fact: Get a specific fact by ID

Use search_docs for general queries. Use get_fact when you have a specific fact ID.

Question: {{ question }}"""
```

### Handling Tool Results

Instruct the model on how to process tool outputs:

```python
prompt = """Research and answer the question.

Instructions:
- Call tools to gather information
- If a tool returns an error, try a different approach
- Synthesize tool results into a coherent answer
- Cite which tools provided which information

Question: {{ question }}"""
```

## See Also

- **[Tool Configurations](tool-configs.md)**: Configure tool access and limits
- **[Traces](../traces.md)**: Capture and inspect tool call history
- **[MCP Providers](mcp-providers.md)**: Configure MCP server connections
