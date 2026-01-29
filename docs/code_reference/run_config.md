# Run Config

The `run_config` module defines runtime settings that control dataset generation behavior,
including early shutdown thresholds, batch sizing, and non-inference worker concurrency.

## Usage

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
data_designer.set_run_config(dd.RunConfig(
    buffer_size=500,
    max_conversation_restarts=3,
))
```

## Trace capture

Data Designer can capture full message traces for LLM columns, which is useful for debugging tool calls and understanding model behavior.

### Per-column traces (recommended)

Enable traces on specific columns using `with_trace=True`:

```python
builder.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="Answer: {{ question }}",
        model_alias="nvidia-text",
        tool_alias="my-tools",
        with_trace=True,  # Capture trace for this column only
    )
)
```

This creates an `answer__trace` column containing the ordered message history.

### Global debug override

For debugging, enable traces for ALL LLM columns at once:

```python
data_designer.set_run_config(dd.RunConfig(
    debug_override_save_all_column_traces=True,
))
```

This overrides all per-column `with_trace` settings and captures traces for every LLM generation. Use during development, but disable in production to reduce dataset size.

### What traces contain

Each trace is a `list[dict]` with the conversation history:

- `system` messages (system prompts)
- `user` messages (rendered prompts)
- `assistant` messages (model responses, may include `tool_calls` and `reasoning_content`)
- `tool` messages (tool execution results with `tool_call_id`)

See [Tool Use & MCP](../concepts/tool_use_and_mcp.md#full-message-traces-trace) for detailed trace structure documentation.

## API Reference

::: data_designer.config.run_config
