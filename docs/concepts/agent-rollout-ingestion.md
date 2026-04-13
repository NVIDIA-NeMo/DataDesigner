# Agent Rollout Ingestion

`AgentRolloutSeedSource` turns existing agent rollouts into a seed dataset for synthetic data workflows. It lets you operate locally on rollout artifacts you already have on disk, then normalizes them into rows you can filter, curate, and distill into training or evaluation data.

## Quick Start

Use `AgentRolloutSeedSource` when you want to work from existing agent traces instead of traces captured during a Data Designer generation run.

=== "Claude Code"

    Uses `~/.claude/projects` and `*.jsonl` by default.

    ```python
    import data_designer.config as dd

    seed_source = dd.AgentRolloutSeedSource(
        format=dd.AgentRolloutFormat.CLAUDE_CODE,
    )
    ```

=== "Codex"

    Uses `~/.codex/sessions` and `*.jsonl` by default.

    ```python
    import data_designer.config as dd

    seed_source = dd.AgentRolloutSeedSource(
        format=dd.AgentRolloutFormat.CODEX,
    )
    ```

=== "Hermes Agent"

    Uses `~/.hermes/sessions` and `*.json*` by default so CLI session logs and gateway transcripts can coexist.

    ```python
    import data_designer.config as dd

    seed_source = dd.AgentRolloutSeedSource(
        format=dd.AgentRolloutFormat.HERMES_AGENT,
    )
    ```

=== "Pi Coding Agent"

    Uses `~/.pi/agent/sessions` and `*.jsonl` by default. Sessions are tree-structured JSONL files; the active conversation path is resolved automatically.

    ```python
    import data_designer.config as dd

    seed_source = dd.AgentRolloutSeedSource(
        format=dd.AgentRolloutFormat.PI_CODING_AGENT,
    )
    ```

=== "ATIF"

    ATIF requires an explicit `path`. See Harbor's [ATIF documentation](https://harborframework.com/docs/trajectory-format) for the format specification.

    ```python
    import data_designer.config as dd

    seed_source = dd.AgentRolloutSeedSource(
        format=dd.AgentRolloutFormat.ATIF,
        path="/data/harbor/runs/swe-bench/job-042",
        recursive=True,
        file_pattern="trajectory*.json",
    )
    ```

You can override `path` and `file_pattern` for any format when your rollout artifacts live outside the built-in defaults.

## Normalized Field Compatibility

All supported rollout formats map into the same seeded row schema. In the table below, `None` means the source artifact does not expose that field directly, and `derived` means Data Designer computes it from normalized `messages`.

| Normalized field | ATIF | Claude Code | Codex | Hermes Agent | Pi Coding Agent |
|---|---|---|---|---|---|
| `trace_id` | `session_id` | `sessionId[:agentId]` | `session_meta.id` or file stem | CLI `session_id` or file stem; gateway file stem | Session header `id` |
| `source_kind` | `"atif"` | `"claude_code"` | `"codex"` | `"hermes_agent"` | `"pi_coding_agent"` |
| `source_path` | Parsed `.json` path | Parsed `.jsonl` trace path | Parsed `rollout-*.jsonl` path | Parsed CLI `.json` or gateway `.jsonl` path | Parsed `.jsonl` session path |
| `root_session_id` | `session_id` | `sessionId` or file stem | `trace_id` | `trace_id` | Session header `id` |
| `agent_id` | `None` | `agentId` | `None` | `None` | `None` |
| `is_sidechain` | `False` | `isSidechain` | `False` | `False` | `False` |
| `cwd` | `agent.extra.cwd` | First non-null record `cwd` | `session_meta.cwd` | `None` | Session header `cwd` |
| `project_path` | `extra.project_path` or `cwd` | `projectPath` or `cwd` | `cwd` | `None` | Session header `cwd` |
| `git_branch` | `agent.extra.git_branch` | First non-null record `gitBranch` | `session_meta.git_branch` | `None` | `None` |
| `started_at` | Earliest step timestamp | Earliest row timestamp | `session_meta.timestamp` or earliest record timestamp | CLI `session_start`; gateway `created_at` | Earliest entry timestamp |
| `ended_at` | Latest step timestamp | Latest row timestamp | Latest record timestamp | CLI `last_updated`; gateway `updated_at` | Latest entry timestamp |
| `messages` | Normalized steps | Normalized trace rows | Normalized response items | Normalized CLI or gateway rows | Normalized active-path messages |
| `source_meta` | ATIF metadata | Claude metadata | Codex metadata | Hermes metadata | Pi session metadata |
| `message_count` | `derived` | `derived` | `derived` | `derived` | `derived` |
| `tool_call_count` | `derived` | `derived` | `derived` | `derived` | `derived` |
| `final_assistant_message` | `derived` | `derived` | `derived` | `derived` | `derived` |

### Notes

- `trace_id`: Claude Code appends `agentId` when present. Hermes uses either the CLI session ID or the gateway transcript file stem. Pi uses the session header `id`.
- `is_sidechain`: ATIF, Hermes, and Pi currently normalize this to `False`. Claude Code preserves `isSidechain` directly.
- `messages`: All formats normalize into the same chat-style message schema. See [Message Traces](traces.md) for the shared block structure. Pi sessions are tree-structured; only the active conversation path (from the last entry back to root) is included.
- `source_meta`: This is where format-specific details live, such as ATIF copied-context metadata, Claude summaries, Codex response-item types, Hermes tool/session metadata, or Pi session version and branch information.

## Example: Summarize a Random Turn

Because the seeded fields are normalized, you can also build lightweight summarization workflows directly from imported rollouts. This example samples one random normalized message from each trace and summarizes it in a single sentence.

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
config_builder = dd.DataDesignerConfigBuilder(
    model_configs=[
        dd.ModelConfig(
            alias="trace-writer",
            model="nvidia/nemotron-3-nano-30b-a3b",
            provider="nvidia",
        )
    ]
)

config_builder.with_seed_dataset(
    dd.AgentRolloutSeedSource(
        format=dd.AgentRolloutFormat.CLAUDE_CODE,
    )
)

config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="sampled_turn",
        expr="{{ messages | random }}",
    )
)

config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="turn_summary",
        model_alias="trace-writer",
        prompt="""\
Summarize this randomly sampled rollout turn in one sentence.
The turn may come from the user, assistant, or a tool result.

Trace: {{ trace_id }}
Turn:
{{ sampled_turn }}
""",
    )
)

preview = data_designer.preview(config_builder, num_records=3)
preview.display_sample_record()
```

This stays fully declarative: no custom seed reader or preprocessing step is required. Because `sampled_turn` is drawn from the normalized `messages` list, the same config works across all supported rollout formats.

## Example: Turn Tool Interactions into a Review Dataset

You can also explode imported rollouts into a tool-interaction dataset. This example scans normalized `messages`, emits one row per tool call and matching tool response, preserves the trace context up to that response, and then uses a structured column to label the interaction as a success, failure, or unclear outcome.

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner
from pydantic import BaseModel, Field
from typing import Literal


@dd.custom_column_generator(
    required_columns=["messages"],
    side_effect_columns=["tool_call", "tool_response", "tool_name"],
)
def explode_tool_interactions(row: dict) -> list[dict]:
    rows = []
    tool_calls_by_id = {}
    context_messages = []

    for message in row["messages"]:
        context_messages.append(message)

        for tool_call in message.get("tool_calls") or []:
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                tool_calls_by_id[tool_call_id] = tool_call

        if message.get("role") != "tool":
            continue

        tool_call = tool_calls_by_id.get(
            message.get("tool_call_id"),
            {
                "id": message.get("tool_call_id"),
                "type": "function",
                "function": {"name": "unknown", "arguments": "{}"},
            },
        )
        tool_name = tool_call.get("function", {}).get("name", "unknown")

        rows.append(
            {
                **row,
                "tool_interaction_context": list(context_messages),
                "tool_call": tool_call,
                "tool_response": message,
                "tool_name": tool_name,
            }
        )

    return rows


class ToolInteractionAnalysis(BaseModel):
    outcome: Literal["success", "failure", "unclear"] = Field(
        description="Whether the tool interaction appears to have succeeded, failed, or is ambiguous."
    )
    summary: str = Field(
        description="One or two sentences summarizing what the tool was asked to do and what the response indicates."
    )


data_designer = DataDesigner()
config_builder = dd.DataDesignerConfigBuilder(
    model_configs=[
        dd.ModelConfig(
            alias="tool-analyst",
            model="nvidia/nemotron-3-nano-30b-a3b",
            provider="nvidia",
        )
    ]
)

config_builder.with_seed_dataset(
    dd.AgentRolloutSeedSource(
        format=dd.AgentRolloutFormat.CLAUDE_CODE,
    )
)

config_builder.add_column(
    dd.CustomColumnConfig(
        name="tool_interaction_context",
        generator_function=explode_tool_interactions,
        allow_resize=True,
    )
)

config_builder.add_column(
    dd.LLMStructuredColumnConfig(
        name="tool_interaction_analysis",
        model_alias="tool-analyst",
        output_format=ToolInteractionAnalysis,
        prompt="""\
You are analyzing one tool interaction from an imported agent rollout.

Context up to the tool response:
{{ tool_interaction_context }}

Tool name: {{ tool_name }}

Tool call:
{{ tool_call }}

Tool response:
{{ tool_response }}

Decide whether this interaction is a success, failure, or unclear outcome.
Then summarize what the tool was asked to do and what happened.
Base your answer on the tool call arguments, the tool response, and the immediate context.
""",
    )
)

preview = data_designer.preview(config_builder, num_records=5)
preview.display_sample_record()
```

This pattern is useful when you want to curate evaluator or monitoring datasets from real traces. The resize-enabled custom column turns each tool interaction into its own record, and the structured column adds a consistent outcome label plus a grounded summary. Because the logic operates on normalized `tool_calls` and `tool` messages, the same pattern transfers across supported rollout formats. If your traces are long, consider adding a second custom or expression column that windows the context before sending it to a model.

## Related Guides

- For the general seed dataset model, see [Seed Datasets](seed-datasets.md).
- For the normalized `messages` structure used in imported rollouts, see [Message Traces](traces.md).
- For an end-to-end distillation example, see [Agent Rollout Trace Distillation](../recipes/trace_ingestion/agent_rollout_distillation.md).
