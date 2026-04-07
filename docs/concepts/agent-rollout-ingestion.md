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

| Normalized field | ATIF | Claude Code | Codex | Hermes Agent |
|---|---|---|---|---|
| `trace_id` | `session_id` | `sessionId[:agentId]` | `session_meta.id` or file stem | CLI `session_id` or file stem; gateway file stem |
| `source_kind` | `"atif"` | `"claude_code"` | `"codex"` | `"hermes_agent"` |
| `source_path` | Parsed `.json` path | Parsed `.jsonl` trace path | Parsed `rollout-*.jsonl` path | Parsed CLI `.json` or gateway `.jsonl` path |
| `root_session_id` | `session_id` | `sessionId` or file stem | `trace_id` | `trace_id` |
| `agent_id` | `None` | `agentId` | `None` | `None` |
| `is_sidechain` | `False` | `isSidechain` | `False` | `False` |
| `cwd` | `agent.extra.cwd` | First non-null record `cwd` | `session_meta.cwd` | `None` |
| `project_path` | `extra.project_path` or `cwd` | `projectPath` or `cwd` | `cwd` | `None` |
| `git_branch` | `agent.extra.git_branch` | First non-null record `gitBranch` | `session_meta.git_branch` | `None` |
| `started_at` | Earliest step timestamp | Earliest row timestamp | `session_meta.timestamp` or earliest record timestamp | CLI `session_start`; gateway `created_at` |
| `ended_at` | Latest step timestamp | Latest row timestamp | Latest record timestamp | CLI `last_updated`; gateway `updated_at` |
| `messages` | Normalized steps | Normalized trace rows | Normalized response items | Normalized CLI or gateway rows |
| `source_meta` | ATIF metadata | Claude metadata | Codex metadata | Hermes metadata |
| `message_count` | `derived` | `derived` | `derived` | `derived` |
| `tool_call_count` | `derived` | `derived` | `derived` | `derived` |
| `final_assistant_message` | `derived` | `derived` | `derived` | `derived` |

### Notes

- `trace_id`: Claude Code appends `agentId` when present. Hermes uses either the CLI session ID or the gateway transcript file stem.
- `is_sidechain`: ATIF and Hermes currently normalize this to `False`. Claude Code preserves `isSidechain` directly.
- `messages`: All formats normalize into the same chat-style message schema. See [Message Traces](traces.md) for the shared block structure.
- `source_meta`: This is where format-specific details live, such as ATIF copied-context metadata, Claude summaries, Codex response-item types, or Hermes tool/session metadata.

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

## Related Guides

- For the general seed dataset model, see [Seed Datasets](seed-datasets.md).
- For the normalized `messages` structure used in imported rollouts, see [Message Traces](traces.md).
- For an end-to-end distillation example, see [Agent Rollout Trace Distillation](../recipes/trace_ingestion/agent_rollout_distillation.md).
