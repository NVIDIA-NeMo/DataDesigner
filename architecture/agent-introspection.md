# Agent Introspection

The agent introspection subsystem provides machine-readable CLI commands that let agents discover DataDesigner's type system, configuration state, and available operations at runtime.

Source: `packages/data-designer/src/data_designer/cli/commands/agent.py` and `packages/data-designer/src/data_designer/cli/utils/agent_introspection.py`

## Overview

Agent introspection solves a specific problem: agents working with DataDesigner need to know what column types, sampler types, validator types, and processor types are available — including any installed plugins. Rather than hardcoding this knowledge or parsing source code, agents can call `data-designer agent` commands to get structured, up-to-date information.

## Key Components

### Commands

All commands live under the `data-designer agent` group:

| Command | Purpose |
|---------|---------|
| `data-designer agent context` | Full context dump: version, paths, type catalogs, model aliases, persona state, available operations |
| `data-designer agent types [family]` | Type catalog for one or all families, with descriptions and source file locations |
| `data-designer agent state model-aliases` | Configured model aliases with usability status (missing provider, missing API key, etc.) |
| `data-designer agent state persona-datasets` | Available persona datasets with download status per locale |

### FamilySpec

Maps a **family name** to a **discriminated union type** and its **discriminator field**:

| Family | Union Type | Discriminator |
|--------|-----------|---------------|
| `column-types` | `ColumnConfigT` | `column_type` |
| `sampler-types` | `SamplerParamsT` | `sampler_type` |
| `validator-types` | `ValidatorParamsT` | `validator_type` |
| `processor-types` | `ProcessorConfigT` | `processor_type` |
| `constraint-types` | `ColumnConstraintT` | `constraint_type` |

### Type Discovery

`discover_family_types` walks `typing.get_args(type_union)`, reads each Pydantic model's discriminator field annotation (must be `Literal[...]`), and builds a map of discriminator string → model class. Detects and reports duplicate discriminator values.

`get_family_catalog` yields the class name and first docstring paragraph for each type — enough for an agent to understand what each type does without reading source code.

`get_family_source_files` uses `inspect.getfile` and normalizes paths under `data_designer/` (absolute path fallback for plugin types outside the tree).

### State Commands

Reuse the CLI's repository stack:
- **Model aliases**: `ModelRepository` + `ProviderRepository` + `get_providers_with_missing_api_keys` to report usability status (configured, missing provider, missing API key)
- **Personas**: `PersonaRepository` + `DownloadService` for locale availability and download status

### Error Handling

`AgentIntrospectionError` carries a `code`, `message`, and `details` dict. Commands catch these and output structured error information to stderr with exit code 1, making errors parseable by agents.

### Command Registration

`AGENT_COMMANDS` in `agent_command_defs.py` drives both the lazy Typer command map in `main.py` and `get_operations()` in introspection. This single source of truth ensures the operations table in `agent context` output stays in sync with the actual commands.

## Data Flow

```
Agent calls: data-designer agent types column-types
  → Typer dispatches to agent.get_types("column-types")
  → FamilySpec maps "column-types" → ColumnConfigT union
  → discover_family_types walks union members
  → get_family_catalog extracts names + descriptions
  → get_family_source_files resolves source locations
  → Formatted output returned to agent
```

## Design Decisions

- **Declarative type discovery from Pydantic unions** rather than maintaining a separate type inventory. The discriminated unions are the source of truth for what types exist (including plugins), so introspection reads directly from them.
- **Structured errors with codes** enable agents to handle failures programmatically (retry, report, escalate) rather than parsing human-readable error messages.
- **Single command registration source** (`AGENT_COMMANDS`) prevents the operations table from drifting out of sync with actual CLI commands.
- **Source file resolution** helps agents navigate to implementations when they need to understand a type's behavior, not just its existence.

## Cross-References

- [System Architecture](overview.md) — where agent introspection fits
- [CLI](cli.md) — the CLI architecture that hosts these commands
- [Config Layer](config.md) — the discriminated unions that introspection reads
- [Plugins](plugins.md) — how plugin types appear in introspection results
