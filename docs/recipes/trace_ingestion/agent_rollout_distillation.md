[Download Code :octicons-download-24:](../../assets/recipes/trace_ingestion/agent_rollout_distillation.py){ .md-button download="agent_rollout_distillation.py" }

This recipe ingests built-in agent rollout traces with `AgentRolloutSeedSource(...)`, selecting the format with
`--format` and optionally overriding the input directory with `--trace-dir`. It works with `atif`, `claude_code`,
`codex`, and `hermes_agent`; `atif` expects standalone `.json` trajectory files and requires `--trace-dir`, while
`claude_code`, `codex`, and `hermes_agent` can use their default locations when `--trace-dir` is omitted. The pipeline turns each
imported trace into a compact task digest, a standalone instruction-response pair for coding-assistant SFT, and a
judge-scored quality signal you can use for downstream filtering. It supports both full dataset creation and in-memory
preview mode via `--preview`.

!!! info "Looking for ingestion details?"
    See [Agent Rollout Ingestion](../../concepts/agent-rollout-ingestion.md) for supported formats, default paths, normalized columns, and rollout-specific parsing behavior. This recipe stays focused on the distillation pipeline.

```python
--8<-- "assets/recipes/trace_ingestion/agent_rollout_distillation.py"
```
