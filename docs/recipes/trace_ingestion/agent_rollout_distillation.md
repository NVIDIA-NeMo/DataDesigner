[Download Code :octicons-download-24:](../../assets/recipes/trace_ingestion/agent_rollout_distillation.py){ .md-button download="agent_rollout_distillation.py" }

This recipe ingests built-in agent rollout traces with `AgentRolloutSeedSource(...)`, selecting the format with
`--format` and optionally overriding the input directory with `--trace-dir`. It works with `claude_code` and `codex`;
both can use their default locations when `--trace-dir` is omitted. The pipeline turns each imported trace into a
compact task digest, a standalone instruction-response pair for coding-assistant SFT, and a judge-scored quality
signal you can use for downstream filtering. It supports both full dataset creation and in-memory preview mode via
`--preview`.

```python
--8<-- "assets/recipes/trace_ingestion/agent_rollout_distillation.py"
```
