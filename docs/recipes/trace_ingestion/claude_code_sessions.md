[Download Code :octicons-download-24:](../../../assets/recipes/trace_ingestion/claude_code_sessions.py){ .md-button download="claude_code_sessions.py" }

This recipe ingests Claude Code session traces with `ClaudeCodeTraceSeedSource`, using the default `~/.claude/projects` location unless `--trace-dir` is provided. It then distills each imported trace into a structured workflow record you can use for downstream SDG or analysis. It supports both full dataset creation and in-memory preview mode via `--preview`.

```python
--8<-- "assets/recipes/trace_ingestion/claude_code_sessions.py"
```
