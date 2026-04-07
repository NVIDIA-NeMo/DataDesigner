# Agent Rollout Ingestion

`AgentRolloutSeedSource` imports agent rollout artifacts from disk and normalizes them into seed rows you can use for analysis, filtering, and distillation pipelines.

This page is the dedicated home for:

- supported rollout formats and default locations
- normalized output columns and message structure
- format-specific ingestion behavior
- downstream usage patterns and troubleshooting

!!! note "Documentation in progress"
    This page now serves as the canonical entry point for agent rollout ingestion docs. The detailed walkthrough and format-specific guidance will be expanded in a follow-up update.

## Quick Start

Use `AgentRolloutSeedSource` when you want to work from existing agent traces instead of traces captured during a Data Designer generation run.

```python
import data_designer.config as dd

seed_source = dd.AgentRolloutSeedSource(
    format=dd.AgentRolloutFormat.CLAUDE_CODE,
)
```

## Related Guides

- For the general seed dataset model, see [Seed Datasets](seed-datasets.md).
- For the normalized `messages` structure used in imported rollouts, see [Message Traces](traces.md).
- For an end-to-end distillation example, see [Agent Rollout Trace Distillation](../recipes/trace_ingestion/agent_rollout_distillation.md).
