# Plan: External model providers in Slurm runs

Issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/1079

The implementation follows the reviewed plan: retain the existing async engine,
add explicit external providers, and measure allocation utilization during SQL
qualification before considering separate generation and judging allocations.

## Design

- Add external provider declarations to invocation configuration. Reuse native DD
  provider settings, with Slurm `SecretRef` values at credential boundaries.
- Partition declared model aliases into managed deployments and external models.
  Require complete, disjoint coverage and reject provider-name collisions.
- Materialize endpoints only for managed aliases. Preserve external model names,
  sampling parameters, and explicit concurrency settings.
- Allow an empty deployment list. Resolve zero GPUs, one native client host, no
  serving images, no GPU discovery, no serving containers, and no proxy ports.
- Reuse existing shard, retry, resume, secret transport, and collection contracts.
  Preserve serialized local-only plans where defaults are unchanged.

## Validation

Exercise authored configs and builders, sourced and inline plan resolution,
native client provider routing, credential isolation, resource directives,
allocation startup and cleanup, retry/resume, and local-only golden fixtures.
Run the Slurm suite and the full package checks before opening the PR.

SQL qualification uses a small mixed-provider async recipe and a CPU-only hosted
recipe. Record per-model timing and actual GPU utilization, and compare idle
windows with the selected cluster policy. Preserve existing SQL prompts, models,
sampling, judge gates, datasets, checkpoints, and completed study results.

Automatic DAG partitioning, new async machinery, model substitutions, and the
separate single-row profiling defect are outside this change.
