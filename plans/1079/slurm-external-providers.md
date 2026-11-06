# Plan: External model providers in Slurm runs

Issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/1079
PR: https://github.com/NVIDIA-NeMo/DataDesigner/pull/1080

Provider ownership follows the [issue feedback](https://github.com/NVIDIA-NeMo/DataDesigner/issues/1079#issuecomment-6369036893).

The implementation retains the existing async engine and resolves hosted models
through native DD provider settings. Slurm configuration describes only models
deployed locally. Mixed runs retain their GPUs during hosted requests; separate
stages remain appropriate when hosted processing is substantial.

## Design

- Configure hosted providers in DD's existing `model_providers.yaml`; recipe model
  configs reference those provider names. Do not introduce `invocation.model_providers`
  or a parallel Slurm provider schema.
- Resolve undeployed aliases against native settings before submission. Capture
  only required hosted providers as immutable native JSON run inputs so compute
  nodes and retries do not depend on the submit host's settings directory or later
  settings edits.
- Keep API-key environment-variable references in snapshots. Validate availability
  before capture and submission, and reuse runtime credential transport to forward
  values only to client steps, including retries. Initially support environment-backed
  API keys and non-secret static headers; reject plaintext keys and sensitive
  headers. Custom secret resolvers are outside this change.
- Partition declared model aliases into managed deployments and external models.
  Require complete, disjoint coverage and reject provider-name collisions.
- Combine captured hosted providers with generated local providers in the client.
  Materialize endpoints only for managed aliases. Preserve external model names,
  sampling parameters, and explicit concurrency settings.
- Allow an empty deployment list. Resolve zero GPUs, one native client host, no
  serving images, no GPU discovery, no serving containers, and no proxy ports.
- Reuse existing shard, retry, resume, secret transport, and collection contracts.
  Preserve serialized local-only plans where defaults are unchanged.
- Allow an explicit per-node submission memory request. The live CPU probe found
  that the cluster otherwise reserves all node memory for a four-CPU job.
- Let DD exclusively create a fresh dataset after the state writer has prepared
  its empty directory. Preserve nonempty directories and reject symlink targets.
- Restrict the generated dataset directory before publishing it as a candidate,
  and validate native DD outputs with the state's real artifact verifier.

## Validation

Exercise authored configs and builders, sourced and inline plan resolution,
native client provider routing, credential isolation, resource directives,
allocation startup and cleanup, retry/resume, and local-only golden fixtures.
Run the Slurm suite and the full package checks before opening the PR.

The live qualification reported in PR #1080 uses a small mixed-provider async
recipe and a CPU-only hosted recipe, combined with the independent vLLM
version-gate fix in #1086. It validates routing and publication, not sustained GPU
utilization, Gym SQL execution, production SQL quality, or a full dataset/SFT rerun.
Existing SQL prompts, models, sampling, judge gates, datasets, checkpoints, and
completed study results remain preserved. Measure per-model timing and actual GPU
utilization against cluster policy before deciding to split production stages.

Automatic DAG partitioning, new async machinery, model substitutions, and the
separate single-row profiling defect are outside this change.
