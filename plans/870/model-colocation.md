# Model co-location

## Context

Different model deployments currently receive disjoint nodes and the full profile GPU budget.

## Goal

Allow explicit single-node groups with disjoint per-deployment GPU quotas in one managed run.
Deliver as one PR against `feat/slurm-execution`.

## Design

Use `resources.shared_node` and `resources.gpus_per_node`. One group must cover every deployment in the run. Members require one node, single-node replicas, GRES allocation, the same resolved serving image, and matching server environment and queue settings. Tensor parallelism divides each quota, and quotas fit the profile allocation budget. The compiler assigns contiguous GPU slices and unique ports. One coordinated worker launches and supervises all model processes using Slurm's step-local CUDA visibility. Client/proxy steps remain GPU-free. Default fields are omitted from serialization to preserve legacy bytes.

## Tasks

- [x] Add authored/preparation validation and deterministic placement/port/GPU contracts, including compatibility tests.
- [x] Compose shared node workers and verify readiness, CUDA mapping, failures, cleanup, and retry resource preservation.
- [x] Document two distinct models and run the package/docs gates and one bounded end-to-end smoke.
- [x] Obtain independent Claude review, address verified findings, and review the final branch.

## Open questions

Runtime semantics and shared-worker restrictions require maintainer review before merge. Model memory fit remains the caller's responsibility.
