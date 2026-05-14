# Issue Map

This file maps GitHub issues to the source-of-truth plan sections. Issues should reference these files and own implementation/quality gates rather than restating the full architecture.

## Source-Of-Truth Rule

Use this directory for architecture, contracts, naming, invariants, and cross-cutting decisions.

Use GitHub issues for:

- implementation slices
- dependencies and target branch
- acceptance criteria for that slice
- tests and validation commands
- benchmark evidence required by that slice
- PR-specific cleanup gates

## Issues

| Issue | Implementation Focus | Source Sections |
| --- | --- | --- |
| #641 | Add `SchedulingMetadata` and generator override contract | [architecture.md](architecture.md), [contracts.md](contracts.md), [task-admission.md](task-admission.md) |
| #646 | Ingest metadata into scheduler grouping through `TaskSchedulingResolver` | [architecture.md](architecture.md), [contracts.md](contracts.md), [task-admission.md](task-admission.md) |
| #653 | Remove legacy hint resolver path | [migration-and-cleanup.md](migration-and-cleanup.md), [contracts.md](contracts.md) |
| #652 | Document plugin-facing metadata behavior | [architecture.md](architecture.md), [contracts.md](contracts.md), [migration-and-cleanup.md](migration-and-cleanup.md) |
| #644 | Implement task admission lease boundary | [task-admission.md](task-admission.md), [contracts.md](contracts.md), [benchmark-plan.md](benchmark-plan.md) |
| #654 | Implement and document capacity vocabulary and snapshots | [capacity-model.md](capacity-model.md), [observability.md](observability.md), [benchmark-plan.md](benchmark-plan.md) |
| #657 | Refactor model-call throttling into request admission | [request-admission.md](request-admission.md), [contracts.md](contracts.md), [migration-and-cleanup.md](migration-and-cleanup.md) |
| #635 | Instrument request admission state | [observability.md](observability.md), [request-admission.md](request-admission.md), [contracts.md](contracts.md) |
| #647 | Instrument scheduler admission state | [observability.md](observability.md), [task-admission.md](task-admission.md), [contracts.md](contracts.md) |
| #648 | Correlate scheduler and request observability | [observability.md](observability.md), [architecture.md](architecture.md) |
| #649 | Build reusable benchmark harness | [benchmark-plan.md](benchmark-plan.md), [capacity-model.md](capacity-model.md), [observability.md](observability.md) |
| #660 | Produce final user/operator docs | [architecture.md](architecture.md), [capacity-model.md](capacity-model.md), [observability.md](observability.md), [migration-and-cleanup.md](migration-and-cleanup.md) |
| #650 | Implement bounded-borrow task policy | [task-admission.md](task-admission.md), [benchmark-plan.md](benchmark-plan.md), [capacity-model.md](capacity-model.md) |
| #651 | Design resource-vector/provider-aware policy | [task-admission.md](task-admission.md), [capacity-model.md](capacity-model.md), [benchmark-plan.md](benchmark-plan.md) |

## Dependency Order

The implementation order remains:

```text
#641 -> #646 -> #653 -> #652 -> #644 -> #654 -> #657
-> #635 -> #647 -> #648 -> #649 -> #660 -> #650 -> #651
```

#641 and #644 may proceed independently only while #646 preserves the adapter contract between them. The accepted end state is `SchedulingMetadata` feeding task admission through `TaskSchedulingResolver`.

## Issue Body Cleanup Pattern

When revising issue bodies, keep:

- priority
- dependency metadata
- target branch
- short problem statement
- links to the relevant plan sections
- implementation checklist
- tests and validation commands
- evidence requirements
- acceptance criteria specific to the slice

Remove or shorten:

- duplicated contract definitions
- duplicated architecture diagrams
- broad cross-cutting non-goals already captured here
- stale naming decisions superseded by this plan
