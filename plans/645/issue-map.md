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
| #641 | Add `SchedulingMetadata` and generator override contract | [architecture.md](architecture.md), [contracts.md](contracts.md), [module-ownership.md](module-ownership.md), [task-admission.md](task-admission.md) |
| #646 | Ingest metadata into scheduler grouping through `TaskSchedulingResolver` | [architecture.md](architecture.md), [contracts.md](contracts.md), [module-ownership.md](module-ownership.md), [task-admission.md](task-admission.md) |
| #653 | Remove legacy hint resolver path | [migration-and-cleanup.md](migration-and-cleanup.md), [module-ownership.md](module-ownership.md), [contracts.md](contracts.md) |
| #652 | Document plugin-facing metadata behavior | [architecture.md](architecture.md), [contracts.md](contracts.md), [migration-and-cleanup.md](migration-and-cleanup.md) |
| #644 | Implement task admission lease boundary | [task-admission.md](task-admission.md), [contracts.md](contracts.md), [module-ownership.md](module-ownership.md), [benchmark-plan.md](benchmark-plan.md) |
| #654 | Implement and document capacity vocabulary and snapshots | [capacity-model.md](capacity-model.md), [observability.md](observability.md), [benchmark-plan.md](benchmark-plan.md) |
| #657 | Refactor model-call request control into request admission | [request-admission.md](request-admission.md), [contracts.md](contracts.md), [module-ownership.md](module-ownership.md), [migration-and-cleanup.md](migration-and-cleanup.md), [benchmark-plan.md](benchmark-plan.md) |
| #635 | Instrument request admission state | [observability.md](observability.md), [request-admission.md](request-admission.md), [contracts.md](contracts.md), [benchmark-plan.md](benchmark-plan.md) |
| #647 | Instrument scheduler admission state | [observability.md](observability.md), [task-admission.md](task-admission.md), [contracts.md](contracts.md), [benchmark-plan.md](benchmark-plan.md) |
| #648 | Correlate scheduler and request observability | [observability.md](observability.md), [architecture.md](architecture.md), [benchmark-plan.md](benchmark-plan.md) |
| #649 | Build reusable benchmark harness and normalize provisional evidence | [benchmark-plan.md](benchmark-plan.md), [capacity-model.md](capacity-model.md), [observability.md](observability.md), [task-admission.md](task-admission.md), [request-admission.md](request-admission.md) |
| #660 | Produce final user/operator docs | [architecture.md](architecture.md), [contracts.md](contracts.md), [module-ownership.md](module-ownership.md), [request-admission.md](request-admission.md), [capacity-model.md](capacity-model.md), [observability.md](observability.md), [benchmark-plan.md](benchmark-plan.md), [migration-and-cleanup.md](migration-and-cleanup.md) |
| #650 | Implement bounded-borrow task policy | [task-admission.md](task-admission.md), [benchmark-plan.md](benchmark-plan.md), [capacity-model.md](capacity-model.md) |
| #651 | Design resource-vector/provider-aware policy | [task-admission.md](task-admission.md), [request-admission.md](request-admission.md), [capacity-model.md](capacity-model.md), [observability.md](observability.md), [benchmark-plan.md](benchmark-plan.md) |

## Dependency Order

The implementation order remains:

```text
#641 -> #646 -> #653 -> #652 -> #644 -> #654 -> #657
-> #635 -> #647 -> #648 -> #649 -> #660 -> #650 -> #651
```

#644 cannot close until task admission uses the final scheduler module homes and the accepted metadata contract. The accepted end state is `SchedulingMetadata` feeding task admission through `TaskSchedulingResolver`; old resolver paths, compatibility adapters, and duplicate module homes are not part of the target architecture.

#660 promotes the stabilized V1 admission/capacity/telemetry docs. #650 and #651 are follow-on policy/design issues; if they change behavior or public/operator guidance, they must update this source-of-truth plan and any promoted docs as part of their own acceptance gates. #651 is design-first unless its issue body explicitly promotes an implementation slice. The request-pressure advisory selection path currently in PR #661 is a narrow implementation slice ahead of the broader #651 design; it must remain read-only with respect to request admission until #651 defines a durable provider/resource-aware policy.

## Evidence Phasing

The native issue order keeps #649 after capacity, request admission, telemetry, and correlation because the reusable harness consumes those contracts. That does not waive evidence for earlier implementation PRs.

Before #649 closes, issues #644, #654, #657, #635, #647, and #648 must produce provisional benchmark/evidence artifacts using the schema in [benchmark-plan.md](benchmark-plan.md). A minimal deterministic smoke writer should exist before those slices rely on one-off evidence. Issue #649 then converts those provisional artifacts into the reusable harness, reruns representative scenarios, and becomes the gate for #660, #650, and #651.

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
