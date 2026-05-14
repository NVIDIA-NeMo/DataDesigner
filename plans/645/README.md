# Async Scheduling Architecture Plan

Source-of-truth architecture plan for the async scheduling epic tracked by issue 645. The UML file is the visual index; the Markdown files in this directory are the durable spec. GitHub issues should point back here and focus on implementation sequencing, quality gates, tests, and evidence.

If an issue body and this plan disagree, update this plan first, then adjust the issue to reference the corrected section.

## Spec

- [Architecture](architecture.md): target system shape, ownership boundaries, invariants, and non-goals.
- [Contracts](contracts.md): durable DTO, protocol, event, and config names.
- [Capacity model](capacity-model.md): layered capacity vocabulary and ownership.
- [Task admission](task-admission.md): scheduler-owned ready selection, task leases, policy hooks, bounded borrowing, and resource-vector direction.
- [Request admission](request-admission.md): model-call admission, AIMD controller shape, dynamic request semantics, and no legacy throttle names.
- [Observability](observability.md): scheduler events, request events, runtime correlation, snapshots, and cardinality rules.
- [Benchmark plan](benchmark-plan.md): scenarios, metrics, A/B baselines, and required artifacts.
- [Migration and cleanup](migration-and-cleanup.md): legacy-name removal, grep gates, and no-shim rules.
- [Issue map](issue-map.md): how the GitHub issues map to this source-of-truth plan.

## Source

- [async-scheduling-epic.puml](async-scheduling-epic.puml): PlantUML source for every diagram on this page.

## Component View

![Component view](AsyncSchedulingEpicComponent.png)

## Task Admission Contracts

![Task admission class model](AsyncSchedulingTaskAdmissionClassModel.png)

## Request Admission Contracts

![Request admission class model](AsyncSchedulingRequestAdmissionClassModel.png)

## Capacity, Telemetry, and Evidence Contracts

![Support contracts class model](AsyncSchedulingSupportContractsClassModel.png)

## Runtime Sequence

![Runtime sequence](AsyncSchedulingEpicRuntimeSequence.png)

## Issue Dependency Map

![Issue dependency map](AsyncSchedulingEpicIssueMap.png)

## Render

```bash
plantuml plans/645/async-scheduling-epic.puml
```

The expected runtime control owner is `AsyncTaskScheduler`:

```text
SchedulingMetadata -> TaskSchedulingResolver -> CompletionTracker
AsyncTaskScheduler -> FairTaskQueue.select_next(...)
AsyncTaskScheduler -> TaskAdmissionController.try_acquire(...)
AsyncTaskScheduler -> ModelRequestExecutor -> RequestAdmissionController -> provider/model endpoint
```

Task admission and request admission each have explicit controller, queue, policy, and lease/state boundaries where applicable. Telemetry observes scheduler admission and request admission separately, then issue 648 correlates the two timelines through the runtime correlation provider.
