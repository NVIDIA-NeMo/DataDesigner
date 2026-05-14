# Async Scheduling Epic UML

Reference diagrams for the async scheduling epic tracked by issue 645. These diagrams are intentionally design artifacts for future implementation PRs, not runtime documentation.

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
