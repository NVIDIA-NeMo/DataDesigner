# Async Scheduling Architecture

This plan moves Data Designer's async engine from implicit scheduling behavior to explicit, layered admission control. The target architecture separates static generator resource metadata, dependency readiness, ready-work ordering, scheduler-level task admission, concrete model-request admission, capacity diagnostics, and runtime observability.

The guiding rule is: each layer owns one question and speaks through typed boundaries.

## Source Of Truth

The Markdown files in `plans/645` are the source of truth for this epic. The UML in [async-scheduling-epic.puml](async-scheduling-epic.puml) is the visual index and must be kept aligned with these files. GitHub issues should reference this plan and own implementation sequencing, validation commands, acceptance gates, and PR-level evidence.

## Target Shape

The durable runtime flow is:

```text
ColumnGenerator / plugin
  -> SchedulingMetadata
  -> TaskSchedulingResolver
  -> CompletionTracker
  -> AsyncTaskScheduler
  -> FairTaskQueue.select_next(TaskAdmissionController.is_eligible)
  -> TaskAdmissionController.try_acquire(selection.item)
  -> TaskAdmissionLease
  -> ModelRequestExecutor
  -> RequestAdmissionController.acquire_async(RequestAdmissionItem)
  -> RequestAdmissionLease
  -> provider/model endpoint
```

This is not a passive pipeline where `FairTaskQueue` or `TaskAdmissionController` pushes work into the scheduler. `AsyncTaskScheduler` is the execution owner. It asks the readiness tracker for work, asks the queue to select a candidate through an admission eligibility callback, asks the task admission controller for a lease, commits the queue selection, executes the task, and releases the lease.

## Layer Responsibilities

`SchedulingMetadata` is a generator-facing static resource declaration. It describes the resource shape a generator expects, such as local work or model-backed work. It does not expose queue internals, admitted limits, request domains, AIMD state, or runtime pressure.

`TaskSchedulingResolver` is the internal bridge from generator metadata to scheduler inputs. It produces `ResolvedTaskScheduling`, including `TaskGroupSpec` and `SchedulerResourceRequest`, and appends scheduler-owned flow identity such as output columns. It is the only scheduler grouping bridge once the legacy resolver is removed.

`CompletionTracker` owns dependency readiness. It reports the ready frontier and completion state. It does not order ready work, admit resources, or inspect provider/model pressure.

`FairTaskQueue` owns ready-work membership and ordering. Its selection operation is non-mutating and takes an eligibility callback supplied by scheduler admission. It does not own dependency readiness, admitted counts, provider metadata, request admission, or policy state.

`TaskAdmissionController` owns scheduler-level task leases and resource accounting. `TaskAdmissionPolicy` decides whether a queued task is eligible under the current queue and admission views. The controller consumes resolved scheduler inputs and must not inspect generators, configs, model registries, or provider registries directly.

`AsyncTaskScheduler` owns runtime control flow. It wires readiness, queue selection, task admission, worker spawn, task execution, salvage/retry behavior, shutdown, and lease release.

`ModelRequestExecutor` is the durable model-call boundary. It maps each concrete provider/model/domain call to a `RequestAdmissionItem`, acquires a request lease, calls the provider, records request timing, and releases that exact lease on success, rate limit, failure, cancellation, or unexpected exception.

`RequestAdmissionController` owns request-level provider/model/domain admission. `AdaptiveRequestAdmissionController` is the V1 AIMD-backed implementation. Internal `RequestFairQueue`, `RequestAdmissionPolicy`, and `AdaptiveRequestLimitState` are implementation components of this controller, not a second public layer.

`SchedulerAdmissionEventSink` and `RequestAdmissionEventSink` observe their own layers separately. `RuntimeCorrelationProvider` supplies shared runtime context, and `CorrelatedRuntimeView` joins timelines without collapsing the two telemetry systems.

## Two-Stage Admission

Task admission controls when ready dataset work may become a running worker. Request admission controls concrete provider/model/domain calls at the moment they are made.

The split is required because arbitrary custom Python can make zero, one, or many model calls dynamically. A task's metadata may help group and schedule the task, but it is not a promise of exact request count and must not reserve every future model call up front.

Task admission may later consume request pressure snapshots as read-only policy input. It must not pre-acquire request permits, emulate AIMD, or wrap provider/model/domain request admission.

## Core Invariants

- Scheduler-level work is not spawned until `TaskAdmissionController` returns a `TaskAdmissionLease`.
- `FairTaskQueue.select_next(...)` does not remove work or mutate virtual-time state. `commit(selection)` is the only queue operation that removes the selected task.
- If `try_acquire(...)` succeeds but `commit(selection)` fails, the scheduler releases the task lease before retrying.
- Every task lease and request lease is released exactly once in all success, failure, retry, cancellation, shutdown, and salvage paths.
- Root/from-scratch work uses the same ready queue and task-admission path as downstream work.
- Request admission happens only at concrete model-call time through `ModelRequestExecutor`.
- Scheduler telemetry and request telemetry remain independently useful when the other subsystem is disabled.
- Capacity and benchmark artifacts must distinguish dependency readiness, ready ordering, scheduler admission wait, request admission wait, provider execution, cooldown/rate-limit behavior, and task completion.

## Non-Goals

- Do not collapse task admission and request admission into one subsystem.
- Do not expose scheduler internals as plugin API.
- Do not put provider retry, cooldown, or AIMD behavior into `AsyncTaskScheduler` or `TaskAdmissionController`.
- Do not put DAG readiness, row-group lifecycle, or task ordering into `RequestAdmissionController`.
- Do not configure OpenTelemetry SDKs or exporters in core runtime.
- Do not add public capacity knobs before benchmark evidence and docs justify them.
- Do not keep durable compatibility shims or aliases for replaced scheduler/request-admission names at epic completion.
