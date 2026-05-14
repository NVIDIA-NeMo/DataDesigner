# Async Scheduling Architecture

This plan moves Data Designer's async engine from implicit scheduling behavior to explicit, layered admission control. The target architecture separates static generator resource metadata, dependency readiness, ready-work ordering, scheduler-level task admission, concrete model-request admission, capacity diagnostics, and runtime observability.

The guiding rule is: each layer owns one question and speaks through typed boundaries.

## Source Of Truth

The Markdown files in `plans/645` are the source of truth for this epic. The UML in [async-scheduling-epic.puml](async-scheduling-epic.puml) is the visual index and must be kept aligned with these files. GitHub issues should reference this plan and own implementation sequencing, validation commands, acceptance gates, and PR-level evidence.

## Target Shape

The durable data-preparation flow is:

```text
ColumnGenerator / plugin
  -> ColumnGenerator.get_scheduling_metadata()
  -> SchedulingMetadata
  -> TaskSchedulingResolver
  -> ResolvedTaskScheduling
  -> SchedulableTask inputs
```

The durable runtime control flow is:

```text
AsyncTaskScheduler
  -> CompletionTracker.ready_frontier()
  -> FairTaskQueue.enqueue(...)
  -> FairTaskQueue.select_next(scheduler-owned eligibility callback)
  -> TaskAdmissionController.try_acquire(selection.item, queue_view)
  -> FairTaskQueue.commit(selection)
  -> execute admitted task/generator code

admitted task/generator code
  -> model facade/provider boundary
  -> ModelRequestExecutor.execute_attempt(...)
  -> RequestAdmissionController.acquire_async(RequestAdmissionItem)
  -> provider/model endpoint
  -> RequestAdmissionController.release(lease, outcome)
```

This is not a passive pipeline where `CompletionTracker`, `FairTaskQueue`, or `TaskAdmissionController` pushes work into the scheduler. `AsyncTaskScheduler` is the execution owner. It asks the readiness tracker for work, enqueues ready tasks, asks the queue to select a candidate through an admission eligibility callback, asks the task admission controller for a lease, commits the queue selection, executes the admitted task, and releases the lease.

`ModelRequestExecutor` is not a scheduler task wrapper. It is reached only when admitted task/generator code makes a concrete model call through the model facade/provider boundary. A task may make zero, one, or many concrete calls; each call attempt receives request admission independently.

## Layer Responsibilities

`SchedulingMetadata` is a generator-facing static resource declaration. It describes the resource shape a generator expects, such as local work or model-backed work. It does not expose queue internals, admitted limits, request domains, AIMD state, or runtime pressure.

`TaskSchedulingResolver` is the internal bridge from generator metadata to scheduler inputs. It produces `ResolvedTaskScheduling`, including `TaskGroupSpec` and `SchedulerResourceRequest`, and appends scheduler-owned flow identity such as output columns. It is the only scheduler grouping bridge once the legacy resolver is removed.

`CompletionTracker` owns dependency readiness. It reports the ready frontier and completion state to `AsyncTaskScheduler`. It does not enqueue into the ready queue, order ready work, admit resources, or inspect provider/model pressure.

`FairTaskQueue` owns ready-work membership and ordering. Its selection operation is non-mutating and takes an eligibility callback supplied by scheduler admission. It does not own dependency readiness, admitted counts, provider metadata, request admission, or policy state.

`TaskAdmissionController` owns scheduler-level task leases and resource accounting. `TaskAdmissionPolicy` decides whether a queued task is eligible under the current queue and admission views. The controller consumes resolved scheduler inputs and its engine-internal `TaskAdmissionConfig`; it must not inspect generators, user config layout, model registries, or provider registries directly.

`AsyncTaskScheduler` owns runtime control flow. It wires readiness, queue selection, task admission, worker spawn, task execution, salvage/retry behavior, shutdown, and lease release.

`ModelRequestExecutor` is the durable model-call boundary. It maps each concrete provider/model/domain call attempt to a `RequestAdmissionItem`, acquires a request lease, calls the provider, records request timing, and releases that exact lease with a classified outcome on success, rate limit, failure, cancellation, timeout, or unexpected exception.

`RequestAdmissionController` owns request-level provider/model/domain admission. `AdaptiveRequestAdmissionController` is the V1 AIMD-backed implementation. Internal `RequestFairQueue`, `RequestAdmissionPolicy`, and `AdaptiveRequestLimitState` are implementation components of this controller, not a second public layer.

`SchedulerAdmissionEventSink` and `RequestAdmissionEventSink` observe their own layers separately. `RuntimeCorrelationProvider` supplies shared runtime context, and `CorrelatedRuntimeView` joins timelines without collapsing the two telemetry systems.

## Audience And API Boundaries

The plan uses several contract categories. Keeping them separate prevents internal scheduling mechanics from becoming accidental plugin API.

| Audience | Durable surface | Must not expose |
| --- | --- | --- |
| Plugin authors | `ColumnGenerator.get_scheduling_metadata()` and `SchedulingMetadata` | queue state, task leases, request domains, AIMD state, runtime pressure |
| Users/operators | documented run config fields, `AsyncCapacityPlan`, benchmark and telemetry artifacts | internal queue/policy classes, per-lease mutation APIs |
| Engine implementers | scheduler/request admission protocols, DTOs, policies, snapshots, events | config-layer imports from engine runtime |
| Diagnostics and benchmarks | event DTOs, snapshots, correlation view, capacity plan | prompts, completions, row data, secrets, unbounded IDs as metric labels |

Package ownership follows Data Designer's structural layering:

| Package | Owns |
| --- | --- |
| `data-designer-config` | public configuration DTOs and generator-facing metadata, including `SchedulingMetadata`, metadata validation errors, and future stable config surfaces only after an issue explicitly promotes them to public API |
| `data-designer-engine` | scheduler runtime DTOs, task/request admission controllers, queues, policies, leases, snapshots, events, capacity plan construction, and benchmark harness internals |
| `data-designer` | public interface wiring, CLI/operator presentation, and integration docs; it may consume engine/config contracts but must not make engine internals plugin API |

When a contract is shared across packages, the lower package owns the data definition and the higher package owns presentation or orchestration. Engine code may import config contracts; config code must not import engine runtime protocols.

## Two-Stage Admission

Task admission controls when ready dataset work may become a running worker. Request admission controls concrete provider/model/domain calls at the moment they are made.

The split is required because arbitrary custom Python can make zero, one, or many model calls dynamically. A task's metadata may help group and schedule the task, but it is not a promise of exact request count and must not reserve every future model call up front.

Task admission may later consume request pressure snapshots as read-only policy input. It must not pre-acquire request permits, emulate AIMD, or wrap provider/model/domain request admission.

In V1, a task waiting inside request admission keeps its scheduler task lease until the task reaches a terminal outcome. This makes request wait visible without adding yield/reacquire complexity to the lease boundary. The cross-provider optimization target, where tasks blocked on one cooled-down provider do not occupy every scheduler slot while another provider has ready work, belongs to #651's provider/resource-aware task policy or an explicit later yield/reacquire design.

## Core Invariants

- Scheduler-level work is not spawned until `TaskAdmissionController` returns a `TaskAdmissionLease`.
- `FairTaskQueue.select_next(...)` does not remove work or mutate virtual-time state. `commit(selection)` is the only queue operation that removes the selected task.
- `select_next(...)`, `try_acquire(...)`, and `commit(selection)` are coordinated by `AsyncTaskScheduler` under a single dispatch critical section or an equivalent versioned-selection protocol.
- If `try_acquire(...)` succeeds but `commit(selection)` fails, the scheduler releases the task lease before retrying.
- Every task lease and request lease is released exactly once in all success, failure, retry, cancellation, shutdown, and salvage paths.
- Root/from-scratch work uses the same ready queue and task-admission path as downstream work.
- Request admission happens only at concrete model-call time through `ModelRequestExecutor`.
- Provider retries are visible to request admission: each outbound attempt either re-enters `ModelRequestExecutor` or is owned by a retry loop inside it that acquires/releases per attempt.
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
