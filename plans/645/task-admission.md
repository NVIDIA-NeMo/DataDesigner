# Task Admission

Task admission controls when dependency-ready dataset work may become a running worker. It is scheduler-level admission, not provider/model request admission.

## Control Owner

`AsyncTaskScheduler` is the control owner. Its dispatch loop follows this shape:

```python
selection = queue.select_next(lambda item, view: admission.is_eligible(item, view))
if selection is None:
    block_summary = admission.explain_blocked(queue.view())
    emit_queue_empty_or_blocked(block_summary)
    wait_for_wake_or_deadline(block_summary.available_after)
    return

decision = admission.try_acquire(selection.item, selection.queue_view)
if isinstance(decision, TaskAdmissionDenied):
    emit_admission_denied(decision)
    wake_dispatch_loop()
    return
lease = decision

committed = queue.commit(selection)
if committed is None:
    admission.release(lease)
    emit_stale_selection(selection, lease)
    wake_dispatch_loop()
    return

try:
    spawn_worker(committed, lease)
except Exception:
    admission.release(lease)
    emit_worker_spawn_failed(committed, lease)
    raise
```

`FairTaskQueue` selects candidates. `TaskAdmissionController` leases scheduler resources. The scheduler coordinates both.

V1 requires a scheduler dispatch mutex around `select_next -> try_acquire -> commit`. No concurrent dispatch iteration may acquire resources for the same selected task. `QueueSelection` still carries a queue version so `commit(selection)` can detect stale selections defensively. If `commit(selection)` fails because the selection is stale, the scheduler releases the exact task lease before retrying and emits a stale-selection event.

Wakeups are required when ready work is enqueued, task admission capacity is released, policy state changes from denied to eligible, an `available_after` deadline expires, shutdown/cancellation is requested, or a stale selection is detected. The implementation must avoid lost wakeups: a sleeper cannot remain asleep while queued work is eligible and task capacity is available.

Lock ordering is part of the contract: the scheduler dispatch mutex coordinates the sequence, but `FairTaskQueue` must not hold queue-internal locks while invoking the scheduler eligibility predicate, and event sinks must not be called while queue or controller locks are held.

## Queue Semantics

`FairTaskQueue` owns ready-work ordering only.

Rules:

- `select_next(...)` is non-mutating.
- `select_next(...)` calls the eligibility callback with candidates and queue view.
- `QueueSelection` returns to `AsyncTaskScheduler`.
- `QueueSelection` carries the queue view/version used to evaluate the candidate.
- `enqueue(...)` returns the accepted task ids; duplicate task ids are accepted idempotently and do not create duplicate queue entries.
- `commit(selection)` removes the selected task and advances queue state.
- The queue does not track admitted/running counts after this epic.
- The queue does not inspect model registries, provider pressure, or request-admission state.
- The queue may scan ready candidates to find the next eligible task, but eligibility is computed only through the scheduler-supplied predicate.

`QueueView` must be strong enough for current strict fairness and future bounded borrow without policy traversal of queue internals. It includes queued counts by group, queued demand by group/resource, and first-candidate resources by group. It does not report admission-aware eligibility. `TaskAdmissionPolicy` computes whether a peer is eligible for a currently available resource by combining `QueueView` with `TaskAdmissionView`.

## Admission Semantics

`TaskAdmissionController` owns:

- scheduler-resource availability
- task-stage leases
- admitted/running resource counts
- per-group accounting used by policy
- release on every worker terminal path
- rollback of acquired resources when the scheduler reports stale queue commit
- the authoritative hard resource ledger; policy debt is stored separately and affects eligibility without changing resource availability counters directly

`TaskAdmissionPolicy` owns:

- eligibility decisions
- acquisition/release policy callbacks
- strict fair admission
- bounded-borrow behavior
- future resource-vector policy decisions

`TaskAdmissionPolicy.evaluate(...)` is a pure decision function. It can be called repeatedly while the queue scans candidates and must not mutate debt, counters, timers, or diagnostics. `on_acquire(...)` and `on_release(...)` return deterministic policy state deltas. They must not directly mutate the controller's authoritative lease/resource ledger. If a policy needs borrow debt or similar mutable state, the controller applies the state transition as part of the same acquire/release transaction and exposes the resulting policy state in `TaskAdmissionView`.

Policy decisions are typed. A denied decision carries the reason used by scheduler telemetry and tests. Bounded-borrow policies return `PolicyStateDelta` values for borrow-debt increments and repayments; the controller applies those deltas atomically with the lease acquire/release path.

`TaskAdmissionController` consumes `SchedulableTask`, `SchedulerResourceRequest`, `QueueView`, and `TaskAdmissionView`. It must not inspect `ColumnGenerator`, config layout, model registry, or provider registry directly.

## V1 Lease Boundary

The first task-admission implementation is lease-only and behavior-preserving. It centralizes resource ownership without changing fairness policy beyond what is required to eliminate hidden waiters and make root work visible.

V1 includes:

- submission capacity for scheduler-spawned work
- task-stage LLM-wait style resource if a distinct scheduler-stage resource remains
- current per-group admitted/running cap behavior
- typed `TaskAdmissionDecision` denial reasons for telemetry, tests, and benchmarks
- unique task lease identities so duplicate, stale, or wrong-controller releases are rejected or diagnosed

V1 request waits remain inside admitted task execution and the task lease is retained until worker completion. That preserves the lease boundary and makes request waits visible, but it does not by itself solve cross-provider utilization when tasks for a cooled-down provider occupy all scheduler task slots. Issue #651 must address provider/resource-aware task admission or an explicit yield/reacquire design before the epic claims cross-provider scheduling optimization as complete.

V1 excludes:

- row-group admission
- concrete provider/model/domain request admission
- public runtime knobs
- distributed scheduling
- token budgets
- provider retry and AIMD behavior

## Root And From-Scratch Work

Root/from-scratch tasks must become `SchedulableTask`s and enter the same `FairTaskQueue` as downstream ready tasks. They must acquire scheduler-level leases through `TaskAdmissionController`.

Initial root materialization is owned by `AsyncTaskScheduler`. `CompletionTracker.ready_frontier()` reports dependency-ready root tasks to the scheduler; the scheduler enqueues them into `FairTaskQueue` through the same path used for downstream work. `CompletionTracker` must not enqueue directly into `FairTaskQueue`.

Readiness handoff is idempotent:

- every `SchedulableTask` has a stable `task_id`
- `ready_frontier()` returns tasks that are ready and not yet acknowledged as enqueued
- `FairTaskQueue.enqueue(...)` is idempotent by `task_id`
- after enqueue succeeds, the scheduler calls `CompletionTracker.mark_enqueued(task_ids)`
- `CompletionTracker.mark_complete(task)` closes the task only after the scheduler records the terminal outcome

No root dispatch path should bypass:

- ready queue membership
- queue selection
- task admission
- lease release accounting
- scheduler admission telemetry

This is required for heavy-root live-traffic evidence and later bounded-borrow policy.

## Resource Handoff

Resource-bound work must not become a spawned worker that waits for scheduler-level resources. The lease is acquired before spawn.

Non-resource-bound work holds the relevant scheduler lease until worker completion. Resource-bound work holds the scheduler resource lease that represents the V1 task-stage resource request. Legacy hidden-wait booleans are not part of the target architecture.

## Lease Lifecycle

Every admitted task has one `TaskAdmissionLease` with a unique lease id. The scheduler releases that exact lease in a terminal `finally` path for success, retryable failure, non-retryable failure, cancellation, shutdown, salvage redispatch, and worker-spawn failure.

Release rules:

- release returns `ReleaseResult` and must not raise from terminal `finally` paths
- duplicate release must not increment capacity
- releasing a stale lease or a lease from another controller generation returns a diagnostic release result and emits an error event
- stale queue commit releases the task lease before any worker is spawned
- salvage/retry may make replacement work visible only after the original lease terminal path is accounted for. Replacement work is recorded through `CompletionTracker` or an explicit retry tracker, then re-enters the normal `ready_frontier() -> enqueue -> mark_enqueued` handoff; it must not be inserted directly into `FairTaskQueue` while the original lease is active.
- task release wakes the dispatch loop if queued work may now be eligible

## Bounded Borrow Policy

`BoundedBorrowTaskAdmissionPolicy` is the first behavior-changing follow-up after the lease boundary. It limits how far one group may borrow ahead while no peer group is queued.

Policy inputs:

- `QueueView`: queued counts and queued resource demand.
- `TaskAdmissionView`: resource limits, availability, leased/running counts, and policy debt by group/resource.
- `TaskGroupSpec`: group key and weight.
- candidate `SchedulerResourceRequest`.
- engine-internal `BoundedBorrowTaskAdmissionPolicyConfig` when enabled by #650, including borrow ceiling by group/resource, strict-share rounding mode, and repayment behavior.

Policy constraints:

- Single-group workloads remain live.
- Borrow debt is measured in admitted scheduler-resource units above strict fair share for a group/resource. Strict share is computed from scheduler-known competing groups and their weights; #650 owns the exact rounding rule and benchmark evidence.
- A group may borrow beyond strict share only up to its configured ceiling while no eligible peer can use the resource.
- When peer queue pressure exists and a group has borrow debt, that group receives no further admissions for the borrowed resource while an eligible peer has queued work and the required resource is available.
- Debt repayment happens when peer pressure exists and the indebted group is withheld, or when policy-defined repayment work completes. Repayment changes policy debt counters only, not hard resource availability.
- The policy must not traverse the DAG inside `FairTaskQueue`.
- No public knob is added until benchmark evidence supports it.

## Resource-Vector Direction

Future policy work may use `SchedulerResourceKey` and `SchedulerResourceRequest` for multi-resource admission. Candidate resources include submission, LLM-wait, provider/model task-stage hints, local resources, and GPU slots if reliable metadata exists.

Resource-vector policy must:

- remain scheduler-internal unless a later design explicitly changes public metadata fields
- consume resolved metadata from `TaskSchedulingResolver`
- avoid duplicating provider/model/domain AIMD request admission
- use `RequestPressureSnapshotProvider` only as read-only pressure input
- preserve single-resource and single-group liveness
- produce benchmark evidence through the benchmark harness
