# Task Admission

Task admission controls when dependency-ready dataset work may become a running worker. It is scheduler-level admission, not provider/model request admission.

## Control Owner

`AsyncTaskScheduler` is the control owner. Its dispatch loop follows this shape:

```python
selection = queue.select_next(admission.is_eligible)
if selection is None:
    wait_for_wake()
    return

lease = admission.try_acquire(selection.item, queue.view())
if lease is None:
    wait_for_wake()
    return

committed = queue.commit(selection)
if committed is None:
    admission.release(lease)
    wake_dispatch_loop()
    return

spawn_worker(committed, lease)
```

`FairTaskQueue` selects candidates. `TaskAdmissionController` leases scheduler resources. The scheduler coordinates both.

## Queue Semantics

`FairTaskQueue` owns ready-work ordering only.

Rules:

- `select_next(...)` is non-mutating.
- `select_next(...)` calls the eligibility callback with candidates and queue view.
- `QueueSelection` returns to `AsyncTaskScheduler`.
- `commit(selection)` removes the selected task and advances queue state.
- The queue does not track admitted/running counts after this epic.
- The queue does not inspect model registries, provider pressure, or request-admission state.

## Admission Semantics

`TaskAdmissionController` owns:

- scheduler-resource availability
- task-stage leases
- admitted/running resource counts
- per-group accounting used by policy
- release on every worker terminal path

`TaskAdmissionPolicy` owns:

- eligibility decisions
- acquisition/release policy callbacks
- strict fair admission
- bounded-borrow behavior
- future resource-vector policy decisions

`TaskAdmissionController` consumes `SchedulableTask`, `SchedulerResourceRequest`, `QueueView`, and `TaskAdmissionView`. It must not inspect `ColumnGenerator`, config layout, model registry, or provider registry directly.

## V1 Lease Boundary

The first task-admission implementation is lease-only and behavior-preserving. It centralizes resource ownership without changing fairness policy beyond what is required to eliminate hidden waiters and make root work visible.

V1 includes:

- submission capacity for scheduler-spawned work
- task-stage LLM-wait style resource if a distinct scheduler-stage resource remains
- current per-group admitted/running cap behavior

V1 excludes:

- row-group admission
- concrete provider/model/domain request admission
- public runtime knobs
- distributed scheduling
- token budgets
- provider retry and AIMD behavior

## Root And From-Scratch Work

Root/from-scratch tasks must become `SchedulableTask`s and enter the same `FairTaskQueue` as downstream ready tasks. They must acquire scheduler-level leases through `TaskAdmissionController`.

No root dispatch path should bypass:

- ready queue membership
- queue selection
- task admission
- lease release accounting
- scheduler admission telemetry

This is required for heavy-root live-traffic evidence and later bounded-borrow policy.

## Resource Handoff

Resource-bound work must not become a spawned worker that waits for scheduler-level resources. The lease is acquired before spawn.

Non-resource-bound work holds the relevant scheduler lease until worker completion. Resource-bound work holds the scheduler resource lease that represents the V1 task-stage resource request. Durable `needs_llm_wait` and `held_llm_wait` fields are not part of the target architecture.

## Bounded Borrow Policy

`BoundedBorrowTaskAdmissionPolicy` is the first behavior-changing follow-up after the lease boundary. It limits how far one group may borrow ahead while no peer group is queued.

Policy inputs:

- `QueueView`: queued counts and peer pressure.
- `TaskAdmissionView`: leased/running counts and resources available.
- `TaskGroupSpec`: group key and weight.
- candidate `SchedulerResourceRequest`.

Policy constraints:

- Single-group workloads remain live.
- A solo heavy group may borrow only to its borrow ceiling.
- When peer queue pressure exists and a group has borrow debt, that group receives no further admissions while an eligible peer has queued work and the required resource is available.
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
