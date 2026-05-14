# Contracts

This file records the durable names and semantics used by the async scheduling architecture. The exact implementation can evolve, but these names are the spec vocabulary for the epic.

## Metadata Contracts

`ColumnGenerator.get_scheduling_metadata()` returns generator-facing scheduling metadata. It is additive and non-abstract so existing generators keep working.

`SchedulingMetadata` is a static declaration with:

- `kind`: initial values are `local`, `model`, and `custom_model`.
- `identity`: deterministic tuple of broad-to-specific resource identity values.
- `weight`: positive static capacity hint.

`SchedulingMetadataError` is the typed failure path for metadata resolution. It can carry fallback metadata when partial resolution is safe.

Rules:

- Metadata identity is resource identity, not a queue key.
- Metadata cannot encode queue depth, admitted limits, runtime pressure, request domains, AIMD state, or provider cooldown.
- Multi-alias metadata deduplicates aliases that resolve to the same provider/model/generation resource before summing weight.
- Alias ordering is canonicalized so equivalent configs produce equivalent metadata.

## Scheduler Input Contracts

`TaskSchedulingResolver` consumes `SchedulingMetadata` and produces scheduler-internal inputs. It owns per-run metadata caching and scheduler flow-identity composition.

`ResolvedTaskScheduling` contains:

- `group: TaskGroupSpec`
- `resource_request: SchedulerResourceRequest`

`TaskGroupSpec` contains a scheduler-internal task group key and static weight.

`SchedulerResourceRequest` contains scheduler-level task-stage resources:

```text
amounts: Mapping[SchedulerResourceKey, int]
```

The first implementation can model submission and LLM-wait style resources. Future resource-vector work may add provider/model, local, GPU, or other scheduler resources, but those remain scheduler-internal unless a later design explicitly changes the public contract.

`SchedulableTask` contains:

- task payload
- task group
- scheduler resource request

## Queue Contracts

`FairTaskQueue` owns ready-task membership and ready ordering:

```text
enqueue(item)
select_next(is_eligible) -> QueueSelection | None
commit(selection) -> SchedulableTask | None
view() -> QueueView
```

`QueueSelection` returns from `FairTaskQueue` to `AsyncTaskScheduler`. It is not delivered to `TaskAdmissionController`.

`QueueSelection` contains the selected item and an opaque sequence/version used by `commit(selection)` to detect stale selections.

`QueueView` is read-only policy input. It exposes queued totals and queued counts by group, including whether a group has queued peers.

## Task Admission Contracts

`TaskAdmissionController` owns task-stage resource accounting and leases:

```text
is_eligible(item, queue_view) -> bool
try_acquire(item, queue_view) -> TaskAdmissionLease | None
release(lease)
view() -> TaskAdmissionView
```

`TaskAdmissionPolicy` owns the decision rule:

```text
is_eligible(item, queue_view, admission_view) -> bool
on_acquire(lease)
on_release(lease)
```

`TaskAdmissionLease` contains:

- `item`
- `resources`
- `acquired_at`

`TaskAdmissionView` exposes task-stage resource availability plus leased/running counts by group.

`TaskAdmissionDecision` is the richer decision shape for implementations that need denial reasons. V1 may expose `try_acquire(...)` as a non-blocking `TaskAdmissionLease | None` helper, but the spec vocabulary treats denials as typed scheduler-admission outcomes for events, tests, and debugging.

`TaskAdmissionConfig` contains scheduler task-stage capacity values such as `submission_capacity` and resource limits.

## Request Admission Contracts

`ModelRequestExecutor` maps concrete model calls into request-admission items and owns exact lease release around provider execution.

`RequestResourceKey` identifies a concrete provider/model/domain request resource:

- `provider_name`
- `model_id`
- `domain`

`RequestGroupSpec` contains the request resource key and static weight.

`RequestAdmissionItem` contains:

- request resource
- request group
- optional timeout

`RequestAdmissionController` owns request-level admission:

```text
try_acquire(item) -> RequestAdmissionDecision
acquire_sync(item) -> RequestAdmissionLease
acquire_async(item) -> RequestAdmissionLease
release_success(lease)
release_rate_limited(lease, retry_after)
release_failure(lease)
pressure -> RequestPressureSnapshotProvider
```

`RequestAdmissionDecision` is a union of `RequestAdmissionLease` and `RequestAdmissionDenied`.

`RequestAdmissionLease` contains:

- item
- acquired timestamp
- current adaptive limit
- effective max

`RequestAdmissionDenied` contains:

- item
- reason
- retry-after or available-after timing where applicable
- optional snapshot

`AdaptiveRequestAdmissionController` is the V1 concrete request controller. It owns AIMD behavior through internal `RequestFairQueue`, `RequestAdmissionPolicy`, and `AdaptiveRequestLimitState`.

`RequestPressureSnapshotProvider` exposes read-only request pressure:

```text
snapshot(resource)
snapshots()
global_snapshot(provider, model)
global_snapshots()
```

It has no mutation or admission methods.

`RequestAdmissionConfig` is the durable request-admission tuning/config vocabulary. `ThrottleConfig` and `RunConfig.throttle` are not durable names.

## Telemetry And Correlation Contracts

`SchedulerAdmissionEventSink` emits scheduler admission events.

`RequestAdmissionEventSink` emits request admission events.

`RuntimeCorrelation` contains primitive runtime context:

- run id
- row group
- task column
- task type
- scheduling group kind
- scheduling group identity hash

`RuntimeCorrelationProvider` owns set/reset/current behavior, likely through context variables. It must not require request admission protocols to import scheduler types.

`CorrelatedRuntimeView` joins scheduler and request timelines for diagnostics, benchmarks, and future operator views.

## Capacity Contracts

`AsyncCapacityPlan` records computed per-run capacity values:

- buffer size
- row-group concurrency
- submission capacity
- task admission config
- request admission config
- static provider/model caps when available
- source of each value

The capacity plan explains observed runtime behavior. It is not itself a policy engine.
