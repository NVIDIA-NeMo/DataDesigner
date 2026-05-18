# Contracts

This file records the durable names and semantics used by the async scheduling architecture. Implementation details inside the owning target modules can evolve, but these names are the normative spec vocabulary for the epic. Topic files may explain behavior, but should not redefine fields or return shapes in ways that conflict with this file.

Durable names in this file are not public API by default. Publicness and final module homes are defined in [Module ownership](module-ownership.md).

## Package Ownership

| Contract family | Owning package | Notes |
| --- | --- | --- |
| Generator metadata and public config DTOs | `data-designer-config` | `SchedulingMetadata`, metadata validation errors, and exposed run-config fields live here when they are public/user-facing. |
| Scheduler/request runtime protocols | `data-designer-engine` | queues, controllers, policies, leases, runtime snapshots, event DTOs, capacity plan construction, and benchmark internals live here. |
| User interface and operator presentation | `data-designer` | consumes config and engine contracts for the public `DataDesigner` interface, CLI, and integrations. |

Config-layer contracts must not import engine runtime protocols. Engine contracts may consume config-layer DTOs.

The final repository layout is specified in [Module ownership](module-ownership.md). Runtime contracts must live in their owning target modules; do not preserve old module paths through aliases, shim files, or broad package reexports.

## Config Surface Status

| Contract | V1 status | Owner |
| --- | --- | --- |
| `SchedulingMetadata` | public plugin-facing DTO | `data-designer-config` |
| `TaskAdmissionConfig` | engine-internal config and benchmark injection surface; not a public `RunConfig` knob in V1 | `data-designer-engine` |
| `RequestAdmissionConfig` | engine-internal config and benchmark injection surface in V1 | `data-designer-engine` |
| `RunConfig.request_admission` | reserved future public spelling if evidence justifies a user-facing request-admission knob | not introduced by this epic unless a later issue explicitly changes the public API |
| `AsyncCapacityPlan` | diagnostic/reporting DTO, emitted to explain a run | `data-designer-engine` |

Public knobs are not added by the task-admission or request-admission implementation slices. Benchmarks may inject capacity values through harness-only configuration so policy evidence can be produced without committing to public API.

## Metadata Contracts

`ColumnGenerator.get_scheduling_metadata()` returns generator-facing scheduling metadata. It is additive and non-abstract so existing generators keep working.

`SchedulingMetadata` is a static declaration with:

- `kind`: initial values are `local`, `model`, and `custom_model`.
- `identity`: deterministic tuple of broad-to-specific resource identity values.
- `weight`: positive static capacity hint.

`SchedulingMetadataError` is the typed failure path for metadata resolution. It can carry fallback metadata when partial resolution is safe. The documented default metadata for generators that do not override `get_scheduling_metadata()` is a normal resolver path, not an error fallback.

Rules:

- Metadata identity is resource identity, not a queue key.
- Metadata cannot encode queue depth, admitted limits, runtime pressure, request domains, AIMD state, or provider cooldown.
- Multi-alias metadata deduplicates aliases that resolve to the same provider/model/generation resource before summing weight.
- Alias ordering is canonicalized so equivalent configs produce equivalent metadata.
- Generators that do not override `get_scheduling_metadata()` receive a documented default metadata value. The default must preserve current behavior and must not infer provider/model pressure dynamically.
- Invalid `kind`, non-deterministic `identity`, non-positive `weight`, or ambiguous alias resolution raises `SchedulingMetadataError`.
- Differing `max_parallel_requests` values for aliases that resolve to the same concrete provider/model endpoint are not, by themselves, ambiguous. They merge through the static-cap min rule in the capacity model. Alias ambiguity is fatal when aliases cannot be resolved to one concrete endpoint, resolve to conflicting endpoint identities, or would require hidden dynamic inference to choose a resource.
- Fallback metadata is safe only when it preserves current scheduling behavior and the resolver can explain the fallback in diagnostics. Ambiguous resource identity or invalid weights are fatal.

Normative V1 metadata shapes:

| Kind | Identity tuple | Weight source | Default/fallback behavior |
| --- | --- | --- | --- |
| `local` | `("local", resource_name)` where `resource_name` defaults to `"default"` | positive integer, default `1` | the default for generators that do not override `get_scheduling_metadata()` is `SchedulingMetadata(kind="local", identity=("local", "default"), weight=1)` |
| `model` | `("model", provider_name, canonical_model_id, generation_kind)` after alias resolution | effective static provider/model capacity hint, normally derived from the model config's `max_parallel_requests` and clamped to at least `1` | safe fallback is allowed only when the resolver can identify the same canonical provider/model resource as the current implementation |
| `custom_model` | `("custom_model", plugin_namespace, resource_name, version)` with deterministic plugin-provided values | positive plugin-provided capacity hint, defaulting to `1` if omitted | no alias inference is performed unless the plugin returns model aliases through the documented metadata API |

`SchedulingMetadataError` contains:

- `code`
- `message`
- optional `fallback: SchedulingMetadata`
- sanitized `diagnostics`

If `fallback` is present, the resolver may continue and must emit diagnostics. If `fallback` is absent, metadata resolution is fatal before scheduler inputs are created.

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

`SchedulerResourceKey` identifies a scheduler-owned task-stage resource such as `submission`, `local`, or a future internal resource-vector key. It is not a provider request-domain key.

The first implementation models scheduler task-stage pressure with explicit scheduler resources. Concrete provider/model/domain request pressure belongs to `RequestResourceKey` and request admission. Future resource-vector work may add local, GPU, or other scheduler resources, but those remain scheduler-internal unless a later design explicitly changes the public contract.

`SchedulableTask` contains:

- stable `task_id`
- task payload
- task group
- scheduler resource request

`CompletionTracker` owns readiness state:

```text
ready_frontier() -> Sequence[SchedulableTask]
mark_enqueued(task_ids)
mark_complete(task)
```

`ready_frontier()` returns dependency-ready tasks that have not yet been acknowledged as enqueued. After `FairTaskQueue.enqueue(...)` accepts a task, `AsyncTaskScheduler` calls `mark_enqueued(...)` with exactly the accepted task ids. `FairTaskQueue.enqueue(...)` is also idempotent by `task_id`, so duplicate frontier reads cannot create duplicate ready membership. If enqueue fails before acceptance, the task remains unacknowledged and appears in a later frontier read.

## Queue Contracts

`FairTaskQueue` owns ready-task membership and ready ordering:

```text
enqueue(items) -> Sequence[task_id]
select_next(is_eligible) -> QueueSelection | None
commit(selection) -> SchedulableTask | None
view() -> QueueView
```

`QueueSelection` returns from `FairTaskQueue` to `AsyncTaskScheduler`. It is not delivered to `TaskAdmissionController`.

`QueueSelection` contains the selected item, the queue view used during selection, and an opaque `sequence_version` used by `commit(selection)` to detect stale selections.

`QueueView` is read-only policy input. It exposes:

- queued totals
- queued counts by group
- queued resource demand by group and `SchedulerResourceKey`
- first-candidate resource request by group where available
- queued peer demand by resource

`QueueView` is produced by `FairTaskQueue`; policies must not traverse queue internals directly. It contains raw queued membership and demand facts only. `TaskAdmissionPolicy` computes eligibility and resource-aware peer pressure from `QueueView` plus `TaskAdmissionView`.

`FairTaskQueue` must not invoke scheduler-supplied eligibility predicates while holding internal queue locks that can be needed by enqueue, commit, release wakeups, or diagnostics. The scheduler dispatch critical section owns the cross-component coordination; queue internals remain local to queue mutation.

## Task Admission Contracts

`TaskAdmissionController` owns task-stage resource accounting and leases:

```text
is_eligible(item, queue_view) -> bool
try_acquire(item, queue_view) -> TaskAdmissionDecision
release(lease) -> ReleaseResult
view() -> TaskAdmissionView
explain_blocked(queue_view) -> TaskAdmissionBlockSummary
```

`TaskAdmissionPolicy` owns the decision rule:

```text
evaluate(item, queue_view, admission_view) -> TaskAdmissionPolicyDecision
on_acquire(lease, decision) -> PolicyStateDelta
on_release(lease) -> PolicyStateDelta
```

`evaluate(...)` is side-effect-free. It may be called while scanning queue candidates and must not mutate borrow debt, counters, timers, diagnostics, or resource ledgers. Only controller-mediated acquire/release paths apply `PolicyStateDelta` values.

`TaskAdmissionPolicyDecision` contains:

- `allowed`
- optional denial `reason`, such as no capacity, group cap, borrow debt, shutdown, or policy denial
- optional `available_after`
- sanitized diagnostic fields

`PolicyStateDelta` contains policy-owned state changes such as borrow-debt increment, repayment, or diagnostic counters. The controller applies the delta in the same transaction as lease acquire/release and exposes the resulting policy counters through `TaskAdmissionView`. Bounded-borrow debt affects eligibility, but it is not part of the hard resource ledger and never changes resource availability counters directly.

`AsyncTaskScheduler` supplies the boolean eligibility callback used by `FairTaskQueue`; that callback delegates to `TaskAdmissionController.is_eligible(...)`. The controller may call `TaskAdmissionPolicy.evaluate(...)` internally, but denial details are surfaced through `try_acquire(...)`, `explain_blocked(...)`, events, and tests rather than through the queue callback.

When `FairTaskQueue.select_next(...)` returns no selection while queued work exists, `AsyncTaskScheduler` calls `TaskAdmissionController.explain_blocked(queue_view)` before sleeping. `TaskAdmissionBlockSummary` contains queued count, dominant denial reasons, optional earliest `available_after`, and sanitized diagnostics. This is the source for `admission_blocked`, `group_capped`, and timed wakeups when no candidate can currently be admitted.

`TaskAdmissionLease` contains:

- `lease_id`
- `item`
- `resources`
- `acquired_at`
- controller identity or generation token sufficient to reject stale/wrong-controller releases

`TaskAdmissionView` exposes a consistent read-only snapshot:

- task resource limits by `SchedulerResourceKey`
- task resources available by `SchedulerResourceKey`
- leased resources by `SchedulerResourceKey`
- leased resources by group and `SchedulerResourceKey`
- running counts by group and resource where tracked
- policy-only debt by group/resource if the active policy uses bounded borrow

`TaskAdmissionDecision` is a union of `TaskAdmissionLease` and `TaskAdmissionDenied`.

`TaskAdmissionDenied` contains:

- item
- reason, such as no capacity, group cap, borrow debt, shutdown, or policy denial
- optional available-after timing
- optional `TaskAdmissionView` snapshot

Implementations may provide a local convenience helper that converts `TaskAdmissionDecision` to an optional lease, but telemetry, tests, and benchmark artifacts use the typed decision vocabulary.

`TaskAdmissionConfig` is engine-internal in V1 and contains scheduler task-stage capacity values such as `submission_capacity`, resource limits, and optional policy-specific config. Bounded-borrow policy config, when enabled by #650, includes borrow ceiling by group/resource, strict-share rounding mode, and repayment behavior. The default V1 lease-boundary policy is behavior-preserving unless #650 explicitly enables bounded borrow.

`ReleaseResult` contains:

- `released: bool`
- `reason`, such as released, duplicate, stale lease, wrong controller generation, or unknown lease
- sanitized diagnostics

Terminal `finally` paths must not raise from release. Duplicate, stale, or wrong-controller releases return `ReleaseResult` and emit diagnostic events without increasing capacity.

## Request Admission Contracts

`ModelRequestExecutor` maps concrete model-call attempts into request-admission items and owns exact lease release around provider execution:

```text
execute_attempt(request) -> provider response
```

`RequestResourceResolver` is the canonical request-resource identity factory. It maps provider alias, model alias, model id, generation kind, endpoint metadata, and `RequestDomain` into `ProviderModelKey` and `RequestResourceKey`. `TaskSchedulingResolver`, `ModelRequestExecutor`, `AsyncCapacityPlan`, and request admission all use the same provider/model canonicalization rules so alias merging, metadata weight, and request caps cannot drift.

`RequestResourceKey` identifies a concrete provider/model/domain request resource:

- `provider_name`, the canonical resolved provider name, not an alias
- `model_id`, the canonical resolved provider/model endpoint id, not a user alias
- `domain`

Aliases are recorded in capacity plans and pressure snapshots for diagnostics, but request admission keys use canonical resolved provider/model identity so aliases cannot bypass aggregate caps.

`ProviderModelKey` is the aggregate request-capacity key:

- canonical provider name
- canonical model endpoint id

`RequestResourceKey` is `ProviderModelKey + RequestDomain`.

`RequestDomain` is the durable domain vocabulary for request admission. V1 includes `chat`, `embedding`, `image`, and `healthcheck`; adding new domains requires updating this plan and the request-admission docs.

`RequestGroupSpec` contains the request fairness group key and static weight. In V1 the group key is the `RequestResourceKey`; a later design may split fairness group from resource key, but must specify the mapping before doing so.

`RequestAdmissionItem` contains:

- request resource
- request group
- optional queue-wait timeout
- optional `RequestEventContext`

`RequestEventContext` is constructed by `ModelRequestExecutor` when it maps a model call attempt into a request item. It contains primitive, telemetry-only context:

- captured `RuntimeCorrelation | None`
- `task_execution_id`
- `request_attempt_id`

The request controller treats this as opaque event context. It does not import scheduler task types or mutate scheduler state.

`RequestFairQueue` owns waiter ordering inside `AdaptiveRequestAdmissionController`:

```text
enqueue(waiter)
select_next(is_eligible) -> RequestQueueSelection | None
commit(selection) -> RequestWaiter | None
remove(waiter_id)
view() -> RequestQueueView
```

`RequestWaiter` contains waiter id, item, enqueue timestamp, deadline/cancellation state, and the waiter completion handle used by the blocking acquire path.

`RequestQueueSelection` contains the selected waiter, item, waiter id, queue view, and opaque `sequence_version` for stale-selection detection.

`RequestQueueView` exposes queued totals, queued counts by request group, queued demand by request resource, and aggregate provider/model waiters. It does not inspect adaptive limit state.

`try_acquire(...)` is non-blocking. It may immediately acquire only when the request is eligible and no queued eligible waiter for the same request resource or provider/model aggregate cap would be selected before the incoming item by `RequestFairQueue`'s weighted ordering. Otherwise it returns `RequestAdmissionDenied` with reason `queued_waiters_ahead` or another specific denial reason.

`RequestAdmissionController` owns request-level admission:

```text
try_acquire(item) -> RequestAdmissionDecision
acquire_sync(item) -> RequestAdmissionLease
acquire_async(item) -> RequestAdmissionLease
release(lease, outcome) -> ReleaseResult
pressure -> RequestPressureSnapshotProvider
```

`acquire_sync(...)` and `acquire_async(...)` wait until a lease is available or a terminal no-lease condition occurs. Timeout, shutdown, or hard denial before a lease is acquired must remove the waiter and raise a typed project error that carries the corresponding `RequestAdmissionDenied` decision. They must not return `None`.

`RequestAdmissionError` is the typed no-lease exception raised by blocking acquire paths. It wraps `RequestAdmissionDenied` and must not be raised after a lease has been returned; post-lease provider outcomes are represented by `RequestReleaseOutcome`.

`acquire_async(...)` must preserve cooperative cancellation. If the awaiting task is cancelled before a lease is acquired, the controller removes the waiter, emits a cancellation/denial event, and re-raises the cancellation exception instead of converting it to `RequestAdmissionError`.

Once a waiter is selected and in-flight counts are incremented, cancellation cannot orphan the lease. The controller either delivers the lease to that waiter's acquire call so caller cleanup can release it, or internally releases the admitted waiter as `local_cancelled` before completing cancellation. A caller's `acquire_async(item)` may only return the lease for its own waiter; if the controller admits another waiter while this caller is awake, it fulfills that other waiter's completion handle and this caller continues waiting.

`RequestAdmissionDecision` is a union of `RequestAdmissionLease` and `RequestAdmissionDenied`.

`RequestAdmissionLease` contains:

- `lease_id`
- item
- acquired timestamp
- current adaptive limit
- effective max
- controller identity or generation token sufficient to reject stale/wrong-controller releases

`RequestAdmissionDenied` contains:

- item
- reason, such as no capacity, cooldown, queue timeout, queued waiters ahead, cancellation, shutdown, or hard policy denial
- `retry_after_seconds` when supplied by the provider or policy
- `available_after_monotonic` when the controller can compute an unblock deadline
- optional snapshot

`RequestReleaseOutcome` contains:

- `kind`: one of `success`, `rate_limited`, `provider_failure`, `provider_timeout`, `local_cancelled`, `local_timeout`, or `unexpected_exception`
- `retry_after_seconds` when rate limited
- provider/status metadata safe for telemetry

Only provider rate-limit outcomes drive multiplicative decrease/cooldown. Provider failures may affect diagnostic counters. Local cancellation and local timeout release capacity and wake waiters but must not be treated as provider pressure unless a later policy explicitly defines that behavior.

`provider_timeout` is a timeout or timeout-shaped transport/provider failure after a lease has been acquired and an outbound provider attempt has started. `local_timeout` is a caller, queue-wait, or controller deadline that is not evidence of provider pressure. Cancellation after lease acquisition is classified as `local_cancelled`; release diagnostics must not mask the original cancellation and the cancellation is re-raised after accounting.

`AdaptiveRequestAdmissionController` is the V1 concrete request controller. It owns AIMD behavior through internal `RequestFairQueue`, `RequestAdmissionPolicy`, and `AdaptiveRequestLimitState`.

Request admission acquires under one controller lock/condition. An admitted lease increments domain in-flight counts and provider/model aggregate in-flight counts before the lease is returned. Release decrements those counts exactly once and wakes eligible waiters.

Cross-domain arbitration under a provider/model aggregate cap uses `RequestFairQueue` ordering by `RequestGroupSpec` weight. V1 uses weighted fair ordering across request groups sharing the aggregate cap; if weights are equal, older waiters are selected first.

V1 AIMD semantics:

- `effective_max = min(provider_model_static_cap, request_config.max_limit_clamp_for_resource_if_present)`
- instantaneous aggregate availability is checked separately as `provider_model_aggregate_in_flight < provider_model_static_cap`
- `initial_limit` is clamped to `[1, effective_max]`
- `current_limit` starts at `initial_limit`
- on `rate_limited`, `current_limit = max(1, floor(current_limit * multiplicative_decrease_factor))`, `blocked_until_monotonic` is set from provider `retry_after_seconds` when supplied or the configured cooldown otherwise, and rate-limit counters increment
- on success outside cooldown, successful releases accumulate; after `increase_after_successes` successes, `current_limit = min(effective_max, current_limit + additive_increase_step)`
- `request_soft_ceiling_recovered` fires when `current_limit` rises above the last rate-limit ceiling
- `request_fully_recovered` fires when `current_limit == effective_max` and cooldown has cleared
- all timing uses a monotonic clock
- waiters use timed waits to the earliest relevant monotonic deadline: queue-wait timeout, cancellation, `available_after_monotonic`, or `blocked_until_monotonic`. Cooldown expiry must wake queued waiters even when no in-flight request releases.

`RequestPressureSnapshotProvider` exposes read-only request pressure:

```text
snapshot(resource)
snapshots()
global_snapshot(provider, model)
global_snapshots()
```

It has no mutation or admission methods.

Snapshots are immutable and internally consistent for their capture point. Domain snapshots include `captured_at`, monotonic `sequence`, resource, effective max, current limit, in-flight count, active lease count, waiters, blocked-until timing, cooldown remaining, rate-limit ceiling, consecutive rate limits, last outcome summary, and leak diagnostic counters. Global provider/model snapshots include aggregate static cap, aggregate in-flight count across domains, aggregate active lease count, aliases contributing to the cap, and per-domain limit summaries.

`RequestAdmissionConfig` is the durable engine-internal request-admission tuning/config vocabulary for V1. It includes request resources, per-resource `initial_limit`, optional `max_limit_clamp`, configured cooldown, `multiplicative_decrease_factor`, `additive_increase_step`, `increase_after_successes`, and default queue-wait timeout. Legacy request-control config names are not durable names.

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
- task execution id

`RuntimeCorrelationProvider` owns set/reset/current behavior, likely through context variables. It must not require request admission protocols to import scheduler types.

Scheduler/request events capture primitive correlation values when event DTOs are constructed. Event sinks must not rely on reading mutable ambient context later, because deferred emission could attach the wrong task context.

Canonical scheduler `event_kind` values are snake_case and versioned as part of the benchmark artifact schema:

```text
dependency_ready
ready_enqueued
selected
queue_empty
admission_blocked
group_capped
task_lease_acquired
admission_denied
worker_spawned
worker_spawn_failed
stale_selection
retry_deferred
non_retryable_dropped
cancelled
salvage_redispatched
queue_drained
task_completed
task_lease_released
release_diagnostic
```

`SchedulerAdmissionEvent` contains:

- `event_kind`
- `captured_at_monotonic`
- monotonic `sequence`
- captured `RuntimeCorrelation | None`
- task id
- task execution id when a worker execution exists
- task lease id when available
- scheduler resource key when applicable
- decision reason or release result when applicable
- optional scheduler snapshot
- sanitized diagnostics

Canonical request `event_kind` values are snake_case and versioned as part of the benchmark artifact schema:

```text
request_resource_registered
request_effective_cap_changed
request_queue_formed
request_wait_started
request_wait_completed
request_wait_timeout
request_wait_cancelled
request_acquire_denied
request_lease_acquired
model_request_started
model_request_completed
request_queue_drained
request_rate_limited
request_limit_decreased
request_limit_increased
request_soft_ceiling_recovered
request_fully_recovered
request_lease_released
request_release_diagnostic
```

`RequestAdmissionEvent` contains:

- `event_kind`
- `captured_at_monotonic`
- monotonic `sequence`
- captured `RuntimeCorrelation | None`
- request attempt id when the event belongs to one concrete model-call attempt
- request lease id when available
- canonical `RequestResourceKey` when the event is resource-specific
- request group key when the event is queue/admission specific
- denial reason or release outcome when applicable
- optional request pressure snapshot
- sanitized diagnostics

Lease ids, task ids, request attempt ids, and raw model ids are trace/artifact fields only; they are not metric labels. Metric exporters use bounded labels such as `metric_model_label`, model family, or allowlisted model label. The OTel bridge must reject raw model ids as metric labels.

`CorrelatedRuntimeView` joins scheduler and request timelines for diagnostics, benchmarks, and future operator views.

## Capacity Contracts

`AsyncCapacityPlan` records computed per-run capacity values:

```text
CapacityValue[T]:
  value: T | None
  source: default | run_config | dataset_builder | model_metadata | engine_internal_config | adapter_config | environment | runtime_snapshot | benchmark_override
  fallback_from: str | None
  missing_reason: str | None

RowGroupAdmission:
  row_group_concurrency: CapacityValue[int]
  observed_in_flight: int | None

ProviderModelStaticCap:
  cap: int
  aliases: Sequence[str]
  raw_caps: Mapping[str, int | None]
  merge_rule: min_same_endpoint

RequestAdmissionConfigSnapshot:
  resources: Sequence[RequestResourceKey]
  initial_limits: Mapping[RequestResourceKey, int]
  max_limit_clamps: Mapping[RequestResourceKey, int | None]
  cooldown_seconds: float
  multiplicative_decrease_factor: float
  additive_increase_step: int
  increase_after_successes: int
  default_queue_wait_timeout_seconds: float | None

AsyncCapacityPlan:
  configured:
    buffer_size: CapacityValue[int]
    row_group_admission: RowGroupAdmission
    submission_capacity: CapacityValue[int]
    task_resource_limits: CapacityValue[Mapping[SchedulerResourceKey, int]]
    request_resources: CapacityValue[Sequence[RequestResourceKey]]
    provider_model_static_caps: CapacityValue[Mapping[ProviderModelKey, ProviderModelStaticCap]]
    request_domain_initial_limits: CapacityValue[Mapping[RequestResourceKey, int]]
    request_admission_config: CapacityValue[RequestAdmissionConfigSnapshot]
    transport_pool_limits: CapacityValue[Mapping[ProviderModelKey, int]]
  runtime_snapshot:
    request_domain_current_limits: Mapping[RequestResourceKey, int] | None
    request_domain_effective_max: Mapping[RequestResourceKey, int] | None
    request_domain_blocked_until: Mapping[RequestResourceKey, float | None] | None
    provider_model_aggregate_in_flight: Mapping[ProviderModelKey, int] | None
  observed_maxima:
    row_groups_in_flight: int
    queued_tasks_by_group: Mapping[str, int]
    task_leases_by_resource: Mapping[SchedulerResourceKey, int]
    request_waiters_by_resource: Mapping[RequestResourceKey, int]
    request_in_flight_by_resource: Mapping[RequestResourceKey, int]
    provider_model_aggregate_in_flight: Mapping[ProviderModelKey, int]
    request_domain_current_limits: Mapping[RequestResourceKey, int]
    transport_pool_utilization: Mapping[ProviderModelKey, int] | None
```

Fields that depend on request-admission runtime state may be `None` in #654 before #657 lands, but the capacity plan and benchmark artifact must still include the field with `missing_reason` or an equivalent `not_available_until_issue` marker.

The capacity plan explains observed runtime behavior. It is not itself a policy engine.
