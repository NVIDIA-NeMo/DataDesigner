# Request Admission

Request admission controls concrete provider/model/domain calls at the moment they are made. It is separate from task admission because task-level scheduling cannot predict every model call inside arbitrary generator Python.

## Runtime Shape

```text
ModelRequestExecutor
  -> ModelRequestExecutor.execute_attempt(request)
  -> RequestAdmissionController.acquire_async(RequestAdmissionItem)
  -> RequestAdmissionLease
  -> provider/model endpoint
  -> RequestAdmissionController.release(lease, RequestReleaseOutcome)
```

`ModelRequestExecutor` is the durable model-call boundary. It maps each concrete call attempt to a request resource, acquires a lease, calls the provider, records timing, and releases the exact lease.

The boundary is per outbound attempt. Provider retry behavior must either live inside `ModelRequestExecutor` and acquire/release a lease for each attempt, or call back through `ModelRequestExecutor` for each attempt. HTTP/provider-client retries that hide multiple outbound attempts under one request lease are not compatible with the target architecture because rate limits and provider timing would be invisible to request admission.

After a lease is acquired, `ModelRequestExecutor` owns release in a non-cancellable cleanup path. Cancellation after lease acquisition is classified as `local_cancelled`, the exact lease is released before the cancellation is re-raised, and release diagnostics must not mask the original cancellation.

## Dynamic Requests

Custom generators may make zero, one, or many model requests depending on row data, branches, retries, validation failures, tool calls, or helper functions. `SchedulingMetadata` can describe static resource shape for task grouping, but it is not an exact request-count promise.

Therefore:

- task admission must not pre-acquire request permits
- request admission happens at concrete model-call time
- each acquired request lease is released exactly once
- each retry attempt is admitted and released independently
- request-level wait and provider execution timing remain visible separately

## Durable Names

The durable interface name is `RequestAdmissionController`.

The durable V1 implementation name is `AdaptiveRequestAdmissionController`.

The durable model-call boundary name is `ModelRequestExecutor`.

The durable internal config vocabulary is `RequestAdmissionConfig`. V1 does not introduce a public `RunConfig.request_admission` knob; that spelling is reserved for a later public surface if benchmark evidence justifies one.

Do not keep production aliases, shims, subclasses, adapters, exports, docs paths, or durable tests for the replaced request-control vocabulary. [Migration and cleanup](migration-and-cleanup.md#request-admission-cleanup) lists the exact search terms.

## Request Resource Model

`RequestResourceKey` identifies canonical resolved request identity:

- provider name, after alias resolution
- model id, after alias resolution
- request domain

Aliases are diagnostic-only after request-key construction. They are recorded in `AsyncCapacityPlan` and snapshots, but request admission must not key aggregate caps by user alias.

`RequestResourceResolver` is the single canonicalization contract for request admission. It resolves provider alias, model alias, model id, generation kind, endpoint metadata, and `RequestDomain` into `ProviderModelKey` and `RequestResourceKey`. Metadata resolution and capacity planning use the same provider/model canonicalization rules; generation kind is folded into the canonical model id only when the provider treats it as a distinct endpoint.

`RequestDomain` V1 values are `chat`, `embedding`, `image`, and `healthcheck`. Additions require updating this plan.

`RequestAdmissionItem` contains resource, group, optional queue-wait timeout, and `RequestEventContext`. `RequestGroupSpec` contains a fairness group key and weight. In V1 the fairness group key is the `RequestResourceKey`; a future policy may split resource identity from fairness identity only after updating this plan.

`RequestEventContext` is created by `ModelRequestExecutor` from the current primitive runtime correlation plus a request-attempt id. It is telemetry context, not scheduler state.

`RequestAdmissionDecision` is `RequestAdmissionLease | RequestAdmissionDenied`.

`RequestAdmissionLease` records a unique lease id, item, acquired timestamp, current adaptive limit, effective max, and controller generation token.

`RequestAdmissionDenied` records item, reason, retry timing, availability timing, and optional snapshot.

`RequestAdmissionController.pressure` exposes the read-only `RequestPressureSnapshotProvider`.

`acquire_sync(...)` and `acquire_async(...)` block until a lease is available or a terminal no-lease condition occurs. Queue-wait timeout, shutdown, or hard denial removes the waiter and raises `RequestAdmissionError`, a typed Data Designer error carrying `RequestAdmissionDenied`. These methods never return `None`. `try_acquire(...)` is the non-blocking path that returns the full decision union.

`acquire_async(...)` preserves cooperative cancellation: if the awaiting task is cancelled before a lease is acquired, the controller removes the waiter, emits a cancellation event, and re-raises the cancellation exception.

Once the controller selects a waiter and increments in-flight counts, cancellation cannot orphan the lease. The selected waiter's acquire call receives the lease for caller cleanup, or the controller internally releases it as `local_cancelled` before completing cancellation. A blocking acquire call may only return a lease for its own waiter; if a wakeup admits a different waiter, that other waiter is fulfilled and the current caller continues waiting.

## Request Queue Semantics

`AdaptiveRequestAdmissionController` owns an internal `RequestFairQueue`. The queue is protected by the controller lock/condition and exposes the same transaction shape as task admission:

```text
enqueue(waiter)
select_next(is_eligible) -> RequestQueueSelection | None
commit(selection) -> RequestWaiter | None
remove(waiter_id)
view() -> RequestQueueView
```

`RequestWaiter` carries waiter id, item, enqueue timestamp, deadline/cancellation state, and the completion handle for the blocking acquire path. `RequestQueueSelection` carries waiter, item, waiter id, queue view, and a `sequence_version`. `commit(selection)` is the only operation that removes an admitted waiter.

Wakeups occur when a request lease releases, cooldown expires, adaptive limit increases, shutdown/cancellation removes waiters, or provider/model aggregate capacity becomes available. Waiters use monotonic timed waits to the earliest queue timeout, `available_after_monotonic`, or `blocked_until_monotonic`; cooldown expiry cannot depend on a later provider release to wake the queue.

Every concrete request attempt emits request-wait timeline events. Immediate acquisition emits `request_wait_started` and `request_wait_completed` as a zero-duration wait before `request_lease_acquired`; queued acquisition emits those events around actual queue wait.

`try_acquire(...)` must not bypass queued work. It may return an immediate lease only when the item is eligible and no queued eligible waiter for the same request resource or provider/model aggregate cap would be selected first by `RequestFairQueue` weighted ordering. Otherwise it returns a typed denial, usually `queued_waiters_ahead`, `cooldown`, or `no_capacity`.

## AdaptiveRequestAdmissionController

`AdaptiveRequestAdmissionController` is the AIMD-backed request controller. It owns:

- request fair queueing
- request admission policy
- adaptive request limit state
- provider/model/domain in-flight counts plus provider/model aggregate in-flight counts
- waiters
- cooldown state
- rate-limit cascades
- additive increase and multiplicative decrease
- request pressure snapshots

Internal `RequestFairQueue`, `RequestAdmissionPolicy`, and `AdaptiveRequestLimitState` are part of the single canonical request-admission implementation. They are not a second public wrapper around request admission.

An admitted request increments domain in-flight count and provider/model aggregate in-flight count before the lease is returned. Release decrements those counts exactly once before waking waiters.

Weighted fairness applies across `RequestGroupSpec` groups that share a provider/model aggregate cap. Equal weights fall back to oldest waiter first.

V1 AIMD contract:

- all timing uses a monotonic clock
- `effective_max = min(provider_model_static_cap, request_config.max_limit_clamp_for_resource_if_present)`
- instantaneous aggregate availability is enforced separately by `provider_model_aggregate_in_flight < provider_model_static_cap`
- `initial_limit` is clamped to `[1, effective_max]`
- `current_limit` starts at `initial_limit`
- provider rate limits apply multiplicative decrease and set `blocked_until_monotonic`
- success outside cooldown contributes to additive recovery
- `request_limit_increased`, `request_soft_ceiling_recovered`, and `request_fully_recovered` events are emitted from state transitions, not inferred later by sinks

## Release Classification

`ModelRequestExecutor` releases the exact acquired lease through the canonical release call:

```text
release(lease, RequestReleaseOutcome)
```

Required outcome kinds:

- `success`
- `rate_limited`, with `retry_after_seconds` when available
- `provider_failure`
- `provider_timeout`
- `local_cancelled`
- `local_timeout`
- `unexpected_exception`

The release path is responsible for exactly-once accounting. Key-only release paths are not durable.

Rate-limit outcomes drive AIMD decrease, cooldown, and waiter wake behavior. Provider failures may drive diagnostic counters but do not automatically imply provider pressure. Local cancellation and local timeout release capacity and wake waiters but must not be treated as rate limits or provider failures.

`provider_timeout` means a timeout or timeout-shaped transport/provider error after a lease has been acquired and an outbound provider attempt has started. `local_timeout` means a caller, queue-wait, or controller deadline that is not evidence of provider pressure.

Release returns `ReleaseResult` and must not raise from terminal cleanup paths. Duplicate release, stale release, or release against the wrong controller generation must return a diagnostic result and emit an error event without corrupting counters.

## Request Pressure Snapshots

`RequestPressureSnapshotProvider` exposes read-only state to diagnostics, benchmarks, telemetry, and future task policies.

Domain snapshots include:

- captured timestamp
- monotonic sequence/version
- request resource
- effective max
- current limit
- in-flight count
- active lease count
- waiters
- blocked-until timing
- cooldown remaining
- rate-limit ceiling
- consecutive rate limits
- last release outcome summary
- leak diagnostic counters

Global snapshots include provider/model effective static caps, aggregate in-flight count across domains, aggregate active lease count, aliases contributing to the cap, and per-domain limit summaries.

Task admission may read these snapshots as advisory input in later policy work. It must not mutate request state or emulate request admission.

## Static And Adaptive Cap Semantics

`max_parallel_requests` remains the provider/model static cap when available. In V1, that cap is enforced as an aggregate upper bound across all request domains for the provider/model. Domain-specific adaptive limits decide how each domain is admitted beneath the aggregate cap; there is no cross-domain aggregate AIMD state beyond the static aggregate cap unless a later issue adds one.

Effective admission for a request must satisfy both:

- the provider/model aggregate static cap has available in-flight capacity
- the request domain's adaptive limit and cooldown state admits the item

## Non-Goals

- Do not make request admission aware of DAG dependencies.
- Do not make request admission own row-group lifecycle or ready-work ordering.
- Do not replace AIMD with token-bucket or leaky-bucket behavior in V1.
- Do not require static prediction of all model calls.
- Do not make task-level `TaskAdmissionController` responsible for provider retry, cooldown, or AIMD updates.
