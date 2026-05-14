# Request Admission

Request admission controls concrete provider/model/domain calls at the moment they are made. It is separate from task admission because task-level scheduling cannot predict every model call inside arbitrary generator Python.

## Runtime Shape

```text
ModelRequestExecutor
  -> RequestAdmissionController.acquire_async(RequestAdmissionItem)
  -> RequestAdmissionLease
  -> provider/model endpoint
  -> release_success | release_rate_limited | release_failure
```

`ModelRequestExecutor` is the durable model-call boundary. It maps each concrete call to a request resource, acquires a lease, calls the provider, records timing, and releases the exact lease.

## Dynamic Requests

Custom generators may make zero, one, or many model requests depending on row data, branches, retries, validation failures, tool calls, or helper functions. `SchedulingMetadata` can describe static resource shape for task grouping, but it is not an exact request-count promise.

Therefore:

- task admission must not pre-acquire request permits
- request admission happens at concrete model-call time
- each acquired request lease is released exactly once
- request-level wait and provider execution timing remain visible separately

## Durable Names

The durable interface name is `RequestAdmissionController`.

The durable V1 implementation name is `AdaptiveRequestAdmissionController`.

The durable model-call boundary name is `ModelRequestExecutor`.

The durable config vocabulary is `RequestAdmissionConfig` and `RunConfig.request_admission` if a run config surface exists.

Do not keep production aliases, shims, subclasses, adapters, exports, docs paths, or durable tests for:

- `ThrottleManager`
- `ThrottleDomain`
- `ThrottleConfig`
- `RunConfig.throttle`
- `throttle_manager.py`

## Request Resource Model

`RequestResourceKey` identifies:

- provider name
- model id
- request domain

`RequestDomain` includes domains such as chat, embedding, image, and healthcheck.

`RequestAdmissionItem` contains resource, group, and optional timeout. `RequestGroupSpec` contains resource key and weight.

`RequestAdmissionDecision` is `RequestAdmissionLease | RequestAdmissionDenied`.

`RequestAdmissionLease` records the item, acquired timestamp, current adaptive limit, and effective max.

`RequestAdmissionDenied` records item, reason, retry timing, availability timing, and optional snapshot.

`RequestAdmissionController.pressure` exposes the read-only `RequestPressureSnapshotProvider`.

## AdaptiveRequestAdmissionController

`AdaptiveRequestAdmissionController` is the AIMD-backed request controller. It owns:

- request fair queueing
- request admission policy
- adaptive request limit state
- provider/model/domain in-flight counts
- waiters
- cooldown state
- rate-limit cascades
- additive increase and multiplicative decrease
- request pressure snapshots

Internal `RequestFairQueue`, `RequestAdmissionPolicy`, and `AdaptiveRequestLimitState` are part of the single canonical request-admission implementation. They are not a second public wrapper around request admission.

## Release Classification

`ModelRequestExecutor` releases the exact acquired lease through:

- `release_success(lease)` after provider success
- `release_rate_limited(lease, retry_after)` after provider rate-limit response
- `release_failure(lease)` after non-rate-limit failure, cancellation, timeout, or unexpected exception

The release path is responsible for exactly-once accounting. Key-only release paths are not durable.

## Request Pressure Snapshots

`RequestPressureSnapshotProvider` exposes read-only state to diagnostics, benchmarks, telemetry, and future task policies.

Domain snapshots include:

- request resource
- effective max
- current limit
- in-flight count
- waiters
- blocked-until timing
- cooldown remaining
- rate-limit ceiling
- consecutive rate limits

Global snapshots include provider/model effective caps and aliases.

Task admission may read these snapshots as advisory input in later policy work. It must not mutate request state or emulate request admission.

## Non-Goals

- Do not make request admission aware of DAG dependencies.
- Do not make request admission own row-group lifecycle or ready-work ordering.
- Do not replace AIMD with token-bucket or leaky-bucket behavior in V1.
- Do not require static prediction of all model calls.
- Do not make task-level `TaskAdmissionController` responsible for provider retry, cooldown, or AIMD updates.
