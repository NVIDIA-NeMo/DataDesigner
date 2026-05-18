# Observability

Observability must explain which layer is limiting progress without collapsing scheduler admission and request admission into one subsystem.

## Separate Event Streams

`SchedulerAdmissionEventSink` emits scheduler-owned task admission events.

`RequestAdmissionEventSink` emits provider/model/domain request admission events.

Both sinks are generic first. OpenTelemetry, structured logs, dashboards, benchmarks, and debug tools are adapters or consumers.

Sink failures must never interrupt generation. Event data can be collected under locks, but event emission should happen after locks are released.

Event DTOs capture primitive correlation fields at construction time. Sinks receive already-captured event data; they must not read ambient context later to discover which task/request an event belongs to.

All event DTOs include `captured_at_monotonic` and a monotonic per-stream `sequence`. Scheduler events include task id, task execution id when a worker execution exists, task lease id when available, scheduler resource key when applicable, denial/release reason when applicable, optional snapshot, and sanitized diagnostics. Request events include request attempt id when they belong to one concrete attempt, request lease id when available, canonical request resource when resource-specific, request group key when queue/admission-specific, denial/release outcome when applicable, optional pressure snapshot, and sanitized diagnostics.

## Scheduler Admission Events

Scheduler events describe dependency-ready work moving through ready ordering, task admission, worker spawn, and task lease release.

Canonical scheduler event kinds:

- `dependency_ready`
- `ready_enqueued`
- `selected`
- `queue_empty`
- `admission_blocked`
- `group_capped`
- `task_lease_acquired`
- `admission_denied`
- `worker_spawned`
- `worker_spawn_failed`
- `stale_selection`
- `retry_deferred`
- `non_retryable_dropped`
- `cancelled`
- `salvage_redispatched`
- `queue_drained`
- `task_completed`
- `task_lease_released`
- `release_diagnostic`

Scheduler snapshots include:

- queued total
- queued by group
- queued demand by group/resource
- admitted/running by group
- resource limits by scheduler resource
- scheduler resources available by resource
- leased resources by group/resource
- active task lease count by resource
- release diagnostic counters
- bounded-borrow debt by group/resource when applicable

Scheduler events must make hidden scheduler-resource waiters derivable and zero after the task-admission lease boundary lands.

## Request Admission Events

Request events describe provider/model/domain request admission and AIMD behavior.

Canonical request event kinds:

- `request_resource_registered`
- `request_effective_cap_changed`
- `request_queue_formed`
- `request_wait_started`
- `request_wait_completed`
- `request_wait_timeout`
- `request_wait_cancelled`
- `request_acquire_denied`
- `request_lease_acquired`
- `model_request_started`
- `model_request_completed`
- `request_queue_drained`
- `request_rate_limited`
- `request_limit_decreased`
- `request_limit_increased`
- `request_soft_ceiling_recovered`
- `request_fully_recovered`
- `request_lease_released`
- `request_release_diagnostic`

Request snapshots include:

- captured timestamp
- monotonic sequence
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
- last release outcome
- leak diagnostic counters

Global provider/model snapshots capture effective static caps, aggregate in-flight counts across domains, aliases, and per-domain summaries.

## Runtime Correlation

`RuntimeCorrelationProvider` carries current task context while the scheduler executes a task. The likely implementation is a context variable with set/reset/current behavior.

`RuntimeCorrelation` contains primitive values only:

- run id
- row group
- task column
- task type
- scheduling group kind
- scheduling group identity hash
- task execution id

`ModelRequestExecutor` reads the current correlation context when constructing `RequestEventContext` for each concrete request attempt. Scheduler event DTOs capture the scheduler's current task/run identity directly. `AdaptiveRequestAdmissionController` remains keyed by provider/model/domain resources and does not import scheduler task types; it may attach the opaque primitive request event context to request events.

Correlation must propagate through child asyncio tasks created as part of admitted task execution. If execution crosses threads, callbacks, or background tasks that cannot preserve context variables, the caller must pass primitive `RuntimeCorrelation` explicitly or mark the event as intentionally uncorrelated. Late/background provider calls after the scheduler has reset task context are not considered part of the admitted task unless they carry explicit correlation.

`CorrelatedRuntimeView` joins the timelines for diagnostics and benchmarks.

## Joined Timeline

The joined timeline should distinguish:

```text
dependency readiness
ready enqueued
selected by fair queue
task lease acquired
worker spawned
request admission wait started
request admission wait completed
request lease acquired
model request started
model request completed
request lease released
task completed
task lease released
```

Runs should be diagnosable as limited by dependency readiness, ready-queue fairness, scheduler capacity, request-admission wait, provider cooldown/rate-limit behavior, transport/provider execution, or downstream completion.

Benchmark-required monotonic timeline fields are derived from these events:

- `dependency_ready_at`
- `ready_enqueued_at`
- `selected_at`
- `task_lease_acquired_at`
- `worker_spawned_at`
- `request_wait_started_at`
- `request_wait_completed_at`
- `request_lease_acquired_at`
- `model_request_started_at`
- `model_request_completed_at`
- `request_lease_released_at`
- `task_completed_at`
- `task_lease_released_at`

## Cardinality And Safety

Metric-safe dimensions:

- event kind
- scheduler resource kind
- request admission event kind
- provider name
- bounded model label, model family, or allowlisted model label
- metric model label
- request domain
- algorithm

Trace-only or sampled fields:

- run id
- row group
- task column
- task type
- scheduling group identity hash
- raw model id
- task id
- task execution id
- task lease id
- request attempt id
- request lease id
- queued maps by group

Never emit:

- prompts
- completions
- row values
- dataset records
- secrets
- raw provider response bodies
- raw exception payloads
- unbounded request IDs as metric labels

## OpenTelemetry Rule

Core runtime may provide an OTel bridge that depends on API-level primitives, but it must not configure OTel SDKs, exporters, or collectors. Applications embedding Data Designer own exporter configuration.
