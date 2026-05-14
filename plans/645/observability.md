# Observability

Observability must explain which layer is limiting progress without collapsing scheduler admission and request admission into one subsystem.

## Separate Event Streams

`SchedulerAdmissionEventSink` emits scheduler-owned task admission events.

`RequestAdmissionEventSink` emits provider/model/domain request admission events.

Both sinks are generic first. OpenTelemetry, structured logs, dashboards, benchmarks, and debug tools are adapters or consumers.

Sink failures must never interrupt generation. Event data can be collected under locks, but event emission should happen after locks are released.

## Scheduler Admission Events

Scheduler events describe dependency-ready work moving through ready ordering, task admission, worker spawn, and task lease release.

Representative event kinds:

- ready enqueued
- selected
- lease acquired
- worker spawned
- admission blocked
- group capped
- stale selection
- retry deferred
- non-retryable dropped
- cancelled
- salvage redispatched
- queue drained
- lease released
- completed

Scheduler snapshots include:

- queued total
- queued by group
- admitted/running by group
- submission available
- scheduler resources available

Scheduler events must make `spawned_waiting_for_llm_lease` derivable and zero after the task-admission lease boundary lands.

## Request Admission Events

Request events describe provider/model/domain request admission and AIMD behavior.

Representative event kinds:

- registered
- effective cap changed
- queue formed
- wait started
- queue wait completed
- queue wait timeout
- queue drained
- rate limited
- limit decreased
- limit increased
- soft ceiling recovered
- fully recovered
- failure released

Request snapshots include:

- request resource
- effective max
- current limit
- in-flight count
- waiters
- cooldown remaining
- rate-limit ceiling
- consecutive rate limits

Global provider/model snapshots capture effective static caps and aliases.

## Runtime Correlation

`RuntimeCorrelationProvider` carries current task context while the scheduler executes a task. The likely implementation is a context variable with set/reset/current behavior.

`RuntimeCorrelation` contains primitive values only:

- run id
- row group
- task column
- task type
- scheduling group kind
- scheduling group identity hash

`ModelRequestExecutor` and event sinks read the current correlation context at event emission time. `AdaptiveRequestAdmissionController` remains keyed by provider/model/domain resources and does not import scheduler types.

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
request lease acquired
provider request started
provider request completed
request lease released
task completed
task lease released
```

Runs should be diagnosable as limited by dependency readiness, ready-queue fairness, scheduler capacity, request-admission wait, provider cooldown/rate-limit behavior, transport/provider execution, or downstream completion.

## Cardinality And Safety

Metric-safe dimensions:

- event kind
- scheduler resource kind
- request admission event kind
- provider name
- model id
- request domain
- algorithm

Trace-only or sampled fields:

- run id
- row group
- task column
- task type
- scheduling group identity hash
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
