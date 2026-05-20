# Capacity Model

The async engine uses layered capacity. Each layer has a different owner and meaning. The epic goal is to make these layers visible, non-overlapping, and traceable in runtime artifacts.

## Layer Vocabulary

| Layer | Owner | Meaning |
| --- | --- | --- |
| Engine selection | Dataset builder / interface | Selects async or sync execution. Not a capacity control. |
| Record window | Dataset builder | Controls row grouping, checkpoint granularity, and memory shape. |
| Row-group admission | Async scheduler | Bounds row groups in flight. |
| Task-stage admission | `TaskAdmissionController` | Bounds scheduler-spawned work and scheduler-level resource pressure. |
| Request-stage admission | `RequestAdmissionController` | Bounds concrete provider/model/domain requests when they are made. |
| Static provider cap | Model config / metadata | User-declared provider/model upper bound and scheduling weight source. |
| Adaptive request-domain limit | `AdaptiveRequestAdmissionController` | Runtime AIMD limit for one provider/model/domain resource under the static provider/model cap. |
| Transport pool | HTTP/model client adapter | Socket/session pool sizing. Not scheduling or fairness policy. |

## AsyncCapacityPlan

`AsyncCapacityPlan` is the run-level explanation of capacity. It should record:

- `buffer_size`
- row-group concurrency
- task admission capacity
- task resource limits
- request admission resources
- provider/model aggregate static caps
- provider/model aggregate in-flight maxima
- static provider/model caps used by the workflow
- adaptive request-admission config snapshot
- request-domain adaptive initial/current/effective limits when captured
- transport/session pool values if they remain distinct
- source of each value, such as default constant, model metadata, run config, request admission state, or environment selection

The plan is emitted for diagnostics, traces, benchmarks, and operator documentation. It does not admit work by itself.

`AsyncCapacityPlan` uses three sections:

```text
configured: values computed before or at run start
runtime_snapshot: point-in-time controller snapshots, nullable until the owning issue lands
observed_maxima: maxima collected during execution or benchmark replay
```

Each configured value is a `CapacityValue` with `value`, `source`, `fallback_from`, and `missing_reason`. Fields that depend on request-admission runtime state may be present with `value = None` and `missing_reason` in #654 before #657 lands.

`CapacityValue.source` uses the durable source vocabulary from [contracts.md](contracts.md#capacity-contracts), including `dataset_builder`, `engine_internal_config`, and `adapter_config` for values that do not come from public run config or model metadata.

Source precedence is per-field, not global:

| Field | V1 precedence |
| --- | --- |
| `buffer_size` | explicit run config, then documented default |
| row-group concurrency | existing dataset-builder/runtime setting if present, then documented default |
| task admission limits | benchmark override for benchmark runs, then engine default |
| provider/model static cap | canonical model/provider metadata; request-admission config may lower but not raise it |
| request-domain initial/adaptive settings | public `RunConfig.request_admission` tuning where supported, benchmark override for non-public harness values, then engine default, all clamped under provider/model static cap |
| transport pool | adapter/client config, then documented default |

If a value is missing, the capacity plan records the missing source and fallback used. If no safe fallback exists, construction fails with a typed configuration/metadata error before work is scheduled.

## Ownership Rules

Task admission capacity is scheduler-level capacity. It controls when a ready task can become a running worker.

Request admission capacity is provider/model/domain request capacity. It controls when a concrete model call can execute.

`max_parallel_requests` remains the user-facing static provider/model cap and scheduling metadata weight source. `AdaptiveRequestAdmissionController.current_limit` is the runtime adaptive request cap for a request domain.

The provider/model static cap is an aggregate in-flight upper bound across all domains for that provider/model in V1. Domain adaptive limits operate under that aggregate cap. V1 intentionally does not define an aggregate cross-domain AIMD policy; adding one requires a later design that specifies fairness, telemetry, and benchmarks.

HTTP transport pools may be larger than the static provider cap. They are transport sizing, not effective request concurrency.

`DATA_DESIGNER_ASYNC_ENGINE` is an execution path selector. It is not a capacity knob.

`RunConfig.buffer_size` shapes record windows and row groups. It is not a request-concurrency knob.

## Row Groups And Record Windows

`buffer_size` defines the record-window shape used by the dataset builder. Row groups are the concrete execution partitions produced from that windowing behavior.

Row-group admission remains scheduler-owned but is not changed by the V1 task-admission lease boundary. For this epic, #654 records row-group configured concurrency and observed row groups in flight through the `RowGroupAdmission` section of `AsyncCapacityPlan`; it does not introduce a new row-group scheduling policy unless a later issue explicitly does so.

Preview, resume, and checkpoint behavior use the existing dataset-builder partitioning rules. `AsyncCapacityPlan` reports the row-group values that the current engine used rather than redefining those rules.

## Transitional Values

Any hidden task-stage capacity concept left from the pre-epic design is transitional. At epic completion those names must be gone or represented by explicit scheduler-resource terminology in `TaskAdmissionConfig` and `AsyncCapacityPlan`.

If a distinct task-stage backpressure resource remains for model-producing work, it must be derived from actually used resolved `SchedulingMetadata`, not every registered model alias. It must be described as scheduler task-stage pressure, not provider request concurrency.

## Alias And Provider Semantics

Scheduling metadata may use model aliases to derive static resource identity and weight. Alias metadata should deduplicate aliases that resolve to the same provider/model/generation resource before summing weight.

Request admission resources are provider/model/domain scoped. A provider/model may have a global effective static cap while each request domain has its own adaptive state. The capacity plan must make that distinction visible.

V1 does not define a cross-domain aggregate AIMD provider cap beyond the documented provider/model effective static cap unless a later issue explicitly adds that policy. The request controller still enforces the static aggregate cap by checking provider/model aggregate in-flight counts before admitting a domain request.

Alias-derived provider/model caps deduplicate aliases that resolve to the same concrete provider/model endpoint. If aliases for the same endpoint specify different `max_parallel_requests` values, V1 uses the minimum as the effective static cap and records every contributing alias and raw cap in `AsyncCapacityPlan`. This min-merge is not a metadata error. Alias resolution is fatal only when endpoint identity is ambiguous or conflicting. If the provider treats generation type as a distinct endpoint, the canonical model id includes that distinction before cap merging.

## Observability Requirements

Operators should be able to answer:

- Which capacity values were used for this run?
- Was progress limited by dependency readiness, queue ordering, task admission, request admission, provider cooldown, or provider execution?
- What static provider caps and adaptive request limits were active?
- Were transport pools distinct from request caps?

Benchmarks and traces must include `AsyncCapacityPlan` plus per-layer observed maxima.

Required per-layer maxima include row groups in flight, queued tasks by group/resource, task leases by resource, request waiters by resource, domain in-flight counts, provider/model aggregate in-flight counts, adaptive current limits, and transport pool utilization when available.

## Public Knob Rule

Do not add new public capacity knobs beyond the documented model `max_parallel_requests`, `buffer_size`, and advanced `RunConfig.request_admission` tuning fields until benchmark evidence shows a specific need and the docs explain the layer. Prefer clear defaults, internal configs, and diagnostics first.
