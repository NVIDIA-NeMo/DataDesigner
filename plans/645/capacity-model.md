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
| Adaptive provider cap | `AdaptiveRequestAdmissionController` | Runtime AIMD limit under the static provider cap. |
| Transport pool | HTTP/model client adapter | Socket/session pool sizing. Not scheduling or fairness policy. |

## AsyncCapacityPlan

`AsyncCapacityPlan` is the run-level explanation of capacity. It should record:

- `buffer_size`
- row-group concurrency
- task admission capacity
- task resource limits
- request admission resources
- static provider/model caps used by the workflow
- adaptive request-admission config
- transport/session pool values if they remain distinct
- source of each value, such as default constant, model metadata, run config, request admission state, or environment selection

The plan is emitted for diagnostics, traces, benchmarks, and operator documentation. It does not admit work by itself.

## Ownership Rules

Task admission capacity is scheduler-level capacity. It controls when a ready task can become a running worker.

Request admission capacity is provider/model/domain request capacity. It controls when a concrete model call can execute.

`max_parallel_requests` remains the user-facing static provider/model cap and scheduling metadata weight source. `AdaptiveRequestAdmissionController.current_limit` is the runtime adaptive request cap.

HTTP transport pools may be larger than the static provider cap. They are transport sizing, not effective request concurrency.

`DATA_DESIGNER_ASYNC_ENGINE` is an execution path selector. It is not a capacity knob.

`RunConfig.buffer_size` shapes record windows and row groups. It is not a request-concurrency knob.

## Transitional Values

Any current `max_llm_wait_tasks`, `needs_llm_wait`, or `held_llm_wait` concept is transitional. At epic completion these names must either be gone or replaced by explicit scheduler-resource terminology in `TaskAdmissionConfig` and `AsyncCapacityPlan`.

If a distinct task-stage LLM backpressure resource remains, it must be derived from actually used resolved scheduling metadata, not every registered model alias. It must be described as scheduler task-stage pressure, not provider request concurrency.

## Alias And Provider Semantics

Scheduling metadata may use model aliases to derive static resource identity and weight. Alias metadata should deduplicate aliases that resolve to the same provider/model/generation resource before summing weight.

Request admission resources are provider/model/domain scoped. A provider/model may have a global effective static cap while each request domain has its own adaptive state. The capacity plan must make that distinction visible.

V1 does not define a cross-domain aggregate AIMD provider cap beyond the documented provider/model effective static cap unless a later issue explicitly adds that policy.

## Observability Requirements

Operators should be able to answer:

- Which capacity values were used for this run?
- Was progress limited by dependency readiness, queue ordering, task admission, request admission, provider cooldown, or provider execution?
- What static provider caps and adaptive request limits were active?
- Were transport pools distinct from request caps?

Benchmarks and traces must include `AsyncCapacityPlan` plus per-layer observed maxima.

## Public Knob Rule

Do not add a new public capacity knob until benchmark evidence shows a specific need and the docs explain its layer. Prefer clear defaults, internal configs, and diagnostics first.
