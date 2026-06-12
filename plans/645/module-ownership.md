# Module Ownership

This page defines the target repository and module ownership for the async scheduling epic. It is an end-state design, not a migration plan. Implementation PRs should move directly toward these homes and must not introduce compatibility aliases, shim modules, transitional reexports, or duplicate old/new module paths.

Durable engine names in this plan are maintainer contracts. They are not public import promises unless this page explicitly marks them plugin-facing or operator-facing.

## Package Ownership

| Package | Owns | Must not own |
| --- | --- | --- |
| `data-designer-config` | public configuration DTOs and generator-facing metadata | engine runtime protocols, queues, admission leases, request domains, AIMD state, runtime pressure |
| `data-designer-engine` | scheduler runtime, task admission, request admission, capacity diagnostics, runtime observability, benchmark internals | public interface orchestration, user-facing docs presentation |
| `data-designer` | public `DataDesigner` interface wiring, CLI presentation, integrations | scheduler internals, plugin-facing scheduling metadata definitions |

Config code must not import engine runtime code. Engine code may import config DTOs.

## Target Module Layout

```text
packages/data-designer-config/src/data_designer/config/
  scheduling.py
    SchedulingMetadata
    SchedulingMetadataError

packages/data-designer-engine/src/data_designer/engine/
  dataset_builders/
    async_scheduler.py
      AsyncTaskScheduler

    scheduling/
      task_model.py
        Task
        SliceRef
        TaskTrace

      completion.py
        CompletionTracker
        FrontierDelta

      resources.py
        TaskGroupKey
        TaskGroupSpec
        SchedulerResourceKey
        SchedulerResourceRequest
        SchedulableTask
        stable_task_id

      resolver.py
        TaskSchedulingResolver
        ResolvedTaskScheduling

      queue.py
        FairTaskQueue
        QueueView
        QueueSelection

      task_admission.py
        TaskAdmissionController
        TaskAdmissionConfig
        TaskAdmissionLease
        TaskAdmissionDenied
        TaskAdmissionDecision
        TaskAdmissionView
        TaskAdmissionBlockSummary
        ReleaseResult

      task_policies.py
        TaskAdmissionPolicy
        TaskAdmissionPolicyDecision
        PolicyStateDelta
        StrictFairTaskAdmissionPolicy
        BoundedBorrowTaskAdmissionPolicy
        BoundedBorrowTaskAdmissionPolicyConfig

  models/
    resources.py
      ProviderModelKey
      ProviderModelStaticCap
      provider/model alias canonicalization helpers

    clients/
      model_request_executor.py
        ModelRequestExecutor

    request_admission/
      resources.py
        RequestDomain
        RequestResourceKey
        RequestGroupSpec
        RequestEventContext
        RequestAdmissionItem

      resolver.py
        RequestResourceResolver
        ResolvedRequestResource

      config.py
        RequestAdmissionConfig

      queue.py
        RequestFairQueue
        RequestWaiter
        RequestQueueView
        RequestQueueSelection

      limits.py
        AdaptiveRequestLimitState
        provider/model aggregate limit state

      pressure.py
        RequestPressureSnapshotProvider
        RequestPressureSnapshot
        ProviderModelPressureSnapshot

      outcomes.py
        RequestReleaseOutcome
        ReleaseResult

      controller.py
        RequestAdmissionController
        AdaptiveRequestAdmissionController
        RequestAdmissionLease
        RequestAdmissionDenied
        RequestAdmissionDecision
        RequestAdmissionError

  capacity.py
    CapacityValue
    AsyncCapacityPlan
    AsyncCapacityConfigured
    AsyncCapacityRuntimeSnapshot
    AsyncCapacityObservedMaxima
    RequestAdmissionConfigSnapshot

  observability.py
    RuntimeCorrelation
    RuntimeCorrelationProvider
    runtime_correlation_provider
    SchedulerAdmissionEvent
    SchedulerAdmissionEventSink
    RequestAdmissionEvent
    RequestAdmissionEventSink
    InMemoryAdmissionEventSink
    CorrelatedRuntimeView

  models/telemetry.py
    product/provider usage telemetry only
```

`AsyncTaskScheduler` is the runtime coordinator only. It owns ready-frontier polling, queue selection, task-lease acquire/release orchestration, worker lifecycle, salvage/retry coordination, shutdown, and row-group lifecycle integration. It does not own queue policy, task admission ledgers, request admission, provider cooldown, AIMD behavior, or model-client wrapping.

`ModelRequestExecutor` remains under `models/clients` because it implements the model-client boundary and wraps concrete provider clients. Request admission itself lives under `models/request_admission` and must not import `ModelClient` or provider adapter classes.

`models/resources.py` owns provider/model identity that is shared across metadata resolution, request admission, and capacity diagnostics. Request admission owns request-domain resources. Capacity consumes both as read-only diagnostic inputs; it does not own admission policy or controller state transitions.

`observability.py` is the cross-layer runtime-observability home. It owns scheduler and request admission event DTOs, primitive runtime correlation, in-memory test/diagnostic sinks, and correlated runtime views. Product/provider usage telemetry remains separate in `models/telemetry.py`.

## Current Module Targets

| Current or legacy module/concept | Target direction |
| --- | --- |
| `dataset_builders/async_scheduler.py` | keep as coordinator; remove durable queue, task-policy, and request-admission ownership |
| `dataset_builders/utils/task_model.py` | move scheduler task DTOs to `dataset_builders/scheduling/task_model.py` |
| `dataset_builders/utils/completion_tracker.py` | move readiness tracking to `dataset_builders/scheduling/completion.py` |
| `dataset_builders/utils/task_scheduling.py` | split scheduler resources into `scheduling/resources.py` and metadata resolution into `scheduling/resolver.py` |
| `dataset_builders/utils/fair_task_queue.py` | move to `dataset_builders/scheduling/queue.py`; keep ready ordering only |
| `dataset_builders/utils/task_admission.py` | split controller/lease DTOs into `scheduling/task_admission.py` and policies into `scheduling/task_policies.py` |
| `models/clients/model_request_executor.py` | keep as the concrete model-client acquire/call/release wrapper |
| `models/clients/request_admission.py` | split into the `models/request_admission/` package |
| `models/clients/__init__.py` request-admission reexports | remove; request-admission internals are imported from their owning modules only |
| `models/telemetry.py` | keep product/provider usage telemetry separate from admission event DTOs |
| `capacity.py` | keep as cross-cutting capacity diagnostic/reporting code that consumes read-only scheduler/request DTOs and snapshots |
| `SchedulingHintResolver`, `SchedulingHint`, and scheduler-side model-bound fallbacks | remove; `SchedulingMetadata` plus `TaskSchedulingResolver` are the only durable path |
| `ThrottleManager`, `ThrottleDomain`, `ThrottledModelClient`, and `throttled_model_client` | remove; request admission and `ModelRequestExecutor` are the only durable request-control path |
| `ThrottleConfig` and `RunConfig.throttle` | keep only as deprecated public config compatibility shims that translate to `RequestAdmissionTuningConfig` and emit `DeprecationWarning`; not durable engine architecture |

## Import Rules

- `data_designer.config.*` must not import `data_designer.engine.*` or `data_designer.interface.*`.
- Engine modules may import `SchedulingMetadata` and `SchedulingMetadataError` from config.
- `dataset_builders/scheduling/*` may import config scheduling metadata, dataset-builder task/readiness concepts, primitive runtime observability, and neutral provider/model identity helpers from `engine.models.resources`.
- `dataset_builders/scheduling/*` must not import model clients, request-admission controllers, request queues, AIMD state, provider adapters, or request leases.
- `models/request_admission/*` may import neutral provider/model identity helpers and primitive observability, but must not import dataset-builder scheduler types or `ModelClient`.
- `models/clients/model_request_executor.py` is the production bridge that imports both model-client types and request-admission protocols. It is the only model-client layer that acquires and releases request leases.
- `capacity.py` may import read-only resource DTOs, config snapshots, pressure snapshots, and event snapshots. It must not call controller mutation APIs or become a controller registry.
- `observability.py` must not import concrete controllers, queues, model clients, provider adapters, or dataset-builder schedulers.
- `data-designer` interface and CLI code may consume engine diagnostics for presentation, but must not reexport scheduler/request internals as plugin API.
- Package `__init__.py` files must not reexport internal queues, policies, leases, waiters, or controllers as broad public-looking APIs.

## Audience Boundaries

| Audience | Exposed surface | Not exposed |
| --- | --- | --- |
| Plugin authors | `ColumnGenerator.get_scheduling_metadata()`, `SchedulingMetadata`, `SchedulingMetadataError` | queues, task groups, scheduler resources, task leases, request domains, pressure snapshots, AIMD state |
| Users/operators | documented public run/model config fields, `AsyncCapacityPlan`, benchmark artifacts, telemetry/event artifacts, correlated runtime views | controller mutation APIs, queues, policies, leases, waiters, internal config objects |
| Engine maintainers | scheduler/request admission modules, DTOs, protocols, policies, snapshots, events, capacity diagnostics | config-layer reverse imports, compatibility aliases, duplicate old/new module paths |
| Tests and benchmarks | local fakes, deterministic model clients, event sinks, benchmark override config | production `engine.testing` helpers, test-module imports from benchmark code, benchmark-module imports from unit tests |

`TaskAdmissionConfig` and `RequestAdmissionConfig` are engine-internal in V1. They may appear inside capacity and benchmark artifacts as explanatory snapshots, but they are not public `RunConfig` knobs. Public request-admission tuning is exposed only through `RequestAdmissionTuningConfig` on `RunConfig.request_admission` and is translated into the engine-internal config at the engine boundary.

## Tests And Benchmarks

Tests mirror target module ownership:

| Area | Target test home |
| --- | --- |
| config metadata | `packages/data-designer-config/tests/config/test_scheduling.py` |
| scheduler task resources/resolver | `packages/data-designer-engine/tests/engine/dataset_builders/scheduling/test_resources.py` and `test_resolver.py` |
| fair task queue | `packages/data-designer-engine/tests/engine/dataset_builders/scheduling/test_queue.py` |
| task admission and policies | `packages/data-designer-engine/tests/engine/dataset_builders/scheduling/test_task_admission.py` and `test_task_policies.py` |
| scheduler integration | `packages/data-designer-engine/tests/engine/dataset_builders/test_async_scheduler.py` |
| request admission | `packages/data-designer-engine/tests/engine/models/request_admission/` |
| model request executor | `packages/data-designer-engine/tests/engine/models/clients/test_model_request_executor.py` |
| capacity diagnostics | `packages/data-designer-engine/tests/engine/test_capacity.py` |
| runtime observability | `packages/data-designer-engine/tests/engine/test_observability.py` |

Test fakes live under tests near their consumers. Benchmark fakes and reusable scenarios live under `scripts/benchmarks/async_scheduling/`, with `scripts/benchmarks/benchmark_async_scheduling.py` as the runnable entrypoint. Production `data_designer.engine.testing` helpers are not part of the target architecture.
