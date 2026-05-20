# Migration And Cleanup

The epic is not complete until replaced names, old module paths, and compatibility paths are removed from production code, current docs, and this source-of-truth plan. The target architecture is defined by [Module ownership](module-ownership.md); implementation PRs should move directly to those final homes.

## Scheduling Metadata Cleanup

The durable scheduling metadata path is:

```text
ColumnGenerator.get_scheduling_metadata()
-> SchedulingMetadata
-> TaskSchedulingResolver
```

Remove the legacy resolver types and any independent scheduler-side model/provider introspection path. All model/provider inference must live behind `ColumnGenerator.get_scheduling_metadata()`, `SchedulingMetadata`, and typed `SchedulingMetadataError` fallback behavior.

Unacceptable end states:

- a parallel fallback that independently introspects generators, configs, model registries, aliases, or admitted policy data under the old resolver contract
- a compatibility adapter, alias, or reexport that preserves the old resolver vocabulary as a durable production path

Final legacy-name search gate should have no production/current-doc matches for these historical strings:

```text
SchedulingHintResolver
SchedulingHint
_model_aliases_for_generator
```

Independent scheduler-side model-bound fallback logic is also migration-only and should be folded behind metadata/resource requests by epic completion.

## Request Admission Cleanup

The durable request-admission names are:

- `ModelRequestExecutor`
- `RequestAdmissionController`
- `AdaptiveRequestAdmissionController`
- `RequestAdmissionConfig`
- `RequestDomain`

Final legacy-name search gate should have no production/current-doc matches for these historical strings:

```text
ThrottleManager
ThrottleDomain
ThrottleConfig
RunConfig.throttle
throttle_manager.py
ThrottledModelClient
throttled_model_client
```

Historical changelog text may remain only if it is clearly marked as historical and not presented as current API.

## Task-Stage Wait Cleanup

The durable architecture does not include:

```text
needs_llm_wait
held_llm_wait
max_llm_wait_tasks
```

If a scheduler-level resource remains for LLM-bound work, it must be represented through `SchedulerResourceRequest`, `TaskAdmissionConfig`, and `AsyncCapacityPlan`, with names that describe scheduler task-stage pressure rather than request concurrency.

## Module Ownership Cleanup

The target scheduler package is:

```text
data_designer.engine.dataset_builders.scheduling
```

The durable architecture does not keep scheduler-owned task models, readiness tracking, queues, task admission, or task policies in `dataset_builders.utils`.

The target request-admission package is:

```text
data_designer.engine.models.request_admission
```

The durable architecture does not keep request-admission controllers, queues, waiters, AIMD state, pressure snapshots, or request leases in `models.clients.request_admission`.

`ModelRequestExecutor` remains under `models.clients` because it wraps concrete model clients. Request admission itself must not be reexported from `models.clients.__init__`.

## Compatibility Shim Rule

Do not leave production compatibility aliases, subclasses, adapters, reexports, docs paths, or durable tests for replaced names at epic completion.

Do not introduce shim modules or deprecation adapters under replaced names. Historical names may appear only in explicit cleanup/search-gate sections like this one or in clearly marked historical changelog/dev-note text.

## Gate Semantics

Before the migration issues land, stale-name matches can exist as current-state evidence.

By #653 close, legacy scheduling-hint production paths are gone and tests have moved to metadata/resolver coverage.

By #657 close, request-admission code has no production `Throttle*` aliases, exports, modules, or durable tests.

By #645 close, production code lives in the target modules from [Module ownership](module-ownership.md), package `__init__.py` files do not reexport internal queues/controllers/leases, and public/current docs plus `plans/645` use only the durable architecture vocabulary except for this cleanup file's explicit legacy-name search lists. Historical changelog or dev-note text can remain only when explicitly marked historical.

## Documentation Cleanup

Current maintainer architecture docs should use durable internal names when they discuss implementation internals:

- `SchedulingMetadata`
- `TaskSchedulingResolver`
- `TaskAdmissionController`
- `TaskAdmissionPolicy`
- `TaskAdmissionLease`
- `ModelRequestExecutor`
- `RequestAdmissionController`
- `AdaptiveRequestAdmissionController`
- `RequestAdmissionConfig`
- `RuntimeCorrelationProvider`

User/operator docs should expose public run config fields, including `RequestAdmissionTuningConfig`, `AsyncCapacityPlan`, benchmark artifacts, telemetry views, and high-level layer names. They must not present `TaskAdmissionConfig`, `RequestAdmissionConfig`, policies, leases, queues, pressure snapshots, or controller mutation APIs as public user knobs. Plugin-facing docs should describe metadata only, then link to architecture docs for maintainers/operators.

Current architecture docs, diagrams, generated assets, and plan files must be checked as part of final cleanup. Existing historical dev notes may retain old names only when the text clearly says the name is historical and no longer current API.

Current user/operator architecture docs must also remove or mark as historical capacity-control descriptions that imply the pre-epic architecture. This includes old model-client request-capacity names and scheduler-slot handoff explanations.

## Validation Commands

Adjust paths as files move, but final PRs should include searches equivalent to:

```bash
rg "SchedulingHintResolver|SchedulingHint|_model_aliases_for_generator|is_llm_bound" packages docs fern architecture plans/645
rg "ThrottleManager|ThrottleDomain|ThrottleConfig|RunConfig\\.throttle|throttle_manager\\.py|ThrottledModelClient|throttled_model_client" packages docs fern architecture plans/645
rg "_submission_semaphore|_llm_wait_semaphore|get_semaphore_permits|TrackingSemaphore" packages docs fern architecture plans/645
rg "throttl(e|ed|ing)|semaphore" docs fern architecture plans/645
rg "needs_llm_wait|held_llm_wait|max_llm_wait_tasks" packages docs fern architecture plans/645
rg "dataset_builders\\.utils\\.(task_model|completion_tracker|task_scheduling|fair_task_queue|task_admission)" packages docs fern architecture plans/645
rg "models\\.clients\\.request_admission|from data_designer\\.engine\\.models\\.clients import .*Request" packages docs fern architecture plans/645
rg "SchedulingMetadata|TaskSchedulingResolver|FairTaskQueue|TaskAdmissionController|TaskAdmissionLease|ModelRequestExecutor|RequestAdmissionController|AdaptiveRequestAdmissionController|AsyncCapacityPlan|SchedulerResourceRequest|RequestResourceKey" docs fern architecture plans/645
```

Any remaining hit must be intentionally historical, not a current implementation or docs path. Allowed plan hits are limited to explicit cleanup/search-gate sections that name the legacy strings so reviewers know what to remove. The task-stage wait-specific search distinguishes obsolete scheduler-slot handoff primitives from unrelated internal synchronization primitives that may remain after review.
