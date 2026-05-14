# Migration And Cleanup

The epic is not complete until replaced names and compatibility paths are removed from production code, current docs, and this source-of-truth plan.

## Scheduling Metadata Cleanup

Remove or collapse the legacy `SchedulingHintResolver` path after `SchedulingMetadata` and `TaskSchedulingResolver` are stable.

Accepted end states:

- delete `SchedulingHintResolver`, or
- refactor/rename it into a metadata-oriented adapter where all model/provider inference lives behind `ColumnGenerator.get_scheduling_metadata()` and typed `SchedulingMetadataError` fallback behavior.

Unacceptable end state:

- a parallel fallback that independently introspects generators, configs, model registries, aliases, or admitted policy data under the old resolver contract.

Final search gate should have no production/current-doc matches for:

```text
SchedulingHintResolver
SchedulingHint
_model_aliases_for_generator
```

Independent scheduler-side `is_llm_bound` fallback is also migration-only and should be folded behind metadata/resource requests by epic completion.

## Request Admission Cleanup

The durable request-admission names are:

- `ModelRequestExecutor`
- `RequestAdmissionController`
- `AdaptiveRequestAdmissionController`
- `RequestAdmissionConfig`
- `RequestDomain`

Final search gate should have no production/current-doc matches for:

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

## Compatibility Shim Rule

Do not leave production compatibility aliases, subclasses, adapters, reexports, docs paths, or durable tests for replaced names at epic completion.

Temporary names may exist inside a PR only if the same PR removes them before merge.

## Gate Semantics

Before the migration issues land, stale-name matches can exist as current-state evidence.

By #653 close, legacy scheduling-hint production paths are gone and tests have moved to metadata/resolver coverage.

By #657 close, request-admission code has no production `Throttle*` aliases, exports, modules, or durable tests.

By #645 close, public/current docs and `plans/645` use only the durable architecture vocabulary except for this cleanup file's explicit legacy-name search lists. Historical changelog or dev-note text can remain only when explicitly marked historical.

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

User/operator docs should expose public run config fields, `AsyncCapacityPlan`, benchmark artifacts, telemetry views, and high-level layer names. They must not present `TaskAdmissionConfig`, `RequestAdmissionConfig`, policies, leases, queues, or controller mutation APIs as public user knobs. Plugin-facing docs should describe metadata only, then link to architecture docs for maintainers/operators.

Current architecture docs, diagrams, generated assets, and plan files must be checked as part of final cleanup. Existing historical dev notes may retain old names only when the text clearly says the name is historical and no longer current API.

Current user/operator architecture docs must also remove or mark as historical semaphore/throttling descriptions that imply the pre-epic architecture. This includes old model-client throttling names and semaphore-based scheduling explanations.

## Validation Commands

Adjust paths as files move, but final PRs should include searches equivalent to:

```bash
rg "SchedulingHintResolver|SchedulingHint|_model_aliases_for_generator|is_llm_bound" packages docs fern architecture plans/645
rg "ThrottleManager|ThrottleDomain|ThrottleConfig|RunConfig\\.throttle|throttle_manager\\.py|ThrottledModelClient|throttled_model_client" packages docs fern architecture plans/645
rg "_submission_semaphore|_llm_wait_semaphore|get_semaphore_permits|TrackingSemaphore" packages docs fern architecture plans/645
rg "throttl(e|ed|ing)|semaphore" docs fern architecture plans/645
rg "needs_llm_wait|held_llm_wait|max_llm_wait_tasks" packages docs fern architecture plans/645
```

Any remaining hit must be intentionally historical, not a current implementation or docs path. Allowed plan hits are limited to explicit cleanup/search-gate sections that name the legacy strings so reviewers know what to remove. The task-stage semaphore-specific search distinguishes obsolete submission/LLM-wait scheduling semaphores from unrelated internal synchronization primitives that may remain after review.
