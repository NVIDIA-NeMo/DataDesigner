# Migration And Cleanup

The epic is not complete until replaced names and compatibility paths are removed from production code and current docs.

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
- `RunConfig.request_admission`
- `RequestDomain`

Final search gate should have no production/current-doc matches for:

```text
ThrottleManager
ThrottleDomain
ThrottleConfig
RunConfig.throttle
throttle_manager.py
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

By #645 close, public/current docs use only the durable architecture vocabulary. Historical changelog or dev-note text can remain only when explicitly marked historical.

## Documentation Cleanup

Current user/operator docs should use:

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

Docs must not present scheduler internals as plugin API. Plugin-facing docs should describe metadata only, then link to architecture docs for maintainers/operators.

## Validation Commands

Adjust paths as files move, but final PRs should include searches equivalent to:

```bash
rg "SchedulingHintResolver|SchedulingHint|_model_aliases_for_generator|is_llm_bound" packages docs fern
rg "ThrottleManager|ThrottleDomain|ThrottleConfig|RunConfig\\.throttle|throttle_manager\\.py" packages docs fern
rg "needs_llm_wait|held_llm_wait|max_llm_wait_tasks" packages docs fern
```

Any remaining hit must be intentionally historical, not a current implementation or docs path.
