# Benchmark Plan

The benchmark harness turns architecture claims into reusable evidence. It prevents each implementation PR from inventing one-off scripts and makes fairness/throughput tradeoffs explicit.

## Harness Requirements

Provide a repo-local benchmark entrypoint that can compare two refs or checkouts.

Required inputs:

- baseline ref
- candidate ref
- scenario
- record count
- buffer size
- row-group concurrency
- task admission capacity
- request latency knobs
- warmups
- measured iterations
- output directory
- seed

Required artifacts:

- JSON and CSV outputs
- concise Markdown summary
- baseline and candidate commit SHAs
- command lines
- machine/runtime information
- environment knobs
- `AsyncCapacityPlan`
- per-layer observed maxima
- completion timeline
- ready-idle/utilization timeline
- deterministic output hashes where applicable

The sync path can be used as a correctness/hash oracle, not as the timing baseline for async scheduling policy.

## Scenario Matrix

### Queue And Admission Microbench

Compare old `admit_next + release` behavior with new `select_next + try_acquire + commit + release` behavior.

Matrix:

- task counts: 1k, 10k, 100k
- group counts: 1, 8, 64, 256
- resource mixes: local only, resource-bound, mixed local/resource-bound, stateful/exclusive if included

Metrics:

- p50/p95 admission cycle cost
- enqueue/select/acquire/commit/release breakdown
- total CPU time
- peak memory
- scaling by group count

### Heavy-Root Downstream Benchmark

Required shape:

```text
true_from_scratch_root_slow -> downstream_fast
```

Optional secondary shape:

```text
seed -> root_slow -> downstream_fast
```

Required metrics:

- first downstream-ready time
- first downstream-dispatch time
- ready-but-not-running gap
- root over-admission debt after first downstream-ready timestamp
- time to first completed record
- time to 50 percent completed records
- p95 row completion time
- final wall time
- max ready-idle gap by group/resource
- active-capacity integral

This scenario must exercise true root/from-scratch dispatch, not only downstream slow tasks.

### Hidden-Waiter Proof

After task admission lands:

```text
max(spawned_waiting_for_llm_lease) == 0
```

Required monotonic timeline fields:

- selected_at
- lease_acquired_at
- worker_spawned_at
- model_request_started_at
- model_request_completed_at
- lease_released_at

Scheduler events own selected/lease/spawn/release. Request/model instrumentation owns model request start/complete.

### Idle And Utilization Proxy

Use mock endpoint pools with request start/end events.

Metrics:

- active-capacity integral
- max ready-idle gap while work is available
- initial idle gap after first downstream-ready task

### End-To-End A/B Timing

Run paired A/B trials with warmup and at least five measured iterations for:

- narrow serial workflow
- wide independent roots
- dual model generate-to-judge workflow
- heavy-root workflow
- dynamic request-count custom generator workflow

### Request Dynamic-Call Benchmark

Use custom generators that make zero, one, and many model calls per task, including branch-dependent request counts.

Metrics:

- request admission acquire/release overhead
- queue wait
- event emission overhead
- emitted event count
- CPU time
- memory
- end-to-end timing

## Baselines

| Consumer | Baseline | Candidate |
| --- | --- | --- |
| Task admission | `origin/main` or the implementation PR merge-base before `TaskAdmissionController` | task-admission PR |
| Bounded borrow | accepted lease-only task-admission SHA | bounded-borrow PR |
| Resource vector | accepted bounded-borrow SHA or named policy baseline | resource-vector policy PR |

## Evidence Thresholds

Neutral scenarios should be no worse than 5 percent mean wall time unless the PR explicitly justifies a fairness/utilization tradeoff.

Heavy-root scenarios should show reduced downstream ready-to-dispatch lag versus the named baseline when the candidate claims to improve heavy-root behavior.

Every run must show no permit leaks and deterministic output equality where applicable.

## CI Smoke

The harness should have a small deterministic smoke mode using mock endpoints. It writes machine-readable artifacts and does not require live providers.
