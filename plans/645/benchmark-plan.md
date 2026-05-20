# Benchmark Plan

The benchmark harness turns architecture claims into reusable evidence. It prevents each implementation PR from inventing one-off scripts and makes fairness/throughput tradeoffs explicit.

Until issue #649 closes, implementation PRs that need scheduling evidence must emit the provisional artifact schema in this file. A minimal deterministic smoke entrypoint and artifact writer should exist before the risky task/request admission implementation slices rely on it; issue #649 formalizes the reusable harness and reruns the provisional evidence against the accepted implementation chain before issue #645 closes. This prevents task/request admission PRs from landing without evidence while still allowing the harness to mature after capacity and telemetry contracts stabilize.

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
- scenario version
- harness version
- mock provider transcript or scripted provider behavior when live providers are not used
- monotonic clock/retry schedule when deterministic replay is claimed

Required artifacts:

- JSON and CSV outputs
- concise Markdown summary
- baseline and candidate commit SHAs
- command lines
- machine/runtime information
- environment knobs
- `AsyncCapacityPlan`
- per-layer observed maxima
- final task admission snapshot
- final request admission snapshot, or explicit `not_available_until_issue` marker before #657 lands
- completion timeline
- ready-idle/utilization timeline
- deterministic output hashes where applicable

Final snapshots must prove zero active task leases, zero request leases, zero request waiters, and no resource-specific permit leaks after all terminal paths complete. Before #657 lands, request snapshot fields remain present but can carry `not_available_until_issue: 657` rather than fabricated zeros.

The sync path can be used as a correctness/hash oracle, not as the timing baseline for async scheduling policy.

## Artifact Schema

The provisional and final JSON artifacts use monotonic seconds for timeline fields and stable scenario ids for comparison:

```text
scenario_id
artifact_schema_version
scenario_version
harness_version
baseline_sha
candidate_sha
inputs
provider_script
clock_script
capacity_plan
iterations[]
  wall_time_seconds
  timeline[]
    event_kind
    captured_at_monotonic
    stream
    sequence
    captured_correlation
      run_id
      row_group
      task_column
      task_type
      scheduling_group_kind
      scheduling_group_identity_hash
      task_execution_id
    task_id
    task_execution_id
    task_lease_id
    request_attempt_id
    request_lease_id
    scheduler_resource_key
    request_resource_key
    reason_or_outcome
  final_task_snapshot
  final_request_snapshot
  output_hashes
derived_metrics
```

Derived metrics:

- `ready_queue_wait = selected_at - ready_enqueued_at`
- `task_admission_wait = task_lease_acquired_at - selected_at`
- `ready_to_lease_gap = task_lease_acquired_at - ready_enqueued_at`
- `ready_idle_gap` is derived from intervals where dependency-ready work exists, scheduler task capacity is available, and no task lease is acquired. Per-task `selected_at -> task_lease_acquired_at` is task admission overhead, not the starvation metric.
- `active_capacity_integral = integral(active_leases / configured_capacity) over wall time`
- `root_over_admission_debt = admitted root work above strict fair share after first downstream-ready timestamp`
- `hidden_scheduler_resource_waiters` is the count of spawned workers waiting for scheduler-level resources that should have been acquired before spawn. After the task-admission lease boundary lands, the event stream should prove this is zero by showing no worker-spawned event before the corresponding task-lease-acquired event and no pre-epic scheduler-resource wait event for the task.
- deterministic hashes include generated output values and stable ordering metadata, not timing or event ids

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
max(hidden_scheduler_resource_waiters) == 0
```

Required monotonic timeline fields:

- dependency_ready_at
- ready_enqueued_at
- selected_at
- task_lease_acquired_at
- worker_spawned_at
- request_wait_started_at
- request_wait_completed_at
- request_lease_acquired_at
- model_request_started_at
- model_request_completed_at
- request_lease_released_at
- task_completed_at
- task_lease_released_at

Scheduler events own selected/task-lease/spawn/task-completion/task-release. Request/model instrumentation owns request wait, request lease, model request start/complete, and request release.

Immediate request acquisition records `request_wait_started_at == request_wait_completed_at` so the timeline can distinguish a zero wait from missing instrumentation.

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
- cross-provider cooldown workflow where provider A is rate-limited or cooling down while provider B has ready independent work

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

All timing gates use paired same-machine runs with at least five measured iterations unless the scenario explicitly raises that count. Reports include mean, p50, p95, min, max, standard deviation, and a noise-floor note. If standard deviation is large enough to make a threshold ambiguous, the PR must either add iterations or treat the timing claim as inconclusive.

Neutral scenarios should be no worse than 5 percent mean wall time unless the PR explicitly justifies a fairness/utilization tradeoff.

Heavy-root scenarios should show reduced downstream ready-to-dispatch lag versus the named baseline when the candidate claims to improve heavy-root behavior.

Every run must show no permit leaks and deterministic output equality where applicable.

Scenario-specific gates:

- Queue/admission microbench: p95 admission cycle cost must not regress more than 10 percent unless the PR documents a fairness or correctness tradeoff.
- Heavy-root benchmark: p95 ready-to-dispatch gap for downstream work must improve versus the named baseline when the candidate claims heavy-root fairness; root over-admission debt must be bounded by the configured policy.
- Hidden-waiter proof: `max(hidden_scheduler_resource_waiters) == 0` across success, failure, cancellation, and salvage paths after task admission lands.
- Idle/utilization proxy: ready-idle gaps while eligible work and capacity are available must be zero except for documented event-loop scheduling granularity.
- Dynamic request benchmark: zero/one/many request tasks must produce matching output hashes, request lease counts must equal concrete outbound attempts, and request wait/execute/release timelines must be monotonic.
- Cross-provider cooldown benchmark: provider B ready work must continue to receive scheduler task leases while provider A is blocked by request cooldown once the provider-aware policy in #651 claims that optimization.
- Variance: measured iterations must report mean, p50, p95, min, max, and standard deviation. Any acceptance claim based on timing should remain directionally true after removing the fastest and slowest measured iteration.

## CI Smoke

The harness should have a small deterministic smoke mode using mock endpoints. It writes machine-readable artifacts and does not require live providers.
