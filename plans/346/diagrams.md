# Async Generators & Task Queue — Reference Diagrams

## 1. Task Lifecycle

How a single task flows through the scheduler, from dispatch to completion or failure.

```mermaid
flowchart TD
    R[Ready Queue] --> A[Acquire scheduler slot]
    A --> P[Prepare request]
    P --> S{LLM-bound?}

    S -->|No| E[Execute task]
    S -->|Yes| RS[Release scheduler slot]
    RS --> T[Acquire throttle permit\nfrom Throttle Manager\nkeyed by provider+model]
    T --> E

    E --> O{Outcome}

    O -->|Success| MC[Write result via update_cell]
    MC --> CT[Mark complete in tracker]
    CT --> W[Wake scheduler\nto check new ready tasks]

    O -->|Retryable failure\n429, 500, timeout| D[Deferred queue\nattempt++, backoff + jitter]

    O -->|Non-retryable failure\n400, validation, schema| DR[Mark row as dropped]
    DR --> W
```

## 2. Scheduler Main Loop

The overall orchestration flow from start to row group checkpoint.

```mermaid
flowchart TD
    START[Start] --> SEED[Dispatch from_scratch tasks]
    SEED --> SEED_CHECK{Stateful generator?}
    SEED_CHECK -->|Yes| SER[Serialize per-instance\nrow group N before N+1]
    SEED_CHECK -->|No| PAR[Dispatch all row groups\nconcurrently]
    SER --> PRE
    PAR --> PRE

    PRE[Pre-batch barrier\nrun processors, reset tracker] --> LOOP

    LOOP[Query tracker:\nget_ready_tasks] --> READY{Tasks ready?}

    READY -->|Yes| DISPATCH[Dispatch tasks\nbehind scheduler semaphore]
    DISPATCH --> COMPLETE[On completion:\nupdate tracker]
    COMPLETE --> LOOP

    READY -->|No| DEFERRED{Deferred queue\nnon-empty?}

    DEFERRED -->|Yes, rounds left| SALVAGE[Run salvage round\nover retryable failures]
    SALVAGE --> LOOP

    DEFERRED -->|No, or budget exhausted| RG_CHECK{Row group\ncomplete?}

    RG_CHECK -->|Yes| POST[Post-batch processors]
    POST --> CP[Checkpoint to parquet\nfree memory]
    CP --> DONE{All row groups\ndone?}

    RG_CHECK -->|No| WAIT[Wait for in-flight\ntasks to complete]
    WAIT --> LOOP

    DONE -->|Yes| FIN[Done]
    DONE -->|No| LOOP
```

## 3. Dependency Resolution Example

A concrete pipeline with 5 columns showing parallel execution opportunities.
Columns `B` and `C` are independent and run in parallel once `A` completes.

```mermaid
gantt
    title Row Group 0 — Task Execution Timeline
    dateFormat YYYY-MM-DD
    axisFormat %d
    tickInterval 1day

    section Seed
    A (from_scratch)                :a, 2026-01-01, 2d

    section Pre-batch
    Pre-batch processors            :pre, after a, 1d

    section Cell-by-cell
    B (LLM text, depends on A)     :b, after pre, 4d
    C (LLM judge, depends on A)    :c, after pre, 3d

    section Full-column
    D (expression, depends on B+C) :d, after b, 1d

    section Post-batch
    E (validation, depends on D)   :e, after d, 1d
    Post-batch + checkpoint        :post, after e, 1d
```

Dependency map for this example:
```
A: {}              ← no dependencies, from_scratch
B: {A}             ← cell-by-cell, waits for A per row
C: {A}             ← cell-by-cell, waits for A per row (parallel with B)
D: {B, C}          ← full-column, waits for B+C on ALL rows
E: {D}             ← full-column, waits for D
```

## 4. Concurrency Layers

The two-layer design: scheduler semaphore (coarse resource guard) and throttle
manager (per-key API concurrency). Tasks release scheduler slots while waiting
for throttle permits.

```mermaid
flowchart TB
    S[Async Scheduler] --> Q[Ready Task Queue]
    Q --> W[Scheduler Semaphore\ncoarse cap ~128]

    W --> T{LLM-bound?}

    T -->|No| N[Run non-LLM task\nexpression, sampler, etc.]
    N --> C[Completion Tracker]

    T -->|Yes| REL[Release scheduler slot]
    REL --> A[Acquire permit from\nThrottle Manager\nkeyed by provider+model+domain]
    A --> R[ModelClient adapter call]
    R --> P[Provider API]
    P --> X{429?}
    X -->|Yes| D[AIMD decrease\ncooldown + retry]
    D --> A
    X -->|No| U[AIMD increase]
    U --> C
```

### Failure mode this design avoids

Without slot release, a throttled key starves unrelated keys:

```mermaid
flowchart TB
    G[Single global semaphore\nno slot release] --> H[Many tasks for throttled key A]
    H --> I[Key A hits 429, backs off]
    I --> J[Tasks wait/retry\nwhile holding scheduler slots]
    J --> K[Unrelated key B tasks\ndelayed or starved]

    style G fill:#fee,stroke:#c33
    style K fill:#fee,stroke:#c33
```

## 5. Row Group Pipelining

Multiple row groups overlap — row group 1 starts its independent columns while
row group 0 is still finishing later columns.

```mermaid
gantt
    title Cross-Row-Group Pipelining
    dateFormat YYYY-MM-DD
    axisFormat %d
    tickInterval 1day

    section Row Group 0
    RG0 seed (A)     :rg0a, 2026-01-01, 2d
    RG0 col B        :rg0b, after rg0a, 4d
    RG0 col C        :rg0c, after rg0a, 3d
    RG0 col D        :rg0d, after rg0b, 1d
    RG0 checkpoint   :rg0cp, after rg0d, 1d

    section Row Group 1
    RG1 seed (A)     :rg1a, after rg0a, 2d
    RG1 col B        :rg1b, after rg1a, 4d
    RG1 col C        :rg1c, after rg1a, 3d
    RG1 col D        :rg1d, after rg1b, 1d
    RG1 checkpoint   :rg1cp, after rg1d, 1d

    section Row Group 2
    RG2 seed (A)     :rg2a, after rg1a, 2d
    RG2 col B        :rg2b, after rg2a, 4d
    RG2 col C        :rg2c, after rg2a, 3d
    RG2 col D        :rg2d, after rg2b, 1d
    RG2 checkpoint   :rg2cp, after rg2d, 1d
```

Note: seed (A) is shown staggered because `SeedDatasetColumnGenerator` is
stateful (`is_stateful = True`), so row groups serialize for that generator.
Columns B and C are stateless and pipeline freely across row groups.
