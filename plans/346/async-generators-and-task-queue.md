# Plan: Async Generators & Task Queue Builder

Created: 2026-02-20
Status: Planning

Issue: [#346](https://github.com/NVIDIA-NeMo/DataDesigner/issues/346)

Related:
- [#260](https://github.com/NVIDIA-NeMo/DataDesigner/issues/260) — original async engine plan
- [PR #280](https://github.com/NVIDIA-NeMo/DataDesigner/pull/280) — async ModelFacade (merged)
- [PR #269](https://github.com/NVIDIA-NeMo/DataDesigner/pull/269) — execution graph reference impl (draft)
- [PR #344](https://github.com/NVIDIA-NeMo/DataDesigner/pull/344) — model facade overhaul plans

## Goal

Transform the dataset builder from sequential column-by-column processing into an
async task queue with dependency-aware scheduling. Generators become async-first,
and the builder dispatches individual cell/batch tasks as soon as their upstream
dependencies are satisfied — enabling pipeline parallelism across columns and rows.

### Current architecture

```
for batch in batches (of buffer_size):       # sequential
    for column in columns:                   # sequential
        if from_scratch: generate_from_scratch(batch)
        elif cell_by_cell: fan_out(cells)    # parallel within column
        elif full_column: generate(df)
    checkpoint(batch)
```

Columns execute sequentially even when they have no mutual dependency. Rows in
different batches never overlap. Only cell-level fan-out within a single column
is parallelised.

### Target architecture

```
all tasks across all row groups submitted to a single async scheduler
scheduler dispatches tasks as dependencies are met, bounded by semaphore
row groups checkpointed to parquet when fully complete
```

Multiple columns can execute in parallel when they don't depend on each other.
Rows from different row groups can pipeline (row group 1 column B starts while
row group 0 column C is still running).

## Key Design Decisions

### 1. Column-level static execution graph

- At setup: build an `ExecutionGraph` (column-level DAG) from each column's
  `config.required_columns` property (already available on all config types via
  Jinja2 template introspection) and the generation strategy from each generator.
- The graph also registers **side-effect output columns** (e.g.,
  `__trace`, `__reasoning_content`) and maps them back to their producer generator.
  A downstream column referencing `summary__trace` resolves to a dependency on
  the `summary` generator. This ensures side-effect columns are never missing from
  the graph or treated as unsatisfied.
- At runtime: a **completion tracker** (columns × rows matrix) determines which
  tasks are ready by checking whether all upstream columns for a given row are done.

The graph is column-granularity only — no cell-level nodes — so it stays small
(N columns, N edges) regardless of row count. Scheduling remains dynamic: the
completion tracker drives readiness checks as tasks complete, with no upfront
planning of execution order. The static graph adds inspectability (visualization,
critical path, upfront task counts, error attribution) without changing how the
scheduler operates at runtime.

### 2. Task granularity

| Generator type | Task unit | Readiness condition |
|---|---|---|
| `FromScratch` (seed, sampler) | `(column, row_group)` | No dependencies (always first) |
| `CELL_BY_CELL` (LLM text/code/structured/judge, image, embedding, custom) | `(column, row)` | All `required_columns` complete for that row |
| `FULL_COLUMN` (expression, validation, sampler-as-transform) | `(column, row_group)` | All `required_columns` complete for ALL rows in row group |

### 3. Row groups as checkpoint units

Rows are partitioned into groups of `buffer_size` (same as current batches).
When all tasks for a row group are complete, write to parquet and free memory.
This preserves the current checkpoint/memory semantics.

Row groups may complete **out of order** (e.g., row group 2 finishes before row
group 1 if RG1 has a slow column). Checkpoint writes use the row group index for
file naming (`batch_0.parquet`, `batch_1.parquet`, etc.), so out-of-order writes
produce correctly named files. When loading the final dataset, files are read in
index order, so row ordering is preserved regardless of write order.

Full-column generators operate on their entire row group at once, same as today.

### 4. Concurrency control

Three independent layers:

1. **Scheduler semaphore** — a coarse resource guard bounding total in-flight
   active execution to limit CPU/memory pressure (e.g., configurable cap, default
   ~128). This is **not** the source of truth for API concurrency.

2. **Throttle manager** (from PR #344) — gates every outbound LLM call, keyed
   by `provider+model(+domain)`. Dynamically adjusts per-key limits on 429s
   via AIMD. This is the real API concurrency control.

3. **Submission budget** — a hard cap on "submitted but not finished" tasks
   (running + waiting on throttle/backoff), e.g., `async_scheduler_max_submitted_tasks`.
   This prevents unbounded parked coroutines when tasks release scheduler slots
   before throttle acquire.

Tasks must **not hold scheduler slots while waiting on throttle backoff**. A task
acquires a scheduler slot, prepares its request, then releases the slot before
awaiting the throttle permit. This ensures a throttled model key doesn't starve
unrelated keys by hogging scheduler capacity.

Task admission is therefore bounded by the submission budget, while active
execution is bounded by the scheduler semaphore. This composes cleanly with
PR #344's adaptive throttling without the two systems fighting each other.

### 5. Generator statefulness and reentrancy

Statefulness and sync/async are orthogonal concerns. Sync vs async is about the
**I/O model** — whether the underlying work is blocking (needs a thread) or
non-blocking (native coroutine). Statefulness is about **concurrency safety** —
whether multiple calls to the same generator instance can safely overlap. A
generator can be async but stateful (e.g., a cursor over an async database), or
sync but stateless (e.g., a random sampler).

Generators declare whether they are stateful via an `is_stateful` property on
the base class (default `False`). Stateful generators maintain internal state
across calls (e.g., `SeedDatasetColumnGenerator` has a DuckDB batch reader
cursor and leftover-row buffer). The scheduler **serializes tasks per-instance**
for stateful generators — row group N must complete before row group N+1 starts
for that generator. Stateless generators (e.g., `SamplerColumnGenerator`) can
dispatch all row groups concurrently.

This is a generator-level attribute, not a type-level assumption. Custom
generators declare their own contract.

The row group admission semaphore (`async_max_concurrent_row_groups`) and stateful
serialization are complementary, not conflicting: the semaphore controls how many row
groups are admitted into the scheduler at once; serialization controls the dispatch
order of seed tasks within the admitted set. A stateful generator with 3 row groups
admitted will still run their seeds in order (0 → 1 → 2); other columns in those row
groups remain free to pipeline.

### 6. Pre/post-batch processors

- **Pre-batch**: runs after seed generators complete for a row group, before
  other columns. Modeled as a barrier task for the row group. If a pre-batch
  processor fails, the entire row group is skipped.
- **Post-batch**: runs after all columns complete for a row group, before
  checkpoint write.

### 7. Retry & salvage policy

In deep pipelines, a transient failure on a late column drops the entire row,
wasting all upstream generation work. Controlled retry rounds recover rows that
would otherwise be lost.

1. **Classify failures**: transient (429, 500, timeout) → retryable; permanent
   (400, validation error, schema mismatch) → non-retryable (immediate drop).
2. **Deferred queue**: retryable failures are placed in a deferred queue with
   `attempt` count, `next_eligible_at` timestamp, and exponential backoff + jitter.
3. **Scheduling priority**: normal ready tasks are dispatched first. When the
   ready queue drains, the scheduler sweeps the deferred queue and re-dispatches
   all tasks whose backoff has elapsed — this is one salvage round. The scheduler
   runs at most `async_salvage_max_rounds` such rounds (default 2) before dropping
   all remaining deferred tasks. A task that keeps failing is retried once per round,
   so the maximum attempts per task is `async_salvage_max_rounds + 1`.
4. **Separate error threshold**: salvage rounds use their own error rate threshold
   (e.g., `async_salvage_error_threshold=0.8`), independent of the main scheduling
   loop, since higher failure rates are expected when retrying.
5. **Throttle-aware**: retries re-enter the throttle manager acquire path, so
   they don't exacerbate rate limiting.
6. **Final drop**: after retry budget is exhausted, mark the cell as failed and
   the row as dropped (via eager row-drop propagation). Continue row-group
   completion checks over remaining rows.

### 8. `allow_resize` scoping

The completion tracker uses row indices as stable identifiers. `allow_resize`
lets any generator change the row count mid-pipeline, which invalidates all
per-row completion state for downstream columns. Supporting this under parallel
execution would require dynamic rescheduling and row identity tracking.

**Async v1 scope**: if any column config has `allow_resize=True` and
`DATA_DESIGNER_ASYNC_ENGINE=1`, the builder raises a `DatasetGenerationError`
at startup (before any generation begins). The user must either remove
`allow_resize=True` from their config or disable the async engine. Silent
fallback is intentionally avoided — the user should know these two settings are
incompatible and make an explicit choice (see Follow-ups).

## Success Criteria

- [ ] All generators expose async-first `agenerate` (cell-by-cell) or async wrappers (full-column/from-scratch)
- [ ] Builder dispatches tasks based on dependency readiness, not column order
- [ ] Multiple columns execute in parallel when dependencies allow
- [ ] Row groups checkpoint to parquet upon full completion
- [ ] Existing sync path (`DATA_DESIGNER_ASYNC_ENGINE=0`) continues to work unchanged
- [ ] All existing tests pass; new tests cover dependency resolution and scheduling
- [ ] `make test-run-recipes` passes with async engine enabled

## Implementation Steps

### Step 1: Execution Graph

Build a column-level static execution graph from column configs at builder init time.
The graph is column-granularity only — no cell-level nodes — so it stays small (N columns,
N edges) regardless of row count and avoids the barrier/checkpoint problems of a cell-level
graph.

- [ ] `ExecutionGraph` class:
  - Backing stores: `dict[str, set[str]]` column → upstream columns;
    `dict[str, GenerationStrategy]` column → generation strategy
  - `upstream(column: str) -> set[str]` — direct dependencies of a column
  - `downstream(column: str) -> set[str]` — columns that depend on this one (for error attribution)
  - `strategy(column: str) -> GenerationStrategy` — cell-by-cell or full-column
  - `topological_order() -> list[str]` — valid DAG execution order (used by scheduler and for validation)
  - `critical_path() -> list[str]` — longest dependency chain (useful for ETA estimates)
  - `task_count(num_records: int, buffer_size: int) -> dict[str, int]` — exact task count per
    column before the run starts; cell-by-cell columns produce `num_records` tasks,
    full-column columns (including from-scratch generators, which report `FULL_COLUMN`)
    produce `ceil(num_records / buffer_size)` tasks
  - `to_mermaid() -> str` — Mermaid diagram string; nodes are annotated with strategy type
- [ ] `build_execution_graph(column_configs, strategies: dict[str, GenerationStrategy]) -> ExecutionGraph` utility:
  - Input: the ordered list of `ColumnConfigT` / `MultiColumnConfig`, plus a pre-computed
    strategy map (available from generators at builder init time via `get_generation_strategy()`)
  - For each config, read `config.required_columns` → set of upstream column names
  - Also register side-effect output columns (`__trace`, `__reasoning_content`, etc.)
    and map them back to their producer column, so downstream references resolve correctly
  - For `MultiColumnConfig`, all sub-columns share the same dependencies
  - Validate: every required column must resolve to a known producer (including
    registered side-effect outputs), and the graph must be acyclic
- [ ] Unit tests for graph construction, validation, critical path, task count, and Mermaid output

**Files**: new module `engine/dataset_builders/utils/execution_graph.py`, tests

### Step 2: Completion Tracker

A lightweight structure tracking which (column, row_group, row_index) tuples are
done. Row indices are **local** to their row group (0-based within each group),
matching the buffer manager's per-row-group addressing.

- [ ] `CompletionTracker` class:
  - Internal: `dict[int, dict[str, set[int]]]` mapping row_group → column → set of completed local row indices
  - `mark_complete(column: str, row_group: int, row_index: int)` / `mark_batch_complete(column: str, row_group: int, row_group_size: int)`
  - `is_ready(column: str, row_group: int, row_index: int, graph: ExecutionGraph) -> bool` — checks all upstream columns for that (row_group, row_index)
  - `is_batch_ready(column: str, row_group: int, row_group_size: int, graph: ExecutionGraph) -> bool` — checks all rows in group
  - `drop_row(row_group: int, row_index: int)` — marks row as dropped across all columns;
    `get_ready_tasks` skips dropped rows, in-flight tasks for dropped rows are ignored on completion
  - `is_row_group_complete(row_group: int, row_group_size: int, all_columns: list[str]) -> bool` — all non-dropped rows have all columns done
  - `get_ready_tasks(graph: ExecutionGraph, row_groups) -> list[Task]` — yields all currently dispatchable tasks, excluding dropped rows; reads `graph.strategy(column)` to determine task granularity per column
- [ ] No locks needed: all access is from the single asyncio event loop thread
- [ ] Unit tests

**Files**: new module `engine/dataset_builders/utils/completion_tracker.py`, tests

### Step 3: Task Model

Simple dataclass representing a unit of work.

- [ ] `Task` dataclass:
  - `column: str`
  - `row_group: int`
  - `row_index: int | None` (None for batch tasks)
  - `task_type: Literal["from_scratch", "cell", "batch", "pre_batch_processor", "post_batch_processor"]`
- [ ] `TaskResult` with status, output, error info
- [ ] `TaskTrace` dataclass (only instantiated when tracing is enabled):
  - `column: str`, `row_group: int`, `row_index: int | None`, `task_type: str`
  - `dispatched_at: float` — `perf_counter()` when `create_task()` fires
  - `slot_acquired_at: float` — after scheduler semaphore acquired
  - `completed_at: float` — in `finally` block after generator returns
  - `status: str`, `error: str | None`
- [ ] Hashable so we can track dispatched/pending sets

**Files**: new module `engine/dataset_builders/utils/task_model.py` — must be its own module
since `CompletionTracker`, `AsyncTaskScheduler`, and the buffer manager all reference `Task`/`TaskResult`;
inlining would create import cycles.

### Step 4: Async Task Scheduler

The core orchestrator that replaces `_run_batch` for the async path.

- [ ] `AsyncTaskScheduler` class:
  - Constructor takes: generators (by column name), `graph: ExecutionGraph`, completion tracker, row group definitions, concurrency limit (`async_scheduler_max_submitted_tasks`), row group semaphore (`async_max_concurrent_row_groups`), salvage config, error/result callbacks, `trace: bool = False`
  - When `trace=True`, populates `scheduler.traces: list[TaskTrace]` (one record per task); otherwise no `TaskTrace` objects are created. See Profiling.
  - `async run()` — main loop:
    1. Acquire the row group semaphore (`async_max_concurrent_row_groups`) before
       admitting a new row group's seed tasks. Dispatch `from_scratch` tasks,
       respecting `is_stateful`: stateful generators serialize per-instance (row group
       N's seed completes before N+1's seed starts for that generator); stateless
       generators dispatch all admitted row groups concurrently
    2. Loop: query `completion_tracker.get_ready_tasks()` → dispatch each via
       `asyncio.create_task()` behind scheduler semaphore → on completion, update
       tracker → repeat until all tasks done or early shutdown
    3. When ready queue drains, run salvage rounds over deferred retryable failures
       (up to `async_salvage_max_rounds` rounds)
    4. After each row group completes: run post-batch processors, checkpoint
  - Task dispatch: acquire scheduler semaphore slot → prepare request → release
    slot → await throttle permit (for LLM tasks) → execute → write result
  - Admission control: never allow more than `async_scheduler_max_submitted_tasks`
    tasks in submitted/running/waiting states; hold remaining ready tasks in the
    scheduler queue until slots free up
  - Error handling: classify failures as retryable vs non-retryable; retryable
    go to deferred queue with backoff; same early-shutdown logic as
    `AsyncConcurrentExecutor` (error rate threshold within sliding window)
  - Progress tracking: create one `ProgressTracker` per column for accounting
    (success/failure counts, rate, ETA), but suppress per-completion interval logs
    in async mode. A separate background coroutine (`asyncio.create_task`) emits a
    single consolidated summary line every 10 seconds across all active columns;
    it is cancelled once all tasks complete. See UX Considerations.
- [ ] Use `asyncio.Event` to wake the scheduler when a task completes (avoids polling).
  `Event` is sufficient — the scheduler resets it and re-checks ready tasks on each wake;
  `Condition` would be needed only if waiting on a specific predicate, which the tracker
  already handles.
- [ ] Unit tests with mock generators

**Files**: new module `engine/dataset_builders/async_scheduler.py`, tests

### Step 5: Generator Async Migration

Make all generator types async-capable and declare statefulness.

**Symmetric `generate` / `agenerate` contract**: only one of the two methods needs
to be implemented. The base class provides automatic bridging in both directions:
- If only `generate()` is implemented → `agenerate()` wraps it via `asyncio.to_thread`
  (already exists from PR #280).
- If only `agenerate()` is implemented → `generate()` uses a safe sync runner
  helper:
  - no running loop in current thread: use `asyncio.run(self.agenerate(data))`
  - running loop detected: submit to the builder's dedicated background event loop
    thread via `asyncio.run_coroutine_threadsafe(...).result(timeout=...)`
  This avoids nested-loop errors while keeping async-first plugins ergonomic.

This means sync-first generators (most built-ins, existing plugins) work unchanged,
and async-first generators (new plugins doing native async I/O) only need to implement
`agenerate()` without writing a redundant sync version.

- [ ] Add symmetric bridging on the base `ColumnGenerator`:
  - `agenerate()` default: `asyncio.to_thread(self.generate, data)` (already exists)
  - `generate()` default: call a safe sync runner helper that:
    - uses `asyncio.run()` if no loop is running in the current thread
    - otherwise submits to the background loop with `run_coroutine_threadsafe(...).result(timeout=...)`
  - Detect which one the subclass overrides to avoid infinite recursion
- [ ] Add `is_stateful` property to base `ColumnGenerator` (default `False`).
  Stateful generators are serialized per-instance by the scheduler.
- [ ] `ColumnGeneratorWithModelChatCompletion.agenerate` — already implemented (PR #280), no changes needed
- [ ] `FromScratchColumnGenerator`: add both async wrappers — `async agenerate_from_scratch(num_records) -> DataFrame`
  (wraps `generate_from_scratch` in `asyncio.to_thread`) and `async agenerate(data: DataFrame) -> DataFrame`
  (wraps `generate` in `asyncio.to_thread` with defensive `df.copy()`). Both are needed because the
  scheduler dispatches subclasses via either path depending on whether the buffer is empty.
- [ ] `ColumnGeneratorFullColumn`: add `async agenerate(data: DataFrame) -> DataFrame` — wraps sync in
  `asyncio.to_thread` with defensive `df.copy()` (see Risks). This intentionally overrides the base
  `ColumnGenerator.agenerate(dict)` with a DataFrame-typed signature; the scheduler dispatches the
  correct variant based on generation strategy.
- [ ] `ExpressionColumnGenerator`: inherits full-column async wrapper
- [ ] `SamplerColumnGenerator`: inherits both wrappers from `FromScratchColumnGenerator`; no custom implementation needed. `is_stateful = False`
- [ ] `SeedDatasetColumnGenerator`: inherits both wrappers from `FromScratchColumnGenerator`; no custom implementation needed. `is_stateful = True` (maintains DuckDB batch reader cursor and leftover-row buffer)
- [ ] `ValidationColumnGenerator`: inherits full-column async wrapper. Note: for `REMOTE` validators
  with `max_parallel_requests > 1`, `generate()` internally uses `ConcurrentThreadExecutor`, so the
  async wrapper spawns a thread that itself spawns more threads — bypassing the scheduler's concurrency
  controls for those HTTP calls. Acceptable for v1 (see Follow-ups).
- [ ] `CustomColumnGenerator`: inherits directly from `ColumnGenerator` (not from
  `ColumnGeneratorFullColumn`), so it does not automatically inherit the full-column async wrapper. Needs its own
  `agenerate` that branches on strategy:
  - `CELL_BY_CELL`: if the user function is a coroutine (`asyncio.iscoroutinefunction`), call it directly;
    otherwise wrap in `asyncio.to_thread`
  - `FULL_COLUMN`: wrap `generate(DataFrame)` in `asyncio.to_thread` with defensive `df.copy()`
  `is_stateful` defaults to `False`; custom implementations can override it.
- [ ] `ImageCellGenerator`, `EmbeddingCellGenerator`: add native `agenerate` using `model.agenerate_image` / `model.agenerate_text_embeddings`

**Files**: `generators/base.py`, `generators/expression.py`, `generators/samplers.py`, `generators/seed_dataset.py`, `generators/image.py`, `generators/embedding.py`, tests

### Step 6: Buffer / Row Group Manager

Adapt `DatasetBatchManager` for concurrent row group processing.

- [ ] Support multiple row groups in-flight simultaneously (currently only one batch's buffer exists)
  - Option A: Multiple buffer instances (one per active row group)
  - Option B: Single shared buffer partitioned by row group offset ranges
  - Recommendation: **Option A** — cleaner isolation, each row group has its own `list[dict]`
- [ ] `update_cell(row_group: int, row_index: int, column: str, value: Any)` — cell-level
  merge is the only write path for the async builder. Whole-record replacement
  (`update_record`) is unsafe under parallel execution (two independent columns
  finishing the same row concurrently would clobber each other's results)
- [ ] `checkpoint_row_group(row_group: int)` — write parquet, free memory
- [ ] Preserve `drop_records` semantics within each row group
- [ ] Keep backward compatibility with sync path (the existing `DatasetBatchManager` is untouched)

**Files**: new class or extension in `dataset_batch_manager.py`, tests

### Step 7: Builder Integration

Wire the new scheduler into `ColumnWiseDatasetBuilder`.

- [ ] New method `_build_async(generators, num_records, buffer_size, ...)`:
  1. Build `ExecutionGraph` from `self._column_configs` and generator strategies
  2. Partition rows into row groups
  3. Create `CompletionTracker`, `AsyncTaskScheduler`
  4. Run scheduler on the background event loop (reuse `_ensure_async_engine_loop()`
     from `dataset_builders/utils/async_concurrency.py` — already exists)
  5. Scheduler handles checkpointing via callbacks
- [ ] `build()` raises `DatasetGenerationError` at startup if `DATA_DESIGNER_ASYNC_ENGINE=1`
    and any column config has `allow_resize=True`, naming the offending column(s);
    otherwise dispatches to `_build_async()`
- [ ] `build_preview()` uses the same async path (single row group, no checkpoint)
- [ ] Error handling: `DatasetGenerationError` wrapping, record dropping, telemetry events
- [ ] Processor integration:
  - Pre-batch: scheduler runs after seed tasks for a row group
  - Post-batch: scheduler runs after all column tasks for a row group, before checkpoint

**Files**: `column_wise_builder.py`

### Step 8: Tests & Validation

- [ ] Unit tests for each new module (execution graph, completion tracker, task model, scheduler)
- [ ] Execution graph: side-effect output columns resolve correctly (e.g., column
  depending on `summary__trace` maps to a dependency on the `summary` generator)
- [ ] Integration test: multi-column config with known dependencies, verify parallel execution
- [ ] Integration test: mixed cell-by-cell + full-column generators
- [ ] Integration test: error rate shutdown
- [ ] Integration test: checkpoint correctness (row groups written in order, parquet valid)
- [ ] Integration test: `allow_resize=True` with async engine raises `DatasetGenerationError` at startup, naming the column
- [ ] Integration test: stateful generator (`is_stateful=True`) serializes per-instance
  across row groups; stateless generators run concurrently
- [ ] Integration test: retry salvage — transient failure is retried and succeeds;
  non-retryable failure drops immediately; retry budget exhaustion drops correctly
- [ ] Integration test: throttling fairness — 429 on model key A does not stall
  unrelated model key B tasks
- [ ] Integration test: bounded submission — with many ready tasks and a tight
  throttle key, submitted task count never exceeds `async_scheduler_max_submitted_tasks`
- [ ] Integration test: eager row-drop — failure on column B drops the row across
  all columns, independent column C does not process the dropped row
- [ ] Integration test: row-drop with in-flight full-column task — completed task
  may still compute dropped rows, but writeback is suppressed and row remains dropped
- [ ] Integration test: out-of-order row group completion produces correctly named
  parquet files; final dataset loads in correct row order
- [ ] Integration test: pre-batch processor failure skips the row group, remaining
  row groups continue
- [ ] Run `make test` — all existing tests pass
- [ ] Run `make test-run-recipes` with `DATA_DESIGNER_ASYNC_ENGINE=1`
- [ ] Benchmark: compare sync vs async on a multi-column recipe with simulated latency; use `trace=True` and load `scheduler.traces` into a DataFrame to measure per-column dispatch and execution times

## Risks & Considerations

- **Memory with concurrent row groups**: Having multiple row groups in-flight increases
  peak memory. Mitigation: a dedicated semaphore caps concurrent in-flight row groups,
  controlled by `async_max_concurrent_row_groups` (default 3). The scheduler only admits
  a new row group's seed tasks once a slot is available.

- **Unbounded parked coroutines during throttle waits**: Releasing scheduler slots
  before throttle acquire improves fairness, but can create large numbers of parked
  tasks if admission is not bounded. Mitigation: enforce
  `async_scheduler_max_submitted_tasks` as a hard cap on submitted/running/waiting
  tasks.

- **Eager row-drop propagation**: When a task fails non-recoverably (non-retryable,
  or retry budget exhausted), the **entire row** must be marked as dropped across all
  columns — not just the failed column. Otherwise, independent columns that don't
  depend on the failed column will continue processing that row, wasting compute on
  a row that can never be complete. The completion tracker needs a `drop_row(row_group,
  row_index)` method that skips all pending tasks for that row; in-flight tasks may
  still complete but their writeback is suppressed once the row is marked dropped.
  Retryable failures go to the deferred queue first; eager drop only happens after
  retries are exhausted. Row group is complete when all non-dropped rows have all
  columns done.

- **Dropped rows vs in-flight batch/full-column work (v1 policy)**: preemptively
  cancelling already-running full-column/batch tasks is complex and error-prone.
  Async v1 keeps this simple: once a row is dropped, scheduler will not enqueue new
  tasks for that row and all write paths must suppress writeback for dropped rows.
  Already-running batch/full-column tasks may still compute values for dropped rows,
  but those outputs are ignored. Dropped-row propagation is strictly row-scoped;
  a row-group/batch is never dropped solely due to row-level failures.

- **Sync bridge in async-hosted contexts**: async-first generators need a safe
  `generate()` fallback that works even when called from environments with an active
  event loop (notebooks/services). Mitigation: use a sync runner helper that uses
  `asyncio.run()` when safe, else routes through the dedicated background event loop
  via `run_coroutine_threadsafe(...).result(timeout=...)`.

- **Full-column generator ordering**: If two full-column generators have no mutual
  dependency, they could run in parallel on the same row group. This is safe as long
  as they operate on independent columns. `asyncio.to_thread` passes **object
  references**, not copies — if two full-column generators share the same DataFrame,
  concurrent mutation is possible. Solution: pass `df.copy()` to each full-column
  generator dispatched to a thread, and merge results back by column name.

- **Pre-batch processors mutating data**: Pre-batch processors (e.g., schema transform)
  can add/remove/modify rows. This changes the row count and invalidates the completion
  tracker's row indices. Solution: treat pre-batch as a barrier that resets the tracker
  state for that row group (re-index rows after processor runs). If a pre-batch
  processor **fails**, the entire row group is skipped (treated as a fatal row-group
  error — log, skip, continue with remaining row groups).

- **Undersized last row group**: If `num_records` is not a multiple of `buffer_size`,
  the last row group has fewer rows. This is the same as the sync path and should not
  require special handling, but full-column generators and batch-level logic must not
  assume uniform row group sizes.

- **`allow_resize` incompatibility**: Any generator with `allow_resize=True` can change
  the row count mid-pipeline, invalidating per-row completion state for all downstream
  columns. Dynamic rescheduling and row identity tracking would be needed to support
  this. **Async v1 raises a `DatasetGenerationError` at startup** when any config uses
  `allow_resize=True` with the async engine enabled, naming the offending column(s).
  The user must resolve the conflict explicitly.

- **Backward compatibility**: The sync path must remain untouched. All new code is
  gated behind `DATA_DESIGNER_ASYNC_ENGINE=1` and sits in new modules.

- **Thread pool sizing**: sync generators wrapped in `asyncio.to_thread` use Python's
  default thread pool executor (`min(32, cpu_count + 4)`). For v1, keep the default —
  the scheduler semaphore and row group cap already bound actual concurrency to levels
  where the default pool is unlikely to be a bottleneck (see Follow-ups).

- **Silent task hangs**: a sync generator wrapped in `asyncio.to_thread` could hang
  indefinitely. For v1, rely on upstream model timeouts (see Follow-ups).

## UX Considerations

- **Interleaved log output**: with parallel columns and row groups, log lines will interleave.
  All async log output should include `(column=X, row_group=N)` context so output remains
  readable during debugging.

- **Progress display**: in the async path, per-column `ProgressTracker` interval logs are
  suppressed to avoid interleaved noise. Instead, the scheduler runs a lightweight background
  coroutine (`asyncio.create_task`) that emits a single consolidated summary line on a fixed
  timer (e.g., every 10 seconds):
  ```
  Progress: col_A 45/100 (45%, 2.1 rec/s) | col_B 32/100 (32%) | col_C 78/100 (78%, eta 12s)
  ```
  The coroutine reads completion counters from existing `ProgressTracker` instances and is
  cancelled once all tasks are done. The sync path is unchanged.

- **Peak memory**: multiple in-flight row groups increase peak memory. The cap is
  `async_max_concurrent_row_groups` (default 3), exposed so users can lower it in
  memory-constrained environments.

- **New config knobs**: `async_scheduler_max_submitted_tasks`, `async_max_concurrent_row_groups`,
  `async_salvage_max_rounds`, and `async_salvage_error_threshold` are new parameters users may
  encounter in error messages or docs. Each must have a sensible default; users should not need
  to tune them in the common case.

- **Async custom columns**: users can now write `async def` functions with
  `@custom_column_generator` and get native async execution without thread overhead. Worth
  surfacing in the changelog and docs.

- **Plugin async upgrade path**: plugin authors who want native async performance can
  override `agenerate()` instead of `generate()` — the symmetric bridging means they don't
  need to implement both. Stateful plugins should override `is_stateful = True`. Worth
  documenting in the plugin authoring guide.

### What stays the same

- **Config API**: no schema changes — existing configs work without modification.
- **Sync path**: `DATA_DESIGNER_ASYNC_ENGINE=0` (the default) is untouched; existing users
  see no behavioral change.
- **Existing plugins and sync custom columns**: all continue to work unchanged.
- **Row ordering**: the final dataset rows are always in the declared order regardless of
  out-of-order row group completion.
- **Checkpoint file naming and format**: parquet files use the same naming scheme and schema.

## Profiling

Task execution tracing is opt-in, enabled by passing `trace=True` to `build()` or setting
`DATA_DESIGNER_ASYNC_TRACE=1`. When disabled (the default), no `TaskTrace` objects are
created and there is no overhead. When enabled, the scheduler collects one `TaskTrace`
record per task and exposes the list as `scheduler.traces: list[TaskTrace]`, which is
surfaced on the result object after the run.

### Instrumentation points

Three timestamps are recorded inside the task coroutine — all on the event loop thread,
so no locking is needed:

1. `dispatched_at` — set in the scheduler loop right before `asyncio.create_task()`
2. `slot_acquired_at` — set inside the coroutine immediately after `await semaphore.acquire()`
3. `completed_at` + `status` — set in a `try/finally` block wrapping the generator call

The `TaskTrace` object is created at dispatch time and passed into the coroutine closure;
the coroutine mutates it in-place. On completion it is appended to `scheduler.traces` via
the result callback.

### What the data shows

- `slot_acquired_at - dispatched_at`: time waiting on the scheduler semaphore (contention indicator)
- `completed_at - slot_acquired_at`: actual generator execution time
- Dispatch timestamps across tasks: verify dependency order and parallelism (e.g. two
  independent columns should show overlapping `slot_acquired_at`–`completed_at` ranges)

### Example output

Two-column config (`topic` → `question` → `answer`), 2 row groups, 3 rows each:

```
column    rg  row  type          dispatched  slot_acquired  completed  duration  status
--------  --  ---  ------------  ----------  -------------  ---------  --------  ------
topic      0   -   from_scratch   0.000       0.001          0.012      0.011    ok
topic      1   -   from_scratch   0.001       0.001          0.013      0.012    ok     ← RG0+1 overlap (stateless)
question   0   0   cell           0.013       0.014          0.142      0.128    ok
question   0   1   cell           0.013       0.014          0.189      0.175    ok
question   0   2   cell           0.013       0.015          0.201      0.186    ok
question   1   3   cell           0.014       0.015          0.155      0.141    ok
question   1   4   cell           0.014       0.016          0.210      0.194    ok
question   1   5   cell           0.015       0.016          0.198      0.182    ok
answer     0   0   cell           0.143       0.143          0.312      0.169    ok     ← dispatched as question[0,0] completes
answer     0   1   cell           0.190       0.190          0.398      0.208    ok
answer     0   2   cell           0.202       0.202          0.445      0.243    ok
answer     1   3   cell           0.156       0.156          0.334      0.178    ok
...
```

`topic` RG0 and RG1 dispatch 1ms apart and run concurrently (stateless). `question` rows
are all dispatched the moment `topic` completes for their row group. `answer[0,0]` is
dispatched at `t=0.143`, exactly when `question[0,0]` finishes — confirming cell-level
pipelining across row groups.

### Usage

```python
result = data_designer.build(num_records=100, trace=True)
df = pd.DataFrame([asdict(t) for t in result.traces])
df["wait"] = df["slot_acquired_at"] - df["dispatched_at"]
df["duration"] = df["completed_at"] - df["slot_acquired_at"]
df.groupby("column")[["wait", "duration"]].describe()
```

## Relation to PR #269

PR #269 ("feat: add execution graph builder plan with reference implementation") is a
companion design by @johnnygreco that we reviewed before finalising this plan. It proposes
a static `ExecutionGraph` with typed node IDs (`CellNodeId`, `BarrierNodeId`), an
`ExecutionTraits` flag enum, and a `CompletionTracker`. It intentionally stops at the
graph/tracker layer; this plan covers the full stack from graph through to deployment.

### What we adopted

- **Static `ExecutionGraph`**: we adopt the concept of building a static graph upfront,
  inspectable before and after a run. Our graph is column-granularity rather than
  cell-granularity — see below for why.
- **Dependency source**: derive the graph from `required_columns` on existing configs,
  extended with a side-effect mapping for columns like `summary__trace`. No config schema
  changes in either approach.
- **Trait inference from properties, not class names**: `GraphBuilder._infer_traits()`
  inspects `can_generate_from_scratch` and `get_generation_strategy()` rather than
  matching class names. We apply the same principle, keeping plugin generators compatible.
- **Lightweight completion tracking**: a `dict[str, set[int]]` mapping column → completed
  rows, rather than materialising O(C × R) cell-level state. Our `CompletionTracker`
  follows the same design.
- **Statefulness as a separate concern from execution strategy**: PR #269 separates
  execution traits (how a generator runs) from per-instance concurrency safety. We
  formalise this with the `is_stateful` property.

### What we changed, and why

**Column-level graph, not cell-level.** PR #269 models every `(row, column)` pair
as a virtual `CellNodeId`. Full-column generators become `BARRIER` nodes — a synthetic
node that must complete before any output cells are ready. This faithfully models
dependencies but creates a problem the PR itself flags as an open issue: a validation
column anywhere in the pipeline blocks all checkpointing until the entire dataset
completes, because no row is "done" until every column, including the barrier, finishes.
Cell-level nodes also scale to O(C × R), which is large for realistic dataset sizes.

We use a **column-level** `ExecutionGraph` instead — N columns, N edges, fixed size
regardless of row count. This still provides the full value of a static graph (visualization,
critical path, upfront task counts, error attribution via `downstream()`) without the
checkpoint problem or the node explosion. Full-column tasks are scoped to a **row group**:
the effective barrier is just that FULL_COLUMN task waiting for all rows *in that group*,
not the whole dataset. Checkpoints happen as each row group completes, so a failure
mid-run loses at most one batch.

**`ExecutionTraits` replaced by `GenerationStrategy` on the graph.** PR #269 attaches an
`ExecutionTraits` flag enum (`CELL`, `BARRIER`, `ROW_STREAMABLE`) to each node. Since our
graph is column-level, we store `GenerationStrategy` (cell-by-cell, full-column) directly
on each column node instead. From-scratch columns are identified by having no upstream
dependencies in the graph; the scheduler checks `can_generate_from_scratch` on the generator
instance to determine which method to call. This serves the same purpose as `ExecutionTraits`
— the scheduler and `CompletionTracker` use it to determine task granularity — without
needing typed node IDs or flag combinations.

**`ROW_STREAMABLE` trait omitted.** PR #269 introduces `is_row_streamable` so full-column
generators that process rows independently (e.g., `ExpressionColumnGenerator`) can be
scheduled row-by-row, recovering some pipelining within a barrier. In our row-group model
this is unnecessary: even a full-column generator only blocks one batch, preserving
checkpoint cadence without subdividing tasks further. Expression columns run in
microseconds and are never the scheduling bottleneck. We note this as a potential
follow-up if profiling shows otherwise.

**Scheduler and concurrency layers added.** PR #269 deliberately stops at the graph and
tracker. Steps 1–3 of this plan (execution graph, completion tracker, task model) are
directly informed by PR #269 and we treat it as the reference design for that layer. The
remaining steps — scheduler, concurrency controls, retry/salvage, buffer manager, and
builder integration — extend that foundation to a deployable implementation.

## Notes

### Out of scope for this PR
- Overhauling `ModelFacade` internals (PR #344's scope)
- Building a cell-level static execution graph (PR #269's `CellNodeId`/`BarrierNodeId`
  approach — we use a column-level graph instead, which avoids the barrier/checkpoint problem)
- Removing the sync/threaded path (it stays as the default)

### Follow-ups
- **`allow_resize` async support**: currently raises at startup when the async engine is
  enabled; full support requires dynamic rescheduling and row identity tracking.
- **Native async `RemoteValidator`**: `ValidationColumnGenerator` wraps `generate()` in
  `asyncio.to_thread`, which spawns a thread that itself uses `ConcurrentThreadExecutor`
  for parallel HTTP calls, bypassing the scheduler's concurrency controls. Fix: native
  async `agenerate` on `RemoteValidator`.
- **Per-generator task timeouts**: sync generators wrapped in `asyncio.to_thread` can
  hang indefinitely. For v1 we rely on upstream model timeouts; optional per-generator
  timeout overrides are the follow-up.
- **Wasted-work telemetry**: in-flight full-column/batch tasks continue computing after
  a row is dropped; add telemetry to track compute wasted on dropped rows.
- **Thread pool sizing**: if profiling shows saturation of the default executor
  (`min(32, cpu_count + 4)`), explicitly size it to match the scheduler caps.

### Impact on plugins and custom columns

This change is **backward-compatible** with all existing plugins and custom columns.
No plugin author needs to modify their code for it to work under the async scheduler.

**Column generator plugins** (registered via entry points): plugins subclass one of
the base generator classes and implement `generate()`. The base class `agenerate()`
fallback wraps `generate()` in `asyncio.to_thread`, so every existing plugin
automatically gets async support. Plugins that want native async performance can
optionally override `agenerate()` instead — the symmetric bridging means they don't
need to implement `generate()` at all. The `is_stateful` property defaults to `False`,
which is correct for most plugins; stateful plugins can override it.

**Custom columns** (`@custom_column_generator`): user-provided sync functions are
wrapped in `asyncio.to_thread` by the framework. If the user provides an async
function, `CustomColumnGenerator` detects this via `asyncio.iscoroutinefunction`
and calls it directly as a coroutine — no thread pool overhead.

**Processor plugins** (`process_before_batch`, `process_after_batch`,
`process_after_generation`): processors run at barrier points in the scheduling loop
where no column generation is concurrent. They remain purely synchronous and are
unaffected by this change.

### Key insight from existing code
Every column config already has a `required_columns` property that returns the
column names referenced in its Jinja2 templates. This gives us explicit dependency
information without any config schema changes. The `ExecutionGraph` dependency
structure starts as `{col.name: set(col.required_columns) for col in configs}`,
extended with a side-effect output mapping so that references to columns like
`summary__trace` resolve to a dependency on the `summary` generator.

### On static graph vs dynamic scheduling
The `ExecutionGraph` is a static column-level DAG built once at init time — it
describes *what* can run and in what order. Scheduling remains dynamic: the
completion tracker drives readiness checks as tasks complete, and new tasks become
eligible without any upfront planning. The two concerns are complementary:
the graph provides inspectability and upfront analysis; the tracker provides
runtime readiness at row granularity.
