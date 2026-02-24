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

### 1. Dynamic dependency resolution, not a static graph

We don't build an explicit graph object (unlike PR #269). Instead:
- At setup: build a **dependency map** `dict[str, set[str]]` from each column's
  `config.required_columns` property (already available on all config types via
  Jinja2 template introspection).
- The dependency map also registers **side-effect output columns** (e.g.,
  `__trace`, `__reasoning_content`) and maps them back to their producer generator.
  A downstream column referencing `summary__trace` resolves to a dependency on
  the `summary` generator. This ensures side-effect columns are never missing from
  the map or treated as unsatisfied.
- At runtime: a **completion tracker** (columns × rows matrix) determines which
  tasks are ready by checking whether all upstream columns for a given row are done.

This is simpler, requires no new data structures beyond what configs already
provide, and naturally handles the "dynamic" aspect — we just check readiness as
tasks complete.

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

Full-column generators operate on their entire row group at once, same as today.

### 4. Concurrency control

Two independent layers:

1. **Scheduler semaphore** — a coarse resource guard bounding total in-flight
   tasks to limit CPU/memory pressure (e.g., configurable cap, default ~128).
   This is **not** the source of truth for API concurrency; it only prevents
   unbounded coroutine fan-out.

2. **Throttle manager** (from PR #344) — gates every outbound LLM call, keyed
   by `provider+model(+domain)`. Dynamically adjusts per-key limits on 429s
   via AIMD. This is the real API concurrency control.

Tasks must **not hold scheduler slots while waiting on throttle backoff**. A task
acquires a scheduler slot, prepares its request, then releases the slot before
awaiting the throttle permit. This ensures a throttled model key doesn't starve
unrelated keys by hogging scheduler capacity.

This replaces per-column fan-out and composes cleanly with PR #344's adaptive
throttling without the two systems fighting each other.

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

### 6. Pre/post-batch processors

- **Pre-batch**: runs after seed generators complete for a row group, before
  other columns. Modeled as a barrier task for the row group.
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
   ready queue drains, the scheduler runs up to `N` salvage rounds over deferred
   tasks (configurable via `async_scheduler_max_retries`, default 2).
4. **Throttle-aware**: retries re-enter the throttle manager acquire path, so
   they don't exacerbate rate limiting.
5. **Final drop**: after retry budget is exhausted, mark the cell as failed and
   the row as dropped. Continue row-group completion checks over remaining rows.

### 8. `allow_resize` scoping

The completion tracker uses row indices as stable identifiers. `allow_resize`
lets any generator change the row count mid-pipeline, which invalidates all
per-row completion state for downstream columns. Supporting this under parallel
execution would require dynamic rescheduling and row identity tracking.

**Async v1 scope**: if any column config has `allow_resize=True`, the builder
falls back to the sync path. This is safe because resize is opt-in and the sync
path handles it naturally. Full async support for resize is a follow-up.

## Success Criteria

- [ ] All generators expose async-first `agenerate` (cell-by-cell) or async wrappers (full-column/from-scratch)
- [ ] Builder dispatches tasks based on dependency readiness, not column order
- [ ] Multiple columns execute in parallel when dependencies allow
- [ ] Row groups checkpoint to parquet upon full completion
- [ ] Existing sync path (`DATA_DESIGNER_ASYNC_ENGINE=0`) continues to work unchanged
- [ ] All existing tests pass; new tests cover dependency resolution and scheduling
- [ ] `make test-run-recipes` passes with async engine enabled

## Implementation Steps

### Step 1: Column Dependency Map

Build the dependency map from column configs at builder init time.

- [ ] Add `build_dependency_map(column_configs) -> dict[str, set[str]]` utility
  - Input: the ordered list of `ColumnConfigT` / `MultiColumnConfig`
  - For each config, read `config.required_columns` → set of upstream column names
  - Also register side-effect output columns (`__trace`, `__reasoning_content`, etc.)
    and map them back to their producer column, so downstream references resolve correctly
  - For `MultiColumnConfig`, all sub-columns share the same dependencies
  - Validate: every required column must appear earlier in the config list or be a
    registered side-effect output (detect cycles / missing refs)
- [ ] Add `topological_order(dependency_map) -> list[str]` — returns columns in valid execution order (should match config order; this is a validation step)
- [ ] Unit tests for dependency map construction and validation

**Files**: new module `engine/dataset_builders/utils/dependency_map.py`, tests

### Step 2: Completion Tracker

A lightweight structure tracking which (column, row) pairs are done.

- [ ] `CompletionTracker` class:
  - Internal: `dict[str, set[int]]` mapping column name → set of completed row indices
  - `mark_complete(column: str, row: int)` / `mark_batch_complete(column: str, row_group: int, row_group_size: int)`
  - `is_ready(column: str, row: int, dependency_map) -> bool` — checks all upstream columns for that row
  - `is_batch_ready(column: str, row_group: int, row_group_size: int, dependency_map) -> bool` — checks all rows in group
  - `is_row_group_complete(row_group: int, row_group_size: int, all_columns: list[str]) -> bool` — all columns done for all rows
  - `get_ready_tasks(dependency_map, columns_with_strategy, row_groups) -> list[Task]` — yields all currently dispatchable tasks
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
- [ ] Hashable so we can track dispatched/pending sets

**Files**: new module `engine/dataset_builders/utils/task_model.py` (or inline in scheduler)

### Step 4: Async Task Scheduler

The core orchestrator that replaces `_run_batch` for the async path.

- [ ] `AsyncTaskScheduler` class:
  - Constructor takes: generators (by column name), dependency map, completion tracker, row group definitions, concurrency limit, error/result callbacks
  - `async run()` — main loop:
    1. Dispatch `from_scratch` tasks, respecting `is_stateful`: stateful generators
       serialize per-instance (row group N completes before N+1 starts for that
       generator); stateless generators dispatch all row groups concurrently
    2. Loop: query `completion_tracker.get_ready_tasks()` → dispatch each via
       `asyncio.create_task()` behind scheduler semaphore → on completion, update
       tracker → repeat until all tasks done or early shutdown
    3. When ready queue drains, run salvage rounds over deferred retryable failures
       (up to `async_scheduler_max_retries` rounds)
    4. After each row group completes: run post-batch processors, checkpoint
  - Task dispatch: acquire scheduler semaphore slot → prepare request → release
    slot → await throttle permit (for LLM tasks) → execute → write result
  - Error handling: classify failures as retryable vs non-retryable; retryable
    go to deferred queue with backoff; same early-shutdown logic as
    `AsyncConcurrentExecutor` (error rate threshold within sliding window)
  - Progress tracking: reuse `ProgressTracker` per column
- [ ] Use `asyncio.Event` or `asyncio.Condition` to wake the scheduler when a task completes (avoids polling)
- [ ] Unit tests with mock generators

**Files**: new module `engine/dataset_builders/async_scheduler.py`, tests

### Step 5: Generator Async Migration

Make all generator types async-capable and declare statefulness.

**Symmetric `generate` / `agenerate` contract**: only one of the two methods needs
to be implemented. The base class provides automatic bridging in both directions:
- If only `generate()` is implemented → `agenerate()` wraps it via `asyncio.to_thread`
  (already exists from PR #280).
- If only `agenerate()` is implemented → `generate()` wraps it via `asyncio.run()`
  (new fallback). This is safe because the sync builder never has a running event loop.

This means sync-first generators (most built-ins, existing plugins) work unchanged,
and async-first generators (new plugins doing native async I/O) only need to implement
`agenerate()` without writing a redundant sync version.

- [ ] Add symmetric bridging on the base `ColumnGenerator`:
  - `agenerate()` default: `asyncio.to_thread(self.generate, data)` (already exists)
  - `generate()` default: `asyncio.run(self.agenerate(data))` (new, for async-first generators)
  - Detect which one the subclass overrides to avoid infinite recursion
- [ ] Add `is_stateful` property to base `ColumnGenerator` (default `False`).
  Stateful generators are serialized per-instance by the scheduler.
- [ ] `ColumnGeneratorWithModelChatCompletion.agenerate` — already implemented (PR #280), no changes needed
- [ ] `FromScratchColumnGenerator`: add `async agenerate_from_scratch(num_records) -> DataFrame` — wraps sync in `asyncio.to_thread` with defensive `df.copy()` on shared data
- [ ] `ColumnGeneratorFullColumn`: add `async agenerate(data: DataFrame) -> DataFrame` — wraps sync in `asyncio.to_thread` with defensive `df.copy()` (see Risks)
- [ ] `ExpressionColumnGenerator`: inherits full-column async wrapper
- [ ] `SamplerColumnGenerator`: inherits from-scratch async wrapper; `is_stateful = False`
- [ ] `SeedDatasetColumnGenerator`: inherits from-scratch async wrapper; `is_stateful = True` (maintains DuckDB batch reader cursor and leftover-row buffer)
- [ ] `ValidationColumnGenerator`: inherits full-column async wrapper
- [ ] `CustomColumnGenerator`: inherits whichever strategy it uses; `is_stateful` should be overridable by custom implementations. For `@custom_column_generator` functions, detect `asyncio.iscoroutinefunction` and call directly if async.
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
  1. Build dependency map from `self._column_configs`
  2. Partition rows into row groups
  3. Create `CompletionTracker`, `AsyncTaskScheduler`
  4. Run scheduler on the background event loop (reuse `_ensure_async_engine_loop()`)
  5. Scheduler handles checkpointing via callbacks
- [ ] `build()` dispatches to `_build_async()` when `DATA_DESIGNER_ASYNC_ENGINE=1`
    **and** no column config has `allow_resize=True`; else existing sync path
- [ ] `build_preview()` uses the same async path (single row group, no checkpoint)
- [ ] Error handling: `DatasetGenerationError` wrapping, record dropping, telemetry events
- [ ] Processor integration:
  - Pre-batch: scheduler runs after seed tasks for a row group
  - Post-batch: scheduler runs after all column tasks for a row group, before checkpoint

**Files**: `column_wise_builder.py`

### Step 8: Tests & Validation

- [ ] Unit tests for each new module (dependency map, completion tracker, task model, scheduler)
- [ ] Dependency map: side-effect output columns resolve correctly (e.g., column
  depending on `summary__trace` maps to a dependency on the `summary` generator)
- [ ] Integration test: multi-column config with known dependencies, verify parallel execution
- [ ] Integration test: mixed cell-by-cell + full-column generators
- [ ] Integration test: error rate shutdown
- [ ] Integration test: checkpoint correctness (row groups written in order, parquet valid)
- [ ] Integration test: `allow_resize=True` falls back to sync path
- [ ] Integration test: stateful generator (`is_stateful=True`) serializes per-instance
  across row groups; stateless generators run concurrently
- [ ] Integration test: retry salvage — transient failure is retried and succeeds;
  non-retryable failure drops immediately; retry budget exhaustion drops correctly
- [ ] Integration test: throttling fairness — 429 on model key A does not stall
  unrelated model key B tasks
- [ ] Run `make test` — all existing tests pass
- [ ] Run `make test-run-recipes` with `DATA_DESIGNER_ASYNC_ENGINE=1`
- [ ] Benchmark: compare sync vs async on a multi-column recipe with simulated latency

## Risks & Considerations

- **Memory with concurrent row groups**: Having multiple row groups in-flight increases
  peak memory. Mitigation: limit max concurrent row groups (e.g., 2-3) via a separate
  semaphore or by feeding row groups into the scheduler incrementally.

- **Record dropping across columns**: If row 5 fails on column B, we must prevent
  column C from processing row 5. The completion tracker naturally handles this — a
  failed task is never marked complete, so downstream tasks never become ready. But we
  need to detect when a row group is "done enough" to checkpoint (some rows dropped).
  Solution: track dropped rows per row group; row group is complete when all non-dropped
  rows have all columns done.

- **Full-column generator ordering**: If two full-column generators have no mutual
  dependency, they could run in parallel on the same row group. This is safe as long
  as they operate on independent columns. `asyncio.to_thread` passes **object
  references**, not copies — if two full-column generators share the same DataFrame,
  concurrent mutation is possible. Solution: pass `df.copy()` to each full-column
  generator dispatched to a thread, and merge results back by column name.

- **Pre-batch processors mutating data**: Pre-batch processors (e.g., schema transform)
  can add/remove/modify rows. This changes the row count and invalidates the completion
  tracker's row indices. Solution: treat pre-batch as a barrier that resets the tracker
  state for that row group (re-index rows after processor runs).

- **`allow_resize` incompatibility**: Any generator with `allow_resize=True` can change
  the row count mid-pipeline, invalidating per-row completion state for all downstream
  columns. Dynamic rescheduling and row identity tracking would be needed to support
  this. **Async v1 falls back to the sync path** when any config uses `allow_resize`.

- **Backward compatibility**: The sync path must remain untouched. All new code is
  gated behind `DATA_DESIGNER_ASYNC_ENGINE=1` and sits in new modules.

## Notes

### What we're NOT doing in this PR
- Overhauling `ModelFacade` internals (PR #344's scope)
- Building a heavyweight static execution graph (PR #269's approach — we take the
  lightweight dynamic approach instead)
- Removing the sync/threaded path (it stays as the default)
- Supporting `allow_resize=True` in the async path (falls back to sync; follow-up)

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
information without any config schema changes. The dependency map starts as
`{col.name: set(col.required_columns) for col in configs}`, extended with a
side-effect output mapping so that references to columns like `summary__trace`
resolve to a dependency on the `summary` generator.

### On "dynamic" graph building
The graph is implicit in the dependency map + completion tracker. We never
materialise a node/edge graph structure. The scheduler dynamically determines
readiness by querying the tracker — this is what "dynamic" means in this context.
As tasks complete, new tasks become eligible. No upfront planning of execution order.
