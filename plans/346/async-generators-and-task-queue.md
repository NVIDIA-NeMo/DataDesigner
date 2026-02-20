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

A single `asyncio.Semaphore` bounds total in-flight tasks. The limit should be
the sum of `max_parallel_requests` across distinct models, or a configured global
cap — whichever is smaller. This replaces per-column fan-out.

### 5. Pre/post-batch processors

- **Pre-batch**: runs after seed generators complete for a row group, before
  other columns. Modeled as a barrier task for the row group.
- **Post-batch**: runs after all columns complete for a row group, before
  checkpoint write.

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
  - For `MultiColumnConfig`, all sub-columns share the same dependencies
  - Validate: every required column must appear earlier in the config list (detect cycles / missing refs)
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
    1. Seed row groups: dispatch all `from_scratch` tasks
    2. Loop: query `completion_tracker.get_ready_tasks()` → dispatch each via `asyncio.create_task()` behind semaphore → on completion, update tracker → repeat until all tasks done or early shutdown
    3. After each row group completes: run post-batch processors, checkpoint
  - Task dispatch: wraps generator call in semaphore-guarded coroutine
  - Error handling: same early-shutdown logic as `AsyncConcurrentExecutor` (error rate threshold within sliding window)
  - Progress tracking: reuse `ProgressTracker` per column
- [ ] Use `asyncio.Event` or `asyncio.Condition` to wake the scheduler when a task completes (avoids polling)
- [ ] Unit tests with mock generators

**Files**: new module `engine/dataset_builders/async_scheduler.py`, tests

### Step 5: Generator Async Migration

Make all generator types async-capable.

- [ ] `ColumnGeneratorWithModelChatCompletion.agenerate` — already implemented (PR #280), no changes needed
- [ ] `FromScratchColumnGenerator`: add `async agenerate_from_scratch(num_records) -> DataFrame` — wraps sync in `asyncio.to_thread`
- [ ] `ColumnGeneratorFullColumn`: add `async agenerate(data: DataFrame) -> DataFrame` — wraps sync in `asyncio.to_thread`
- [ ] `ExpressionColumnGenerator`: inherits full-column async wrapper
- [ ] `SamplerColumnGenerator`: inherits from-scratch async wrapper
- [ ] `SeedDatasetColumnGenerator`: inherits from-scratch async wrapper
- [ ] `ValidationColumnGenerator`: inherits full-column async wrapper
- [ ] `CustomColumnGenerator`: inherits whichever strategy it uses
- [ ] `ImageCellGenerator`, `EmbeddingCellGenerator`: add native `agenerate` using `model.agenerate_image` / `model.agenerate_text_embeddings`
- [ ] Base class `ColumnGenerator.agenerate` fallback (already exists via `asyncio.to_thread`) is sufficient for non-LLM generators

**Files**: `generators/base.py`, `generators/expression.py`, `generators/samplers.py`, `generators/seed_dataset.py`, `generators/image.py`, `generators/embedding.py`, tests

### Step 6: Buffer / Row Group Manager

Adapt `DatasetBatchManager` for concurrent row group processing.

- [ ] Support multiple row groups in-flight simultaneously (currently only one batch's buffer exists)
  - Option A: Multiple buffer instances (one per active row group)
  - Option B: Single shared buffer partitioned by row group offset ranges
  - Recommendation: **Option A** — cleaner isolation, each row group has its own `list[dict]`
- [ ] `update_cell(row_group: int, row_index: int, column: str, value: Any)` — fine-grained update (vs. replacing entire record dict)
  - Or keep the current `update_record(index, record_dict)` approach per row group
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
- [ ] `build()` dispatches to `_build_async()` when `DATA_DESIGNER_ASYNC_ENGINE=1`, else existing sync path
- [ ] `build_preview()` uses the same async path (single row group, no checkpoint)
- [ ] Error handling: `DatasetGenerationError` wrapping, record dropping, telemetry events
- [ ] Processor integration:
  - Pre-batch: scheduler runs after seed tasks for a row group
  - Post-batch: scheduler runs after all column tasks for a row group, before checkpoint

**Files**: `column_wise_builder.py`

### Step 8: Tests & Validation

- [ ] Unit tests for each new module (dependency map, completion tracker, task model, scheduler)
- [ ] Integration test: multi-column config with known dependencies, verify parallel execution
- [ ] Integration test: mixed cell-by-cell + full-column generators
- [ ] Integration test: error rate shutdown
- [ ] Integration test: checkpoint correctness (row groups written in order, parquet valid)
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
  as they operate on independent columns. Need to ensure the DataFrame isn't mutated
  concurrently — use `asyncio.to_thread` (which copies into a thread) or serialize
  full-column tasks within a row group.

- **Pre-batch processors mutating data**: Pre-batch processors (e.g., schema transform)
  can add/remove/modify rows. This changes the row count and invalidates the completion
  tracker's row indices. Solution: treat pre-batch as a barrier that resets the tracker
  state for that row group (re-index rows after processor runs).

- **Backward compatibility**: The sync path must remain untouched. All new code is
  gated behind `DATA_DESIGNER_ASYNC_ENGINE=1` and sits in new modules.

## Notes

### What we're NOT doing in this PR
- Overhauling `ModelFacade` internals (PR #344's scope)
- Building a heavyweight static execution graph (PR #269's approach — we take the
  lightweight dynamic approach instead)
- Removing the sync/threaded path (it stays as the default)

### Key insight from existing code
Every column config already has a `required_columns` property that returns the
column names referenced in its Jinja2 templates. This gives us explicit dependency
information without any config schema changes. The dependency map is just
`{col.name: set(col.required_columns) for col in configs}`.

### On "dynamic" graph building
The graph is implicit in the dependency map + completion tracker. We never
materialise a node/edge graph structure. The scheduler dynamically determines
readiness by querying the tracker — this is what "dynamic" means in this context.
As tasks complete, new tasks become eligible. No upfront planning of execution order.
