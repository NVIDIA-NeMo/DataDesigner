# Dataset Builders

The dataset builder subsystem orchestrates the end-to-end generation of a dataset from compiled column configs. It supports two execution modes: a sequential batch loop and an async DAG-based scheduler.

Source: `packages/data-designer-engine/src/data_designer/engine/dataset_builders/`

## Overview

`DatasetBuilder` is the central orchestrator. It receives a compiled `DataDesignerConfig`, instantiates column generators from the registry, and executes them in dependency order. The execution mode is selected by the `DATA_DESIGNER_ASYNC_ENGINE` environment variable.

Both modes produce the same output: batched parquet files managed by `DatasetBatchManager`, with post-generation processing and profiling.

## Key Components

### DatasetBuilder

Entry point for generation. `build()` branches:
- **Sequential path** (default): `DatasetBatchManager.start` → batch loop → `_run_batch` per batch → `finish()` → `ProcessorRunner.run_after_generation` → `model_registry.log_model_usage`
- **Async path** (`DATA_DESIGNER_ASYNC_ENGINE=1`): `_prepare_async_run` → `AsyncTaskScheduler.run()` → telemetry and metadata

### Sequential Execution (`_run_batch`)

Iterates compiled column order. For each generator:
1. `log_pre_generation()` — logs model and optional MCP tool alias
2. **From-scratch generators** (empty buffer): `generate_from_scratch` → optional `run_pre_batch` after first seed column
3. **`CELL_BY_CELL` generators**: `_fan_out_with_threads` or `_fan_out_with_async` — parallel cell generation
4. **`FULL_COLUMN` generators**: `generate` on the whole batch DataFrame; output row count must match input row count

### Async Execution (`_build_async`)

Preparation (`_prepare_async_run`):
1. Builds `gen_map` — maps each column name to its generator instance (multi-column configs share a single instance)
2. Creates `ExecutionGraph` from column dependencies
3. Partitions rows into row groups by `buffer_size`
4. Constructs `CompletionTracker`, `RowGroupBufferManager`, `AsyncTaskScheduler`
5. Hooks `ProcessorRunner` for pre-batch and post-batch stages

`AsyncTaskScheduler` runs on a dedicated async loop with frontier-driven dispatch, task-admission leases, salvage rounds for failed tasks, and order-dependent locks for columns that must execute sequentially. Ready frontier tasks enter `FairTaskQueue`, are selected through virtual-time ordering, and are committed only after `TaskAdmissionController` acquires the required scheduler resources. Salvage-exhausted tasks are dropped except for preserved retryable failures: provider rate limits and local request-admission queue timeouts stay deferred and retry after cooldown/backoff so scheduler-local pressure delays records rather than discarding them.

### Execution Graph

`ExecutionGraph` (in `dataset_builders/utils/execution_graph.py`) models column dependencies:
- Upstream/downstream sets derived from `required_columns`, side-effect columns, and `skip.when` references
- `GenerationStrategy` per column (CELL_BY_CELL or FULL_COLUMN)
- Kahn topological sort for execution order
- `split_upstream_by_strategy` — separates batch-level from cell-level dependencies
- Skip metadata per column — `get_skip_config`, `should_propagate_skip`, `get_required_columns`, and `get_side_effect_columns` — queried at runtime by both engines to evaluate skip decisions

### CompletionTracker

Tracks per-row-group, per-column completion state:
- **Cell-level**: completed cell indices for `CELL_BY_CELL` columns
- **Batch-level**: full-column completion flags for `FULL_COLUMN` columns
- **Frontier**: computes ready tasks when backed by `ExecutionGraph`
- Handles dropped rows and downstream task enqueuing

### Conditional Generation (Skip)

Columns can be conditionally skipped per-row via `SkipConfig` (defined in `data_designer.config.base`). Two mechanisms control skipping:

1. **Expression gate** — `skip=SkipConfig(when="{{ expr }}")` on a `SingleColumnConfig`. The Jinja2 expression is evaluated per-row; when truthy, the column is skipped for that row and the configured `value` (default `None`) is written instead of calling the generator.
2. **Skip propagation** — when an upstream column was skipped, downstream columns auto-skip unless they set `propagate_skip=False`. Propagation checks `required_columns` against the row's `__internal_skipped_columns` set.

Skip evaluation is handled by two utility modules:

- **`skip_evaluator.py`** — `evaluate_skip_when` renders the expression in a `NativeSandboxedEnvironment` (native Python types, `StrictUndefined`). `should_skip_by_propagation` checks set intersection between required columns and skipped columns.
- **`skip_tracker.py`** — manages the `__internal_skipped_columns` metadata key on record dicts. Each record carries a `__internal_skipped_columns` set listing which columns were skipped for that row. `apply_skip_to_record` adds the column name to that set, writes the skip value into the cell, and clears any side-effect columns. `strip_skip_metadata_from_records` removes the `__internal_skipped_columns` key before DataFrame construction so it never reaches parquet (called by `DatasetBatchManager`, `RowGroupBufferManager`, and inline in both engines).

Both execution modes integrate skip at the same points:

- **Sequential**: `_run_full_column_generator` and the fan-out methods (`_fan_out_with_threads`, `_fan_out_with_async`) call `_should_skip_cell` per record. Skipped rows are excluded from the generator input, then merged back with skip metadata preserved. A fast `_column_can_skip` check short-circuits the per-record evaluation when no skip config or propagation applies.
- **Async**: `_run_cell` and `_run_batch` in `AsyncTaskScheduler` call `_should_skip_record` / `_apply_skip_to_record` with the same logic. Skipped cells report as skipped (not success) in progress tracking.

DAG edges are added for `skip.when` column references in both `topologically_sort_column_configs` (compile-time sort) and `ExecutionGraph.create` (async runtime) so skip-gate columns are generated before the gated column.

### DatasetBatchManager

Manages in-memory row buffers and persistence:
- `finish_batch` → writes parquet via `ArtifactStorage`
- Updates dataset metadata between batches
- The async path uses `RowGroupBufferManager` for per-row-group DataFrames and checkpointing

### Resume Checkpointing

`DatasetBuilder.build(..., resume=ResumeMode.*)` can continue an interrupted run from the last durable checkpoint:

- `ResumeMode.NEVER` always starts a fresh run, using a timestamped dataset directory when needed.
- `ResumeMode.ALWAYS` resumes the existing dataset directory and raises on incompatible state.
- `ResumeMode.IF_POSSIBLE` resumes when the persisted config fingerprint matches; otherwise it starts a fresh timestamped run.

Checkpoint state lives in `metadata.json`. Each metadata write includes the config fingerprint (`config_hash`, `config_hash_algo`, and `config_hash_version`) so compatibility checks do not need to deserialize `builder_config.json` for the common path. `builder_config.json` remains the human-readable record of the run configuration and the fallback for older datasets.

Both engines resume the same way: they scan `parquet-files/batch_*.parquet` and read parquet metadata to recover the completed row-group IDs and their actual persisted row counts. `metadata.json` remains the source of truth for the run *configuration* (`buffer_size`, `target_num_records`, `original_target_num_records`, config fingerprint), but the filesystem is the source of truth for *progress* (`num_completed_batches`, `actual_num_records`). Splitting the two sources is what lets resume survive a crash between writing a batch parquet and updating metadata — the filesystem reflects the durable state even when metadata lags by a step. Reading actual row counts also matters for async early-shutdown salvage, where a completed parquet file can contain fewer rows than the requested row-group size. The async engine tolerates non-contiguous IDs because row groups can complete out of order; the sync engine writes batches sequentially and rejects holes (likely external mutation or a directory written by an incompatible engine).

Resume relies on stable row-group boundaries within a run. It treats datasets that have completed `process_after_generation()` as terminal: after-generation processors operate on the whole dataset and can re-chunk rows or change schema, invalidating row-group identity for later resume/extension. The terminal-state check raises a clear `DatasetGenerationError` (not a `TypeError`) when the persisted metadata is missing required fields such as `target_num_records`.

After-generation processors run unconditionally on the on-disk dataset whenever they are configured — including the case where resume sees every row group already on disk. This closes the crash window between the final row-group parquet write and the `post_generation_state="started"` marker write: in that window, the dataset is complete but post-generation never ran, and the on-disk parquet files are still clean (no processor has touched them). The `post_generation_state="started"` short-circuit still rejects the other direction (`process_after_generation()` crashed mid-rewrite, leaving the parquet files in an ambiguous state), so resume only re-runs after-generation when it is safe to do so.

Metadata writes are atomic (`tmp` file + `fsync` + `os.replace`) because `metadata.json` is the crash-recovery checkpoint. Corrupt or partially written metadata raises a clear `DatasetGenerationError` rather than falling through as a generic config mismatch.

`DatasetCreationResults` from a resume invocation reflects the full on-disk dataset for anything that reads the artifact directory (`load_dataset`, `count_records`, `load_analysis`, `export`, `push_to_hub`), but per-run observability (`task_traces`, model-usage logs, telemetry events) is scoped to the current invocation — the original run's in-memory state is not persisted across process boundaries.

## Data Flow

### Sequential
```
DatasetBuilder.build()
  → DatasetBatchManager.start()
  → for each batch:
      → for each generator (topological order):
          → generate_from_scratch / generate (FULL_COLUMN) / fan_out (CELL_BY_CELL)
      → DatasetBatchManager.finish_batch() → parquet
  → ProcessorRunner.run_after_generation()
  → model_registry.log_model_usage()
```

### Async
```
DatasetBuilder.build()
  → _build_async()
  → _prepare_async_run()
      → ExecutionGraph.create()
      → CompletionTracker.with_graph()
      → AsyncTaskScheduler(task admission, fair queue, salvage_rounds)
  → scheduler.run()
      → admit row groups under the configured row-group cap
      → fairly admit ready tasks from the frontier through task admission
      → tasks execute generators, update CompletionTracker
      → checkpoints via RowGroupBufferManager
  → collect TaskTraces, emit telemetry
```

Row-group admission is fixed by default in the dataset-builder path: the configured row-group concurrency is the hard in-flight cap. The scheduler also has an internal adaptive row-group mode for direct use that only raises a soft target up to that cap; it is additive ramp-up, not AIMD shrink/recovery behavior.

When request admission is available, async scheduling may use request-pressure snapshots as a read-only advisory during fair-queue selection. A request-pressured task can be skipped for an eligible peer without mutating request-admission state; provider/model/domain request limits remain owned by request admission.

## Design Decisions

- **Dual execution engines behind one API.** The sequential engine is simpler and easier to debug; the async engine adds row-group parallelism for throughput. Users switch via an environment variable without changing their code.
- **DAG-driven ordering** ensures columns with dependencies (e.g., a judge column that depends on a text column) are generated in the correct order, regardless of the order they appear in the config.
- **Fair async admission with bounded borrow by default** keeps the scheduler flowing across ready columns and model groups. `FairTaskQueue.select_next(...)` chooses eligible ready work, `TaskAdmissionController` leases scheduler resources before spawn, and `FairTaskQueue.commit(...)` removes the selected task only after admission succeeds. The default `BoundedBorrowTaskAdmissionPolicyConfig` computes a strict per-group share, lets solo groups borrow only up to a capacity-derived reserve, and makes borrowed groups yield when eligible peer pressure appears. Passing `bounded_borrow=None` selects strict-fair admission for tests and benchmark comparisons. Per-group virtual-time ordering prevents a large ready frontier from degenerating into a column-by-column wave, and scheduler-resource accounting remains separate from provider/model request admission.
- **Salvage rounds in async mode** retry failed tasks after all other tasks in a round complete, improving resilience against transient LLM failures without blocking the entire generation.
- **Unified DAG construction.** `topologically_sort_column_configs` (in `execution_graph.py`) determines column ordering using Kahn's algorithm; the runtime `ExecutionGraph` adds strategy-aware dependency tracking for the async scheduler.

## Cross-References

- [System Architecture](overview.md) — end-to-end data flow
- [Engine Layer](engine.md) — compilation and generator hierarchy
- [Models](models.md) — how generators access LLMs
- [Config Layer](config.md) — column configs and dependency declarations
