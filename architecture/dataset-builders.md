# Dataset Builders

The dataset builder subsystem orchestrates the end-to-end generation of a dataset from compiled column configs using an async DAG-based scheduler.

Source: `packages/data-designer-engine/src/data_designer/engine/dataset_builders/`

## Overview

`DatasetBuilder` is the high-level orchestrator. It receives a compiled `DataDesignerConfig`, initializes column generators from the registry, and owns shared scheduler and resource state. Ordinary builds execute directly through `AsyncTaskScheduler`; record-selection builds delegate their feature lifecycle to a per-build `RecordSelectionRunner`.

The scheduler produces row-group DataFrames managed by `RowGroupBufferManager`. `DatasetBuilder` composes generic processor callbacks for ordinary checkpointing, while `RecordSelectionRunner` treats each completed candidate scheduler as a one-row-group producer and then owns selection, durable state, resume, and terminal publication.

## Key Components

### DatasetBuilder

Entry point for generation. `build()` initializes shared generators and routes ordinary runs to `_build_async()` or selection runs to `RecordSelectionRunner(self).run(...)`. `_prepare_async_run()` remains the shared scheduler-construction path.

### RecordSelectionRunner

Per-build collaborator for record-selection runs. It holds a reference to its `DatasetBuilder` so it can reuse scheduler construction, processors, artifact storage, resource configuration, and build-level run state without duplicating them in a context object. It owns the bounded candidate loop, selection decisions, per-batch media and marker commits, unresolved-path resume compatibility and cleanup, committed-marker reconstruction, exhaustion, and terminal publication.

### Async Execution (`_build_async`)

Preparation (`_prepare_async_run`):
1. Builds `gen_map` — maps each column name to its generator instance (multi-column configs share a single instance)
2. Creates `ExecutionGraph` from column dependencies
3. Partitions rows into row groups by `buffer_size`
4. Constructs `CompletionTracker`, `RowGroupBufferManager`, `AsyncTaskScheduler`
5. Composes a generic before-checkpoint callback for configured post-batch processors

`AsyncTaskScheduler` runs on a dedicated async loop with frontier-driven dispatch, task-admission leases, salvage rounds for failed tasks, and order-dependent locks for columns that must execute sequentially. Ready frontier tasks enter `FairTaskQueue`, are selected through virtual-time ordering, and are committed only after `TaskAdmissionController` acquires the required scheduler resources. Salvage-exhausted tasks are dropped except for preserved retryable failures: provider rate limits and local request-admission queue timeouts stay deferred and retry after cooldown/backoff so scheduler-local pressure delays records rather than discarding them.

`AsyncTaskScheduler` has no record-selection-specific callback. `on_before_checkpoint` is generic. A result-oriented caller can set `retain_row_group_result=True` for a single-row-group run; the scheduler then retains even a zero-row result and defers checkpoint and terminal events until the caller finishes its durable commit. Record selection uses that one contract so accepted and dropped telemetry reflects the marker on disk and `scheduler_job_completed` remains the terminal event. If caller-side selection, processing, commit, or cleanup fails, the deferred terminal event is emitted with an error outcome and no pre-marker checkpoint is reported.

Post-batch processors run only after selection. When a candidate accepts no rows, their zero-row batch-addressed side files are removed before the marker is committed; this prevents an empty first parquet file from defining an empty directory schema that masks accepted processor output from a later candidate.

### Execution Graph

`ExecutionGraph` (in `dataset_builders/utils/execution_graph.py`) models column dependencies:
- Upstream/downstream sets derived from `required_columns`, side-effect columns, and `skip.when` references
- `GenerationStrategy` per column (CELL_BY_CELL or FULL_COLUMN)
- Kahn topological sort for execution order
- `split_upstream_by_strategy` — separates batch-level from cell-level dependencies
- Skip metadata per column — `get_skip_config`, `should_propagate_skip`, `get_required_columns`, and `get_side_effect_columns` — queried at runtime to evaluate skip decisions

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
- **`skip_tracker.py`** — manages the `__internal_skipped_columns` metadata key on record dicts. Each record carries a `__internal_skipped_columns` set listing which columns were skipped for that row. `apply_skip_to_record` adds the column name to that set, writes the skip value into the cell, and clears any side-effect columns. `strip_skip_metadata_from_records` removes the `__internal_skipped_columns` key before DataFrame construction so it never reaches parquet.

`_run_cell` and `_run_batch` in `AsyncTaskScheduler` call `_should_skip_record` / `_apply_skip_to_record`. Skipped cells report as skipped (not success) in progress tracking.

DAG edges are added for `skip.when` column references in both `topologically_sort_column_configs` (compile-time sort) and `ExecutionGraph.create` (runtime graph) so skip-gate columns are generated before the gated column.

### RowGroupBufferManager

Manages per-row-group DataFrames and persistence:
- `checkpoint_row_group` → writes parquet via `ArtifactStorage`
- Updates dataset metadata between row groups
- Tracks dropped rows and actual record counts for resume

### Resume Checkpointing

`DatasetBuilder.build(..., resume=ResumeMode.*)` can continue an interrupted run from the last durable checkpoint:

- `ResumeMode.NEVER` always starts a fresh run, using a timestamped dataset directory when needed.
- `ResumeMode.ALWAYS` resumes the existing dataset directory and raises on incompatible state.
- `ResumeMode.IF_POSSIBLE` resumes when the persisted config fingerprint matches; otherwise it starts a fresh timestamped run.

Checkpoint state lives in `metadata.json`. Each metadata write includes the config fingerprint (`config_hash`, `config_hash_algo`, and `config_hash_version`) so compatibility checks do not need to deserialize `builder_config.json` for the common path. `builder_config.json` remains the human-readable record of the run configuration and the fallback for older datasets.

For ordinary generation, resume scans `parquet-files/batch_*.parquet` and reads parquet metadata to recover the completed row-group IDs and their actual persisted row counts. `metadata.json` remains the source of truth for the run *configuration* (`buffer_size`, `target_num_records`, `original_target_num_records`, config fingerprint), but the filesystem is the source of truth for *progress* (`num_completed_batches`, `actual_num_records`). Splitting the two sources is what lets resume survive a crash between writing a row-group parquet and updating metadata - the filesystem reflects the durable state even when metadata lags by a step. Reading actual row counts also matters for early-shutdown salvage, where a completed parquet file can contain fewer rows than the requested row-group size. Resume tolerates non-contiguous IDs because row groups can complete out of order.

Ordinary resume relies on stable row-group boundaries within a run. It treats datasets that have completed `process_after_generation()` as terminal: after-generation processors operate on the whole dataset and can re-chunk rows or change schema, invalidating row-group identity for later resume/extension. The terminal-state check raises a clear `DatasetGenerationError` (not a `TypeError`) when the persisted metadata is missing required fields such as `target_num_records`.

On the ordinary path, after-generation processors run unconditionally on the on-disk dataset whenever they are configured — including the case where resume sees every row group already on disk. This closes the crash window between the final row-group parquet write and the `post_generation_state="started"` marker write: in that window, the dataset is complete but post-generation never ran, and the on-disk parquet files are still clean (no processor has touched them). The `post_generation_state="started"` short-circuit still rejects the other direction (`process_after_generation()` crashed mid-rewrite, leaving the parquet files in an ambiguous state), so resume only re-runs after-generation when it is safe to do so.

Record selection uses a separate two-phase checkpoint model. Each completed candidate batch commits a marker under `selection-checkpoints/` and, when it accepted rows, an immutable accepted-row partition under `selection-accepted/`; those artifacts, rather than the published `parquet-files/`, are the durable source of selection progress. Once enough rows are accepted—or the candidate budget is exhausted with `on_exhausted="return_partial"`—the engine publishes the final dataset from the immutable partitions and then runs after-generation processors. A valid completed publication is reused as-is on resume, while a missing or incomplete publication can be rebuilt from the immutable partitions without regenerating candidates. This keeps candidate progress resumable even when publication or after-generation processing is interrupted.

Before `ArtifactStorage` resolves or caches a possibly timestamped dataset path, `RecordSelectionRunner._selection_resume_probe()` reads the unresolved requested directory (`artifact_path / dataset_name`). The probe distinguishes missing, readable, and unreadable metadata; whether readable metadata contains record-selection state; stored target and buffer values; checkpoint-directory presence; and actual marker presence. Both runtime compatibility and cleanup use this state model. Missing metadata with no marker is compatible (including an empty selection-checkpoint directory), while unreadable or readable non-selection metadata is incompatible. Selection metadata or a checkpoint directory establishes selection ownership for cleanup; unreadable metadata without either is preserved rather than destructively claimed. Raw JSON reading is intentional here because `ArtifactStorage.read_metadata()` would resolve and cache the dataset path too early. After path resolution, the same runner hydrates markers, validates committed partitions, cleans uncommitted artifacts, and reconstructs publication.

Metadata writes are atomic (`tmp` file + `fsync` + `os.replace`). Metadata remains authoritative for run configuration and publication state; ordinary parquet files and record-selection markers are authoritative for generation progress. With `ResumeMode.ALWAYS`, corrupt or partially written metadata raises a clear `DatasetGenerationError`. During record-selection preflight with `ResumeMode.IF_POSSIBLE`, unreadable metadata is treated as incompatible: selection-owned state is cleared, unowned state is preserved, and generation restarts under normal fresh-path semantics. Ordinary-run corrupt-metadata behavior is unchanged.

`DatasetCreationResults` from a resume invocation reflects the full on-disk dataset for anything that reads the artifact directory (`load_dataset`, `count_records`, `load_analysis`, `export`, `push_to_hub`), but per-run observability (`task_traces`, model-usage logs, telemetry events) is scoped to the current invocation — the original run's in-memory state is not persisted across process boundaries.

When `RunConfig.write_scheduler_events` is enabled, each invocation also appends its scheduler event segment to `scheduler_events.jsonl` in the dataset directory.

## Data Flow

```
DatasetBuilder.build()
  → initialize generators and graph
  ├─ ordinary → _build_async()
  │    → _prepare_async_run()
  │    → AsyncTaskScheduler.run()
  └─ record selection → RecordSelectionRunner(self).run()
       → probe unresolved resume state and restore committed markers
       → for each immutable candidate batch
            → DatasetBuilder._prepare_async_run()
            → AsyncTaskScheduler.run() retains one completed candidate DataFrame
            → select and trim accepted rows
            → promote accepted media and run strict post-batch processors
            → commit accepted partition and marker
            → report the durable accepted/dropped checkpoint
       → materialize terminal publication
```

Dataset-builder row-group admission is fixed: `RunConfig.max_concurrent_row_groups` (default `3`) is the hard in-flight cap. `buffer_size` controls the records per group, so raising the cap can increase active memory. The scheduler's adaptive row-group mode remains internal and is not wired through the public interface.

When request admission is available, async scheduling may use request-pressure snapshots as a read-only advisory during fair-queue selection. A request-pressured task can be skipped for an eligible peer without mutating request-admission state; provider/model/domain request limits remain owned by request admission.

## Design Decisions

- **One execution engine behind the API.** The async scheduler handles row-group parallelism, DAG-aware dispatch, resume, and checkpointing for all generation runs.
- **Selection collaborator.** `DatasetBuilder` owns shared engine setup; the per-build `RecordSelectionRunner` owns the feature lifecycle and resume state, keeping selection-specific orchestration and recovery out of the central builder.
- **Feature-agnostic scheduler.** The scheduler produces candidate DataFrames without selection callbacks. `retain_row_group_result` is one generic contract for result ownership and caller-managed completion reporting.
- **DAG-driven ordering** ensures columns with dependencies (e.g., a judge column that depends on a text column) are generated in the correct order, regardless of the order they appear in the config.
- **Fair async admission with bounded borrow by default** keeps the scheduler flowing across ready columns and model groups. `FairTaskQueue.select_next(...)` chooses eligible ready work, `TaskAdmissionController` leases scheduler resources before spawn, and `FairTaskQueue.commit(...)` removes the selected task only after admission succeeds. The default `BoundedBorrowTaskAdmissionPolicyConfig` computes a strict per-group share, lets solo groups borrow only up to a capacity-derived reserve, and makes borrowed groups yield when eligible peer pressure appears. Passing `bounded_borrow=None` selects strict-fair admission for tests and benchmark comparisons. Per-group virtual-time ordering prevents a large ready frontier from degenerating into a column-by-column wave, and scheduler-resource accounting remains separate from provider/model request admission.
- **Salvage rounds** retry failed tasks after all other tasks in a round complete, improving resilience against transient LLM failures without blocking the entire generation.
- **Unified DAG construction.** `topologically_sort_column_configs` (in `execution_graph.py`) determines column ordering using Kahn's algorithm; the runtime `ExecutionGraph` adds strategy-aware dependency tracking for the async scheduler.

## Cross-References

- [System Architecture](overview.md) - end-to-end data flow
- [Engine Layer](engine.md) — compilation and generator hierarchy
- [Models](models.md) — how generators access LLMs
- [Config Layer](config.md) — column configs and dependency declarations
