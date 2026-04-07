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
4. **`FULL_COLUMN` generators**: `generate` on the whole batch DataFrame; optional resize via `allow_resize`

### Async Execution (`_build_async`)

Preparation (`_prepare_async_run`):
1. Builds `gen_map` — maps each column name to its generator instance (multi-column configs share a single instance)
2. Creates `ExecutionGraph` from column dependencies
3. Partitions rows into row groups by `buffer_size`
4. Constructs `CompletionTracker`, `RowGroupBufferManager`, `AsyncTaskScheduler`
5. Hooks `ProcessorRunner` for pre-batch and post-batch stages

`AsyncTaskScheduler` runs on a dedicated async loop with semaphore-based concurrency, salvage rounds for failed tasks, and order-dependent locks for columns that must execute sequentially.

### Execution Graph

`ExecutionGraph` (in `dataset_builders/utils/execution_graph.py`) models column dependencies:
- Upstream/downstream sets derived from `required_columns`, side-effect columns, and `skip.when` references
- `GenerationStrategy` per column (CELL_BY_CELL or FULL_COLUMN)
- Kahn topological sort for execution order
- `split_upstream_by_strategy` — separates batch-level from cell-level dependencies
- Skip metadata: `SkipConfig`, `propagate_skip`, `required_columns`, and `side_effects_by_producer` per column — queried at runtime by both engines to evaluate skip decisions

### CompletionTracker

Tracks per-row-group, per-column completion state:
- **Cell-level**: completed cell indices for `CELL_BY_CELL` columns
- **Batch-level**: full-column completion flags for `FULL_COLUMN` columns
- **Frontier**: computes ready tasks when backed by `ExecutionGraph`
- Handles dropped rows and downstream task enqueuing

### Conditional Generation (Skip)

Columns can be conditionally skipped per-row via `SkipConfig` (defined in `data_designer.config.base`). Two mechanisms control skipping:

1. **Expression gate** — `skip=SkipConfig(when="{{ expr }}")` on a `SingleColumnConfig`. The Jinja2 expression is evaluated per-row; when truthy, the column is skipped for that row and the configured `value` (default `None`) is written instead of calling the generator.
2. **Skip propagation** — when an upstream column was skipped, downstream columns auto-skip unless they set `propagate_skip=False`. Propagation checks `required_columns` against the row's `__skipped__` set.

Skip evaluation is handled by two utility modules:

- **`skip_evaluator.py`** — `evaluate_skip_when` renders the expression in a `NativeSandboxedEnvironment` (native Python types, `StrictUndefined`). `should_skip_by_propagation` checks set intersection between required columns and skipped columns.
- **`skip_tracker.py`** — manages the `__skipped__` metadata key on record dicts. Each record carries a `__skipped__` set listing which columns were skipped for that row. `apply_skip_to_record` adds the column name to that set, writes the skip value into the cell, and clears any side-effect columns. `strip_skip_metadata_from_records` removes the `__skipped__` key before DataFrame construction so it never reaches parquet (called by `DatasetBatchManager`, `RowGroupBufferManager`, and inline in both engines).

Both execution modes integrate skip at the same points:

- **Sequential**: `_run_full_column_generator` and the fan-out methods (`_fan_out_with_threads`, `_fan_out_with_async`) call `_should_skip_cell` per record. Skipped rows are excluded from the generator input, then merged back with skip metadata preserved. A fast `_column_can_skip` check short-circuits the per-record evaluation when no skip config or propagation applies.
- **Async**: `_run_cell` and `_run_batch` in `AsyncTaskScheduler` call `_should_skip_record` / `_apply_skip_to_record` with the same logic. Skipped cells report as skipped (not success) in progress tracking.

DAG edges are added for `skip.when` column references (both in `dag.py` and `ExecutionGraph.create`) so skip-gate columns are generated before the gated column.

### DAG (Config-Level)

`dataset_builders/utils/dag.py` provides `topologically_sort_column_configs` — builds a NetworkX graph from `required_columns`, side-effect columns, and `skip.when` references, returns a topological ordering. Used by both execution modes for initial column ordering.

### DatasetBatchManager

Manages in-memory row buffers and persistence:
- `finish_batch` → writes parquet via `ArtifactStorage`
- Updates dataset metadata between batches
- The async path uses `RowGroupBufferManager` for per-row-group DataFrames and checkpointing

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
      → AsyncTaskScheduler(semaphores, salvage_rounds)
  → scheduler.run()
      → for each row group, dispatch ready tasks from frontier
      → tasks execute generators, update CompletionTracker
      → checkpoints via RowGroupBufferManager
  → collect TaskTraces, emit telemetry
```

## Design Decisions

- **Dual execution engines behind one API.** The sequential engine is simpler and easier to debug; the async engine adds row-group parallelism for throughput. Users switch via an environment variable without changing their code.
- **DAG-driven ordering** ensures columns with dependencies (e.g., a judge column that depends on a text column) are generated in the correct order, regardless of the order they appear in the config.
- **Salvage rounds in async mode** retry failed tasks after all other tasks in a round complete, improving resilience against transient LLM failures without blocking the entire generation.
- **Separate config-level and runtime DAGs.** The config-level DAG (`dag.py`) determines column ordering; the runtime `ExecutionGraph` adds strategy-aware dependency tracking for the async scheduler.

## Cross-References

- [System Architecture](overview.md) — end-to-end data flow
- [Engine Layer](engine.md) — compilation and generator hierarchy
- [Models](models.md) — how generators access LLMs
- [Config Layer](config.md) — column configs and dependency declarations
