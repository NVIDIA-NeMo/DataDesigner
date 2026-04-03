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
- Upstream/downstream sets derived from `required_columns` and side-effect columns
- `GenerationStrategy` per column (CELL_BY_CELL or FULL_COLUMN)
- Kahn topological sort for execution order
- `split_upstream_by_strategy` — separates batch-level from cell-level dependencies

### CompletionTracker

Tracks per-row-group, per-column completion state:
- **Cell-level**: completed cell indices for `CELL_BY_CELL` columns
- **Batch-level**: full-column completion flags for `FULL_COLUMN` columns
- **Frontier**: computes ready tasks when backed by `ExecutionGraph`
- Handles dropped rows and downstream task enqueuing

### DAG (Config-Level)

`dataset_builders/utils/dag.py` provides `topologically_sort_column_configs` — builds a NetworkX graph from `required_columns` and side-effect columns, returns a topological ordering. Used by both execution modes for initial column ordering.

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
