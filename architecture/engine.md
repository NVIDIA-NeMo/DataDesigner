# Engine Layer

The engine layer (`data_designer.engine`) compiles declarative configs into executable generation plans and runs them. It owns column generators, dataset builders, model access, MCP integration, sampling, validation, and profiling.

Source: `packages/data-designer-engine/src/data_designer/engine/`

## Overview

The engine is the largest package, organized into focused subsystems:

| Subsystem | Path | Role |
|-----------|------|------|
| Column generators | `column_generators/` | Registry + concrete generators for each column type |
| Dataset builders | `dataset_builders/` | Sync/async orchestration, DAG, batching |
| Models | `models/` | Facade, registry, clients, parsers, recipes, usage |
| MCP | `mcp/` | Tool registry, facade, I/O service |
| Sampling | `sampling_gen/` | Schema, DAG, data sources, person/entity helpers |
| Processing | `processing/` | Processors, Ginja (Jinja for generation), Gsonschema |
| Validators | `validators/` | Runtime row/batch validation |
| Analysis | `analysis/` | Dataset/column profiling |
| Registry | `registry/` | Generic `TaskRegistry` base + `DataDesignerRegistry` aggregator |
| Resources | `resources/` | Seed/person readers, managed datasets |
| Storage | `storage/` | Artifact and media storage |

Top-level modules handle cross-cutting concerns: `compiler.py` (config compilation), `validation.py` (static config validation), `context.py` (execution context), `configurable_task.py` (base for all tasks), `secret_resolver.py`, `model_provider.py`.

## Key Components

### Compilation Pipeline

`compiler.py` transforms a `DataDesignerConfig` into an execution-ready form:

1. Enriches the config with seed columns and an internal UUID column
2. Runs static validation (`validation.py`) — checks Jinja references, code columns, processor targets, constraint consistency
3. Produces `Violation` objects with typed `ViolationType` for structured error reporting

### Registry System

`TaskRegistry` (in `registry/base.py`) is the generic base: maps an enum value to a task class + config class. Uses `__new__`-based singleton per subclass to prevent duplicate instances.

`DataDesignerRegistry` bundles the three registries used by `DatasetBuilder`:
- `ColumnGeneratorRegistry` — column type → generator class
- `ColumnProfilerRegistry` — column type → profiler class
- Processor registry

`create_default_column_generator_registry()` registers all built-in types and merges plugin entry points.

### Column Generator Hierarchy

```
ConfigurableTask
  └── ColumnGenerator (abstract: get_generation_strategy, generate/agenerate)
        ├── FromScratchColumnGenerator (can_generate_from_scratch)
        ├── ColumnGeneratorWithModelRegistry
        │     └── ColumnGeneratorWithModel (cached model, inference params, MCP)
        ├── ColumnGeneratorCellByCell (strategy: CELL_BY_CELL, generate(dict))
        └── ColumnGeneratorFullColumn (strategy: FULL_COLUMN, generate(DataFrame))
```

Each concrete generator (e.g., `SamplerColumnGenerator`, `LLMTextColumnGenerator`) combines the appropriate base classes. The `GenerationStrategy` enum (`CELL_BY_CELL` or `FULL_COLUMN`) determines how the dataset builder dispatches work.

### ResourceProvider

Bundles everything a generator needs at runtime: `ModelRegistry`, `MCPRegistry`, `ArtifactStorage`, seed readers, person readers, secret resolver. Passed to generators during initialization.

## Data Flow

1. `DatasetBuilder` receives a `DataDesignerConfig` and a `DataDesignerRegistry`
2. Compilation produces a topologically sorted list of column configs
3. Generators are instantiated from the registry for each column config
4. The builder executes generators in dependency order (see [Dataset Builders](dataset-builders.md))
5. Post-generation processors and profilers run on the completed dataset

## Design Decisions

- **Registry + strategy pattern** decouples column type definitions (config) from generation behavior (engine). Adding a new column type means registering a config class and a generator class — no changes to orchestration code.
- **`ConfigurableTask` as the universal base** ensures all tasks (generators, profilers, processors) share config validation and resource access patterns.
- **Static validation before execution** catches config errors (missing references, invalid templates) before any LLM calls are made, failing fast and cheaply.
- **Sync/async bridge** on `ColumnGenerator` allows generators to be written as async and called from sync contexts via `_run_coroutine_sync` / `asyncio.to_thread`.

## Cross-References

- [System Architecture](overview.md) — package relationships
- [Config Layer](config.md) — column configs and builder API
- [Dataset Builders](dataset-builders.md) — sync/async execution, DAG
- [Models](models.md) — model facade and client adapters
- [MCP](mcp.md) — tool execution integration
- [Sampling](sampling.md) — sampler generators
- [Plugins](plugins.md) — how plugins register generators
