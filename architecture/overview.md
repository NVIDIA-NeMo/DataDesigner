# System Architecture

DataDesigner is split across three installable packages that merge at runtime into a single `data_designer` namespace via PEP 420 implicit namespace packages (no top-level `__init__.py`).

## Overview

```
┌─────────────────────────────────────────────────────────┐
│  data-designer  (interface + CLI + integrations)        │
│    DataDesigner class, CLI commands, HuggingFace Hub    │
├─────────────────────────────────────────────────────────┤
│  data-designer-engine  (execution)                      │
│    Generators, builders, models, MCP, sampling,         │
│    validators, profilers, processing                    │
├─────────────────────────────────────────────────────────┤
│  data-designer-config  (declaration)                    │
│    Column configs, model configs, sampler params,       │
│    builder API, plugin system, lazy imports             │
└─────────────────────────────────────────────────────────┘
```

**Dependency direction:** interface → engine → config. No reverse imports.

Users declare what their data should look like through config objects (columns, types, relationships, validation rules). The engine compiles those configs into an execution plan and generates the dataset. The interface package provides the public `DataDesigner` class and CLI that wire everything together.

## Key Components

| Component | Package | Entry Point |
|-----------|---------|-------------|
| `DataDesigner` | `data-designer` | Public API — `create()`, `preview()`, `validate()` |
| `DataDesignerConfigBuilder` | `data-designer-config` | Fluent builder for dataset configs |
| `DatasetBuilder` | `data-designer-engine` | Orchestrates generation (sync or async) |
| `ModelFacade` / `ModelRegistry` | `data-designer-engine` | LLM client abstraction with retry, throttle, usage tracking |
| `MCPFacade` / `MCPRegistry` | `data-designer-engine` | Tool execution via Model Context Protocol |
| `ColumnGeneratorRegistry` | `data-designer-engine` | Maps column types to generator implementations |
| `PluginRegistry` | `data-designer-config` | Discovers and registers entry-point plugins |
| CLI (`data-designer`) | `data-designer` | Typer-based CLI with lazy command loading |

## Data Flow

1. **Declaration** — User builds a `DataDesignerConfig` via the builder API or loads YAML/JSON. Columns are a discriminated union on `column_type`; sampler columns add a second discriminated layer on `sampler_type`.

2. **Compilation** — `compile_data_designer_config` enriches the config (seed columns, internal UUID column), runs static validation (Jinja references, code columns, processors), and produces a compiled column order via topological sort.

3. **Generation** — `DatasetBuilder` instantiates column generators from the registry, then executes one of two paths:
   - **Sequential** (default): batch loop over columns in topological order. Each generator produces its column via `CELL_BY_CELL` (threaded fan-out) or `FULL_COLUMN` strategy.
   - **Async** (`DATA_DESIGNER_ASYNC_ENGINE=1`): builds an `ExecutionGraph`, partitions rows into groups, and dispatches tasks via `AsyncTaskScheduler` with semaphore-based concurrency, salvage rounds, and per-row-group checkpointing.

4. **Post-processing** — `ProcessorRunner` applies transformations (pre-batch, post-batch, after-generation). Profilers analyze the generated dataset.

5. **Results** — `DatasetCreationResults` wraps the artifact storage, analysis, config, and metadata. Supports `load_dataset()`, record sampling, and `push_to_hub()`.

## Design Decisions

- **PEP 420 namespace packages** allow the three packages to be installed independently while sharing the `data_designer` namespace. This enables lighter installs (e.g., config-only for validation tooling) without import conflicts.
- **Lazy imports throughout** — `__getattr__`-based lazy loading in `data_designer.config` and `data_designer.interface`, plus `lazy_heavy_imports` for numpy/pandas, keep startup fast.
- **Dual execution engines** share the same `DatasetBuilder` API. The async engine adds row-group parallelism and DAG-aware scheduling without changing the public interface.
- **`TaskRegistry` subclasses: one instance per class** — `TaskRegistry.__new__` (`registry/base.py`) ensures a single instance of each concrete registry (column generators, profilers, processors). **`ModelRegistry`** and **`MCPRegistry`** are ordinary classes, constructed per run with injected dependencies. **`PluginRegistry`** (`plugins/registry.py`) uses `__new__` so entry points are discovered once per process.

## Cross-References

- [Config Layer](config.md) — builder API, column types, model configs, plugin system
- [Engine Layer](engine.md) — compilation, generators, registries
- [Models](models.md) — model facade, adapters, retry/throttle
- [Dataset Builders](dataset-builders.md) — sync/async orchestration, DAG, batching
- [MCP](mcp.md) — tool execution, session pooling
- [Sampling](sampling.md) — statistical generators, person/entity data
- [CLI](cli.md) — command structure, controller/service/repo pattern
- [Agent Introspection](agent-introspection.md) — type discovery, state commands
- [Plugins](plugins.md) — entry-point discovery, registry injection
