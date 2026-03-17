## Identity

You are developing **Data Designer** — a framework for creating synthetic datasets from scratch. Users declare *what* data they want (columns, types, relationships, quality criteria) and the system figures out *how* to generate it.

The core contract is **declare, don't orchestrate**. General users should never have to think about execution order, batching, or model plumbing. Every change you make should preserve or strengthen that contract. If a change forces users to reason about runtime mechanics, it's moving in the wrong direction.

## The Layering Is Structural

Three packages share the `data_designer` namespace via PEP 420 implicit namespace packages (no `__init__.py` at the `data_designer/` level — this is intentional): **config** (schema) → **engine** (execution) → **interface** (entry-point).

- **Config** (`data-designer-config`): Schema layer. Pydantic models for columns, pipelines, and model routing. Builder API for config construction. Pure data — no I/O.
- **Engine** (`data-designer-engine`): Execution layer. Compiler resolves configs into a DAG; runtime executes it via LLM calls, batching, and task scheduling. Registries map config types to implementations.
- **Interface** (`data-designer`): Entry-point layer. `DataDesigner` class, CLI, and result types. Orchestrates engine on behalf of users.

**Import direction is one-way and absolute.** Config must never import engine. Engine must never import interface. Violations create circular dependencies and break the namespace package structure.

## Core Concepts

- **Columns** — the primary abstraction. Each column declares its dependencies and extra outputs. The DAG of these declarations determines execution order automatically — users never specify ordering.
- **Samplers** — statistical distributions, category sets, and persona generators used by sampler columns to produce values without model calls.
- **Seed datasets** — existing data that bootstraps generation, providing real-world context for downstream columns.
- **Processors** — batch-level transformations (schema reshaping, column dropping) that run outside the column DAG.
- **Models** — model configurations are routed per-column via `model_alias`. The framework is model-agnostic and provider-agnostic.
- **Plugins** — extend any of the above via `setuptools` entry points. Define a config, implement the behavior, register both.

## Core Design Principles

- **Config is declarative, engine is imperative.** Users build configs that describe intent; the engine is where I/O, LLM calls, and state mutation happen. The **compiler** bridges the two: it validates the config, resolves the full DAG, and produces the runtime plan the engine executes.
- **Registries connect types to behavior.** `TaskRegistry` maps config type → implementation class. Plugins extend this at runtime.
- **Errors normalize at boundaries.** Each layer defines canonical error types. Callers depend on these public types, never on vendor-specific or internal exceptions.

## Structural Invariants

- **Imports flow downstream only** — config must not import from engine or interface; engine must not import from interface
- **Fast imports** — Avoid new import-time side effects. Prefer lazy initialization and defer heavy work until use-time
- **No relative imports** — absolute only
- **No untyped code** — all code carries type annotations
- **Follow established patterns** — before introducing new conventions, study surrounding code and match existing style, structure, and idioms
- **No untested code paths** — new and changed logic must have associated unit tests

## Development

Use `make` targets — not raw tool commands — for all standard workflows:

- `make check-all-fix` — format + lint (ruff)
- `make test` — run all tests
- `make update-license-headers` — add SPDX license headers (never add them manually)
- `make perf-import CLEAN=1` — profile import time after dependency changes
