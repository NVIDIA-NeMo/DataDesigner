# AGENTS.md

This file is for agents **developing** DataDesigner — the codebase you are working in.
If you are an agent helping a user **build a dataset**, use the [`data-designer` skill](skills/data-designer/SKILL.md) and the [product documentation](https://nvidia-nemo.github.io/DataDesigner/) instead.

**DataDesigner** is an NVIDIA NeMo framework for creating synthetic datasets from scratch. Users declare what their data should look like (columns, types, relationships, validation rules); the engine figures out how to generate it. Every change you make should preserve this "declare, don't orchestrate" contract.

## The Layering Is Structural

The `data_designer` namespace is split across three installable packages that merge at runtime via PEP 420 implicit namespace packages (no top-level `__init__.py`).

| Package | Path | Owns |
|---------|------|------|
| `data-designer-config` | `packages/data-designer-config/` | `data_designer.config` — column configs, model configs, sampler params, builder API, plugin system, lazy imports |
| `data-designer-engine` | `packages/data-designer-engine/` | `data_designer.engine` — column generators, dataset builders, DAG execution, model facade, validators, sampling |
| `data-designer` | `packages/data-designer/` | `data_designer.interface` — public `DataDesigner` class, results, errors; `data_designer.cli` — CLI entry point; `data_designer.integrations` |

**Dependency direction (left depends on right):** interface → engine → config. Never import against this flow.

## Core Concepts

- **Column** — a named field in the output dataset, defined by a column config
- **Sampler** — a built-in statistical generator (UUID, Category, Uniform, Gaussian, Person, DateTime, etc.)
- **Seed dataset** — an existing dataset used as input for generation
- **Processor** — a post-generation transformation applied to column values
- **Model** — an LLM endpoint configured via `ModelConfig` and accessed through the model facade
- **Plugin** — a user-supplied extension registered via entry points (custom column generators, validators, profilers)

## Core Design Principles

1. **Declarative config, imperative engine.** Users build configs; the engine compiles them into an execution plan. Config objects are data; they never call the engine directly.
2. **Registries connect types to behavior.** Column generators, validators, and profilers are discovered through registries. Adding a new type means registering it, not modifying orchestration code.
3. **Errors normalize at boundaries.** Third-party exceptions are wrapped into canonical project error types at module boundaries. Callers depend on `data_designer` errors, not leaked internals.

## Structural Invariants

- **Import direction** — interface → engine → config (left depends on right). No reverse imports.
- **Fast imports** — heavy third-party libraries are lazy-loaded via `data_designer.lazy_heavy_imports`. See [STYLEGUIDE.md](STYLEGUIDE.md) for the pattern.
- **No relative imports** — absolute imports only, enforced by ruff rule `TID`.
- **Typed code** — all functions, methods, and class attributes require type annotations. Modern syntax: `list[str]`, `str | None`.
- **`from __future__ import annotations`** — required in every Python source file.
- **Follow established patterns** — match the conventions of the module you're editing. When in doubt, read the neighboring code.
- **No untested code paths** — new logic requires tests. See [DEVELOPMENT.md](DEVELOPMENT.md) for testing guidance.

## Development

```bash
make check-all-fix        # format + lint (ruff)
make test                 # run all test suites
make update-license-headers  # add SPDX headers to new files
make perf-import CLEAN=1  # profile import time (run after adding heavy deps)
```

For full setup, testing, and workflow details see [DEVELOPMENT.md](DEVELOPMENT.md).
For code style, naming, and import conventions see [STYLEGUIDE.md](STYLEGUIDE.md).
For deeper dives into specific subsystems see [`architecture/`](architecture/).
