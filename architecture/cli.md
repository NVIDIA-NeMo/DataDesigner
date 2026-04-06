# CLI

The CLI (`data-designer`) provides an interactive command-line interface for configuring models, providers, tools, and personas, as well as running dataset generation. It uses a layered architecture for config management and delegates generation to the public `DataDesigner` API.

Source: `packages/data-designer/src/data_designer/cli/`

## Overview

The CLI is built on Typer with lazy command loading to keep startup fast. Config management commands follow a **command → controller → service → repository** layering pattern. Generation commands bypass this stack and use the public `DataDesigner` class directly.

## Key Components

### Entry Point

`data-designer` is registered as a console script pointing to `data_designer.cli.main:main`. On startup:
1. `ensure_cli_default_model_settings()` initializes default model/provider configs
2. `app()` launches the Typer application

### Lazy Command Loading

`create_lazy_typer_group` and `_LazyCommand` stubs defer importing command modules until a command is actually invoked. This keeps `data-designer --help` fast — only the command names and descriptions are loaded eagerly; the full module (and its dependencies) loads on first use.

### Layering Pattern (Config Management)

Config management commands (models, providers, tools, personas) follow a consistent four-layer pattern:

| Layer | Role | Example |
|-------|------|---------|
| **Command** | Thin Typer entry, wires `DATA_DESIGNER_HOME` | `models_command` → `ModelController(DATA_DESIGNER_HOME).run()` |
| **Controller** | UX flow: menus, forms, success/error display | `ModelController` composes repos + services + `ModelFormBuilder` |
| **Service** | Domain rules: uniqueness, merge, delete-all | `ModelService.add/update/delete` over `ModelRepository` |
| **Repository** | File I/O for typed config registries | `ModelRepository` extends `ConfigRepository[ModelConfigRegistry]` |

Repositories: `ModelRepository`, `ProviderRepository`, `ToolRepository`, `MCPProviderRepository`, `PersonaRepository`.

Services mirror the repository domains with business logic (validation, conflict resolution).

### Generation Commands

`preview`, `create`, and `validate` commands use `GenerationController`, which:
1. Loads config via `load_config_builder`
2. Calls `DataDesigner.preview()`, `DataDesigner.create()`, or `DataDesigner.validate()` directly
3. Handles output display and error formatting

This keeps generation aligned with the public Python API — the CLI is a thin wrapper, not a separate code path.

### UI Utilities

- `cli/ui.py` — Rich console helpers for formatted output
- `cli/forms/` — interactive form builders for config creation/editing
- `cli/utils/config_loader.py` — config file resolution and loading
- `sample_records_pager.py` — paginated display of generated records

## Data Flow

### Config Management
```
User invokes command (e.g., `data-designer models add`)
  → Command function wires DATA_DESIGNER_HOME
  → Controller presents interactive form
  → Service validates and applies changes
  → Repository reads/writes config files
```

### Generation
```
User invokes command (e.g., `data-designer create config.yaml`)
  → GenerationController loads config
  → DataDesigner.create() runs the full pipeline
  → Results displayed via Rich console
```

## Design Decisions

- **Lazy command loading** keeps `data-designer --help` responsive: command modules (and their heavy dependencies, such as the engine and model stacks) load only when a command is invoked, not at process startup.
- **Controller/service/repo for config, direct API for generation** — config management benefits from the layered pattern (testable services, swappable repositories). Generation doesn't need this indirection; it delegates to the same `DataDesigner` class that Python users call directly.
- **`DATA_DESIGNER_HOME`** centralizes all CLI-managed state (model configs, provider configs, tool configs, personas) in a single directory, defaulting to `~/.data_designer/`.
- **Rich-based UI** provides formatted tables, progress bars, and interactive prompts without requiring a web interface.

## Cross-References

- [System Architecture](overview.md) — where the CLI fits
- [Agent Introspection](agent-introspection.md) — the `agent` command group
- [Config Layer](config.md) — config objects the CLI manages
- [Models](models.md) — model/provider configuration
