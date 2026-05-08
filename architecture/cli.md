# CLI

The CLI (`data-designer`) provides an interactive command-line interface for configuring models, providers, tools, and personas, discovering/installing plugins from catalogs, and running dataset generation. It uses a layered architecture for setup workflows and delegates generation to the public `DataDesigner` API.

Source: `packages/data-designer/src/data_designer/cli/`

## Overview

The CLI is built on Typer with lazy command loading to keep startup fast. Config management and plugin catalog commands follow a **command → controller → service → repository** layering pattern. Generation commands bypass this stack and use the public `DataDesigner` class directly.

## Key Components

### Entry Point

`data-designer` is registered as a console script pointing to `data_designer.cli.main:main`. On startup:
1. `ensure_cli_default_model_settings()` initializes default model/provider configs
2. `app()` launches the Typer application

### Lazy Command Loading

`create_lazy_typer_group` and `_LazyCommand` stubs defer importing command modules until a command is actually invoked. This keeps `data-designer --help` fast — only the command names and descriptions are loaded eagerly; the full module (and its dependencies) loads on first use.

### Layering Pattern (Setup Workflows)

Config management commands (models, providers, tools, personas) follow a consistent four-layer pattern:

| Layer | Role | Example |
|-------|------|---------|
| **Command** | Thin Typer entry, wires `DATA_DESIGNER_HOME` | `models_command` → `ModelController(DATA_DESIGNER_HOME).run()` |
| **Controller** | UX flow: menus, forms, success/error display | `ModelController` composes repos + services + `ModelFormBuilder` |
| **Service** | Domain rules: uniqueness, merge, delete-all | `ModelService.add/update/delete` over `ModelRepository` |
| **Repository** | File I/O for typed config registries | `ModelRepository` extends `ConfigRepository[ModelConfigRegistry]` |

Repositories: `ModelRepository`, `ProviderRepository`, `ToolRepository`, `MCPProviderRepository`, `PersonaRepository`.

Services mirror the repository domains with business logic (validation, conflict resolution).

Plugin catalog commands use the same layering shape:

| Layer | Role | Example |
|-------|------|---------|
| **Command** | Thin Typer entry, wires `DATA_DESIGNER_HOME` and command options | `plugins list/search/info/install/installed/catalogs` → `PluginCatalogController(DATA_DESIGNER_HOME)` |
| **Controller** | UX flow: catalog tables, package metadata, compatibility display, install confirmation | `PluginCatalogController` composes catalog + install services |
| **Service** | Domain rules: package-first flattening, compatibility checks, install planning, entry point verification | `PluginCatalogService`, `PluginInstallService` |
| **Repository** | File/cache I/O for catalog aliases and catalog documents | `PluginCatalogRepository` |

The built-in `nvidia` catalog points at `https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json`. `NVIDIA-NeMo/DataDesignerPlugins` defines the package-first catalog shape: top-level packages carry install metadata, compatibility constraints, docs, and nested runtime plugins. The CLI flattens nested plugins for list/search display, but `info` and `install` resolve back to the package so installation targets the package requirement.

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
User invokes command (e.g., `data-designer config models`)
  → Command function wires DATA_DESIGNER_HOME
  → Controller presents interactive menu
  → Service validates and applies changes
  → Repository reads/writes config files
```

### Plugin Catalog Discovery
```
User invokes command (e.g., `data-designer plugins list`)
  → Command function wires DATA_DESIGNER_HOME and catalog options
  → PluginCatalogController resolves the catalog alias
  → PluginCatalogService loads and filters package-first catalog entries
  → PluginCatalogRepository reads local config and cached/remote catalog JSON
```

### Plugin Install
```
User invokes command (e.g., `data-designer plugins install text-transform`)
  → PluginCatalogController resolves runtime plugin or package name
  → PluginCatalogService evaluates Python and Data Designer compatibility
  → PluginInstallService builds a pip/uv install plan for the package requirement
  → PluginInstallService verifies declared entry points after installation
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
- **Controller/service/repo for setup workflows, direct API for generation** — config and plugin catalog workflows benefit from the layered pattern (testable services, swappable repositories). Generation doesn't need this indirection; it delegates to the same `DataDesigner` class that Python users call directly.
- **`DATA_DESIGNER_HOME`** centralizes all CLI-managed state (model configs, provider configs, tool configs, personas) in a single directory, defaulting to `~/.data_designer/`.
- **Package-first plugin catalogs** keep install metadata at the package boundary while allowing one package to expose multiple runtime plugins through entry points.
- **Rich-based UI** provides formatted tables, progress bars, and interactive prompts without requiring a web interface.

## Cross-References

- [System Architecture](overview.md) — where the CLI fits
- [Agent Introspection](agent-introspection.md) — the `agent` command group
- [Config Layer](config.md) — config objects the CLI manages
- [Models](models.md) — model/provider configuration
