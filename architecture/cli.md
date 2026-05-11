# CLI

The CLI (`data-designer`) provides an interactive command-line interface for configuring models, providers, MCP providers, and tools, downloading managed persona datasets, discovering, installing, and uninstalling plugin packages from catalogs, and running dataset generation. It uses a layered architecture for setup workflows and delegates generation to the public `DataDesigner` API.

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

Config management commands (models, providers, MCP providers, tools) follow a consistent four-layer pattern:

| Layer | Role | Example |
|-------|------|---------|
| **Command** | Thin Typer entry, wires `DATA_DESIGNER_HOME` | `models_command` → `ModelController(DATA_DESIGNER_HOME).run()` |
| **Controller** | UX flow: menus, forms, success/error display | `ModelController` composes repos + services + `ModelFormBuilder` |
| **Service** | Domain rules: uniqueness, merge, delete-all | `ModelService.add/update/delete` over `ModelRepository` |
| **Repository** | File I/O for typed config registries | `ModelRepository` extends `ConfigRepository[ModelConfigRegistry]` |

Repositories: `ModelRepository`, `ProviderRepository`, `MCPProviderRepository`, and `ToolRepository`.
`PersonaRepository` provides read-only locale metadata for managed persona dataset downloads.

Services mirror the repository domains with business logic (validation, conflict resolution).

Plugin catalog commands use the same layering shape:

| Layer | Role | Example |
|-------|------|---------|
| **Command** | Thin Typer entry, wires `DATA_DESIGNER_HOME` and command options | `plugin` subcommands (`list`, `search`, `info`, `install`, `uninstall`, `installed`, `catalog`) → `PluginCatalogController(DATA_DESIGNER_HOME)` |
| **Controller** | UX flow: catalog tables, package metadata, compatibility display, install/uninstall confirmations | `PluginCatalogController` composes catalog + install services |
| **Service** | Domain rules: package listing, compatibility checks, uv/pip install and uninstall commands, plugin discovery verification | `PluginCatalogService`, `PluginInstallService` |
| **Repository** | File/cache I/O for catalog aliases and catalog documents | `PluginCatalogRepository` |

The built-in `nvidia` catalog points at `https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json`. `NVIDIA-NeMo/DataDesignerPlugins` defines the catalog format. Each catalog entry is an installable package with docs, install metadata, compatibility constraints, and one or more runtime plugins. Users install and uninstall packages, not individual runtime plugins. Commands that take a package name also accept the package alias from the `data-designer-{alias}` package-name pattern; for example, `data-designer-calculator` can be addressed as `calculator`. If a user passes a runtime plugin name where a package is required, the CLI reports the package that owns that runtime plugin.

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
User invokes command (e.g., `data-designer plugin list`)
  → Command function wires DATA_DESIGNER_HOME and catalog options
  → PluginCatalogController resolves the catalog alias and chooses table or narrow-terminal layout
  → PluginCatalogService loads packages and filters out incompatible packages by default
  → PluginCatalogRepository reads local config and cached/remote catalog JSON
```

### Plugin Install/Uninstall
```
User invokes command (e.g., `data-designer plugin install calculator`)
  → PluginCatalogController resolves the plugin package name or package alias
  → PluginCatalogService evaluates Python and Data Designer compatibility
  → PluginInstallService chooses uv or pip and builds the command.
    In active uv projects it uses `uv add` so the package is recorded in
    `pyproject.toml`; otherwise it installs into the current Python environment.
    Data Designer itself is already installed, so its packages are not reinstalled
    or replaced while installing plugin dependencies.
  → PluginInstallService verifies Data Designer can discover the package's runtime plugins
```

```
User invokes command (e.g., `data-designer plugin uninstall calculator`)
  → PluginCatalogController resolves the plugin package name or package alias
  → PluginInstallService chooses uv or pip and builds the uninstall command.
    Active uv projects remove the dependency from project metadata and uninstall
    the package from the current environment.
  → PluginInstallService verifies Data Designer no longer discovers the package's runtime plugins
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
- **`DATA_DESIGNER_HOME`** centralizes CLI-managed state (model configs, provider configs, MCP provider configs, tool configs, managed assets, plugin catalog aliases, and catalog caches) in a single directory, defaulting to `~/.data-designer/`.
- **Package-first plugin catalogs** match how users install plugins: one package can provide one or more runtime plugins, but install and uninstall commands always target the package.
- **Rich-based UI** provides formatted tables, progress bars, and interactive prompts without requiring a web interface.

## Cross-References

- [System Architecture](overview.md) — where the CLI fits
- [Agent Introspection](agent-introspection.md) — the `agent` command group
- [Config Layer](config.md) — config objects the CLI manages
- [Models](models.md) — model/provider configuration
