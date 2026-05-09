# 🎨 NeMo Data Designer CLI

This directory contains the Command-Line Interface (CLI) for configuring model providers, model configurations, MCP providers, tool configs, managed assets, and plugin catalogs used in Data Designer.

## Overview

The CLI provides an interactive interface for managing:
- **Model Providers**: LLM API endpoints (NVIDIA, OpenAI, Anthropic, custom providers)
- **Model Configs**: Specific model configurations with inference parameters
- **MCP Providers**: MCP server configurations for tool integration
- **Tool Configs**: Tool definitions used by configured models and workflows
- **Managed Assets**: Persona dataset downloads under the Data Designer home directory
- **Plugin Catalogs**: Catalog aliases for finding Data Designer plugin packages
- **Plugin Packages**: Install and uninstall packages from catalogs, check version compatibility first, and verify Data Designer can discover the plugins they provide

Configuration files and CLI-managed state are stored in `~/.data-designer/` by default.

## Architecture

The CLI follows a **layered architecture** pattern, separating concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                         Commands                            │
│  Entry points for CLI commands (config, download, plugin)   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       Controllers                           │
│  Orchestrate user workflows and coordinate between layers   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌──────────────────┐                 ┌──────────────────┐
│    Services      │                 │      Forms       │
│  Business logic  │                 │  Interactive UI  │
└──────────────────┘                 └──────────────────┘
        │
        ▼
┌──────────────────┐
│   Repositories   │
│  Data persistence│
└──────────────────┘
```

### Layer Responsibilities

#### 1. **Commands** (`commands/`)
- **Purpose**: Define CLI command entry points using Typer
- **Responsibilities**:
  - Parse command-line arguments and options
  - Initialize controllers with appropriate configuration
  - Handle top-level error reporting
- **Files**:
  - `list.py`: List current configurations
  - `mcp.py`: Configure MCP providers
  - `models.py`: Configure models
  - `providers.py`: Configure providers
  - `download.py`: Download managed assets
  - `plugin.py`: Discover, install, and uninstall plugin packages from catalogs
  - `reset.py`: Reset/delete configurations
  - `tools.py`: Configure tool configs

#### 2. **Controllers** (`controllers/`)
- **Purpose**: Orchestrate user workflows and coordinate between services, forms, and UI
- **Responsibilities**:
  - Implement the main workflow logic (add, update, delete, etc.)
  - Coordinate between services and interactive forms
  - Handle user navigation and session state
  - Manage associated resource deletion (e.g., deleting models when provider is deleted)
- **Files**:
  - `download_controller.py`: Orchestrates managed asset download workflows
  - `mcp_provider_controller.py`: Orchestrates MCP provider configuration workflows
  - `model_controller.py`: Orchestrates model configuration workflows
  - `provider_controller.py`: Orchestrates provider configuration workflows
  - `plugin_catalog_controller.py`: Orchestrates plugin catalog browsing, alias management, and package workflows
  - `tool_controller.py`: Orchestrates tool configuration workflows

**Key Features**:
- **Associated Resource Management**: When deleting a provider, the controller checks for associated models and prompts the user to delete them together
- **Interactive Navigation**: Supports add/update/delete/delete_all operations with user-friendly menus

#### 3. **Services** (`services/`)
- **Purpose**: Implement business logic and enforce domain rules
- **Responsibilities**:
  - Validate business rules (e.g., unique names, required fields)
  - Implement CRUD operations with validation
  - Coordinate between multiple repositories when needed
  - Handle default management (e.g., default provider selection)
- **Files**:
  - `mcp_provider_service.py`: MCP provider configuration business logic
  - `model_service.py`: Model configuration business logic
  - `provider_service.py`: Provider business logic
  - `plugin_catalog_service.py`: Plugin catalog loading, search, compatibility checks, and installed plugin listing
  - `plugin_install_service.py`: Chooses and runs uv or pip commands for installing/uninstalling plugin packages, keeps installed Data Designer packages in place, and verifies installed plugins
  - `tool_service.py`: Tool configuration business logic

**Key Methods**:
- `list_all()`: Get all configured items
- `get_by_*()`: Retrieve specific items
- `add()`: Add new item with validation
- `update()`: Update existing item
- `delete()`: Delete single item
- `delete_by_aliases()`: Batch delete (models only)
- `find_by_provider()`: Find models by provider (models only)
- `set_default()`, `get_default()`: Manage default provider (providers only)

#### 4. **Repositories** (`repositories/`)
- **Purpose**: Handle data persistence and read-only reference metadata
- **Responsibilities**:
  - Load configuration from YAML files
  - Save configuration to YAML files
  - Check file existence and delete configuration files where applicable
  - Provide read-only metadata for built-in managed assets
- **Files**:
  - `base.py`: Abstract base repository with common operations
  - `mcp_provider_repository.py`: MCP provider configuration persistence
  - `model_repository.py`: Model configuration persistence
  - `persona_repository.py`: Read-only persona locale metadata
  - `provider_repository.py`: Provider persistence
  - `plugin_catalog_repository.py`: Plugin catalog aliases, catalog fetching, and URL-keyed catalog cache
  - `tool_repository.py`: Tool configuration persistence

**Base Repository Pattern**:
```python
class ConfigRepository(ABC, Generic[T]):
    def load(self) -> T | None: ...
    def save(self, config: T) -> None: ...
    def exists(self) -> bool: ...
    def delete(self) -> None: ...
```

#### 5. **Forms** (`forms/`)
- **Purpose**: Interactive form-based data collection from users
- **Responsibilities**:
  - Define form fields with validation
  - Collect user input interactively
  - Support navigation (back, cancel)
  - Build configuration objects from form data
- **Files**:
  - `builder.py`: Abstract form builder base
  - `field.py`: Form field types (TextField, SelectField, NumericField)
  - `form.py`: Form container and prompt orchestration
  - `mcp_provider_builder.py`: Interactive MCP provider configuration builder
  - `model_builder.py`: Interactive model configuration builder
  - `provider_builder.py`: Interactive provider configuration builder
  - `tool_builder.py`: Interactive tool configuration builder

**Form Features**:
- Field-level validation
- Auto-completion support
- History navigation (arrow keys)
- Current value display when editing (`(current value: X)` instead of `(default: X)`)
- Value clearing support (type `'clear'` to remove optional parameter values)
- Back navigation support
- Empty input handling (Enter key keeps current value or skips optional fields)

#### 6. **UI Utilities** (`ui.py`)
- **Purpose**: User interface utilities for terminal output and input
- **Responsibilities**:
  - Interactive menus with arrow key navigation
  - Text input prompts with validation
  - Confirmation dialogs
  - Styled output (success, error, warning, info)
  - Configuration preview displays
- **Key Functions**:
  - `select_with_arrows()`: Interactive arrow-key menu
  - `prompt_text_input()`: Text input with validation and completion
  - `confirm_action()`: Yes/no confirmation
  - `print_*()`: Styled console output
  - `display_config_preview()`: YAML preview with syntax highlighting


## Configuration Files

The CLI manages YAML configuration files, managed assets, and plugin catalog caches under `~/.data-designer/`:

### `~/.data-designer/model_providers.yaml`

Stores model provider configurations (API endpoints):

```yaml
providers:
  - name: nvidia
    endpoint: https://integrate.api.nvidia.com/v1
    provider_type: openai
    api_key: NVIDIA_API_KEY
  - name: openai
    endpoint: https://api.openai.com/v1
    provider_type: openai
    api_key: OPENAI_API_KEY
default: nvidia
```

### `~/.data-designer/model_configs.yaml`

Stores model configurations:

```yaml
model_configs:
  - alias: llama3-70b
    model: meta/llama-3.1-70b-instruct
    provider: nvidia
    inference_parameters:
      generation_type: chat-completion
      temperature: 0.7
      top_p: 0.9
      max_tokens: 2048
      max_parallel_requests: 4
      timeout: 60
  - alias: gpt-4
    model: gpt-4-turbo
    provider: openai
    inference_parameters:
      generation_type: chat-completion
      temperature: 0.8
      top_p: 0.95
      max_tokens: 4096
      max_parallel_requests: 4
  - alias: embedder
    model: text-embedding-3-large
    provider: openai
    inference_parameters:
      generation_type: embedding
      encoding_format: float
      dimensions: 1024
      max_parallel_requests: 4
```

### `~/.data-designer/mcp_providers.yaml`

Stores MCP provider configurations:

```yaml
providers:
  - name: local-tools
    provider_type: stdio
    command: python
    args:
      - "-m"
      - my_mcp_server
```

### `~/.data-designer/tool_configs.yaml`

Stores tool configurations that reference MCP providers:

```yaml
tool_configs:
  - tool_alias: research-tools
    providers:
      - local-tools
    max_tool_call_turns: 5
```

### `~/.data-designer/managed-assets/`

Stores managed assets downloaded by CLI commands such as
`data-designer download personas`. Set `DATA_DESIGNER_MANAGED_ASSETS_PATH` to
store managed assets outside `DATA_DESIGNER_HOME`.

### `~/.data-designer/plugin_catalogs.yaml`

Stores user-added plugin catalog aliases. The built-in NVIDIA catalog points at
`https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json`, is
always available, and is not written to this file. Set
`DATA_DESIGNER_DEFAULT_PLUGIN_CATALOG_URL` to repoint the built-in catalog for QA or
staging.

```yaml
catalogs:
  - alias: research
    url: https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json
```

### `~/.data-designer/plugin-catalog-cache/`

Stores fetched plugin catalog payloads as JSON cache files keyed by catalog alias and URL hash. This prevents a re-pointed alias from serving stale catalog data from a previous URL.

Plugin package arguments accept either the full package name or the package
alias. For packages named `data-designer-{alias}`, the alias is `{alias}`. For
example, `data-designer-github` can be addressed as `github` in `info`,
`install`, and `uninstall`.

## Usage Examples

### Configure Providers

```bash
# Interactive provider configuration
data-designer config providers

# Options:
#  - Add a new provider (predefined: nvidia, openai, anthropic, or custom)
#  - Update an existing provider
#  - Delete a provider (with associated model cleanup)
#  - Delete all providers
#  - Change default provider
```

### Configure Models

```bash
# Interactive model configuration
data-designer config models

# Options:
#  - Add a new model
#  - Update an existing model
#  - Delete a model
#  - Delete all models
```

### List Configurations

```bash
# Display current configurations
data-designer config list
```

### Reset Configurations

```bash
# Delete configuration files (with confirmation)
data-designer config reset
```

### Discover, Install, and Uninstall Plugin Packages

```bash
# List compatible plugin packages from the default NVIDIA catalog
data-designer plugin list

# Search a specific catalog
data-designer plugin --catalog research search transform

# Show package metadata, compatibility, docs, and the install command
data-designer plugin info github

# Install a plugin package from a catalog and verify Data Designer can discover its plugins
data-designer plugin install github --yes

# Preview without changing the current environment
data-designer plugin install github --dry-run

# Uninstall a plugin package and verify Data Designer no longer discovers its plugins
data-designer plugin uninstall github --yes

# Preview without changing the current environment
data-designer plugin uninstall github --dry-run

# Add and manage catalog aliases
data-designer plugin catalog add research https://github.com/acme/dd-plugins
data-designer plugin catalog list
data-designer plugin catalog remove research

# List installed runtime plugin entry points without importing plugin modules
data-designer plugin installed
```

When installing a plugin package, the CLI first checks the package's Python and
Data Designer version requirements. The plugin package and its other
dependencies are installed normally, but the currently installed Data Designer
packages (`data-designer`, `data-designer-config`, and `data-designer-engine`)
are kept in place. This prevents a plugin dependency from upgrading,
downgrading, or reinstalling Data Designer itself.

In an active virtual environment with a user `pyproject.toml`, `uv` uses
`uv add` so the plugin package is recorded in the project. Otherwise the CLI
installs into the current Python environment with `uv pip install` or `pip`.
`uv` plugin installs require `uv >= 0.6.0`; auto mode falls back to `pip` when
`uv` is missing or too old. `pip` remains supported for pip-only environments.
