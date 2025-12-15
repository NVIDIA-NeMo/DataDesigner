# 🎨 NeMo Data Designer CLI

This directory contains the Command-Line Interface (CLI) for configuring model providers and model configurations used in Data Designer.

## Overview

The CLI provides an interactive interface for managing:
- **Model Providers**: LLM API endpoints (NVIDIA, OpenAI, Anthropic, custom providers)
- **Model Configs**: Specific model configurations with inference parameters

Configuration files are stored in `~/.data-designer/` by default and can be referenced by Data Designer workflows.

## Architecture

The CLI follows a **layered architecture** pattern, separating concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                         Commands                            │
│  Entry points for CLI commands (list, providers, models)    │
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
  - `models.py`: Configure models
  - `providers.py`: Configure providers
  - `reset.py`: Reset/delete configurations

#### 2. **Controllers** (`controllers/`)
- **Purpose**: Orchestrate user workflows and coordinate between services, forms, and UI
- **Responsibilities**:
  - Implement the main workflow logic (add, update, delete, etc.)
  - Coordinate between services and interactive forms
  - Handle user navigation and session state
  - Manage associated resource deletion (e.g., deleting models when provider is deleted)
- **Files**:
  - `model_controller.py`: Orchestrates model configuration workflows
  - `provider_controller.py`: Orchestrates provider configuration workflows

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
  - `model_service.py`: Model configuration business logic
  - `provider_service.py`: Provider business logic

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
- **Purpose**: Handle data persistence (YAML file I/O)
- **Responsibilities**:
  - Load configuration from YAML files
  - Save configuration to YAML files
  - Check file existence
  - Delete configuration files
- **Files**:
  - `base.py`: Abstract base repository with common operations
  - `model_repository.py`: Model configuration persistence
  - `provider_repository.py`: Provider persistence

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
  - `model_builder.py`: Interactive model configuration builder
  - `provider_builder.py`: Interactive provider configuration builder

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

The CLI manages two YAML configuration files:

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
