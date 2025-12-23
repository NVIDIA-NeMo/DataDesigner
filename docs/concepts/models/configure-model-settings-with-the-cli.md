# Configuring Model Settings Using The CLI

The Data Designer CLI provides an interactive interface for creating and managing default model providers and model configurations stored in your Data Designer home directory (default: `~/.data-designer/`).

## Configuration Files

The CLI manages two YAML configuration files:

- **`model_providers.yaml`**: Model provider configurations
- **`model_configs.yaml`**: Model configurations

!!! info "Automatic Configuration"
    If these configuration files don't already exist, the Data Designer library automatically creates them with default settings at runtime when first initialized.

!!! note "Custom Directory"
    You can customize the configuration directory location with the `DATA_DESIGNER_HOME` environment variable:
    ```bash
    export DATA_DESIGNER_HOME="/path/to/your/custom/directory"
    ```

## CLI Commands

The Data Designer CLI provides four main configuration commands:

```bash
# Configure model providers
data-designer config providers

# Configure models
data-designer config models

# List current configurations
data-designer config list

# Reset all configurations
data-designer config reset
```

!!! tip "Getting help"
    See available commands
    ```bash
    data-designer --help
    ```

    See available sub-commands
    ```bash
    data-designer config --help
    ```

## Managing Model Providers

Run the interactive provider configuration command:

```bash
data-designer config providers
```

### Available Operations

**Add a new provider**: Define a new provider by entering its name, endpoint URL, provider type, and optionally an API key (as plain text or as an environment variable name).

**Update an existing provider**: Modify an existing provider's settings. All fields are pre-filled with current values.

**Delete a provider**: Remove a provider and its associated models.

**Delete all providers**: Remove all providers and their associated models.

**Change default provider**: Set which provider is used by default. This option is only available when multiple providers are configured.

## Managing Model Configurations

Run the interactive model configuration command:

```bash
data-designer config models
```

!!! info "Provider Required"
    You need at least one provider configured before adding models. Run `data-designer config providers` first if none exist.

### Available Operations

**Add a new model configuration**

Create a new model configuration with the following fields:

- **Alias**: A unique name for referencing this model in a column configuration.
- **Model ID**: The model identifier (e.g., `nvidia/nemotron-3-nano-30b-a3b`)
- **Provider**: Select from available providers (if multiple exist)
- **Temperature**: Sampling temperature (0.0 to 2.0)
- **Top P**: Nucleus sampling parameter (0.0 to 1.0)
- **Max Tokens**: Maximum output length (1 to 100000)

!!! note "Additional Settings"
    To configure additional inference parameter settings or use distribution-based inference parameters, edit the `model_configs.yaml` file directly.

**Update an existing model configuration**: Modify an existing model's configuration. All fields are pre-filled with current values.

**Delete a model configuration**: Remove a single model configuration.

**Delete all model configurations**: Remove all model configurations. The CLI will ask for confirmation before proceeding.

## Listing Configurations

View all current configurations:

```bash
data-designer config list
```

This command displays:

- **Model Providers**: All configured providers with their endpoints (API keys are masked)
- **Default Provider**: The currently selected default provider
- **Model Configurations**: All configured models with their settings

## Resetting Configurations

Delete all configuration files:

```bash
data-designer config reset
```

The CLI will show which configuration files exist and ask for confirmation before deleting them.

!!! danger "Destructive Operation"
    This command permanently deletes all configuration files and resets to the default model providers and configurations. You'll need to reconfigure your custom configurations from scratch.

## See Also

- **[Default Model Settings](default-model-settings.md)**: Pre-configured providers and model settings included with Data Designer
- **[Custom Model Settings](custom-model-settings.md)**: Learn how to create custom providers and model configurations
- **[Model Providers](model-providers.md)**: Learn about the `ModelProvider` class and provider configuration
- **[Model Configurations](model-configs.md)**: Learn about `ModelConfig`
- **[Quick Start Guide](../../quick-start.md)**: Get started with a simple example
