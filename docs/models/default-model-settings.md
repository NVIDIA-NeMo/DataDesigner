# Default Model Settings

Data Designer ships with pre-configured model providers and model configurations that make it easy to start generating synthetic data without manual setup.

## Model Providers

Data Designer includes two default model providers that are configured automatically:

### NVIDIA Provider (`nvidia`)

- **Endpoint**: `https://integrate.api.nvidia.com/v1`
- **API Key**: Set via `NVIDIA_API_KEY` environment variable
- **Models**: Access to NVIDIA's hosted models from [build.nvidia.com](https://build.nvidia.com)
- **Getting Started**: Sign up and get your API key at [build.nvidia.com](https://build.nvidia.com)

The NVIDIA provider gives you access to state-of-the-art models including Nemotron and other NVIDIA-optimized models.

### OpenAI Provider (`openai`)

- **Endpoint**: `https://api.openai.com/v1`
- **API Key**: Set via `OPENAI_API_KEY` environment variable
- **Models**: Access to OpenAI's model catalog
- **Getting Started**: Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

The OpenAI provider gives you access to GPT models and other OpenAI offerings.

## Model Configurations

Data Designer provides pre-configured model aliases for common use cases. When you create a `DataDesignerConfigBuilder` without specifying `model_configs`, these default configurations are automatically available.

### NVIDIA Models

The following model configurations are automatically available when `NVIDIA_API_KEY` is set:

| Alias | Model | Use Case | Temperature | Top P |
|-------|-------|----------|-------------|-------|
| `nvidia-text` | `nvidia/nvidia-nemotron-nano-9b-v2` | General text generation | 0.85 | 0.95 |
| `nvidia-reasoning` | `openai/gpt-oss-20b` | Reasoning and analysis tasks | 0.35 | 0.95 |
| `nvidia-vision` | `nvidia/nemotron-nano-12b-v2-vl` | Vision and image understanding | 0.85 | 0.95 |


### OpenAI Models

The following model configurations are automatically available when `OPENAI_API_KEY` is set:

| Alias | Model | Use Case | Temperature | Top P |
|-------|-------|----------|-------------|-------|
| `openai-text` | `gpt-4.1` | General text generation | 0.85 | 0.95 |
| `openai-reasoning` | `gpt-5` | Reasoning and analysis tasks | 0.35 | 0.95 |
| `openai-vision` | `gpt-5` | Vision and image understanding | 0.85 | 0.95 |


### How Default Model Providers and Configurations Work

When the Data Designer library or the CLI is initialized, default model configurations and providers are stored in the Data Designer home directory for easy access and customization if they do not already exist. These configuration files serve as the single source of truth for model settings. By default they are saved to the following paths:

- **Model Configs**: `~/.data-designer/model_configs.yaml`
- **Model Providers**: `~/.data-designer/model_providers.yaml`

!!! tip Tip
    While these files provide a convenient way to specify settings for your model providers and configuration you use most often, they can always be set programatically in your SDG workflow.

You can customize the home directory location by setting the `DATA_DESIGNER_HOME` environment variable:

```bash
# In your .bashrc, .zshrc, or similar
export DATA_DESIGNER_HOME="/path/to/your/custom/directory"
```

These configuration files can be modified in two ways:

1. **Using the CLI**: Run CLI commands to add, update, or delete model configurations and providers
2. **Manual editing**: Directly edit the YAML files with your preferred text editor

Both methods operate on the same files, ensuring consistency across your entire Data Designer setup.

## Important Notes

!!! warning "API Key Requirements"
    While default model configurations are always available, you need to set the appropriate API key environment variable (`NVIDIA_API_KEY` or `OPENAI_API_KEY`) to actually use the corresponding models for data generation. Without a valid API key, any attempt to generate data using that provider's models will fail.

!!! tip "Environment Variables"
    Store your API keys in environment variables rather than hardcoding them in your scripts:

    ```bash
    # In your .bashrc, .zshrc, or similar
    export NVIDIA_API_KEY="your-api-key-here"
    export OPENAI_API_KEY="your-openai-api-key-here"
    ```

## See Also

- **[Configure Model Settings With the CLI](configure-model-settings-with-the-cli.md)**: Learn how to use the CLI to manage model settings.
- **[Quick Start Guide](../quick-start.md)**: Get started with a simple example
- **[Model Configurations](model-configs.md)**: Learn about model configurations
