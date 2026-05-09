# Model Providers

Model providers are external services that host and serve models. Data Designer uses the `ModelProvider` class to configure connections to these services.

## Overview

A `ModelProvider` defines how Data Designer connects to a provider's API endpoint. When you create a `ModelConfig`, you reference a provider by name, and Data Designer uses that provider's settings to make API calls to the appropriate endpoint.

## ModelProvider Configuration

The `ModelProvider` class has the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the provider (e.g., `"nvidia"`, `"openai"`, `"openrouter"`) |
| `endpoint` | `str` | Yes | API endpoint URL (e.g., `"https://integrate.api.nvidia.com/v1"`) |
| `provider_type` | `str` | No | Provider type: `"openai"` (default) or `"anthropic"`. See [Supported Provider Types](#supported-provider-types) below |
| `api_key` | `str` | No | API key or environment variable name (e.g., `"NVIDIA_API_KEY"`) |
| `extra_body` | `dict[str, Any]` | No | Additional parameters to include in the request body of all API requests to the provider. |
| `extra_headers` | `dict[str, str]` | No | Additional headers to include in all API requests to the provider. |

## Supported Provider Types

Data Designer supports two provider types:

| Type | Description |
|------|-------------|
| `"openai"` | OpenAI-compatible chat completion API. This is the default and works with most providers, including NVIDIA NIM, vLLM, TGI, OpenRouter, Together AI, and OpenAI itself. |
| `"anthropic"` | Anthropic's native Messages API for Claude models. Use this when connecting directly to Anthropic's API. |

Most self-hosted and third-party endpoints expose an OpenAI-compatible API, so `provider_type="openai"` is the right choice in the majority of cases. Only use `"anthropic"` when connecting directly to Anthropic's API at `https://api.anthropic.com`.

> **Note:** Previous versions of Data Designer supported additional provider types (e.g., `"azure"`, `"bedrock"`, `"vertex_ai"`) via a LiteLLM bridge. These are no longer supported. If you were using one of these types, switch to `provider_type="openai"` and point the `endpoint` to an OpenAI-compatible proxy or gateway for that service.

## API Key Configuration

The `api_key` field can be specified in two ways:

1. **Environment variable name** (recommended): Set `api_key` to the name of an environment variable (e.g., `"NVIDIA_API_KEY"`). Data Designer will automatically resolve it at runtime.

2. **Plain-text value**: Set `api_key` to the actual API key string. This is less secure and not recommended for production use.

```python
# Method 1: Environment variable (recommended)
provider = ModelProvider(
    name="nvidia",
    endpoint="https://integrate.api.nvidia.com/v1",
    api_key="NVIDIA_API_KEY",  # Will be resolved from environment
)

# Method 2: Direct value (not recommended)
provider = ModelProvider(
    name="nvidia",
    endpoint="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-abc123...",  # Direct API key
)
```

## See Also

- **[Model Configurations](model-configs.md)**: Learn about configuring models
- **[Inference Parameters](inference-parameters.md)**: Detailed guide to inference parameters and how to configure them
- **[Default Model Settings](default-model-settings.md)**: Pre-configured providers and model settings included with Data Designer
- **[Custom Model Settings](custom-model-settings.md)**: Learn how to create custom providers and model configurations
- **[Model Configurations](model-configs.md)**: Learn about configuring models
- **[Inference Parameters](inference-parameters.md)**: Detailed guide to inference parameters and how to configure them
- **[Configure Model Settings With the CLI](configure-model-settings-with-the-cli.md)**: Use the CLI to manage providers and model settings
- **[Getting Started](../../index.md)**: Installation and basic usage example
