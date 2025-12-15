# Custom Model Settings

While Data Designer ships with pre-configured model providers and configurations, you can create custom configurations to use different models, adjust inference parameters, or connect to custom API endpoints.

## When to Use Custom Settings

Use custom model settings when you need to:

- Use models not included in the defaults
- Adjust inference parameters (temperature, top_p, max_tokens) for specific use cases
- Add distribution-based inference parameters for variability
- Connect to self-hosted or custom model endpoints
- Create multiple variants of the same model with different settings

## Creating and Using Custom Settings

### Custom Models with Default Providers

Create custom model configurations that use the default providers (no need to define providers yourself):

```python
from data_designer.essentials import (
    CategorySamplerParams,
    ChatCompletionInferenceParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    ModelConfig,
    SamplerColumnConfig,
    SamplerType,
)

# Create custom models using default providers
custom_models = [
    # High-temperature for more variability
    ModelConfig(
        alias="creative-writer",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",  # Uses default NVIDIA provider
        inference_parameters=ChatCompletionInferenceParams(
            temperature=1.2,
            top_p=0.98,
            max_tokens=4096,
        ),
    ),
    # Low-temperature for less variability
    ModelConfig(
        alias="fact-checker",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",  # Uses default NVIDIA provider
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
        ),
    ),
]

# Create DataDesigner (uses default providers)
data_designer = DataDesigner()

# Pass custom models to config builder
config_builder = DataDesignerConfigBuilder(model_configs=custom_models)

# Add a topic column using a categorical sampler
config_builder.add_column(
    SamplerColumnConfig(
        name="topic",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Artificial Intelligence", "Space Exploration", "Ancient History", "Climate Science"],
        ),
    )
)

# Use your custom models
config_builder.add_column(
    LLMTextColumnConfig(
        name="creative_story",
        model_alias="creative-writer",
        prompt="Write a creative short story about {{topic}}.",
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="facts",
        model_alias="fact-checker",
        prompt="List 3 facts about {{topic}}.",
    )
)

# Preview your dataset
preview_result = data_designer.preview(config_builder=config_builder)
preview_result.display_sample_record()
```

!!! note "Default Providers Always Available"
    When you only specify `model_configs`, the default model providers (NVIDIA and OpenAI) are still available. You only need to create custom providers if you want to connect to different endpoints or modify provider settings.

!!! tip "Mixing Custom and Default Models"
    When you provide custom `model_configs` to `DataDesignerConfigBuilder`, they **replace** the defaults entirely. To use custom model configs in addition to the default configs, use the add_model_config method:

    ```python
    # Load defaults first
    config_builder = DataDesignerConfigBuilder()

    # Add custom model to defaults
    config_builder.add_model_config(
        ModelConfig(
            alias="my-custom-model",
            model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
            provider="nvidia",  # Uses default provider
            inference_parameters=ChatCompletionInferenceParams(
                temperature=0.6,
                max_tokens=8192,
            ),
        )
    )

    # Now you can use both default and custom models
    # Default: nvidia-text, nvidia-reasoning, nvidia-vision, etc.
    # Custom: my-custom-model
    ```

### Custom Providers with Custom Models

Define both custom providers and custom model configurations when you need to connect to services not included in the defaults:

!!! warning "Network Accessibility"
    The custom provider endpoints must be reachable from where Data Designer runs. Ensure network connectivity, firewall rules, and any VPN requirements are properly configured.

```python
from data_designer.essentials import (
    CategorySamplerParams,
    ChatCompletionInferenceParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    ModelConfig,
    ModelProvider,
    SamplerColumnConfig,
    SamplerType,
)

# Step 1: Define custom providers
custom_providers = [
    ModelProvider(
        name="my-custom-provider",
        endpoint="https://api.my-llm-service.com/v1",
        provider_type="openai",  # OpenAI-compatible API
        api_key="MY_SERVICE_API_KEY",  # Environment variable name
    ),
    ModelProvider(
        name="my-self-hosted-provider",
        endpoint="https://my-org.internal.com/llm/v1",
        provider_type="openai",
        api_key="SELF_HOSTED_API_KEY",
    ),
]

# Step 2: Define custom models
custom_models = [
    ModelConfig(
        alias="my-text-model",
        model="openai/some-model-id",
        provider="my-custom-provider",  # References provider by name
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.85,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    ModelConfig(
        alias="my-self-hosted-text-model",
        model="openai/some-hosted-model-id",
        provider="my-self-hosted-provider",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        ),
    ),
]

# Step 3: Create DataDesigner with custom providers
data_designer = DataDesigner(model_providers=custom_providers)

# Step 4: Create config builder with custom models
config_builder = DataDesignerConfigBuilder(model_configs=custom_models)

# Step 5: Add a topic column using a categorical sampler
config_builder.add_column(
    SamplerColumnConfig(
        name="topic",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Technology", "Healthcare", "Finance", "Education"],
        ),
    )
)

# Step 6: Use your custom model by referencing its alias
config_builder.add_column(
    LLMTextColumnConfig(
        name="short_news_article",
        model_alias="my-text-model",  # Reference custom alias
        prompt="Write a short news article about the '{{topic}}' topic in 10 sentences.",
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="long_news_article",
        model_alias="my-self-hosted-text-model",  # Reference custom alias
        prompt="Write a detailed news article about the '{{topic}}' topic.",
    )
)

# Step 7: Preview your dataset
preview_result = data_designer.preview(config_builder=config_builder)
preview_result.display_sample_record()
```

## See Also

- **[Default Model Settings](default-model-settings.md)**: Pre-configured providers and model settings
- **[Configure Model Settings With the CLI](configure-model-settings-with-the-cli.md)**: CLI-based configuration
- **[Quick Start Guide](../../quick-start.md)**: Basic usage example
