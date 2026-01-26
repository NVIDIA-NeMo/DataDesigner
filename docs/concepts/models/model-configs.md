# Model Configurations

Model configurations define the specific models you use for synthetic data generation and their associated inference parameters. Each `ModelConfig` represents a named model that can be referenced throughout your data generation workflows.

## Overview

A `ModelConfig` specifies which LLM model to use and how it should behave during generation. When you create column configurations (like `LLMText`, `LLMCode`, or `LLMStructured`), you reference a model by its alias. Data Designer uses the model configuration to determine which model to call and with what parameters.

## ModelConfig Structure

The `ModelConfig` class has the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alias` | `str` | Yes | Unique identifier for this model configuration (e.g., `"my-text-model"`, `"reasoning-model"`) |
| `model` | `str` | Yes | Model identifier as recognized by the provider (e.g., `"nvidia/nemotron-3-nano-30b-a3b"`, `"gpt-4"`) |
| `inference_parameters` | `InferenceParamsT` | No | Controls model behavior during generation. Use `ChatCompletionInferenceParams` for text/code/structured generation or `EmbeddingInferenceParams` for embeddings. Defaults to `ChatCompletionInferenceParams()` if not provided. The generation type is automatically determined by the inference parameters type. See [Inference Parameters](inference-parameters.md) for details. |
| `provider` | `str` | No | Reference to the name of the Provider to use (e.g., `"nvidia"`, `"openai"`, `"openrouter"`). If not specified, one set as the default provider, which may resolve to the first provider if there are more than one |
| `skip_health_check` | `bool` | No | Whether to skip the health check for this model. Defaults to `False`. Set to `True` to skip health checks when you know the model is accessible or want to defer validation. |


## Examples

### Basic Model Configuration

```python
import data_designer.config as dd

# Simple model configuration with fixed parameters
model_config = dd.ModelConfig(
    alias="my-text-model",
    model="nvidia/nemotron-3-nano-30b-a3b",
    provider="nvidia",
    inference_parameters=dd.ChatCompletionInferenceParams(
        temperature=0.85,
        top_p=0.95,
        max_tokens=2048,
    ),
)
```

### Multiple Model Configurations for Different Tasks

```python
import data_designer.config as dd

model_configs = [
    # Creative tasks
    dd.ModelConfig(
        alias="creative-model",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.9,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    # Critic tasks
    dd.ModelConfig(
        alias="critic-model",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.25,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    # Reasoning and structured tasks
    dd.ModelConfig(
        alias="reasoning-model",
        model="openai/gpt-oss-20b",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=4096,
        ),
    ),
    # Vision tasks
    dd.ModelConfig(
        alias="vision-model",
        model="nvidia/nemotron-nano-12b-v2-vl",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    # Embedding tasks
    dd.ModelConfig(
        alias="embedding_model",
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        provider="nvidia",
        inference_parameters=dd.EmbeddingInferenceParams(
            encoding_format="float",
            extra_body={
                "input_type": "query"
            }
        )
    )
]
```

!!! tip "Experiment with max_tokens for Task-Specific Model Configurations"
    The number of tokens required to generate a single data entry can vary significantly with use case. For example, reasoning models often need more tokens to "think through" problems before generating a response. Note that `max_tokens` specifies the **maximum number of output tokens** to generate in the response, so set this value based on the expected length of the generated content.

### Skipping Health Checks

By default, Data Designer runs a health check for each model before starting data generation to ensure the model is accessible and configured correctly. You can skip this health check for specific models by setting `skip_health_check=True`:

```python
import data_designer.config as dd

model_config = dd.ModelConfig(
    alias="my-model",
    model="nvidia/nemotron-3-nano-30b-a3b",
    provider="nvidia",
    inference_parameters=dd.ChatCompletionInferenceParams(
        temperature=0.85,
        top_p=0.95,
        max_tokens=2048,
    ),
    skip_health_check=True,  # Skip health check for this model
)
```

!!! info "When to Skip Health Checks"
    Skipping health checks can be useful when:

    - You've already verified the model is accessible and want to speed up initialization
    - You're using a model that doesn't support the standard health check format
    - You want to defer model validation until the model is actually used

    Note that skipping health checks means errors will only be discovered during actual data generation.

## See Also

- **[Inference Parameters](inference-parameters.md)**: Detailed guide to inference parameters and how to configure them
- **[Model Providers](model-providers.md)**: Learn about configuring model providers
- **[Default Model Settings](default-model-settings.md)**: Pre-configured model settings included with Data Designer
- **[Custom Model Settings](custom-model-settings.md)**: Learn how to create custom providers and model configurations
- **[Inference Parameters](inference-parameters.md)**: Detailed guide to inference parameters and how to configure them
- **[Model Providers](model-providers.md)**: Learn about configuring model providers
- **[Configure Model Settings With the CLI](configure-model-settings-with-the-cli.md)**: Use the CLI to manage model settings
- **[Column Configurations](../../code_reference/column_configs.md)**: Learn how to use models in column configurations
