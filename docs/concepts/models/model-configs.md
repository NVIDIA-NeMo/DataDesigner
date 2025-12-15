# Model Configurations

Model configurations define the specific models you use for synthetic data generation and their associated inference parameters. Each `ModelConfig` represents a named model that can be referenced throughout your data generation workflows.

## Overview

A `ModelConfig` specifies which LLM model to use and how it should behave during generation. When you create column configurations (like `LLMText`, `LLMCode`, or `LLMStructured`), you reference a model by its alias. Data Designer uses the model configuration to determine which model to call and with what parameters.

## ModelConfig Structure

The `ModelConfig` class has the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alias` | `str` | Yes | Unique identifier for this model configuration (e.g., `"my-text-model"`, `"reasoning-model"`) |
| `model` | `str` | Yes | Model identifier as recognized by the provider (e.g., `"nvidia/nvidia-nemotron-nano-9b-v2"`, `"gpt-4"`) |
| `inference_parameters` | `InferenceParamsT` | No | Controls model behavior during generation. Use `ChatCompletionInferenceParams` for text/code/structured generation or `EmbeddingInferenceParams` for embeddings. Defaults to `ChatCompletionInferenceParams()` if not provided. The generation type is automatically determined by the inference parameters type. See [Inference Parameters](inference_parameters.md) for details. |
| `provider` | `str` | No | Reference to the name of the Provider to use (e.g., `"nvidia"`, `"openai"`). If not specified, one set as the default provider, which may resolve to the first provider if there are more than one |


## Examples

### Basic Model Configuration

```python
from data_designer.essentials import ChatCompletionInferenceParams, ModelConfig

# Simple model configuration with fixed parameters
model_config = ModelConfig(
    alias="my-text-model",
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    provider="nvidia",
    inference_parameters=ChatCompletionInferenceParams(
        temperature=0.85,
        top_p=0.95,
        max_tokens=2048,
    ),
)
```

### Multiple Model Configurations for Different Tasks

```python
from data_designer.essentials import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    GenerationType,
    ModelConfig
)

model_configs = [
    # Creative tasks
    ModelConfig(
        alias="creative-model",
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        provider="nvidia",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.9,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    # Critic tasks
    ModelConfig(
        alias="critic-model",
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        provider="nvidia",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.25,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    # Reasoning and structured tasks
    ModelConfig(
        alias="reasoning-model",
        model="openai/gpt-oss-20b",
        provider="nvidia",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=4096,
        ),
    ),
    # Vision tasks
    ModelConfig(
        alias="vision-model",
        model="nvidia/nemotron-nano-12b-v2-vl",
        provider="nvidia",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    # Embedding tasks
    ModelConfig(
        alias="embedding_model",
        model=-"nvidia/llama-3.2-nv-embedqa-1b-v2",
        provider="nvidia",
        inference_parameters=EmbeddingInferenceParams(
            encoding_format="float"
            extra_body={
                "input_type": "query"
            }
        )
    )
]
```

!!! tip "Experiment with max_tokens for Task-Specific Model Configurations"
    The number of tokens required to generate a single data entry can vary significantly with use case. For example, reasoning models often need more tokens to "think through" problems before generating a response. Note that `max_tokens` includes **both input and output tokens** (the total context window used), so factor in your prompt length, any context data, and the expected response length when setting this parameter.

## See Also

- **[Inference Parameters](inference-parameters.md)**: Detailed guide to inference parameters and how to configure them
- **[Model Providers](model-providers.md)**: Learn about configuring model providers
- **[Default Model Settings](default-model-settings.md)**: Pre-configured model settings included with Data Designer
- **[Configure Model Settings With the CLI](configure-model-settings-with-the-cli.md)**: Use the CLI to manage model settings
- **[Column Configurations](../../code_reference/column_configs.md)**: Learn how to use models in column configurations
