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
| `inference_parameters` | `InferenceParameters` | No | Controls model behavior during generation (temperature, top_p, max_tokens, etc). Defaults from constructing an empty `InferenceParameters` object are picked up when not provided.|
| `provider` | `str` | No | Reference to the name of the Provider to use (e.g., `"nvidia"`, `"openai"`). If not specified, one set as the default provider, which may resolve to the first provider if there are more than one |

## InferenceParameters

The `InferenceParameters` class controls how the model generates responses. It provides fine-grained control over generation behavior and supports both static values and dynamic distribution-based sampling.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `temperature` | `float` or `Distribution` | No | Controls randomness in generation (0.0 to 2.0). Higher values = more creative/random |
| `top_p` | `float` or `Distribution` | No | Nucleus sampling parameter (0.0 to 1.0). Controls diversity by filtering low-probability tokens |
| `max_tokens` | `int` | No | Maximum number of tokens for the request, including both input and output tokens (≥ 1) |
| `max_parallel_requests` | `int` | No | Maximum concurrent API requests (default: 4, ≥ 1) |
| `timeout` | `int` | No | API request timeout in seconds (≥ 1) |
| `extra_body` | `dict[str, Any]` | No | Additional parameters to include in the API request body |

!!! note "Default Values"
    If `temperature`, `top_p`, or `max_tokens` are not provided, the model provider's default values will be used. Different providers and models may have different defaults.

!!! tip "Controlling Reasoning Effort for GPT-OSS Models"
    For gpt-oss models like `gpt-oss-20b` and `gpt-oss-120b`, you can control the reasoning effort using the `extra_body` parameter:

    ```python
    # High reasoning effort (more thorough, slower)
    inference_parameters = InferenceParameters(
        extra_body={"reasoning_effort": "high"}
    )

    # Medium reasoning effort (balanced)
    inference_parameters = InferenceParameters(
        extra_body={"reasoning_effort": "medium"}
    )

    # Low reasoning effort (faster, less thorough)
    inference_parameters = InferenceParameters(
        extra_body={"reasoning_effort": "low"}
    )
    ```

### Temperature and Top P Guidelines

- **Temperature**:
    - `0.0-0.3`: Highly deterministic, focused outputs (ideal for structured/reasoning tasks)
    - `0.4-0.7`: Balanced creativity and coherence (general purpose)
    - `0.8-1.0`: Creative, diverse outputs (ideal for creative writing)
    - `1.0+`: Highly random and experimental

- **Top P**:
    - `0.1-0.5`: Very focused, only most likely tokens
    - `0.6-0.9`: Balanced diversity
    - `0.95-1.0`: Maximum diversity, including less likely tokens

!!! tip "Adjusting Temperature and Top P Together"
    When tuning both parameters simultaneously, consider these combinations:

    - **For deterministic/structured outputs**: Low temperature (`0.0-0.3`) + moderate-to-high top_p (`0.8-0.95`)
        - The low temperature ensures focus, while top_p allows some token diversity
    - **For balanced generation**: Moderate temperature (`0.5-0.7`) + high top_p (`0.9-0.95`)
        - This is a good starting point for most use cases
    - **For creative outputs**: Higher temperature (`0.8-1.0`) + high top_p (`0.95-1.0`)
        - Both parameters work together to maximize diversity

    **Avoid**: Setting both very low (overly restrictive) or adjusting both dramatically at once. When experimenting, adjust one parameter at a time to understand its individual effect.

## Distribution-Based Inference Parameters

For `temperature` and `top_p`, you can specify distributions instead of fixed values. This allows Data Designer to sample different values for each generation request, introducing controlled variability into your synthetic data.

### Uniform Distribution

Samples values uniformly between a low and high bound:

```python
from data_designer.essentials import (
    InferenceParameters,
    UniformDistribution,
    UniformDistributionParams,
)

inference_params = InferenceParameters(
    temperature=UniformDistribution(
        params=UniformDistributionParams(low=0.7, high=1.0)
    ),
)
```

### Manual Distribution

Samples from a discrete set of values with optional weights:

```python
from data_designer.essentials import (
    InferenceParameters,
    ManualDistribution,
    ManualDistributionParams,
)

# Equal probability for each value
inference_params = InferenceParameters(
    temperature=ManualDistribution(
        params=ManualDistributionParams(values=[0.5, 0.7, 0.9])
    ),
)

# Weighted probabilities (normalized automatically)
inference_params = InferenceParameters(
    top_p=ManualDistribution(
        params=ManualDistributionParams(
            values=[0.8, 0.9, 0.95],
            weights=[0.2, 0.5, 0.3]  # 20%, 50%, 30% probability
        )
    ),
)
```

## Examples

### Basic Model Configuration

```python
from data_designer.essentials import InferenceParameters, ModelConfig

# Simple model configuration with fixed parameters
model_config = ModelConfig(
    alias="my-text-model",
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    provider="nvidia",
    inference_parameters=InferenceParameters(
        temperature=0.85,
        top_p=0.95,
        max_tokens=2048,
    ),
)
```

### Multiple Model Configurations for Different Tasks

```python
from data_designer.essentials import InferenceParameters, ModelConfig

model_configs = [
    # Creative tasks
    ModelConfig(
        alias="creative-model",
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        provider="nvidia",
        inference_parameters=InferenceParameters(
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
        inference_parameters=InferenceParameters(
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
        inference_parameters=InferenceParameters(
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
        inference_parameters=InferenceParameters(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
]
```

!!! tip "Experiment with max_tokens for Task-Specific model configurations"
    The number of tokens required to generate a single data entry can vary significantly with use case. For example, reasoning models often need more tokens to "think through" problems before generating a response. Note that `max_tokens` includes **both input and output tokens** (the total context window used), so factor in your prompt length, any context data, and the expected response length when setting this parameter.
### Using Distribution-Based Parameters

```python
from data_designer.essentials import (
    InferenceParameters,
    ManualDistribution,
    ManualDistributionParams,
    ModelConfig,
    UniformDistribution,
    UniformDistributionParams,
)

# Model with variable temperature and top_p
model_config = ModelConfig(
    alias="variable-model",
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    inference_parameters=InferenceParameters(
        # Temperature varies uniformly between 0.7 and 1.0
        temperature=UniformDistribution(
            params=UniformDistributionParams(low=0.7, high=1.0)
        ),
        # Top P samples from discrete values with equal probability
        top_p=ManualDistribution(
            params=ManualDistributionParams(values=[0.85, 0.90, 0.95])
        ),
        max_tokens=2048,
    ),
)
```

## See Also

- **[Model Providers](model-providers.md)**: Learn about configuring model providers
- **[Default Model Settings](default-model-settings.md)**: Pre-configured model settings included with Data Designer
- **[Configure Model Settings With the CLI](configure-model-settings-with-the-cli.md)**: Use the CLI to manage model settings
- **[Column Configurations](../code_reference/column_configs.md)**: Learn how to use models in column configurations
