# Inference Parameters

Inference parameters control how models generate responses during synthetic data generation. Data Designer provides two types of inference parameters: `ChatCompletionInferenceParams` for text/code/structured generation and `EmbeddingInferenceParams` for embedding generation.

## Overview

When you create a `ModelConfig`, you can specify inference parameters to adjust model behavior. These parameters control aspects like randomness (temperature), diversity (top_p), context size (max_tokens), and more. Data Designer supports both static values and dynamic distribution-based sampling for certain parameters.

## Chat Completion Inference Parameters

The `ChatCompletionInferenceParams` class controls how models generate text completions (for text, code, and structured data generation). It provides fine-grained control over generation behavior and supports both static values and dynamic distribution-based sampling.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `temperature` | `float` or `Distribution` | No | Controls randomness in generation (0.0 to 2.0). Higher values = more creative/random |
| `top_p` | `float` or `Distribution` | No | Nucleus sampling parameter (0.0 to 1.0). Controls diversity by filtering low-probability tokens |
| `max_tokens` | `int` | No | Maximum number of tokens to generate in the response (≥ 1) |
| `max_parallel_requests` | `int` | No | Maximum concurrent API requests to this model (default: 4, ≥ 1). See [Concurrency Control](#concurrency-control) below. |
| `timeout` | `int` | No | API request timeout in seconds (≥ 1) |
| `extra_body` | `dict[str, Any]` | No | Additional parameters to include in the API request body |

!!! note "Default Values"
    If `temperature`, `top_p`, or `max_tokens` are not provided, the model provider's default values will be used. Different providers and models may have different defaults.

!!! tip "Controlling Reasoning Effort for GPT-OSS Models"
    For gpt-oss models like `gpt-oss-20b` and `gpt-oss-120b`, you can control the reasoning effort using the `extra_body` parameter:

    ```python
    import data_designer.config as dd

    # High reasoning effort (more thorough, slower)
    inference_parameters = dd.ChatCompletionInferenceParams(
        extra_body={"reasoning_effort": "high"}
    )

    # Medium reasoning effort (balanced)
    inference_parameters = dd.ChatCompletionInferenceParams(
        extra_body={"reasoning_effort": "medium"}
    )

    # Low reasoning effort (faster, less thorough)
    inference_parameters = dd.ChatCompletionInferenceParams(
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

For `temperature` and `top_p` in `ChatCompletionInferenceParams`, you can specify distributions instead of fixed values. This allows Data Designer to sample different values for each generation request, introducing controlled variability into your synthetic data.

### Uniform Distribution

Samples values uniformly between a low and high bound:

```python
import data_designer.config as dd

inference_params = dd.ChatCompletionInferenceParams(
    temperature=dd.UniformDistribution(
        params=dd.UniformDistributionParams(low=0.7, high=1.0)
    ),
)
```

### Manual Distribution

Samples from a discrete set of values with optional weights:

```python
import data_designer.config as dd

# Equal probability for each value
inference_params = dd.ChatCompletionInferenceParams(
    temperature=dd.ManualDistribution(
        params=dd.ManualDistributionParams(values=[0.5, 0.7, 0.9])
    ),
)

# Weighted probabilities (normalized automatically)
inference_params = dd.ChatCompletionInferenceParams(
    top_p=dd.ManualDistribution(
        params=dd.ManualDistributionParams(
            values=[0.8, 0.9, 0.95],
            weights=[0.2, 0.5, 0.3]  # 20%, 50%, 30% probability
        )
    ),
)
```

## Concurrency Control

The `max_parallel_requests` parameter controls how many concurrent API calls Data Designer makes to a specific model. This is a critical performance tuning parameter.

### How It Works

Within each batch of records (controlled by `buffer_size` in `RunConfig`), Data Designer generates columns sequentially but parallelizes cell generation within each column:

```
Batch (1000 records)
├── Column A: 1000 cells processed with max_parallel_requests concurrent LLM calls
├── Column B: 1000 cells processed with max_parallel_requests concurrent LLM calls
└── Column C: 1000 cells processed with max_parallel_requests concurrent LLM calls
```

### Recommended Values

| Inference Backend | Recommended `max_parallel_requests` |
|-------------------|-------------------------------------|
| NVIDIA API Catalog (build.nvidia.com) | 4-8 |
| Self-hosted vLLM (single GPU) | 8-16 |
| Self-hosted vLLM (multi-GPU) | 16-64 |
| OpenAI API | 4-8 (tier-dependent) |
| NeMo Inference Microservices | 16-32 |

### Example

```python
import data_designer.config as dd

# High-throughput configuration for self-hosted vLLM
inference_params = dd.ChatCompletionInferenceParams(
    temperature=0.7,
    max_tokens=2048,
    max_parallel_requests=16,  # Higher for local deployment
    timeout=120,
)
```

!!! warning "Match Your Inference Capacity"
    Setting `max_parallel_requests` too high can cause:

    - Rate limit errors (429) on cloud APIs
    - Request queuing and timeouts on self-hosted servers
    - Out-of-memory errors on GPU inference servers

    Start with conservative values and increase based on monitoring.

For detailed performance optimization, see the [Performance Tuning](../performance-tuning.md) guide.

## Embedding Inference Parameters

The `EmbeddingInferenceParams` class controls how models generate embeddings. This is used when working with embedding models for tasks like semantic search or similarity analysis.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `encoding_format` | `Literal["float", "base64"]` | No | Format of the embedding encoding (default: "float") |
| `dimensions` | `int` | No | Number of dimensions for the embedding |
| `max_parallel_requests` | `int` | No | Maximum concurrent API requests (default: 4, ≥ 1) |
| `timeout` | `int` | No | API request timeout in seconds (≥ 1) |
| `extra_body` | `dict[str, Any]` | No | Additional parameters to include in the API request body |


## See Also

- **[Default Model Settings](default-model-settings.md)**: Pre-configured model settings included with Data Designer
- **[Custom Model Settings](custom-model-settings.md)**: Learn how to create custom providers and model configurations
- **[Model Configurations](model-configs.md)**: Learn about configuring model settings
- **[Model Providers](model-providers.md)**: Learn about configuring model providers
- **[Inference Architecture](../inference-architecture.md)**: Understanding separation of concerns with inference servers
- **[Performance Tuning](../performance-tuning.md)**: Comprehensive guide to optimizing concurrency and batching
