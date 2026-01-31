# Inference Architecture

Data Designer is an **orchestration framework** that coordinates synthetic data generation workflows. It is a **client** of LLM inference servers—it does not host models itself.

This guide explains the separation of concerns and how to configure Data Designer for optimal performance with your inference infrastructure.

## Separation of Concerns

```
┌─────────────────────────────────────┐          ┌─────────────────────────────────────┐
│         Data Designer               │          │       Inference Server(s)           │
│         (Orchestration)             │  HTTP    │       (LLM Hosting)                 │
│                                     │  ─────►  │                                     │
│  • Dataset workflow management      │          │  • Model weights and execution      │
│  • Column dependency resolution     │          │  • GPU allocation and scheduling    │
│  • Batching and parallelism         │          │  • Request queuing                  │
│  • Retry and error handling         │          │  • Token generation                 │
│  • Data validation and quality      │          │  • Rate limiting (optional)         │
└─────────────────────────────────────┘          └─────────────────────────────────────┘
              ▲                                                    ▲
              │                                                    │
        Your workflow                                    Your infrastructure
         configuration                                    (or cloud API)
```

### What Data Designer Does

- **Orchestrates** the generation workflow across multiple columns
- **Resolves dependencies** between columns (DAG-based execution)
- **Batches** work into manageable chunks (`buffer_size`)
- **Parallelizes** LLM calls within batches (`max_parallel_requests`)
- **Handles errors** with retries and early shutdown logic
- **Validates** generated data against schemas and constraints

### What Data Designer Does NOT Do

- **Host models**: You must provide LLM endpoints
- **Manage GPUs**: Your inference server handles GPU allocation
- **Scale inference**: You must provision sufficient capacity
- **Rate limit**: Your server or API gateway handles this

---

## Your Responsibilities

### Inference Server Side

You are responsible for ensuring your inference infrastructure can handle Data Designer's workload:

| Responsibility | Description |
|----------------|-------------|
| **Provision capacity** | Ensure enough GPUs/replicas for your expected throughput |
| **Configure queuing** | Set up request queues to handle burst traffic |
| **Monitor utilization** | Watch GPU utilization and scale as needed |
| **Set timeouts** | Configure server timeouts that exceed Data Designer's timeout settings |
| **Handle rate limits** | Implement rate limiting if using shared infrastructure |

### Data Designer Side

Configure Data Designer to match your inference capacity:

| Configuration | Purpose |
|---------------|---------|
| `max_parallel_requests` | Match your server's concurrent request capacity |
| `buffer_size` | Balance throughput vs. memory usage |
| `timeout` | Set below your server's timeout to get useful errors |

---

## Common Deployment Scenarios

### Scenario 1: NVIDIA API Catalog (build.nvidia.com)

Cloud-hosted endpoints with rate limits.

```python
from data_designer.config import ModelConfig, ChatCompletionInferenceParams

model = ModelConfig(
    alias="nim-llama",
    model="meta/llama-3.1-70b-instruct",
    provider="nvidia",
    inference_parameters=ChatCompletionInferenceParams(
        max_parallel_requests=4,  # Respect rate limits
        timeout=60,
    ),
)
```

**Tips**:

- Start with `max_parallel_requests=4` and increase based on your tier
- Monitor for rate limit errors (429) and adjust accordingly
- Consider multiple model aliases to distribute load across endpoints

### Scenario 2: Self-Hosted vLLM

Single or multi-GPU deployment you control.

```python
model = ModelConfig(
    alias="local-llm",
    model="meta-llama/Llama-3.1-8B-Instruct",
    provider="openai",  # vLLM is OpenAI-compatible
    base_url="http://localhost:8000/v1",
    inference_parameters=ChatCompletionInferenceParams(
        max_parallel_requests=16,  # Higher for local deployment
        timeout=120,
    ),
)
```

**Tips**:

- Set `max_parallel_requests` based on vLLM's `--max-num-seqs` setting
- Monitor GPU utilization to find optimal parallelism
- For multi-GPU, scale `max_parallel_requests` proportionally

### Scenario 3: Enterprise LLM Gateway

Centralized gateway with RBAC and rate limiting.

```python
model = ModelConfig(
    alias="enterprise-llm",
    model="gpt-4-turbo",
    provider="openai",
    base_url="https://llm-gateway.company.com/v1",
    api_key_env_var="GATEWAY_API_KEY",
    inference_parameters=ChatCompletionInferenceParams(
        max_parallel_requests=8,  # Based on your quota
        timeout=90,
    ),
)
```

**Tips**:

- Check your quota/rate limit allocation
- Work with your platform team to understand capacity
- Consider dedicated capacity for large generation jobs

### Scenario 4: NeMo Inference Microservices (NIMs)

Kubernetes-deployed NIMs with auto-scaling.

```python
model = ModelConfig(
    alias="nim-endpoint",
    model="meta/llama-3.1-70b-instruct",
    provider="nvidia",
    base_url="http://nim-service.namespace.svc:8000/v1",
    inference_parameters=ChatCompletionInferenceParams(
        max_parallel_requests=32,  # NIMs can handle high parallelism
        timeout=120,
    ),
)
```

**Tips**:

- NIMs auto-scale based on load; higher `max_parallel_requests` is fine
- Monitor NIM pod scaling to ensure adequate capacity
- Consider pre-warming NIMs before large jobs

---

## Scaling Considerations

### Matching Capacity

Your inference capacity should match your Data Designer configuration:

| Workload | Inference Needs | Data Designer Config |
|----------|-----------------|----------------------|
| **Small** (< 1K records) | Single GPU, low concurrency | Default settings |
| **Medium** (1K-100K records) | Multi-GPU or multiple replicas | Tune `max_parallel_requests` to 8-16 |
| **Large** (> 100K records) | Horizontally scaled cluster | Optimize both `buffer_size` and `max_parallel_requests` |

### Multi-Model Workflows

When using multiple models (e.g., different models for different columns):

```python
# Each model has independent concurrency
fast_model = ModelConfig(
    alias="fast",
    model="meta/llama-3.1-8b-instruct",
    inference_parameters=ChatCompletionInferenceParams(
        max_parallel_requests=16,  # Small model, high parallelism
    ),
)

quality_model = ModelConfig(
    alias="quality",
    model="meta/llama-3.1-70b-instruct",
    inference_parameters=ChatCompletionInferenceParams(
        max_parallel_requests=4,  # Large model, lower parallelism
    ),
)
```

**Note**: Columns are processed sequentially. When Column A (using `fast`) completes, Column B (using `quality`) begins. The `quality` endpoint is idle while `fast` is processing.

---

## Troubleshooting

### "Request timed out"

**Cause**: Data Designer's timeout is shorter than the server's generation time.

**Fix**: Increase `timeout` in inference parameters:

```python
ChatCompletionInferenceParams(
    timeout=180,  # Increase for long generations
)
```

### "Rate limit exceeded" (429 errors)

**Cause**: `max_parallel_requests` exceeds your API rate limit.

**Fix**: Reduce `max_parallel_requests`:

```python
ChatCompletionInferenceParams(
    max_parallel_requests=2,  # Lower to stay within limits
)
```

### Low GPU utilization

**Cause**: Not enough concurrent requests to saturate the GPU.

**Fix**: Increase `max_parallel_requests` and/or `buffer_size`:

```python
# Increase parallelism
ChatCompletionInferenceParams(
    max_parallel_requests=16,
)

# And/or increase batch size
RunConfig(
    buffer_size=2000,
)
```

### Memory errors on inference server

**Cause**: Too many concurrent requests for available GPU memory.

**Fix**: Reduce `max_parallel_requests`:

```python
ChatCompletionInferenceParams(
    max_parallel_requests=4,  # Lower to reduce memory pressure
)
```

---

## Next Steps

- [Performance Tuning](performance-tuning.md): Detailed guidance on optimizing concurrency and batching
- [Model Configuration](models/model-configs.md): Complete reference for model settings
