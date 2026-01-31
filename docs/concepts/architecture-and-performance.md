# 🏗️ Architecture & Performance

Data Designer is an **orchestration framework** that coordinates synthetic data generation workflows. It is a **client** of LLM inference servers—it does not host models itself.

This guide explains the architecture, execution model, and how to tune performance for your specific use case.

---

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

## Execution Model

Data Designer processes datasets in **batches**, with **parallel** operations within each batch.

```
┌─────── Batch 1 (buffer_size records) ────────────────────────────────────────┐
│                                                                               │
│  Column 1 (Sampler):  ════════►  (non_inference_max_parallel_workers)        │
│  Column 2 (LLM):      ════════════════════════►  (max_parallel_requests)     │
│  Column 3 (LLM):      ══════════════════►  (max_parallel_requests)           │
│  Column 4 (Expression): ══►  (computed from existing columns)                │
│                                                                               │
│  Post-batch processors → Write batch to disk                                 │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────── Batch 2 (buffer_size records) ────────────────────────────────────────┐
│  ... repeat ...                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **Batching**: Records are split into batches of `buffer_size`. Each batch completes entirely before the next begins.
2. **Sequential columns**: Within a batch, columns are generated one at a time (respecting dependencies).
3. **Parallel cells**: Within a column, individual cells (records) are generated in parallel.

### Concurrency Formula

The effective concurrency at any moment is:

```
Concurrent requests = min(
    buffer_size,                    # Records in current batch
    max_parallel_requests,          # Per-model limit
    remaining_cells_in_column       # Cells left to generate
)
```

---

## Configuration Parameters

### `buffer_size` (RunConfig)

Controls how many records are processed per batch.

```python
from data_designer.config import RunConfig

run_config = RunConfig(buffer_size=2000)
```

| Value | Memory Usage | Throughput | Error Feedback |
|-------|--------------|------------|----------------|
| **Low** (100-500) | Lower | May not saturate inference | Fast |
| **Default** (1000) | Moderate | Good for most cases | Moderate |
| **High** (2000-5000) | Higher | Better for deep pipelines | Slower |

**When to increase**: High-capacity inference server, single-model workflows, memory not constrained

**When to decrease**: Memory-constrained environments, development/debugging, complex multi-model pipelines

---

### `max_parallel_requests` (InferenceParams)

Controls concurrent LLM API calls **per model alias**.

```python
from data_designer.config import ModelConfig, ChatCompletionInferenceParams

model = ModelConfig(
    alias="my-model",
    model="meta/llama-3.1-70b-instruct",
    inference_parameters=ChatCompletionInferenceParams(
        max_parallel_requests=8,
    ),
)
```

| Inference Backend | Recommended Value |
|-------------------|-------------------|
| NVIDIA API Catalog | 4-8 |
| Self-hosted vLLM (single GPU) | 8-16 |
| Self-hosted vLLM (multi-GPU) | 16-64 |
| OpenAI API | 4-8 (tier-dependent) |
| NeMo Inference Microservices | 16-32 |

**Key insight**: This is **per model**. If you have two models with `max_parallel_requests=8` each, you could have up to 8 concurrent calls to each model (but columns are processed sequentially, so typically only one model is active at a time).

---

### `non_inference_max_parallel_workers` (RunConfig)

Controls thread pool size for non-LLM operations (samplers, expressions, validators).

```python
run_config = RunConfig(non_inference_max_parallel_workers=8)
```

**Default**: 4

**When to increase**: Many CPU-bound columns (complex expressions, heavy sampling)

---

## Deployment Scenarios

### NVIDIA API Catalog (build.nvidia.com)

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

### Self-Hosted vLLM

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

### Enterprise LLM Gateway

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

### NeMo Inference Microservices (NIMs)

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

## Common Problems & Solutions

### Throughput is too low

**Symptoms**: Low GPU utilization, generation slower than expected

**Solutions**:

```python
# Increase LLM parallelism
ChatCompletionInferenceParams(
    max_parallel_requests=16,  # Up from default 4
)

# Increase batch size
RunConfig(
    buffer_size=2000,  # Up from default 1000
)
```

---

### Long tail of slow generations

**Symptoms**: Most records complete quickly, a few take much longer

**Causes**: Reasoning models with variable output, structured output retries, complex prompts

**Solutions**:

1. **Tune retry settings**:

```python
run_config = RunConfig(
    max_conversation_restarts=3,           # Reduce from default 5
    max_conversation_correction_steps=1,   # Allow 1 in-conversation correction
)
```

2. **Simplify structured output schemas** — flatter schemas parse more reliably

3. **Improve prompts** — clearer prompts = fewer retries

---

### Multi-model pipeline has idle periods

**Symptoms**: One model busy while others idle, GPU utilization varies

**Explanation**: With large `buffer_size` and multiple models, columns are processed sequentially:

```
Time →
┌──────────────────┬──────────────────┬──────────────────┐
│ Column A (fast)  │ Column B (slow)  │ Column C (fast)  │
│ Model 1 active   │ Model 2 active   │ Model 1 active   │
│ Model 2 IDLE     │ Model 1 IDLE     │ Model 2 IDLE     │
└──────────────────┴──────────────────┴──────────────────┘
```

**Solutions**:

1. **Reduce `buffer_size`** — smaller batches cycle through models faster
2. **Accept the trade-off** — total GPU time is the same, just not concurrent
3. **Consolidate models** — use one model for multiple columns if quality permits

---

### Memory errors during generation

**Symptoms**: Out of memory errors, process crashes

**Solutions**:

```python
RunConfig(buffer_size=500)  # Lower memory footprint

ChatCompletionInferenceParams(
    max_parallel_requests=4,  # Lower GPU memory usage (if local inference)
)
```

---

## Error Handling

### Early Shutdown

Data Designer monitors error rates and can shut down early if too many failures occur:

```python
run_config = RunConfig(
    disable_early_shutdown=False,  # Enable early shutdown (default)
    shutdown_error_rate=0.5,       # Shut down if >50% errors
    shutdown_error_window=10,      # After at least 10 tasks complete
)
```

**When to disable**: Debugging (want to see all errors), noisy endpoints with transient errors

### Retry Configuration

For LLM generation with parsing (structured outputs, code extraction):

```python
run_config = RunConfig(
    max_conversation_restarts=5,           # Full conversation restarts (default: 5)
    max_conversation_correction_steps=0,   # In-conversation corrections (default: 0)
)
```

| Scenario | `max_conversation_restarts` | `max_conversation_correction_steps` |
|----------|-----------------------------|------------------------------------|
| Development/debugging | 3 | 2 |
| Production (well-tuned prompts) | 5 | 0 |
| Strict schemas | 7 | 2 |
| Simple text generation | 3 | 0 |

---

## Quick Reference

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `buffer_size` | `RunConfig` | 1000 | Records per batch |
| `max_parallel_requests` | `ChatCompletionInferenceParams` | 4 | Concurrent LLM calls per model |
| `non_inference_max_parallel_workers` | `RunConfig` | 4 | Threads for non-LLM work |
| `shutdown_error_rate` | `RunConfig` | 0.5 | Error rate threshold for early shutdown |
| `shutdown_error_window` | `RunConfig` | 10 | Minimum tasks before error monitoring |
| `max_conversation_restarts` | `RunConfig` | 5 | Full retry attempts |
| `max_conversation_correction_steps` | `RunConfig` | 0 | In-conversation corrections |
| `disable_early_shutdown` | `RunConfig` | False | Disable early shutdown on errors |

---

## Tuning Workflow

1. **Start with defaults** for initial development
2. **Profile your workload**: How many LLM columns? How many records? What models?
3. **Identify bottleneck**: Low GPU util → increase `max_parallel_requests`. Memory issues → decrease `buffer_size`. Long tails → tune retry settings.
4. **Iterate**: Make one change at a time, measure impact before next change

---

## Related Documentation

- [Deployment Options](deployment-options.md): Choosing between library and microservice
- [Model Configuration](models/model-configs.md): Complete model settings reference
- [Inference Parameters](models/inference-parameters.md): Detailed parameter reference
