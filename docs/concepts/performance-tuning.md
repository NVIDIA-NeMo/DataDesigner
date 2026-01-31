# Performance and Concurrency Tuning

This guide explains how Data Designer executes generation workflows and how to tune performance for your specific use case.

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

**When to increase**:

- Your inference server has high capacity
- Single-model workflows (no multi-model idle time)
- Memory is not a constraint

**When to decrease**:

- Memory-constrained environments
- Development/debugging (faster feedback)
- Complex multi-model pipelines (reduces idle time)

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

**When to decrease**: Resource-constrained environments

---

## Concurrency Formula

The effective concurrency at any moment is:

```
Concurrent requests = min(
    buffer_size,                    # Records in current batch
    max_parallel_requests,          # Per-model limit
    remaining_cells_in_column       # Cells left to generate
)
```

**Example**:

- `buffer_size=1000`
- `max_parallel_requests=8`
- Generating column with 1000 cells

→ 8 concurrent LLM calls, processing 1000 cells in ~125 rounds

---

## Common Scenarios and Solutions

### Scenario: Throughput is too low

**Symptoms**:

- Low GPU utilization on inference server
- Generation is slower than expected

**Diagnosis**:

1. Check if `max_parallel_requests` is too low
2. Check if `buffer_size` is too small
3. Verify inference server isn't the bottleneck

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

### Scenario: Long tail of slow generations

**Symptoms**:

- Most records complete quickly
- A few records take much longer
- Total time dominated by slowest records

**Causes**:

- Reasoning models with variable output length
- Structured output retries on schema failures
- Complex prompts causing model "thinking"

**Solutions**:

1. **Tune retry settings**:

```python
run_config = RunConfig(
    max_conversation_restarts=3,           # Reduce from default 5
    max_conversation_correction_steps=1,   # Allow 1 in-conversation correction
)
```

2. **Simplify structured output schemas**:

```python
# Instead of deeply nested schemas
class ComplexOutput(BaseModel):
    nested: NestedType
    another: AnotherNestedType
    # ... many fields

# Consider flatter schemas
class SimpleOutput(BaseModel):
    field1: str
    field2: int
    field3: list[str]
```

3. **Improve prompts**: Clearer prompts = fewer retries

---

### Scenario: Multi-model pipeline has idle periods

**Symptoms**:

- One model is busy while others are idle
- GPU utilization varies over time

**Explanation**:

With large `buffer_size` and multiple models, columns are processed sequentially:

```
Time →
┌──────────────────┬──────────────────┬──────────────────┐
│ Column A (fast)  │ Column B (slow)  │ Column C (fast)  │
│ Model 1 active   │ Model 2 active   │ Model 1 active   │
│ Model 2 IDLE     │ Model 1 IDLE     │ Model 2 IDLE     │
└──────────────────┴──────────────────┴──────────────────┘
```

**Solutions**:

1. **Reduce `buffer_size`**: Smaller batches cycle through models faster

```python
RunConfig(buffer_size=500)  # Faster cycling
```

2. **Accept the trade-off**: This is inherent to column-wise generation. The total GPU time is the same; it's just not concurrent.

3. **Consolidate models**: Use one model for multiple columns if quality permits

---

### Scenario: Memory errors during generation

**Symptoms**:

- Out of memory errors
- Process crashes with large datasets

**Solutions**:

1. **Reduce `buffer_size`**:

```python
RunConfig(buffer_size=500)  # Lower memory footprint
```

2. **Reduce `max_parallel_requests`** (if using local inference):

```python
ChatCompletionInferenceParams(
    max_parallel_requests=4,  # Lower GPU memory usage
)
```

---

## Error Handling Configuration

### Early Shutdown

Data Designer monitors error rates and can shut down early if too many failures occur:

```python
run_config = RunConfig(
    disable_early_shutdown=False,  # Enable early shutdown (default)
    shutdown_error_rate=0.5,       # Shut down if >50% errors
    shutdown_error_window=10,      # After at least 10 tasks complete
)
```

**When to disable early shutdown**:

- Debugging: You want to see all errors, not just the first batch
- Noisy endpoints: Transient errors that you expect to recover

```python
run_config = RunConfig(
    disable_early_shutdown=True,  # Continue despite errors
)
```

---

### Retry Configuration

For LLM generation with parsing (structured outputs, code extraction):

```python
run_config = RunConfig(
    max_conversation_restarts=5,           # Full conversation restarts (default: 5)
    max_conversation_correction_steps=0,   # In-conversation corrections (default: 0)
)
```

**`max_conversation_restarts`**: Complete restart with fresh conversation. Use when the model goes off-track.

**`max_conversation_correction_steps`**: Feed error back to model within same conversation. Useful for schema validation errors.

**Recommendations**:

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
2. **Profile your workload**:
   - How many LLM columns?
   - How many records?
   - What models (size, speed)?
3. **Identify bottleneck**:
   - Low GPU util → increase `max_parallel_requests`
   - Memory issues → decrease `buffer_size`
   - Long tails → tune retry settings
4. **Iterate**:
   - Make one change at a time
   - Measure impact before next change

---

## Related Documentation

- [Inference Architecture](inference-architecture.md): Understanding separation of concerns
- [Model Configuration](models/model-configs.md): Complete model settings reference
- [RunConfig Reference](../code_reference/run_config.md): API documentation
