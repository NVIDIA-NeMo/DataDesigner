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

!!! note "Column-Wise Generator"
    This describes Data Designer's current **column-wise dataset generator**. Other dataset generation strategies are in development.

Data Designer processes datasets in **batches**, with **parallel** operations within each batch.

### How It Works

**Step 1: Split into batches**

Your dataset is divided into batches of `buffer_size` records. Each batch is processed completely before moving to the next.

**Step 2: Process columns sequentially**

Within a batch, columns are generated one at a time following the dependency graph. The order depends on column dependencies—expression columns may come before LLM columns if the LLM columns depend on them.

Example workflow:

```
Batch 1 (100 records)
│
├─► Column 1: category (Sampler)      ──── All 100 values generated
├─► Column 2: prompt (LLM Text)       ──── All 100 values generated
├─► Column 3: response (LLM Text)     ──── All 100 values generated
├─► Column 4: score (Expression)      ──── All 100 values computed
│
└─► Write batch to disk
    │
    ▼
Batch 2 (100 records)
    ...repeat...
```

**Step 3: Generate cells in parallel**

Within each column, cells are processed **in parallel** up to the configured limit:

| Column Type | Parallelism Control |
|-------------|---------------------|
| Sampler | `non_inference_max_parallel_workers` |
| LLM (Text, Code, Structured, Judge) | `max_parallel_requests` |
| Expression | Sequential (fast, CPU-bound) |

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Batching** | Records are split into batches of `buffer_size`. Each batch completes entirely before the next begins. |
| **Sequential columns** | Within a batch, columns are generated one at a time, respecting the dependency graph. |
| **Parallel cells** | Within a column, individual cells (records) are generated in parallel up to the configured limit. |

### Concurrency Formula

At any moment, the number of concurrent LLM requests is:

```python
concurrent_requests = min(
    buffer_size,                # Records in current batch
    max_parallel_requests,      # Per-model limit
    remaining_cells_in_column   # Cells left to generate
)
```

**Example**: With `buffer_size=100` and `max_parallel_requests=8`, Data Designer sends up to 8 LLM requests at a time until all 100 cells in the column are complete.

---

## Configuration Parameters

### `buffer_size` (RunConfig)

Controls how many records are processed per batch.

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

run_config = dd.RunConfig(buffer_size=2000)

designer = DataDesigner()
designer.set_run_config(run_config)
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
import data_designer.config as dd

model = dd.ModelConfig(
    alias="my-model",
    model="nvidia/nemotron-3-nano-30b-a3b",
    inference_parameters=dd.ChatCompletionInferenceParams(
        max_parallel_requests=8,
    ),
)
```

**Default**: 4

**When to increase**: Your inference backend has high throughput capacity, you're using a cloud API with generous rate limits, or you're running vLLM/TensorRT-LLM with multiple GPUs

**When to decrease**: You're hitting rate limits or 429 errors, the inference server is overloaded, or you want more predictable/debuggable execution

!!! tip "Finding the optimal value"
    The right value depends on your inference stack and model. Self-hosted vLLM servers can often handle values as high as 256, 512, or even 1024 depending on your hardware.

    **Benchmark approach**: Run a small dataset (e.g., 100 records) with increasing `max_parallel_requests` values (4 → 8 → 16 → 32 → ...) and measure generation time. Stop increasing when the runtime stops decreasing—that's when your inference server is saturated.

---

### `non_inference_max_parallel_workers` (RunConfig)

Controls thread pool size for non-LLM operations (samplers, expressions, validators).

```python
run_config = dd.RunConfig(non_inference_max_parallel_workers=8)
designer.set_run_config(run_config)
```

**Default**: 4

**When to increase**: Many CPU-bound columns (complex expressions, heavy sampling)

---

### Error Handling (RunConfig)

Control retry behavior and early shutdown for failed generations.

```python
run_config = dd.RunConfig(
    max_conversation_restarts=5,           # Full conversation restarts (default: 5)
    max_conversation_correction_steps=0,   # In-conversation corrections (default: 0)
    disable_early_shutdown=False,          # Enable early shutdown (default)
    shutdown_error_rate=0.5,               # Shut down if >50% errors
    shutdown_error_window=10,              # Min tasks before error monitoring
)
designer.set_run_config(run_config)
```

**When to adjust**:

- **Strict schemas**: Increase `max_conversation_restarts` to 7, add `max_conversation_correction_steps=2`
- **Debugging**: Set `disable_early_shutdown=True` to see all errors
- **Simple text**: Reduce `max_conversation_restarts` to 3

---

## Common Problems

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Low throughput** | Low GPU utilization | Increase `max_parallel_requests` and/or `buffer_size` |
| **Long tail of slow generations** | Most records fast, few very slow | Reduce `max_conversation_restarts`, simplify schemas, improve prompts |
| **Multi-model idle periods** | One model busy, others idle | Reduce `buffer_size` for faster cycling, or consolidate models |
| **Memory errors** | OOM crashes | Reduce `buffer_size` and `max_parallel_requests` |
| **Too many errors** | Generation fails frequently | Check prompts/schemas; adjust `shutdown_error_rate` or disable early shutdown for debugging |

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
