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
│  • Adaptive concurrency (AIMD)      │          │  • Rate limiting (optional)         │
│  • Data validation and quality      │          │                                     │
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
- **Adapts to rate limits** automatically via AIMD concurrency control
- **Handles errors** with retries and early shutdown logic
- **Validates** generated data against schemas and constraints

### What Data Designer Does NOT Do

- **Host models**: You must provide LLM endpoints
- **Manage GPUs**: Your inference server handles GPU allocation
- **Scale inference**: You must provision sufficient capacity
- **Impose rate limits**: Your server or API gateway sets rate limits (Data Designer *reacts* to them automatically)

---

## Execution Model

!!! note "Two execution engines"
    The default execution path is the **async engine**, which dispatches work at the cell level and overlaps independent columns — see [Async Engine](#async-engine) below for its semantics. The legacy **sync engine** is still available for one transitional release via `DATA_DESIGNER_ASYNC_ENGINE=0` and is what this section describes. The configuration knobs documented below (`buffer_size`, `max_parallel_requests`, AIMD throttle config, error handling) apply to both engines; the differences are flagged inline.

The sync engine processes datasets in **batches**, with **parallel** operations within each batch.

### How It Works (sync engine)

**Step 1: Split into batches**

Your dataset is divided into batches of `buffer_size` records. Each batch is processed completely before moving to the next.

**Step 2: Process columns sequentially**

Within a batch, columns are generated one at a time following the dependency graph. The order depends on column dependencies—expression columns may come before LLM columns if the LLM columns depend on them. (The async engine relaxes this: columns whose per-cell dependencies are satisfied can run concurrently with columns earlier in the order.)

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
| **Batching** | Records are split into batches of `buffer_size`. In the sync engine, each batch completes entirely before the next begins; in the async engine, multiple row groups (the async equivalent) can be in flight concurrently. |
| **Sequential columns** | Sync-engine only: columns within a batch are generated one at a time, respecting the dependency graph. The async engine schedules at the cell level instead. |
| **Parallel cells** | Within a column, individual cells (records) are generated in parallel up to the configured limit. Same on both engines. |

### Concurrency Formula

At any moment, the number of concurrent LLM requests is:

```python
concurrent_requests = min(
    buffer_size,                # Records in current batch
    current_throttle_limit,     # AIMD-managed limit (≤ max_parallel_requests)
    remaining_cells_in_column   # Cells left to generate
)
```

`max_parallel_requests` sets the **ceiling**. The actual limit (`current_throttle_limit`) is managed at runtime by an AIMD (Additive Increase / Multiplicative Decrease) controller that reacts to rate-limit signals from the inference server:

- **On the first 429 in a burst**: the limit is reduced by a configurable factor (default: 25% reduction) and a cooldown is applied. Further 429s from already in-flight requests in the same burst do not reduce the limit again — they release their permits and hold the limit steady.
- **After consecutive successes**: the limit increases by 1 (by default) until it reaches the ceiling or a stabilized rate-limit threshold.

This means Data Designer automatically finds the right concurrency level for your server without manual tuning.

!!! note "Engine paths"
    AIMD adaptive concurrency is fully active on the default **async engine**. The legacy **sync engine** is available for one transitional release via `DATA_DESIGNER_ASYNC_ENGINE=0`; on that path 429s are first retried at the HTTP transport layer and AIMD only engages as a fallback. See [Async engine](#async-engine) below.

**Example**: With `buffer_size=100` and `max_parallel_requests=32`, Data Designer starts sending up to 32 requests in parallel. If the server returns 429s, concurrency drops automatically (e.g., to 24, then 18) and recovers once the server catches up.

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

Sets the **maximum** concurrent LLM API calls **per model**. This is the ceiling that the AIMD throttle controller can ramp up to — the actual concurrency at runtime may be lower if the server signals rate limits.

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

**When to increase**: Your inference backend has high throughput capacity, you're using a cloud API with generous rate limits, or you're running vLLM/TensorRT-LLM with multiple GPUs. With AIMD, setting an aggressively high value is safer than before — the system will self-correct downward if the server can't keep up. The salvage queue on the async engine (default) reclaims failed rows; on the sync engine the initial burst of 429s before AIMD stabilizes can drop rows, so start with a more conservative ceiling if you've opted into sync.

**When to decrease**: You want to cap resource usage to a known safe level, or you want more predictable/debuggable execution.

!!! tip "Finding the optimal value"
    The right value depends on your inference stack and model. Self-hosted vLLM servers can often handle values as high as 256, 512, or even 1024 depending on your hardware.

    With AIMD, a practical approach is to set `max_parallel_requests` to the **upper bound** you're comfortable with and let the throttle controller find the sustainable level automatically. If you see frequent 429 → recovery cycles in the logs, your ceiling is above the server's true capacity but the system is handling it. If you never see any throttle activity, you may have room to increase the ceiling further.

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

### Adaptive Throttling (RunConfig)

Data Designer uses an AIMD (Additive Increase / Multiplicative Decrease) controller to automatically adjust concurrency per model based on rate-limit feedback from the inference server. The defaults work well for most workloads. Override them via `ThrottleConfig` only when you understand the trade-offs.

!!! note "Engine paths"
    Adaptive throttling is fully active on the default **async engine**, where 429 responses propagate directly to the AIMD controller. On the legacy **sync engine** (`DATA_DESIGNER_ASYNC_ENGINE=0`), 429s are first retried at the HTTP transport layer; `ThrottleConfig` settings only take effect as a fallback if transport retries are exhausted.

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

run_config = dd.RunConfig(
    throttle=dd.ThrottleConfig(
        reduce_factor=0.75,       # Multiply limit by this on a 429 (default: 0.75)
        additive_increase=1,      # Add this many slots after success_window successes (default: 1)
        success_window=25,        # Consecutive successes before increasing (default: 25)
        cooldown_seconds=2.0,     # Pause after a 429 when no Retry-After header (default: 2.0)
        ceiling_overshoot=0.10,   # Probe 10% above observed server limit (default: 0.10)
    ),
)

designer = DataDesigner()
designer.set_run_config(run_config)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `reduce_factor` | 0.75 | How aggressively to cut concurrency on a 429. Lower = more aggressive. |
| `additive_increase` | 1 | Slots added per recovery step. Higher = faster ramp-up, but riskier. |
| `success_window` | 25 | Consecutive successes required before each increase step. |
| `cooldown_seconds` | 2.0 | Pause duration after a 429 (used when the server doesn't send `Retry-After`). |
| `ceiling_overshoot` | 0.10 | Fraction above the observed rate-limit ceiling the controller is allowed to probe. |

!!! tip "How it works in practice"
    When a model endpoint returns HTTP 429, the controller reduces the concurrency limit for that model and pauses briefly. After enough consecutive successes, it begins ramping back up. If the server rate-limits again, the controller records that level as a ceiling and stabilizes just below it, with a small overshoot band to detect when the server can handle more load.

    You can observe this in the logs — look for messages like `concurrency reduced from X → Y` and `concurrency increased from X → Y`.

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

## Async Engine

The async engine is the default execution path. It dispatches work at the cell level rather than the column level, so independent columns overlap in time and per-(provider, model) AIMD pools tune themselves independently. See the [Async All the Way Down](../devnotes/posts/async-all-the-way-down.md) dev note for the full architecture.

### Per-model timeouts drive every deadline

The `inference_parameters.timeout` field on a `ModelConfig` sets the per-request HTTP timeout. The same value also drives the sync→async bridge that custom columns use when they call `model.generate()`. There is no separate queue-wait deadline — waits scale with provider speed and AIMD's adaptive concurrency. Slow self-hosted endpoints (e.g. large models on a single GPU) only need this one knob raised:

```python
import data_designer.config as dd

config_builder.add_model_config(
    dd.ModelConfig(
        alias="slow-model",
        model="my/slow-model",
        provider="my-provider",
        inference_parameters=dd.ChatCompletionInferenceParams(
            timeout=600,
        ),
    )
)
```

### Run outcomes

A run can finish with fewer records than requested when non-retryable errors drop rows. Inspect `len(result.load_dataset())` to detect.

If the rate of non-retryable errors crosses `RunConfig.shutdown_error_rate`, generation stops early and raises `DataDesignerEarlyShutdownError` (a subclass of `DataDesignerGenerationError`). Catch it separately when a typed retry path is appropriate:

```python
from data_designer.interface.errors import DataDesignerEarlyShutdownError

try:
    result = dd_instance.create(config_builder, num_records=1000)
except DataDesignerEarlyShutdownError:
    # e.g. retry against a different model alias
    ...
```

### Opting out

The legacy sync engine is available as a transitional opt-out:

```bash
DATA_DESIGNER_ASYNC_ENGINE=0 python my_pipeline.py
```

This switch will be removed in a future release.

---

## Common Problems

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Low throughput** | Low GPU utilization | Increase `max_parallel_requests` and/or `buffer_size`. If the throttle has self-reduced due to earlier 429s (check logs for "concurrency reduced" messages), the server may need more capacity or you can wait for AIMD recovery. |
| **Frequent 429 → recovery cycles** | Logs show repeated concurrency drops and ramp-ups | The `max_parallel_requests` ceiling is above the server's sustained capacity. This is handled automatically, but you can lower the ceiling to reduce the sawtooth or tune `reduce_factor` / `success_window`. |
| **Long tail of slow generations** | Most records fast, few very slow | Reduce `max_conversation_restarts`, simplify schemas, improve prompts |
| **Multi-model idle periods** | One model busy, others idle | Reduce `buffer_size` for faster cycling, or consolidate models |
| **Memory errors** | OOM crashes | Reduce `buffer_size` and `max_parallel_requests` |
| **Too many errors** | Generation fails frequently | Check prompts/schemas; adjust `shutdown_error_rate` or disable early shutdown for debugging |

---

## Tuning Workflow

1. **Start with defaults** for initial development — AIMD handles rate-limit adaptation automatically
2. **Profile your workload**: How many LLM columns? How many records? What models?
3. **Identify bottleneck**: Low GPU util → increase `max_parallel_requests` (AIMD will self-correct if you overshoot). Memory issues → decrease `buffer_size`. Long tails → tune retry settings.
4. **Check throttle logs**: Look for "concurrency reduced" / "concurrency increased" messages to understand whether rate limits are the bottleneck
5. **Iterate**: Make one change at a time, measure impact before next change

---

## Related Documentation

- [Deployment Options](deployment-options.md): Choosing between library and microservice
- [Model Configuration](models/model-configs.md): Complete model settings reference
- [Inference Parameters](models/inference-parameters.md): Detailed parameter reference
