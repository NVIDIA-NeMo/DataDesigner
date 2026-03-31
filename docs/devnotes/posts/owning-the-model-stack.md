---
date: 2026-03-25
authors:
  - nmulepati
---

# **Owning the Model Stack: Adaptive Concurrency FTW!**

Picture this: you're generating a million-record dataset. Thirty two concurrent requests per model, three models in the pipeline, two providers. Everything hums along for the first ten minutes — then one provider starts returning 429s, your retry logic kicks in, and suddenly you're in a feedback loop where retries *cause* more 429s. The run stalls. You restart with lower concurrency, waste throughput for hours, and wonder if there's a better way.

There is. This post is about the native model client layer we built with adaptive throttling (a system that discovers provider capacity at runtime) replacing our dependency on LiteLLM along the way.

<!-- more -->

![From chaotic request flow to calibrated concurrency via adaptive throttling](assets/owning-the-model-stack/native-model-client-hero.png)

## **Why We Made the Move**

LiteLLM gave us a fast path to multi-provider support early on: "just call any model" without writing HTTP adapters from scratch. As Data Designer's workloads scaled to millions of records across multiple models and providers, we wanted more control over what happens between our orchestrator and the provider API.

The biggest opportunity was **adaptive concurrency**. When you start hitting rate limits at scale, you don't want to just retry. You want the system to *learn* the provider's actual capacity and adjust on the fly. That adaptation needs to be aware of your pipeline's topology. Which models share an endpoint? Which routes share a rate-limit budget? Building that required owning the transport layer.

We also saw a chance to simplify. We were only using a slice of what LiteLLM provides. A purpose-built stack meant less surface area, faster startup, and a transport lifecycle we could reason about end to end.

So we built a native client layer. Thin HTTP adapters with adaptive rate-limit handling, deterministic retry policy, and canonical error normalization. The rest of this post walks through how it works.

## **Architecture: The Native Client Layer**

The replacement is a layered stack where each layer does one thing. `ModelFacade`, the public orchestration surface that column generators call, didn't change at all. Everything below it is new.

![Native model client architecture: six layers from ModelFacade down to provider HTTP APIs](assets/owning-the-model-stack/native-model-client-layers.png){ style="max-width:75%; height:auto" }

From top to bottom:

1. **ModelFacade**: orchestrates correction loops, MCP tool-calling, and usage tracking. This is the public API. Column generators talk to this layer, and it was untouched during the migration. If you've written a Data Designer pipeline, nothing about your code changes.

2. **ThrottledModelClient**: the new layer. It's a decorator around `HttpModelClient` — same `ModelClient` protocol, but every outbound call is wrapped with a throttle permit: acquire a concurrency slot before the call, release it after, and feed the outcome (success, 429, or error) back to `ThrottleManager`. This is where adaptive throttling lives.

3. **ThrottleManager**: the Additive Increase / Multiplicative Decrease (AIMD) controller that `ThrottledModelClient` delegates to. A single instance is created at pipeline startup and shared across all model clients. It owns all the mutable concurrency state — per-domain AIMD counters, global caps, cascade dampening, and cooldown timers.

4. **HttpModelClient**: an abstract base class that defines the interface for all provider adapters. It owns the shared `httpx` transport lifecycle — connection pooling, timeouts, and transport-level retries for transient failures (502, 503, 504). Boring but important.

5. **Provider Adapters**: `OpenAICompatibleClient` and `AnthropicClient`, both extending `HttpModelClient`. Each adapter translates between our canonical request/response types and the provider's wire format. Provider-specific shapes are contained here and never leak upward.

6. **Provider HTTP APIs**: the actual endpoints (OpenAI, NVIDIA NIM, vLLM, Anthropic Messages API).

The boundary between `ModelFacade` and the client layer is defined by canonical types. `ChatCompletionRequest`, `ChatCompletionResponse`, `EmbeddingRequest`, `EmbeddingResponse`, `ImageGenerationRequest`, `ImageGenerationResponse`, and `ProviderError`. These are plain dataclasses. No provider SDK objects cross this line. A `ModelClient` protocol defines the contract that all adapters implement, and that's the only interface the rest of the system sees.

## **Adaptive Throttling: The Centerpiece**

With this client stack in place, we had the foundation to build something that wasn't possible before. Adaptive concurrency control. Let's start with the problem.

### **The guessing game**

When you're calling LLM APIs at scale, you need to pick a concurrency level: how many requests to keep in flight at once. Providers publish RPM and TPM limits, but the actual capacity you can sustain depends on factors they don't tell you (current load, your prompt lengths, what other tenants are doing). You could run benchmarking passes to get a better estimate, but that's time-consuming, costs real tokens, and the answer can shift between runs anyway. Set concurrency too high and you trigger 429 storms that cascade through your pipeline. Set it too low and you leave throughput on the table for hours.

What you actually want is a system that *discovers* the provider's capacity at runtime and adjusts automatically. That's what AIMD does.

### **AIMD: Additive Increase / Multiplicative Decrease**

If you've studied networking, this will sound familiar. [AIMD](https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease) is the algorithm behind TCP congestion control. We apply the same idea to LLM API concurrency:

- **On success**: after a window of consecutive successful requests (default: 25), increase the concurrency limit by 1. Slow, cautious growth.
- **On 429**: multiply the current limit by a reduce factor (default: 0.75, a 25% cut). Fast, decisive pullback. Then apply a cooldown using the provider's `Retry-After` header when available, or a default of 2 seconds.

The asymmetry is deliberate. You probe upward slowly because overshooting wastes requests. You pull back quickly because staying above the limit wastes *everything* because every request in the burst gets rejected. This is the same insight that makes TCP work: be optimistic cautiously, be pessimistic decisively.

The result is that the system converges on the provider's actual capacity without you setting it. It starts at your configured `max_parallel_requests`, discovers the real limit through 429 signals, and settles into a steady state that tracks the provider's capacity as it changes.

![AIMD concurrency control over time: initial phase, 429 drop, recovery, ceiling stabilization, steady state](assets/owning-the-model-stack/aimd-concurrency-over-time.png){ style="max-width:75%; height:auto" }

This is especially useful when you're self-hosting your inference stack (running vLLM or NVIDIA NIM on your own hardware) as long as the serving framework returns 429s when it's at capacity. The capacity of a self-hosted endpoint depends on your GPU count, model size, quantization, batch settings, and whatever else is sharing the cluster. That capacity might change between runs, or even mid-run if other workloads spin up. If your serving layer signals overload with 429s, you don't need to figure any of that out. Point Data Designer at your endpoint, set `max_parallel_requests` to a generous upper bound, and the system self-adjusts to whatever your infrastructure can actually handle.

### **Ceiling stabilization**

Classic AIMD has a well-known problem, the sawtooth. After a 429 drops the limit, additive increase climbs all the way back to the configured max, hits another 429, drops again, and repeats. Every climb wastes requests, and the 429 bursts are predictable.

We dampen it with **ceiling stabilization**. After the first 429, the system records the pre-decrease limit as a `rate_limit_ceiling`. Subsequent additive increases don't climb all the way back to `max_parallel_requests` — they stop at `ceiling * (1 + ceiling_overshoot)` (by default 10% above the observed limit). This lets the system probe gently above what it knows works — the 10% overshoot band — without repeatedly slamming into the wall. If the probe succeeds (no 429), the limit keeps rising within the overshoot band while the ceiling stays put. If a 429 fires at or below the existing ceiling, the ceiling is updated downward to the lower observed limit via `min(existing_ceiling, prev_limit)`, tightening the band over time. The result is that oscillations shrink and the system converges on a tight band around the provider's real capacity.

### **Cascade dampening**

Here's a subtlety that bit us during testing. When the system is running at capacity and 429s start coming back, it's not just one request that fails. Multiple in-flight requests hit the rate limit at the same time. Without dampening, each 429 triggers its own multiplicative decrease. If you have 5 concurrent 429s and each one cuts the limit by 25%, you've collapsed from 20 to 4 in a single burst. That's way too aggressive.

**Cascade dampening** fixes this. Only the first 429 in a burst triggers a decrease. Subsequent 429s in the same cascade are counted (for observability) but don't further reduce the limit. The cascade resets on the next successful request. Simple, but it makes the difference between a graceful pullback and a collapse.

### **Two-level keying**

Real pipelines aren't simple. A single provider+model combination might serve chat completions, embeddings, and image generation, potentially on different rate-limit budgets. And multiple [model aliases](../../concepts/models/model-configs.md) in your pipeline might point to the same underlying provider and model (say, one alias for generation and another for judging, both hitting the same NVIDIA endpoint).

The throttle manager handles this with two-level keying:

![Two-level throttle keying: global cap per provider+model, independent domain states for chat, embedding, image](assets/owning-the-model-stack/throttle-keying.png){ style="max-width:75%; height:auto" }

- **Global cap**: keyed by `(provider_name, model_id)`. When multiple model aliases target the same provider and model, the effective max is `min()` of their configured `max_parallel_requests`. This enforces the most conservative limit for shared upstream capacity, because the provider doesn't care what you *call* the model, it sees the same API key.

- **Domain state**: keyed by `(provider_name, model_id, throttle_domain)`. Each domain (`chat`, `embedding`, `image`, `healthcheck`) maintains its own AIMD state: `current_limit`, `in_flight`, `blocked_until`, `success_streak`, and `rate_limit_ceiling`. Domains float independently but are always capped by the global max.

The practical effect is that a burst of 429s on the chat route doesn't starve embedding requests, and vice versa. Each route adapts to its own capacity independently while respecting the shared upstream limit.

## **The Retry Boundary**

There's a design choice here that isn't obvious until you think about it, and getting it wrong would break the entire throttling system.

The transport layer (via `httpx` with `RetryTransport`) handles transient server failures like 502, 503, 504, and connection errors. These are hiccups. The server is temporarily broken. Retry with exponential backoff and jitter, and move on.

But **429 is explicitly excluded from transport retries**.

![Retry boundary: 502/503/504 retried at transport, 429 passed through to ThrottledModelClient for AIMD feedback](assets/owning-the-model-stack/retry-boundary.png){ style="max-width:75%; height:auto" }

Why? Because if the retry layer swallows 429s, the throttle manager never learns the provider is overloaded. The whole AIMD feedback loop depends on seeing raw rate-limit signals. A 429 must bubble up to `ThrottledModelClient` so it can call `release_rate_limited()`, cut the concurrency limit, apply the cooldown, and record the ceiling. The next attempt then re-enters the throttle acquire path, waiting for a permit, before making another HTTP call.

The split is clean and worth remembering. Transport retries handle *server problems*. Throttle adaptation handles *capacity problems*. The provider is working fine, you're just sending too many requests. Conflating the two is how you get retry storms.

One caveat: this boundary behaves differently depending on the execution mode. In async mode (currently experimental, enabled with `DATA_DESIGNER_ASYNC_ENGINE=1`), 429s bypass transport retries entirely and flow straight to `ThrottledModelClient` for AIMD feedback — this is the full adaptive loop described above. In sync mode, 429s are retried at the transport layer since there's no salvage queue to re-attempt failed rows. AIMD is still wired up but only fires if all transport retries are exhausted. This is temporary — once the async engine graduates from experimental, it will become the default path and the sync codepath will be retired. Stay tuned for a dedicated dev note on the async engine.

## **Configuration**

The throttle system is designed to work well out of the box. The defaults are conservative and handle most workloads without tuning. The primary user-facing knob is still `max_parallel_requests` on your model's inference parameters, which sets the hard upper bound for concurrency. AIMD floats below it.

For workloads where you want to fine-tune the adaptation behavior, `ThrottleConfig` is available on `RunConfig`:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
data_designer.set_run_config(
    dd.RunConfig(
        throttle=dd.ThrottleConfig(
            reduce_factor=0.75,
            success_window=25,
            cooldown_seconds=2.0,
            ceiling_overshoot=0.10,
        )
    )
)
config_builder = dd.DataDesignerConfigBuilder(
    model_configs=[
        dd.ModelConfig(
            alias="reasoning-model",
            model="nvidia/nemotron-3-super-120b-a12b",
            provider="nvidia",
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_parallel_requests=32,
            ),
        ),
    ],
)

# ... add columns to config_builder ...

create_result = data_designer.create(
    config_builder,
    num_records=10_000,
)
```

| Parameter | Default | What it does |
|---|---|---|
| `reduce_factor` | 0.75 | Multiplicative decrease on 429 (0.75 = reduce by 25%) |
| `additive_increase` | 1 | How much to increase the limit after a success window |
| `success_window` | 25 | Consecutive successes before additive increase |
| `cooldown_seconds` | 2.0 | Default cooldown when no `Retry-After` header |
| `ceiling_overshoot` | 0.10 | How far above the observed ceiling to probe (10%) |

In practice, the parameter most worth adjusting is `success_window`. A smaller window (say, 10) makes the system more aggressive about reclaiming throughput after a pullback, useful when you know the provider's capacity fluctuates quickly. A larger window (say, 50) makes it more conservative, better for providers with strict, stable rate limits where you'd rather not probe at all.

Most users will never need to touch any of these. The system adapts automatically.

## **What It Looks Like in the Logs**

`ThrottleManager` logs every state transition at `INFO` level, so the adaptation story is visible in your terminal as the run progresses.

```
# When the system hits a 429 and cuts concurrency:
🪫📉 'nvidia/nemotron-3-super-120b-a12b' [chat] server rate-limited — concurrency reduced from 20 → 15 (retrying in 2s)

# If the provider's capacity is lower than a previously observed ceiling, the log includes the estimated server limit:
🪫📉 'nvidia/nemotron-3-super-120b-a12b' [chat] server rate-limited at 15 (server limit ~12) — concurrency reduced to 11 (retrying in 2s)

# As successes accumulate and the limit climbs back:
🪫📈🔥 'nvidia/nemotron-3-super-120b-a12b' [chat] concurrency increased from 11 → 12

# When the limit reaches the ceiling band:
🔋✅ 'nvidia/nemotron-3-super-120b-a12b' [chat] concurrency recovered to 13 parallel requests

# And if no 429s have been observed and the limit reaches the configured max:
🔋✅ 'nvidia/nemotron-3-super-120b-a12b' [chat] concurrency fully recovered (20 parallel requests)
```

Reading these lines in sequence tells you exactly what happened: where the system started, when it hit the wall, how far it pulled back, and how it recovered. No guessing, no metrics pipeline required.

## **Where This Leaves Us**

This shipped in Data Designer v0.5.4. If you're using Data Designer today, nothing changes in your pipeline code. `ModelFacade` is the same API it's always been. What changes is what happens underneath. The system now discovers provider capacity at runtime, isolates throttle state per route, and separates retry logic from rate-limit adaptation. Adaptive throttling is enabled by default for all providers. You don't opt in or configure anything; it just starts learning. If you want to see this fully in action, turn on async mode — it's experimental today, but soon to be stable.

For most workloads, the defaults are all you need. Set `max_parallel_requests` to a generous upper bound and let AIMD find the right level. If you're running against a self-hosted stack that returns 429s, the system adapts to your hardware without any tuning. If you want finer control, `ThrottleConfig` is there, but the goal is that you shouldn't have to think about it.

The goal is that you spend your time designing datasets, not tuning concurrency knobs. The system handles the transport-level complexity automatically.

Key Resources:

1. [NeMo Data Designer on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)
2. [Design Principles Dev Note](design-principles.md)

*Want to learn more about NeMo Data Designer? Check out our [documentation](https://nvidia-nemo.github.io/DataDesigner/) and start building your own synthetic data pipelines today.*
