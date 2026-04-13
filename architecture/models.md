# Models

The model subsystem provides a unified interface for LLM access: chat completions, embeddings, and image generation. It handles client creation, retry, rate-limit throttling, usage tracking, and MCP tool integration.

Source: `packages/data-designer-engine/src/data_designer/engine/models/`

## Overview

The model subsystem is layered:

```
ModelRegistry (lazy facade-per-alias)
  └── ModelFacade (completion, embeddings, image gen, MCP tool loops)
        └── ThrottledModelClient (AIMD rate limiting)
              └── ModelClient (OpenAI-compatible or Anthropic adapter)
                    └── RetryTransport (httpx-level retries)
```

Generators never interact with HTTP clients directly. They request a `ModelFacade` by alias from the `ModelRegistry`, which handles lazy construction and shared throttle state.

## Key Components

### ModelClient (Protocol)

Defines the contract: sync/async chat, embeddings, image generation, `supports_*` capability checks, `close` / `aclose`. Two implementations:

- **`OpenAICompatibleClient`** — native httpx adapter for OpenAI-compatible endpoints (NIM, vLLM, etc.)
- **`AnthropicClient`** — native httpx adapter for the Anthropic Messages API

### Client Factory

`create_model_client` routes by provider type to the appropriate adapter. Optionally wraps with:
- **`RetryTransport`** — httpx-level retries via `httpx_retries.RetryTransport`. `HttpModelClient` sets `strip_rate_limit_codes=True` for the async client and `False` for the sync client (`http_model_client.py`), which controls whether 429 responses are eligible for transport-layer retries.
- **`ThrottledModelClient`** — AIMD (Additive Increase, Multiplicative Decrease) concurrency control per throttle domain.

### ThrottleManager

Manages concurrency limits per `ThrottleDomain` (CHAT, EMBEDDING, IMAGE, HEALTHCHECK), keyed by `(provider_name, model_id)`. Thread-safe with a shared lock for sync/async access.

`ThrottledModelClient` wraps each API call in a context manager that acquires/releases throttle capacity and adjusts limits on success (additive increase) or rate-limit errors (multiplicative decrease).

### ModelFacade

The primary interface for generators. Holds a `ModelConfig`, `ModelClient`, optional `MCPRegistry`, and `ModelUsageStats`.

- **`completion` / `acompletion`** — consolidates kwargs from inference params + provider extras, calls the client, tracks usage
- **`embeddings` / `aembeddings`** — embedding generation
- **`image_generation` / `aimage_generation`** — image generation
- **MCP tool loops** — when a tool config is active, processes tool calls from completions via `MCPFacade`, feeds results back, and tracks tool usage stats

### ModelRegistry

Lazy `ModelFacade` construction per alias. Registers a shared `ThrottleManager` across all facades for coordinated rate limiting. Provides `get_model_usage_stats` and `log_model_usage` for post-build reporting.

### Usage Tracking

`ModelUsageStats` aggregates `TokenUsageStats`, `RequestUsageStats`, `ToolUsageStats`, and `ImageUsageStats` per model. Tracked on every successful or failed request for cost and performance visibility.

## Data Flow

1. Generator requests a model by alias from `ModelRegistry`
2. Registry lazily creates `ModelFacade` with the appropriate client and throttle config
3. Generator calls `completion()` with prompt/messages
4. `ModelFacade` builds kwargs, calls `ThrottledModelClient`
5. Throttle layer acquires capacity, delegates to `ModelClient`
6. `ModelClient` makes the HTTP request through `RetryTransport`
7. Response flows back; usage is tracked; if MCP tools are configured, tool calls are executed and results fed back for another completion round

## Design Decisions

- **Facade pattern** hides HTTP, retry, throttle, and MCP complexity from generators. Generators see `completion()` and get back parsed results.
- **AIMD throttling at the application layer** rather than relying solely on HTTP retries. This provides smoother throughput under rate limits — the transport layer still handles many transient failures, while the throttle manager adjusts concurrency to avoid sustained 429 storms.
- **429 handling depends on sync vs async `HttpModelClient`** — The async client uses `strip_rate_limit_codes=True`, so 429s are not retried at the transport layer and rate-limit signals reach `ThrottledModelClient` / AIMD quickly. The sync client uses `strip_rate_limit_codes=False`, so 429s may still be retried transparently at the transport layer before surfacing to callers.
- **Distribution-valued inference parameters** (`temperature`, `top_p` as `UniformDistribution` or `ManualDistribution`) enable controlled randomness across a dataset without per-row config changes.
- **Lazy facade construction** avoids health-checking or connecting to models that are configured but never used in a particular generation run.

## Cross-References

- [System Architecture](overview.md) — where models fit in the stack
- [Engine Layer](engine.md) — how generators use models
- [MCP](mcp.md) — tool execution integrated into completions
- [Config Layer](config.md) — `ModelConfig` and `ModelProvider` definitions
