---
date: 2026-03-06
authors:
  - nmulepati
---

# Model Facade Overhaul PR-3 Architecture Notes

This document captures the architecture intent for PR-3 from
`plans/343/model-facade-overhaul-plan-step-1.md`.

## Goal

Introduce the first native HTTP adapter (`OpenAICompatibleClient`) with shared
retry infrastructure and a standalone adaptive throttle resource.  After this
PR, the client factory routes `provider_type="openai"` to the native adapter
while all other provider types continue through the `LiteLLMBridgeClient`.

## What Changes

### 1. Shared retry module (`clients/retry.py`)

`RetryConfig` is a frozen dataclass whose defaults mirror the current LiteLLM
router settings (`LiteLLMRouterDefaultKwargs`):

- `max_retries = 3`
- `backoff_factor = 2.0` (exponential)
- `backoff_jitter = 0.2`
- `retryable_status_codes = {429, 502, 503, 504}`

`create_retry_transport()` builds an `httpx_retries.RetryTransport` from a
`RetryConfig`.  The transport handles both sync and async requests (it inherits
from `httpx.BaseTransport` and `httpx.AsyncBaseTransport`).

### 2. Adaptive throttle module (`clients/throttle.py`)

`ThrottleManager` implements AIMD (additive-increase / multiplicative-decrease)
concurrency control keyed at two levels:

- **Global cap** `(provider_name, model_id)` — shared hard ceiling derived as
  `min(max_parallel_requests)` across all aliases targeting the same provider
  and model.
- **Domain** `(provider_name, model_id, throttle_domain)` — per-route AIMD
  state (`chat`, `embedding`, `image`, `healthcheck`) that floats between 1
  and the global effective max.

AIMD behaviour:

- *Decrease* — on a 429 the domain limit is multiplied by `reduce_factor`
  (default 0.5) and a cooldown block is applied.
- *Increase* — after every `success_window` (default 50) consecutive
  successful releases the limit grows by `additive_increase` (default 1),
  up to the global effective max.  Both `additive_increase` and
  `success_window` are constructor parameters for tuning recovery speed.
- *Recovery cost* — after a single halve from *L* to *L/2*, full recovery
  requires `(L/2) × success_window / additive_increase` successful requests.

Core state methods are non-blocking so both sync and async wrappers reuse
the same thread-safe state:

- `try_acquire(now) -> wait_seconds` (0 = acquired)
- `release_success(now)`
- `release_rate_limited(now, retry_after)`
- `release_failure(now)`

`acquire_sync` and `acquire_async` wrap `try_acquire` in a poll loop with a
configurable `timeout` (default 300s) to prevent indefinite blocking when a
domain is persistently at capacity or in cooldown.

#### Ownership — standalone resource, not adapter-owned

`ThrottleManager` is **not** owned by the adapter.  It lives as a shared
resource on `ModelRegistry` and is intended to be called by the orchestration
layer (the `AsyncTaskScheduler` from plan 346).

Rationale:

- **Separation of concerns** — the adapter is pure HTTP transport (request,
  retry, parse).  Concurrency policy is an orchestration concern.
- **Scheduler optimization** — the async scheduler needs to release its
  execution semaphore slot *while waiting* for a throttle permit, then
  reacquire it before executing.  This is only possible if the scheduler
  owns the acquire/release lifecycle directly.
- **Sync path** — the current sync builder is sequential (one call at a time),
  so it cannot exceed concurrency limits and does not need throttle gating.

The layered responsibility is:

| Layer | Responsibility |
|---|---|
| **Scheduler / Builder** | Concurrency policy: execution slots + throttle acquire/release |
| **ModelFacade** | Business logic: prompt assembly, usage tracking, correction loops |
| **Adapter** | Transport: HTTP, retry, response parsing |

### 3. OpenAI-compatible adapter (`clients/adapters/openai_compatible.py`)

`OpenAICompatibleClient` implements the `ModelClient` protocol using `httpx`
with `RetryTransport` for resilient HTTP calls.  The adapter is pure transport
— it has no knowledge of throttle or concurrency policy.

Routes:

- `POST /chat/completions` — chat completion and autoregressive image generation
- `POST /embeddings` — text embeddings
- `POST /images/generations` — diffusion-style image generation

Image routing is request-shape-based: if `request.messages is not None` the
chat route is used, otherwise the dedicated image route.

Response parsing reuses the shared `parsing.py` helpers.  The `get_value_from()`
utility handles both dict and object access, so raw JSON dicts from `httpx`
responses are passed directly to the parsing functions.

### 4. Reasoning field migration (`clients/parsing.py`)

`extract_reasoning_content(message)` checks `message.reasoning` first
(vLLM >= 0.16.0 canonical field), falling back to `message.reasoning_content`
(legacy / LiteLLM-normalized).  Both `parse_chat_completion_response` and
`aparse_chat_completion_response` now use this helper.

Internal canonical field remains `reasoning_content` — no downstream contract
change.

Ref: [GitHub issue #374](https://github.com/NVIDIA-NeMo/DataDesigner/issues/374)

### 5. Client factory routing (`clients/factory.py`)

`create_model_client` accepts an optional `retry_config` parameter and routes
based on provider type via sequential early returns:

1. If `DATA_DESIGNER_MODEL_BACKEND=litellm_bridge` → always `LiteLLMBridgeClient`
   (rollback safety during migration).
2. If `provider_type == "openai"` → `OpenAICompatibleClient`.
3. Otherwise → `LiteLLMBridgeClient` (Anthropic native adapter is PR-4).

The factory does not pass a `ThrottleManager` to adapters — throttle is an
orchestration concern (see §2).

### 6. Registry integration (`models/factory.py`, `models/registry.py`)

`create_model_registry` creates a shared `ThrottleManager` (held on
`ModelRegistry` for the scheduler to access) and a shared `RetryConfig`
(passed through to each `create_model_client` call).  The throttle manager
is not forwarded to adapters.

`ModelRegistry._get_model()` calls `throttle_manager.register()` when it
lazily creates each `ModelFacade`.  This ensures the throttle manager's
per-`(provider_name, model_id)` global caps are populated before the
scheduler (or any other caller) attempts to acquire permits.

## What Does NOT Change

1. `ModelFacade` public method signatures — callers see the same API.
2. MCP tool-loop behavior — tool turns, refusal, parallel execution all preserved.
3. Usage accounting semantics — token, request, image, and tool usage remain identical.
4. Error boundaries — `@catch_llm_exceptions` / `@acatch_llm_exceptions` decorators
   and `DataDesignerError` subclass hierarchy remain stable.
5. `consolidate_kwargs` merge semantics for `extra_body` / `extra_headers`.
6. `generate` / `agenerate` parser correction/restart loop logic.

## Files Touched

| File | Change |
|---|---|
| `clients/retry.py` | New — `RetryConfig` + `create_retry_transport` |
| `clients/throttle.py` | New — `ThrottleManager` with AIMD (standalone resource) |
| `clients/adapters/openai_compatible.py` | New — native OpenAI-compatible adapter (pure transport, no throttle) |
| `clients/errors.py` | Extract `infer_error_kind_from_exception` as shared function |
| `clients/base.py` | Add docstring to `ModelClient` protocol |
| `clients/parsing.py` | Add `extract_reasoning_content` helper; use `get_value_from` consistently |
| `clients/factory.py` | Route `provider_type=openai` to native adapter (sequential early returns) |
| `clients/__init__.py` | Export new public names |
| `clients/adapters/__init__.py` | Export `OpenAICompatibleClient` |
| `models/factory.py` | Create shared `ThrottleManager` (on registry) and `RetryConfig` |
| `models/registry.py` | Hold `ThrottleManager` as shared resource; expose via property |
| `models/facade.py` | Minor log message tweak (drop LiteLLM-specific wording) |

## Planned Follow-On

PR-4 introduces the Anthropic native adapter.  At that point, the client
factory gains a third adapter option alongside the LiteLLM bridge and
OpenAI-compatible adapter.

The `AsyncTaskScheduler` (plan 346) will call `ThrottleManager` directly
from the orchestration layer, acquiring throttle permits as part of its
execution-slot lifecycle:

1. Acquire execution slot
2. Release execution slot, await `throttle_manager.acquire_async(...)`
3. Reacquire execution slot, execute via `ModelFacade`

This pattern lets the scheduler free execution slots while waiting for
throttle permits (e.g., during 429 cooldowns), maximizing throughput.
