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
retry and adaptive throttle infrastructure.  After this PR, the client factory
routes `provider_type="openai"` to the native adapter while all other provider
types continue through the `LiteLLMBridgeClient`.

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

Core state methods are non-blocking so both sync and async wrappers reuse
the same thread-safe state:

- `try_acquire(now) -> wait_seconds` (0 = acquired)
- `release_success(now)`
- `release_rate_limited(now, retry_after)`
- `release_failure(now)`

### 3. OpenAI-compatible adapter (`clients/adapters/openai_compatible.py`)

`OpenAICompatibleClient` implements the `ModelClient` protocol using `httpx`
with `RetryTransport` for resilient HTTP calls and `ThrottleManager` for
adaptive concurrency.

Routes:

- `POST /chat/completions` — chat completion and autoregressive image generation
- `POST /embeddings` — text embeddings
- `POST /images/generations` — diffusion-style image generation

Image routing is request-shape-based: if `request.messages is not None` the
chat route is used, otherwise the dedicated image route.

Response parsing reuses the shared `parsing.py` helpers via a `_DictProxy`
wrapper that gives dict-style JSON responses attribute-style access expected
by `get_value_from()`.

### 4. Reasoning field migration (`clients/parsing.py`)

`extract_reasoning_content(message)` checks `message.reasoning` first
(vLLM >= 0.16.0 canonical field), falling back to `message.reasoning_content`
(legacy / LiteLLM-normalized).  Both `parse_chat_completion_response` and
`aparse_chat_completion_response` now use this helper.

Internal canonical field remains `reasoning_content` — no downstream contract
change.

Ref: [GitHub issue #374](https://github.com/NVIDIA-NeMo/DataDesigner/issues/374)

### 5. Client factory routing (`clients/factory.py`)

`create_model_client` now accepts optional `throttle_manager` and `retry_config`
parameters and routes based on provider type:

- `provider_type == "openai"` (and no bridge override) → `OpenAICompatibleClient`
- All other provider types → `LiteLLMBridgeClient`

The `DATA_DESIGNER_MODEL_BACKEND` environment variable can force bridge mode
for rollback safety during migration.

### 6. Registry integration (`models/factory.py`)

`create_model_registry` creates a shared `ThrottleManager` and `RetryConfig`
and passes them through to each `create_model_client` call.

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
| `clients/throttle.py` | New — `ThrottleManager` with AIMD |
| `clients/adapters/openai_compatible.py` | New — native OpenAI-compatible adapter |
| `clients/parsing.py` | Add `extract_reasoning_content` helper |
| `clients/factory.py` | Route `provider_type=openai` to native adapter |
| `clients/__init__.py` | Export new public names |
| `clients/adapters/__init__.py` | Export `OpenAICompatibleClient` |
| `models/factory.py` | Create shared `ThrottleManager` and `RetryConfig` |

## Planned Follow-On

PR-4 introduces the Anthropic native adapter.  At that point, the client
factory gains a third adapter option alongside the LiteLLM bridge and
OpenAI-compatible adapter.
