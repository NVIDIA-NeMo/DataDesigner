---

## date: 2026-03-19
authors:
  - nmulepati

# Model Facade Overhaul PR-5 Architecture Notes

This document captures the architecture intent and implementation state
for PR-5 from `plans/343/model-facade-overhaul-plan-step-1.md`.

PR-5 was originally scoped as "Config/CLI auth schema rollout". It has
been repurposed to address a design-level lifecycle issue in
`HttpModelClient` that surfaced during PR-4 review (#426). The auth
schema work shifts to PR-6.

## Goal

Constrain each `HttpModelClient` instance to a single execution mode --
sync or async -- at construction time. This eliminates the dual-mode
lifecycle complexity that causes transport leaks and cross-mode teardown
bugs. Additionally, add `ModelRegistry.arun_health_check()` so that
health checks use the async path when `DATA_DESIGNER_ASYNC_ENGINE=1`,
keeping the entire async engine flow consistent.

## Problem

`HttpModelClient` (introduced in PR-4) creates a single `RetryTransport`
eagerly, then lazily creates both `httpx.Client` (sync) and
`httpx.AsyncClient` (async) on top of it. This causes:

- **Transport leak**: `close()` only tears down the sync side of
`RetryTransport`; `aclose()` never touches the transport at all.
- **Cross-mode teardown**: `close()` cannot `await aclient.aclose()`;
`aclose()` must also handle sync cleanup.
- **Shared transport fragility**: both clients share one `RetryTransport`
with separate internal sync/async transports, making teardown
order-sensitive.
- **Excessive complexity**: the `close()`/`aclose()` logic must reason
about which combination of sync client, async client, and shared
transport were initialized, close them in the right order, avoid
double-closing the transport, and handle the fundamental mismatch of
needing `await` for async teardown from a sync `close()` call. Every
review round surfaced new edge cases (Andre's PR #426 feedback found
two separate leak paths). This complexity is a design smell, not a
patching problem.

The root cause is that no caller ever needs both sync and async on the
same instance. The sync engine uses sync methods; the async engine uses
async methods. They are mutually exclusive paths selected by
`DATA_DESIGNER_ASYNC_ENGINE`.

## What Changes

### 1. `ClientConcurrencyMode` enum (`clients/adapters/http_model_client.py`)

A new `StrEnum` replaces the `Literal["sync", "async"]` type alias:

```python
class ClientConcurrencyMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"
```

Using `StrEnum` over a bare `Literal` provides runtime type identity,
IDE autocomplete on members, and consistency with the project's existing
enum pattern (e.g. `ResourceType`, `StorageMode`). Since `StrEnum`
inherits from `str`, bare string comparisons and default values remain
backward-compatible.

### 2. `HttpModelClient` mode flag and enforcement (`clients/adapters/http_model_client.py`)

The constructor accepts
`concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC`.
Each instance owns exactly one transport and one httpx client type.

Mode enforcement:

- `_get_sync_client()` raises `RuntimeError` if `mode == ASYNC`.
- `_get_async_client()` raises `RuntimeError` if `mode == SYNC`.

Simplified lifecycle -- each method only handles its own mode:

- `close()`: tears down `self._transport` and `self._client`. No-op if
async mode.
- `aclose()`: awaits `self._transport.aclose()` and
`self._aclient.aclose()`. No-op if sync mode.

The `threading.Lock` is retained for both modes. It is only strictly
needed in sync mode (thread pool), but is harmless in async mode and
provides safety if callers ever share an instance across threads.

### 3. Factory chain threading

The `mode` parameter flows through the factory chain so that the
`DATA_DESIGNER_ASYNC_ENGINE` environment variable, read once at resource
provider creation time, determines the mode for all native HTTP adapters.


| Layer                     | File                             | Change                                                                                                                              |
| ------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Client factory            | `clients/factory.py`             | `create_model_client` accepts `client_concurrency_mode`, forwards as `concurrency_mode` to `OpenAICompatibleClient` and `AnthropicClient`. `LiteLLMBridgeClient` ignores it. |
| Model registry factory    | `models/factory.py`              | `create_model_registry` accepts `client_concurrency_mode`, captures it in the `model_facade_factory` closure.                                          |
| Resource provider factory | `resources/resource_provider.py` | Reads `DATA_DESIGNER_ASYNC_ENGINE` env var, passes `ClientConcurrencyMode.ASYNC` or `ClientConcurrencyMode.SYNC` to `create_model_registry`. |


### 4. `ModelRegistry.arun_health_check()` (`models/registry.py`)

An async mirror of `run_health_check()` that calls the async facade
methods:

- `await model.agenerate_text_embeddings(...)` for embedding models
- `await model.agenerate(...)` for chat completion models
- `await model.agenerate_image(...)` for image models

This ensures that when the registry holds async-mode clients, health
checks exercise the correct code path instead of hitting the mode
enforcement guard.

### 5. Async health check dispatch (`dataset_builders/column_wise_builder.py`)

`_run_model_health_check_if_needed` now branches on
`DATA_DESIGNER_ASYNC_ENGINE`:

- **Sync path** (default): calls `registry.run_health_check()` directly.
- **Async path**: submits `registry.arun_health_check()` to the
background event loop via `asyncio.run_coroutine_threadsafe` and blocks
on `future.result()`.

Health checks remain blocking from the main thread's perspective (same
UX), but use the async client internally when in async-engine mode.

### 6. Test changes

`**test_native_http_clients.py`**:

- Lifecycle tests are split into sync-mode and async-mode groups.
- Tests that injected both `sync_client` and `async_client`
simultaneously are removed (that scenario no longer exists).
- New mode enforcement tests verify that calling `_get_sync_client()` on
an async-mode client (and vice versa) raises `RuntimeError`.
- New cross-mode no-op tests verify that `close()` on an async-mode
client and `aclose()` on a sync-mode client are no-ops.

`**test_factory.py**`:

- New tests verify that `client_concurrency_mode` is forwarded to native
adapter constructors and that the default is `ClientConcurrencyMode.SYNC`.

`**test_model_registry.py**`:

- New async tests for `arun_health_check` covering success and
authentication error paths.

## What Does NOT Change

1. `**ModelClient` protocol** (`clients/base.py`) -- still requires both
  sync and async method signatures. Adapters implement both; mode
   enforcement happens inside `HttpModelClient`, not at the protocol
   level.
2. `**ModelFacade`** -- continues to expose both `generate` /
  `agenerate`, `completion` / `acompletion`, etc. No changes needed.
3. **Subclass adapters** -- `OpenAICompatibleClient` and
  `AnthropicClient` inherit mode enforcement from `HttpModelClient`.
   Their provider-specific logic is unchanged.
4. `**LiteLLMBridgeClient`** -- does not inherit from `HttpModelClient`
  and is unaffected. Its `close` / `aclose` are already no-ops.
5. **Error boundaries** -- `@catch_llm_exceptions` /
  `@acatch_llm_exceptions` decorators and `DataDesignerError` subclass
   hierarchy remain stable.
6. **Usage accounting** -- token tracking works unchanged via canonical
  `Usage`.
7. **MCP tool-loop behavior** -- `ModelFacade` continues to orchestrate
  tool turns using canonical `ToolCall` objects regardless of provider.

## Files Touched


| File                                      | Change                                                                                                                                                       |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `clients/adapters/http_model_client.py`   | Add `ClientConcurrencyMode` enum, `concurrency_mode` constructor param, mode enforcement in `_get_sync_client` / `_get_async_client`, simplified single-mode `close()` / `aclose()` |
| `clients/factory.py`                      | Add `client_concurrency_mode` parameter to `create_model_client`, forward as `concurrency_mode` to native adapter constructors                                                             |
| `models/factory.py`                       | Add `client_concurrency_mode` parameter to `create_model_registry`, capture in facade factory closure                                                                                      |
| `resources/resource_provider.py`          | Read `DATA_DESIGNER_ASYNC_ENGINE`, pass `ClientConcurrencyMode` to `create_model_registry`                                                                                                 |
| `models/registry.py`                      | Add `arun_health_check()` async mirror                                                                                                                       |
| `dataset_builders/utils/async_concurrency.py` | Rename `_ensure_async_engine_loop` to `ensure_async_engine_loop` (public, used cross-module)                                                             |
| `dataset_builders/column_wise_builder.py` | Dispatch health checks via `run_coroutine_threadsafe` when `DATA_DESIGNER_ASYNC_ENGINE=1`; import `ensure_async_engine_loop` in top-level async guard    |
| `tests/.../test_native_http_clients.py`   | Rewrite lifecycle tests for single-mode semantics, add mode enforcement and cross-mode no-op tests                                                           |
| `tests/.../test_factory.py`               | Add mode forwarding and default mode tests                                                                                                                   |
| `tests/.../test_model_registry.py`        | Add `arun_health_check` success and error tests                                                                                                              |


## Test Coverage


| Category                | Tests                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Sync-mode lifecycle     | `close()` delegates to httpx client, idempotent close, no-op when no client created, completion raises after close             |
| Async-mode lifecycle    | `aclose()` delegates to httpx async client, idempotent aclose, no-op when no client created, acompletion raises after aclose   |
| Cross-mode no-ops       | `aclose()` is no-op on sync-mode client, `close()` is no-op on async-mode client                                               |
| Mode enforcement        | `_get_sync_client()` raises on async-mode, `_get_async_client()` raises on sync-mode, `concurrency_mode` property reflects constructor arg |
| Lazy initialization     | Sync completion lazily creates `httpx.Client`, async acompletion lazily creates `httpx.AsyncClient`                            |
| Factory mode forwarding | `client_concurrency_mode` forwarded to OpenAI client, `client_concurrency_mode` forwarded to Anthropic client, default is sync |
| Async health check      | `arun_health_check` success (chat + embedding), `arun_health_check` authentication error propagation                           |


## Planned Follow-On

PR-6 picks up the config/CLI auth schema rollout that was originally
scoped for PR-5, adding typed provider-specific auth objects
(`AnthropicAuth`, `OpenAIApiKeyAuth`) to `ModelProvider` with
backward-compatible `api_key` fallback.
