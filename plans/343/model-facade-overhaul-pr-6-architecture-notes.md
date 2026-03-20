---

## date: 2026-03-20
authors:
  - nmulepati

# Model Facade Overhaul PR-6 Architecture Notes

This document captures the architecture intent and implementation plan
for PR-6 from `plans/343/model-facade-overhaul-plan-step-1.md`.

PR-6 was originally scoped as "Config/CLI auth schema rollout". That
work is deferred to PR-7. PR-6 instead integrates the `ThrottleManager`
(introduced in PR-3, keyed and registered in PR-4) with the model
client layer and `AsyncTaskScheduler`. This closes the last gap in the
concurrency control stack: the throttle manager exists and models
register into it, but no execution path actually acquires or releases
throttle permits.

The design uses a **dual-layer** approach: a `ThrottledModelClient`
wrapper handles per-HTTP-request AIMD acquire/release, while the
`AsyncTaskScheduler` manages submission slot release/reacquire for
LLM-bound tasks to prevent cross-key starvation.

## Goal

Wire `ThrottleManager` into the model client layer so that every
outbound HTTP request acquires a throttle permit and releases it on
completion, rate-limit, or failure. Simultaneously, update the
`AsyncTaskScheduler` to release submission slots for LLM-bound tasks
so that throttle waits don't cause cross-key starvation. This enables
per-provider+model AIMD concurrency control that is accurate at the
per-request level and fair at the task level.

## Problem

Today the `AsyncTaskScheduler` uses two concurrency controls:

1. **Row group semaphore** (`_rg_semaphore`) — bounds admitted row groups.
2. **Submission semaphore** (`_submission_semaphore`) — bounds total
   submitted tasks.

Neither is API-aware. All LLM tasks compete for the same flat pool of
submission slots regardless of which provider/model they target. This
means:

- A 429 from provider A causes retries that hold submission slots,
  starving unrelated provider B tasks.
- There is no AIMD feedback loop — the scheduler cannot reduce
  concurrency for a throttled key while keeping other keys at full
  capacity.
- The `ThrottleManager` is instantiated and models register into it
  (via `ModelRegistry._get_model`), but `try_acquire` / `release_*`
  are never called.
- The shared HTTP retry transport can currently retry `429` before the
  throttling layer sees it, which would mask the exact rate-limit
  signal PR-6 needs for AIMD backoff.

Additionally, a scheduler-only throttle integration (one permit per
task) would miss multi-call patterns: correction loops, conversation
restarts, tool-calling loops, and custom columns that call
`ModelFacade` directly. The per-request accuracy requires throttling
at the HTTP call boundary.

## Design Overview

Two layers work together:

| Layer | Where | What it does |
| --- | --- | --- |
| **Client wrapper** | `ThrottledModelClient` (new) | Acquires/releases a throttle permit around every HTTP call (`completion`, `acompletion`, `embeddings`, `aembeddings`, `generate_image`, `agenerate_image`). Provides per-request AIMD accuracy. |
| **Scheduler** | `AsyncTaskScheduler` | Releases the submission slot before dispatching LLM-bound tasks and reacquires after. Prevents cross-key starvation. No throttle logic. |

The `ModelFacade` is **untouched** — throttling is a transport concern
handled below it. The facade's correction loops, tool-calling loops,
and custom column calls all go through the client, so they are
automatically gated.

## What Changes

### 1. `ThrottledModelClient` wrapper

A new class in `models/clients/throttled.py` that wraps any
`ModelClient` and adds throttle acquire/release around every call:

```python
class ThrottledModelClient:
    def __init__(
        self,
        inner: ModelClient,
        throttle_manager: ThrottleManager,
        provider_name: str,
        model_id: str,
    ) -> None:
        self._inner = inner
        self._tm = throttle_manager
        self._provider_name = provider_name
        self._model_id = model_id
```

The wrapper implements the `ModelClient` protocol by delegating to
`self._inner` with throttle acquire/release around each call. The
`ThrottleDomain` is determined by the method:

- `completion` / `acompletion` → `ThrottleDomain.CHAT`
- `embeddings` / `aembeddings` → `ThrottleDomain.EMBEDDING`
- `generate_image` / `agenerate_image` → domain depends on the
  request: `ThrottleDomain.IMAGE` when `request.messages is None`
  (diffusion path), `ThrottleDomain.CHAT` when `request.messages` is
  set (chat-backed image generation). This matches the actual HTTP
  route chosen by `OpenAICompatibleClient`.

The acquire/release pattern for each method:

```python
async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
    await self._tm.acquire_async(
        provider_name=self._provider_name,
        model_id=self._model_id,
        domain=ThrottleDomain.CHAT,
    )
    try:
        return await self._inner.acompletion(request)
    except ProviderError as exc:
        if exc.kind == ProviderErrorKind.RATE_LIMIT:
            self._tm.release_rate_limited(
                provider_name=self._provider_name,
                model_id=self._model_id,
                domain=ThrottleDomain.CHAT,
                retry_after=exc.retry_after,
            )
        else:
            self._tm.release_failure(
                provider_name=self._provider_name,
                model_id=self._model_id,
                domain=ThrottleDomain.CHAT,
            )
        raise
    except Exception:
        self._tm.release_failure(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=ThrottleDomain.CHAT,
        )
        raise
    else:
        self._tm.release_success(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=ThrottleDomain.CHAT,
        )
```

The sync methods (`completion`, `embeddings`, `generate_image`) use
`acquire_sync` instead of `acquire_async`. The pattern is identical
otherwise.

**Key advantage over facade-level integration:** The client layer is
*below* `@catch_llm_exceptions`, so the wrapper sees `ProviderError`
directly — not the translated `ModelRateLimitError`. This means:

- `ProviderError.retry_after` is available natively (parsed from the
  `Retry-After` header by `_extract_retry_after` in `clients/errors.py`).
- No need to propagate `retry_after` through `ModelRateLimitError`.
- `ProviderError.kind == RATE_LIMIT` is a clean discriminator.

**Critical retry-boundary requirement:** This only works if the shared
HTTP transport does **not** auto-retry `429`. PR-6 therefore updates
`clients/retry.py` to remove `429` from transport-level retryable
statuses. Non-rate-limit transient failures (`502` / `503` / `504`,
connection/transport errors) remain retried in the shared HTTP layer.

**DRY:** The six methods share the same acquire/release pattern. A
private helper `_release_on_error` and paired context managers
`_throttled_sync` / `_athrottled` can centralize the logic:

```python
@contextlib.asynccontextmanager
async def _athrottled(self, domain: ThrottleDomain):
    await self._tm.acquire_async(
        provider_name=self._provider_name,
        model_id=self._model_id,
        domain=domain,
    )
    try:
        yield
    except ProviderError as exc:
        self._release_on_provider_error(domain, exc)
        raise
    except Exception:
        self._tm.release_failure(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=domain,
        )
        raise
    else:
        self._tm.release_success(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=domain,
        )
```

Each method then becomes a one-liner delegation:

```python
async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
    async with self._athrottled(ThrottleDomain.CHAT):
        return await self._inner.acompletion(request)
```

### 2. Factory wiring

`create_model_client` in `models/clients/factory.py` wraps the inner
client with `ThrottledModelClient` when a `ThrottleManager` is
provided:

```python
def create_model_client(
    model_config: ModelConfig,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    *,
    retry_config: RetryConfig | None = None,
    client_concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    throttle_manager: ThrottleManager | None = None,
) -> ModelClient:
    ...
    client = _create_inner_client(...)  # existing logic

    if throttle_manager is not None:
        provider = model_provider_registry.get_provider(model_config.provider)
        client = ThrottledModelClient(
            inner=client,
            throttle_manager=throttle_manager,
            provider_name=provider.name,
            model_id=model_config.model,
        )

    return client
```

The `ThrottleManager` flows from `create_model_registry` →
`model_facade_factory` closure → `create_model_client`. The
`model_facade_factory` closure in `create_model_registry` already has
access to the `ThrottleManager` instance (it creates it). The closure
just needs to pass it through to `create_model_client`.

**Registration ordering:** `ThrottleManager.register()` must be called
before `try_acquire()`. Registration happens in
`ModelRegistry._get_model()` *after* the facade (and thus the client)
is created. This is safe because `try_acquire` is only called when
the client makes an actual HTTP request, which happens later during
generation. The sequence is:

1. `_get_model()` calls `model_facade_factory()` → creates client
   (with `ThrottledModelClient` wrapper)
2. `_get_model()` calls `throttle_manager.register(...)` for the model
3. Later, during generation, the client calls `acquire_async()` →
   `try_acquire()` — registration is already done.

### 3. AIMD tuning via `RunConfig`

`RunConfig` gains four new optional fields that mirror the
`ThrottleManager` constructor parameters:

```python
class RunConfig(ConfigBase):
    # ... existing fields ...

    throttle_reduce_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    throttle_additive_increase: int = Field(default=1, ge=1)
    throttle_success_window: int = Field(default=50, ge=1)
    throttle_block_seconds: float = Field(default=2.0, gt=0.0)
```

These fields have the same defaults as the current `throttle.py`
constants, so existing behavior is unchanged when users don't set them.

**Wiring:** `create_resource_provider` already has `run_config` at the
call site where `create_model_registry` is invoked. Two changes are
needed in the factory chain:

1. **`create_model_registry` signature** gains a new parameter
   `run_config: RunConfig | None = None`. Today the function accepts
   `model_configs`, `secret_resolver`, `model_provider_registry`,
   `mcp_registry`, and `client_concurrency_mode` — it has no access to
   `RunConfig`. Adding the parameter keeps the factory self-contained.

2. **`create_resource_provider`** forwards `run_config` to the
   `create_model_registry` call.

Inside `create_model_registry`, the `ThrottleManager` is constructed
with the `RunConfig` values:

```python
throttle_manager = ThrottleManager(
    reduce_factor=run_config.throttle_reduce_factor,
    additive_increase=run_config.throttle_additive_increase,
    success_window=run_config.throttle_success_window,
    default_block_seconds=run_config.throttle_block_seconds,
) if run_config else ThrottleManager()
```

### 4. Scheduler: submission slot management for LLM tasks

The `AsyncTaskScheduler` does **not** acquire or release throttle
permits. That is the client wrapper's job. The scheduler's only
throttle-related change is managing submission slots to prevent
cross-key starvation.

**Dispatch pattern change in `_execute_task_inner`:**

**Before (current):**
```
acquire submission slot → execute generator → release submission slot
```

**After:**
```
acquire submission slot →
  if LLM-bound:
    release submission slot
    execute generator (client wrapper handles throttle per HTTP call)
    reacquire submission slot
  else:
    execute generator
→ release submission slot
```

LLM-bound tasks release the submission slot *before* the generator
runs. The generator may block inside the client wrapper's throttle
acquire, but the submission slot is already free, so other tasks can
proceed. When the generator completes (success or failure), the
scheduler reacquires the submission slot for bookkeeping (completion
tracking, error classification, checkpoint logic).

**LLM-bound detection:** The scheduler needs to know which generators
are LLM-bound. A simple `_is_llm_bound` lookup is built at init time:
`dict[str, bool]` mapping column name → whether the generator
subclasses `ColumnGeneratorWithModel`. This replaces the more complex
`ThrottleKey` / `build_throttle_keys` approach from the scheduler-only
design — the scheduler doesn't need to know provider/model/domain,
just whether the task will make HTTP calls.

```python
def _build_llm_bound_lookup(
    generators: dict[str, ColumnGenerator],
) -> dict[str, bool]:
    from data_designer.engine.column_generators.generators.base import (
        ColumnGeneratorWithModel,
        ColumnGeneratorWithModelRegistry,
    )
    result: dict[str, bool] = {}
    for col, gen in generators.items():
        result[col] = isinstance(gen, (ColumnGeneratorWithModel, ColumnGeneratorWithModelRegistry))
    return result
```

`ColumnGeneratorWithModelRegistry` is included because custom columns
subclass it (not `ColumnGeneratorWithModel`) but still make model calls.

**Error handling:** The scheduler's `except` block no longer needs to
distinguish `ModelRateLimitError` for throttle release — the client
wrapper already handled it. The scheduler only classifies retryability
for the salvage queue. The existing `_is_retryable` check is
unchanged.

**`TimeoutError` from throttle acquire:** The client wrapper's
`acquire_async` can raise a builtin `TimeoutError`. This propagates
through the generator and reaches the scheduler's `except` block.
`TimeoutError` is added to `_is_retryable` so the task is deferred to
a salvage round:

```python
@staticmethod
def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, AsyncTaskScheduler._RETRYABLE_MODEL_ERRORS)
```

### 5. Submission pool sizing

The submission semaphore is currently hardcoded at 256. With the
submission slot release/reacquire pattern, the pool must be large
enough to not bottleneck the aggregate throttle capacity.

`ModelRegistry` exposes a new public method:

```python
def get_aggregate_max_parallel_requests(self) -> int:
    """Sum of max_parallel_requests across all registered model configs."""
    return sum(
        mc.inference_parameters.max_parallel_requests
        for mc in self._model_configs.values()
    )
```

The hardcoded `256` floor is extracted to named constants:

```python
MIN_SUBMITTED_TASKS: int = 256
SUBMISSION_POOL_MULTIPLIER: int = 2
```

The builder derives `max_submitted_tasks`:

```python
aggregate = model_registry.get_aggregate_max_parallel_requests()
max_submitted_tasks = max(MIN_SUBMITTED_TASKS, SUBMISSION_POOL_MULTIPLIER * aggregate)
```

**Precision note:** This is a deliberate **heuristic over-estimate**.
The real shared upstream cap in `ThrottleManager` is
`min(max_parallel_requests)` across aliases sharing the same
`(provider_name, model_id)` key. Summing individual values overstates
the true capacity when aliases share a key. However, the submission
pool is an upper bound on in-flight tasks, not a concurrency target —
oversizing wastes a few coroutine slots but doesn't cause incorrect
behavior. The `ThrottleManager` enforces the real per-key limit.

### 6. Builder wiring

`_build_async` in `ColumnWiseDatasetBuilder` passes the computed
`max_submitted_tasks` to the `AsyncTaskScheduler`:

```python
registry = self._resource_provider.model_registry
aggregate = registry.get_aggregate_max_parallel_requests()

scheduler = AsyncTaskScheduler(
    ...,
    max_submitted_tasks=max(MIN_SUBMITTED_TASKS, SUBMISSION_POOL_MULTIPLIER * aggregate),
)
```

The scheduler no longer receives a `throttle_manager` parameter —
throttling is handled by the client wrapper inside each `ModelFacade`.

### 7. Rate-limit error integration with retry/salvage

PR-6 changes the retry boundary before a `ModelRateLimitError` occurs:

1. `RetryTransport` no longer retries `429`. It continues to retry
   non-rate-limit transient failures (`502` / `503` / `504`,
   connection/transport errors).
2. The first throttled HTTP response is mapped to
   `ProviderError(kind=RATE_LIMIT)` with the provider's `retry_after`
   value when present.
3. The client wrapper has already called `release_rate_limited` with
   the provider's `retry_after` value (from the `ProviderError`).
   The AIMD state is updated before the error reaches the scheduler.
4. `@catch_llm_exceptions` translates `ProviderError` →
   `ModelRateLimitError` and propagates to the generator.
5. The generator raises to the scheduler.
6. The scheduler classifies it as retryable via `_is_retryable` and
   defers to a salvage round.
7. When retried, the generator calls the facade again, which calls
   the client wrapper, which calls `acquire_async` before the next
   HTTP attempt — respecting the reduced limit and cooldown.

This creates a feedback loop: 429s reduce concurrency at the client
level, salvage retries respect the reduced limit, and successful
completions gradually restore capacity via AIMD additive increase.

**`ModelQuotaExceededError` note:** Quota-exceeded errors (HTTP 403)
are semantically similar to rate limits. However,
`ModelQuotaExceededError` is **not** in `_RETRYABLE_MODEL_ERRORS`
today, so it is treated as non-retryable and the affected row is
dropped. This is intentional: quota exhaustion is typically a
billing/account issue that won't resolve within a generation run. The
client wrapper still calls `release_failure` for quota errors, which
decrements `in_flight` without triggering AIMD backoff.

### 8. Health check interaction

Health checks call `ModelFacade.generate()` / `agenerate()` (or the
embedding/image equivalents) with `skip_usage_tracking=True`. These
calls go through the `ThrottledModelClient` wrapper and will acquire
throttle permits.

This is **acceptable** for v1:

- Health checks run once at startup, before generation begins. The
  throttle limits are at their initial maximums, so the acquire
  succeeds immediately.
- If a health check triggers a 429 (unlikely but possible with
  aggressive rate limits), the AIMD state is correctly updated before
  generation starts — the system begins with a reduced limit, which
  is the right behavior.

If health check throttling becomes problematic (e.g. slow startup
under load), a future enhancement can add a `skip_throttle` parameter
to the client wrapper or use `ThrottleDomain.HEALTHCHECK` (already
in the enum) with permissive limits.

### 9. Test changes

**`test_throttled_model_client.py`** (new):

- Acquire/release on success for each modality (chat, embedding, image)
- `release_rate_limited` with `retry_after` on `ProviderError` with
  `kind=RATE_LIMIT`
- `release_failure` on non-rate-limit `ProviderError`
- `release_failure` on non-`ProviderError` exceptions
- `TimeoutError` from `acquire_async` propagates without release
  (acquire failed, no slot held)
- Image generation: `request.messages is None` → `IMAGE` domain;
  `request.messages` set → `CHAT` domain
- `ThrottleManager=None` path (no wrapper, inner client called directly)
- Sync methods use `acquire_sync`

**`test_async_scheduler.py`**:

- Submission slot released before LLM-bound generator runs
- Submission slot NOT released for non-LLM generators
- `TimeoutError` is retryable via `_is_retryable`
- Existing tests pass unchanged

**`test_model_registry.py`**:

- `get_aggregate_max_parallel_requests` returns correct sum

**`test_client_factory.py`** (or existing factory tests):

- `create_model_client` with `throttle_manager` returns
  `ThrottledModelClient` wrapping the inner client
- `create_model_client` without `throttle_manager` returns inner
  client directly

**Retry tests** (existing or new):

- `429` is excluded from transport-level retryable statuses
- `502` / `503` / `504` remain transport-retryable

## What Does NOT Change

1. **`ThrottleManager` API** — no changes to `throttle.py`. The
   existing `try_acquire`, `release_success`, `release_rate_limited`,
   `release_failure`, `acquire_async`, `acquire_sync` methods are
   used as-is. The constructor already accepts AIMD tuning params.
2. **`ModelFacade`** — untouched. Throttling is below it in the client
   layer. Correction loops, tool-calling loops, and custom columns
   are automatically gated because they all go through the client.
3. **`ModelRegistry`** — already holds `ThrottleManager` and registers
   models. Only addition is `get_aggregate_max_parallel_requests()`.
4. **Sync path** — `DATA_DESIGNER_ASYNC_ENGINE=0` is unaffected. The
   sync builder does not use `AsyncTaskScheduler`. The client wrapper
   still provides per-request throttling for the sync path (using
   `acquire_sync`), which is a bonus.
5. **Error model** — `ModelRateLimitError` and other error types are
   unchanged. The client wrapper works with `ProviderError` directly,
   so `retry_after` propagation through `ModelRateLimitError` is not
   needed.
6. **Generator code** — generators are unaware of throttling.
7. **Non-rate-limit transport retries** — `502` / `503` / `504` and
   connection failures still use the shared HTTP retry layer. PR-6
   only removes `429` from that retry set.

## Files Touched

| File | Change |
| --- | --- |
| `models/clients/throttled.py` (new) | `ThrottledModelClient` wrapper with `_throttled_sync` / `_athrottled` context managers |
| `models/clients/retry.py` | Remove `429` from transport retryable statuses; retain non-rate-limit transient retries |
| `models/clients/factory.py` | Accept optional `throttle_manager`; wrap inner client with `ThrottledModelClient` when provided |
| `models/factory.py` | Forward `ThrottleManager` to `model_facade_factory` closure → `create_model_client`; accept `run_config` parameter; construct `ThrottleManager` with `RunConfig` values; remove TODO comment |
| `models/registry.py` | Add `get_aggregate_max_parallel_requests()` public method |
| `config/run_config.py` | Add `throttle_reduce_factor`, `throttle_additive_increase`, `throttle_success_window`, `throttle_block_seconds` fields |
| `resources/resource_provider.py` | Forward `run_config` to `create_model_registry` |
| `dataset_builders/async_scheduler.py` | Add `_is_llm_bound` lookup, submission slot release/reacquire for LLM tasks, `MIN_SUBMITTED_TASKS` / `SUBMISSION_POOL_MULTIPLIER` constants, add `TimeoutError` to `_is_retryable` |
| `dataset_builders/column_wise_builder.py` | Forward data-driven `max_submitted_tasks` to scheduler |
| `tests/.../test_throttled_model_client.py` (new) | Throttle wrapper unit tests |
| `tests/.../test_async_scheduler.py` | Submission slot management tests |
| `tests/.../test_model_registry.py` | Test for `get_aggregate_max_parallel_requests` |
| `tests/.../test_client_factory.py` | Test wrapper wiring in `create_model_client` |

## Test Coverage

| Category | Tests |
| --- | --- |
| Client wrapper: acquire/release on success | Chat, embedding, image (diffusion + chat-backed) all call `release_success` |
| Client wrapper: rate limit | `ProviderError(kind=RATE_LIMIT)` calls `release_rate_limited` with `retry_after` |
| Client wrapper: failure | Non-rate-limit errors call `release_failure` |
| Client wrapper: acquire timeout | `TimeoutError` from `acquire_async` propagates, no release (no slot held) |
| Client wrapper: image domain routing | `request.messages is None` → `IMAGE`; `request.messages` set → `CHAT` |
| Client wrapper: sync path | Sync methods use `acquire_sync` / `release_*` |
| Client wrapper: no throttle | `ThrottleManager=None` → inner client called directly |
| Transport retry boundary | `429` excluded from transport retryable statuses; `502` / `503` / `504` remain retried |
| Scheduler: submission slot for LLM tasks | Slot released before generator runs, reacquired after |
| Scheduler: non-LLM tasks | Slot held during generator execution (no release/reacquire) |
| Scheduler: TimeoutError retryable | Builtin `TimeoutError` classified as retryable, deferred to salvage |
| Scheduler: backward compat | Existing tests pass unchanged |
| Submission pool sizing | `max(MIN_SUBMITTED_TASKS, SUBMISSION_POOL_MULTIPLIER * aggregate)` |
| RunConfig → AIMD wiring | Custom `RunConfig` throttle fields forwarded to `ThrottleManager` constructor |
| Factory wiring | `create_model_client` with `throttle_manager` returns `ThrottledModelClient` |

## Design Rationale

Key points:

- **Why client-level, not scheduler-level throttling?** The scheduler
  sees one task = one throttle permit, but generators can make multiple
  HTTP calls (correction loops, tool calls, custom columns). The client
  wrapper sees every actual HTTP request.

- **Why keep the scheduler's submission slot dance?** Without it, a
  throttled key holds its submission slot for the entire generator
  invocation (which may include multiple throttle waits inside the
  client wrapper). Releasing the slot before the generator runs
  prevents cross-key starvation.

- **Why a client wrapper, not facade instrumentation?** The client
  layer is below `@catch_llm_exceptions`, so it sees `ProviderError`
  directly with `retry_after` already parsed. No need to propagate
  `retry_after` through `ModelRateLimitError`. The facade is untouched.

- **Why not remove all HTTP-layer retries?** The throttling loop only
  needs raw `429` signals. Keeping retries for `502` / `503` / `504`
  and connection failures preserves resilience without masking AIMD
  backoff triggers.

- **Why not a `ThrottleKey` / `build_throttle_keys` in the scheduler?**
  The scheduler only needs to know if a task is LLM-bound (for
  submission slot management). The client wrapper handles
  provider/model/domain resolution internally.

## Planned Follow-On

- **Exact submission pool sizing** — replace the sum-based heuristic
  with a `get_effective_throttle_capacity()` method that deduplicates
  by `(provider_name, model_id)` and uses the real `min()` cap.

PR-7 picks up the config/CLI auth schema rollout that was originally
scoped for PR-6, adding typed provider-specific auth objects
(`AnthropicAuth`, `OpenAIApiKeyAuth`) to `ModelProvider` with
backward-compatible `api_key` fallback.

PR-8 flips the default backend to native while retaining the bridge path.

PR-9 removes the LiteLLM dependency after the soak window.
