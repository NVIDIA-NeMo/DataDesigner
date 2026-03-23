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
`AsyncTaskScheduler` uses a one-way semaphore handoff for LLM-bound
tasks to prevent cross-key starvation.

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
| **Scheduler** | `AsyncTaskScheduler` | Dual-semaphore model with one-way handoff: acquires LLM-wait slot then releases submission slot for LLM-bound tasks (never reacquires). Prevents cross-key starvation while bounding live coroutines. No throttle logic. |

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

The acquire/release contract for each method:

- On entry: acquire a throttle slot (with `TimeoutError` → `ProviderError(kind=TIMEOUT)` normalization)
- On `ProviderError(kind=RATE_LIMIT)`: call `release_rate_limited` with `retry_after`
- On other `ProviderError`: call `release_failure`
- On any other `BaseException` (including `asyncio.CancelledError`): call `release_failure`
- On success: call `release_success`

**Cancellation safety:** `asyncio.CancelledError` is a `BaseException`,
not an `Exception`. The context managers catch `BaseException` (not
just `Exception`) so that cancelling an in-flight async request always
releases the throttle permit via `release_failure` before re-raising.
Without this, a single cancelled request would permanently reduce
available concurrency for that provider/model/domain. The same pattern
is applied to `_throttled_sync` for consistency, even though sync
callers are not subject to `asyncio.CancelledError`.

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
statuses for **async-mode** clients (which have the salvage queue and
AIMD feedback loop). **Sync-mode** clients keep `429` in the transport
retry list because the sync engine has no salvage queue — a surfaced
429 would simply drop the row via `_worker_error_callback`.

The mode-awareness is implemented in `HttpModelClient`: the sync path
calls `create_retry_transport(config, strip_rate_limit_codes=False)`
while the async path calls `create_retry_transport(config,
strip_rate_limit_codes=True)`. Non-rate-limit transient failures
(`502` / `503` / `504`, connection/transport errors) remain retried
in the shared HTTP layer for both modes.

**DRY:** The six methods share the same acquire/release pattern. A
private helper `_release_on_error` and paired context managers
`_throttled_sync` / `_athrottled` can centralize the logic:

```python
@contextlib.asynccontextmanager
async def _athrottled(self, domain: ThrottleDomain):
    try:
        await self._tm.acquire_async(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=domain,
        )
    except TimeoutError as exc:
        raise ProviderError(
            kind=ProviderErrorKind.TIMEOUT,
            message=str(exc),
            provider_name=self._provider_name,
            model_name=self._model_id,
        ) from exc
    exc_to_reraise: BaseException | None = None
    try:
        yield
    except ProviderError as exc:
        exc_to_reraise = exc
        self._release_on_provider_error(domain, exc)
    except BaseException as exc:
        exc_to_reraise = exc
        self._tm.release_failure(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=domain,
        )
    else:
        self._tm.release_success(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=domain,
        )
    if exc_to_reraise is not None:
        raise exc_to_reraise
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

**LiteLLM bridge scope:** The `ThrottledModelClient` wrapper is
applied to **all** clients returned by `create_model_client`,
including `LiteLLMBridgeClient`. However, the "first raw 429 reaches
AIMD" contract only holds for native adapters (`OpenAICompatibleClient`,
`AnthropicClient`), because LiteLLM's internal router can retry 429s
before the `ProviderError` reaches the wrapper. This means:

- **Native adapters**: Full AIMD accuracy. The transport layer does
  not retry 429 (PR-6 removes it), so the first 429 reaches the
  wrapper with `retry_after` intact.
- **LiteLLM bridge**: Best-effort AIMD. The bridge's internal retries
  may mask some 429s, so the wrapper sees fewer rate-limit signals
  than actually occurred. The AIMD state is still updated on the 429s
  that do surface, but the feedback loop is less precise.

This is acceptable for PR-6 because:

1. The bridge is the **fallback** path — native adapters are preferred
   for `openai` and `anthropic` provider types.
2. PR-8 flips the default to native, and PR-9 removes the bridge.
3. Wrapping the bridge is still better than not wrapping it — even
   imprecise AIMD is better than no concurrency control.

A test explicitly documents this limitation:

```python
def test_bridge_client_is_wrapped_with_throttle_manager():
    """LiteLLMBridgeClient is wrapped, but AIMD accuracy is best-effort
    because the bridge's internal router may retry 429s before the
    wrapper sees them. See architecture notes for scope."""
```

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

### 3. AIMD tuning via `RunConfig` / `ThrottleConfig`

The four AIMD tuning knobs are grouped into a dedicated
`ThrottleConfig` Pydantic model in `config/run_config.py`, and
`RunConfig` exposes them as a single nested field:

```python
class ThrottleConfig(ConfigBase):
    DEFAULT_REDUCE_FACTOR: ClassVar[float] = 0.75
    DEFAULT_ADDITIVE_INCREASE: ClassVar[int] = 1
    DEFAULT_SUCCESS_WINDOW: ClassVar[int] = 25
    DEFAULT_BLOCK_SECONDS: ClassVar[float] = 2.0

    reduce_factor: float = Field(default=DEFAULT_REDUCE_FACTOR, gt=0.0, lt=1.0)
    additive_increase: int = Field(default=DEFAULT_ADDITIVE_INCREASE, ge=1)
    success_window: int = Field(default=DEFAULT_SUCCESS_WINDOW, ge=1)
    block_seconds: float = Field(default=DEFAULT_BLOCK_SECONDS, gt=0.0)

class RunConfig(ConfigBase):
    # ... existing fields ...
    throttle: ThrottleConfig = Field(default_factory=ThrottleConfig)
```

`ThrottleConfig` is the single source of truth for AIMD defaults —
`ThrottleManager` accepts an optional `ThrottleConfig` object directly
(defaulting to `ThrottleConfig()` when `None`), eliminating duplicated
default constants. The `ClassVar` constants on `ThrottleConfig` are
also used in tests for assertion values.

`ThrottleConfig` is exported from `data_designer.config` alongside
`RunConfig`.

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
from the `ThrottleConfig`:

```python
throttle_config = (run_config or RunConfig()).throttle
throttle_manager = ThrottleManager(config=throttle_config)
```

### 4. Scheduler: dual-semaphore model for LLM tasks

The `AsyncTaskScheduler` does **not** acquire or release throttle
permits. That is the client wrapper's job. The scheduler's change is
a **dual-semaphore** model that separates task admission from
provider-wait capacity, preventing cross-key starvation without
removing the global task cap.

**New semaphore:** `_llm_wait_semaphore` — a separate semaphore that
LLM-bound tasks hold while executing the generator (including any
throttle waits inside the client wrapper). This bounds the number of
coroutines parked inside throttle waits, preventing unbounded memory
growth from scheduler fan-out.

**One-way handoff (no reacquire):** LLM-bound tasks acquire the
LLM-wait slot *before* releasing the submission slot (so the task is
never holding neither), then hold only the LLM-wait slot for the
duration of the generator. On completion, the task releases the
LLM-wait slot and does **not** reacquire the submission slot. The
`finally` bookkeeping (trace, `_in_flight`, `_dispatched`,
`in_flight_count`, wake) is all in-memory state mutation that does not
require holding the submission semaphore.

This avoids circular wait: both semaphores are only ever acquired in
one order (submission → LLM-wait), never the reverse.

**Dispatch pattern change in `_execute_task_inner`:**

**Before (current):**
```
acquire submission slot → execute generator → release submission slot
```

**After (LLM-bound):**
```
acquire submission slot →
  acquire LLM-wait slot         ← while still holding submission
  release submission slot       ← one-way: never reacquired
  execute generator (client wrapper handles throttle per HTTP call)
  release LLM-wait slot
  bookkeeping (no semaphore held)
```

**After (non-LLM):**
```
acquire submission slot → execute generator → release submission slot
```

The `finally` block uses two flags to track which semaphores are
actually held, so release is conditional on successful acquire. This
is critical for cancellation safety: if a task is cancelled while
awaiting `_llm_wait_semaphore.acquire()`, it never owned that permit,
and releasing it would over-increment the semaphore.

```python
holds_submission = True
holds_llm_wait = False
try:
    if is_llm_bound:
        await self._llm_wait_semaphore.acquire()
        holds_llm_wait = True
        self._submission_semaphore.release()
        holds_submission = False
    # ... execute generator ...
finally:
    if holds_llm_wait:
        self._llm_wait_semaphore.release()
    if holds_submission:
        self._submission_semaphore.release()
    # ... bookkeeping (trace, _in_flight, _dispatched, wake) ...
```

**Bound:** The total number of live coroutines is bounded by
`max_submitted_tasks + max_llm_wait_tasks`. During the generator
execution phase, every coroutine holds exactly one of the two
semaphores. After the generator completes, the `finally` block
releases whichever semaphore is held and then runs bookkeeping with
no semaphore held — this is safe because the bookkeeping is all
in-memory state mutation. The two semaphores have independent sizes:

- `_submission_semaphore`: bounds total admitted tasks (unchanged role)
- `_llm_wait_semaphore`: bounds coroutines executing LLM generators

```python
self._submission_semaphore = asyncio.Semaphore(max_submitted_tasks)
self._llm_wait_semaphore = asyncio.Semaphore(max_llm_wait_tasks)
```

`max_llm_wait_tasks` is derived from aggregate throttle capacity (see
section 5). This ensures the scheduler cannot spawn more parked
coroutines than the throttle system can eventually service.

**LLM-bound detection:** The scheduler needs to know which generators
are LLM-bound. Rather than using `isinstance` checks against the
class hierarchy (which is brittle for custom generators), the
`ColumnGenerator` base class exposes an `is_llm_bound` property:

```python
class ColumnGenerator(ConfigurableTask[TaskConfigT], ABC):
    @property
    def is_llm_bound(self) -> bool:
        """Whether this generator makes LLM/HTTP calls during generation."""
        return False
```

`ColumnGeneratorWithModelRegistry` overrides this to return `True`:

```python
class ColumnGeneratorWithModelRegistry(ColumnGenerator[TaskConfigT], ABC):
    @property
    def is_llm_bound(self) -> bool:
        return True
```

The scheduler builds its lookup from this property at init time:

```python
def _build_llm_bound_lookup(
    generators: dict[str, ColumnGenerator],
) -> dict[str, bool]:
    return {col: gen.is_llm_bound for col, gen in generators.items()}
```

This approach is inheritance-agnostic: custom generators that make
HTTP calls can override `is_llm_bound` to return `True` without
needing to subclass `ColumnGeneratorWithModelRegistry`. The scheduler
never inspects the class hierarchy.

**Error handling:** The scheduler's `except` block no longer needs to
distinguish `ModelRateLimitError` for throttle release — the client
wrapper already handled it. The scheduler only classifies retryability
for the salvage queue. The existing `_is_retryable` check is
unchanged.

**`TimeoutError` from throttle acquire:** The client wrapper's
`acquire_async` can raise a builtin `TimeoutError`. Because the
wrapper sits *below* `@catch_llm_exceptions`, a raw `TimeoutError`
would be caught by the decorator's generic `case _:` branch and
re-raised as a `DataDesignerError` — which is **not** in
`_RETRYABLE_MODEL_ERRORS`, causing silent row drops instead of
salvage retries.

To stay within the existing error contract, the `ThrottledModelClient`
wrapper catches `TimeoutError` from `acquire_async` / `acquire_sync`
and normalizes it into a `ProviderError(kind=TIMEOUT)`:

```python
try:
    await self._tm.acquire_async(...)
except TimeoutError as exc:
    raise ProviderError(
        kind=ProviderErrorKind.TIMEOUT,
        message=str(exc),
        provider_name=self._provider_name,
        model_name=self._model_id,
    ) from exc
```

This `ProviderError` flows through `@catch_llm_exceptions` →
`_raise_from_provider_error` → `ModelTimeoutError`, which is already
in `_RETRYABLE_MODEL_ERRORS`. No changes to `_is_retryable` are
needed — the existing retry path handles it correctly.

**No release on acquire timeout:** When `acquire_async` raises, no
throttle slot was acquired, so no `release_*` call is made. The
`_athrottled` context manager catches the `TimeoutError` *before*
entering the `try` body, so the `except` / `else` release branches
are never reached.

### 5. Semaphore sizing

Two semaphores, two sizing strategies:

**Submission semaphore** (`_submission_semaphore`): Unchanged role —
bounds total admitted tasks. The hardcoded `256` floor is extracted to
a named constant but the value is unchanged:

```python
MIN_SUBMITTED_TASKS: int = 256
```

The builder keeps the existing default:

```python
max_submitted_tasks = MIN_SUBMITTED_TASKS
```

This is a true cap. LLM-bound tasks temporarily release their slot
while parked in throttle waits, but the cap still governs how many
tasks can be admitted at any point in time.

**LLM-wait semaphore** (`_llm_wait_semaphore`): Bounds the number of
coroutines parked inside throttle waits. Sized from aggregate throttle
capacity so it cannot exceed what the throttle system can service:

`ModelRegistry` exposes a new public method:

```python
def get_aggregate_max_parallel_requests(self) -> int:
    """Sum of max_parallel_requests across all registered model configs."""
    return sum(
        mc.inference_parameters.max_parallel_requests
        for mc in self._model_configs.values()
    )
```

```python
LLM_WAIT_POOL_MULTIPLIER: int = 2
```

```python
aggregate = model_registry.get_aggregate_max_parallel_requests()
max_llm_wait_tasks = max(MIN_SUBMITTED_TASKS, LLM_WAIT_POOL_MULTIPLIER * aggregate)
```

**Why the multiplier?** A task may make multiple HTTP calls (correction
loops, tool calls), so the wait pool should be somewhat larger than
the raw throttle capacity. The `2x` multiplier is a heuristic — the
`ThrottleManager` enforces the real per-key limit regardless.

**Precision note:** Summing `max_parallel_requests` across all model
configs overstates the true capacity when aliases share a
`(provider_name, model_id)` key. However, the LLM-wait pool is an
upper bound on parked coroutines, not a concurrency target —
oversizing wastes a few coroutine slots but doesn't cause incorrect
behavior. The `ThrottleManager` enforces the real per-key limit.

### 6. Builder wiring

`_build_async` in `ColumnWiseDatasetBuilder` passes both semaphore
sizes to the `AsyncTaskScheduler`:

```python
registry = self._resource_provider.model_registry
aggregate = registry.get_aggregate_max_parallel_requests()

scheduler = AsyncTaskScheduler(
    ...,
    max_submitted_tasks=MIN_SUBMITTED_TASKS,
    max_llm_wait_tasks=max(MIN_SUBMITTED_TASKS, LLM_WAIT_POOL_MULTIPLIER * aggregate),
)
```

The scheduler no longer receives a `throttle_manager` parameter —
throttling is handled by the client wrapper inside each `ModelFacade`.

### 7. Rate-limit error integration with retry/salvage

PR-6 changes the retry boundary before a `ModelRateLimitError` occurs:

1. `RetryTransport` no longer retries `429`. The `RetryConfig`
   docstring has been updated to document this boundary (replacing the
   old "mirrors LiteLLM router" wording). It continues to retry
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

### 8. AIMD behavioral refinements

Three refinements to `ThrottleManager` improve AIMD behavior under
real-world conditions:

**Cascade dampening:** When many in-flight requests return 429
simultaneously (e.g. 32 requests all rate-limited), each calls
`release_rate_limited`. Without dampening, the limit drops
exponentially (`32 * 0.75^9 ≈ 2`), requiring minutes to recover.

The fix uses a `consecutive_429s` counter in `DomainThrottleState`.
Only the **first** 429 in a cascade (`consecutive_429s == 0` before
increment) applies the multiplicative decrease. Subsequent cascade
429s still decrement `in_flight`, increment `consecutive_429s`, set
`blocked_until`, and reset `success_streak`, but skip the limit
reduction. `release_success` resets `consecutive_429s` to 0.

Result: `32 → 24` on first 429, remaining cascade 429s just release
permits. Recovery starts from 24 instead of 1.

**Ceiling stabilization:** Without a ceiling, AIMD creates a
sawtooth pattern — concurrency climbs to `max_parallel_requests`,
hits a 429, drops, climbs again, repeats. The `rate_limit_ceiling`
field in `DomainThrottleState` tracks the highest observed stable
concurrency before a rate-limit. Combined with `ceiling_overshoot`
(default 10%), the additive increase phase caps at
`ceiling + max(1, floor(ceiling * overshoot))` instead of climbing
all the way to `effective_max`. This allows cautious probing above
the known server limit without triggering repeated 429 cascades.

The ceiling is set on the first 429 in a cascade and lowered (via
`min()`) on subsequent isolated 429s after recovery. It is never
raised — only lowered or preserved — ensuring the system converges
toward the server's actual limit.

**Early shutdown exclusion:** `ModelRateLimitError` exceptions are
expected AIMD behavior, not critical failures. The
`AsyncTaskScheduler._execute_task_inner` method now skips the
`_check_error_rate(success=False)` call when the exception is a
`ModelRateLimitError`, preventing initial 429 cascades from
triggering early shutdown.

### 9. Health check interaction

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
- `TimeoutError` from `acquire_async` is normalized to
  `ProviderError(kind=TIMEOUT)`, no release (acquire failed, no slot
  held)
- Image generation: `request.messages is None` → `IMAGE` domain;
  `request.messages` set → `CHAT` domain
- Cancellation safety: `asyncio.CancelledError` during in-flight async
  request releases throttle permit via `release_failure`; `in_flight`
  returns to 0
- Sync methods use `acquire_sync`
- E2E AIMD feedback loop: success → 429 halves limit → successes
  recover via additive increase (real `ThrottleManager`, aggressive
  tuning with `success_window=2`)
- E2E concurrent request bounding: 5 concurrent async calls with
  `max_parallel_requests=2` — peak `in_flight` never exceeds 2, all
  5 complete successfully

**`test_async_scheduler.py`**:

- One-way handoff: LLM-wait slot acquired before submission slot
  released; submission slot never reacquired (no circular wait)
- Submission slot held (no release/reacquire) for non-LLM generators
- Deadlock regression: `max_submitted_tasks=1`, `max_llm_wait_tasks=1`,
  two ready LLM tasks — completes without deadlock
- Cancellation safety: scheduler cancelled while LLM-bound tasks are
  blocked mid-flight — semaphore slots are recovered (real cancellation
  via `task.cancel()`, not run-to-completion)
- `is_llm_bound` property drives lookup (no isinstance)
- Rate-limit errors do not trigger early shutdown
  (`ModelRateLimitError` excluded from error rate check)
- Existing tests pass unchanged

**`test_model_registry.py`**:

- `get_aggregate_max_parallel_requests` returns correct sum

**`test_client_factory.py`** (or existing factory tests):

- `create_model_client` with `throttle_manager` returns
  `ThrottledModelClient` wrapping the inner client
- `create_model_client` without `throttle_manager` returns inner
  client directly

**`test_retry.py`** (updated):

- `test_retry_config_defaults` (renamed from
  `test_retry_config_defaults_match_litellm_router`): asserts
  `retryable_status_codes == {502, 503, 504}` (429 excluded)
- `502` / `503` / `504` remain transport-retryable

## What Does NOT Change

1. **`ThrottleManager` public method signatures** — the existing
   `try_acquire`, `release_success`, `release_rate_limited`,
   `release_failure`, `acquire_async`, `acquire_sync` methods are
   unchanged. The constructor was updated to accept an optional
   `ThrottleConfig` object instead of individual tuning parameters,
   and the module was renamed from `throttle.py` to
   `throttle_manager.py` for discoverability. Internal AIMD behavior
   was refined (see "AIMD behavioral refinements" below).
   `try_acquire` now returns `CAPACITY_POLL_INTERVAL` (50ms) instead
   of `default_block_seconds` (2s) when at capacity but not
   rate-limited, so callers poll responsively instead of waiting the
   full cooldown duration when a slot could free in milliseconds.
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
6. **Generator code** — generators are unaware of throttling. The new
   `is_llm_bound` property on `ColumnGenerator` is a scheduling hint,
   not throttling logic — generators do not acquire or release permits.
7. **Non-rate-limit transport retries** — `502` / `503` / `504` and
   connection failures still use the shared HTTP retry layer. PR-6
   only removes `429` from that retry set.

## Files Touched

| File | Change |
| --- | --- |
| `models/clients/throttle.py` → `models/clients/throttle_manager.py` (rename) | Renamed for discoverability; constructor updated to accept `ThrottleConfig`; cascade dampening (`consecutive_429s`), ceiling stabilization (`rate_limit_ceiling`, `ceiling_overshoot`), refined logging |
| `models/clients/throttled.py` (new) | `ThrottledModelClient` wrapper with cancellation-safe `_throttled_sync` / `_athrottled` context managers |
| `models/clients/retry.py` | `create_retry_transport` gains `strip_rate_limit_codes` kwarg (default `True`); async-mode strips 429 for AIMD, sync-mode keeps 429 for transport-level retry |
| `models/clients/adapters/http_model_client.py` | Sync transport: `strip_rate_limit_codes=False`; async transport: `strip_rate_limit_codes=True` |
| `models/clients/factory.py` | Accept optional `throttle_manager`; wrap inner client with `ThrottledModelClient` when provided; docstring documents registration ordering invariant |
| `models/factory.py` | Forward `ThrottleManager` to `model_facade_factory` closure → `create_model_client`; accept `run_config` parameter; construct `ThrottleManager` from `run_config.throttle` (`ThrottleConfig`); remove TODO comment |
| `models/registry.py` | Add `get_aggregate_max_parallel_requests()` public method |
| `config/run_config.py` | Add `ThrottleConfig` sub-model with `ClassVar` defaults (`reduce_factor=0.75`, `additive_increase=1`, `success_window=25`, `block_seconds=2.0`); add `RunConfig.throttle: ThrottleConfig` field. `block_seconds` now only applies to rate-limit cooldowns; capacity waits use `CAPACITY_POLL_INTERVAL` (50ms) |
| `resources/resource_provider.py` | Forward `run_config` to `create_model_registry` |
| `column_generators/generators/base.py` | Add `is_llm_bound` property to `ColumnGenerator` (default `False`), override to `True` in `ColumnGeneratorWithModelRegistry` |
| `dataset_builders/async_scheduler.py` | Add `_llm_wait_semaphore`, one-way semaphore handoff for LLM tasks, `MIN_SUBMITTED_TASKS` / `LLM_WAIT_POOL_MULTIPLIER` constants; exclude `ModelRateLimitError` from early shutdown error rate |
| `dataset_builders/column_wise_builder.py` | Forward `max_submitted_tasks` and `max_llm_wait_tasks` to scheduler |
| `tests/.../test_throttle.py` → `tests/.../test_throttle_manager.py` (rename) | Renamed to match source module rename; updated for `ThrottleConfig`-based construction, `reduce_factor=0.75` assertions, cascade dampening (`test_cascade_only_first_429_reduces_limit`), ceiling stabilization tests |
| `tests/.../test_throttled_model_client.py` (new) | Throttle wrapper unit tests + cancellation safety + E2E AIMD tests |
| `tests/.../test_retry.py` | Updated assertion and test name for new `retryable_status_codes` default |
| `tests/.../test_async_scheduler.py` | Dual-semaphore tests + rewritten cancellation test with real task cancellation + rate-limit early shutdown exclusion test |
| `tests/.../test_model_registry.py` | Test for `get_aggregate_max_parallel_requests` |
| `tests/.../test_client_factory.py` | Test wrapper wiring in `create_model_client` |

## Test Coverage

| Category | Tests |
| --- | --- |
| Client wrapper: acquire/release on success | Chat, embedding, image (diffusion + chat-backed) all call `release_success` |
| Client wrapper: rate limit | `ProviderError(kind=RATE_LIMIT)` calls `release_rate_limited` with `retry_after` |
| Client wrapper: failure | Non-rate-limit errors call `release_failure` |
| Client wrapper: acquire timeout | `TimeoutError` from `acquire_async` normalized to `ProviderError(kind=TIMEOUT)`, no release (no slot held) |
| Client wrapper: image domain routing | `request.messages is None` → `IMAGE`; `request.messages` set → `CHAT` |
| Client wrapper: sync path | Sync methods use `acquire_sync` / `release_*` |
| Client wrapper: cancellation safety | `asyncio.CancelledError` during in-flight async request releases throttle permit via `release_failure`; `in_flight` returns to 0 |
| Client wrapper: E2E AIMD loop | Success → 429 halves limit → successes recover via additive increase (real `ThrottleManager`, `success_window=2`) |
| Client wrapper: E2E concurrency | 5 concurrent async calls with `max_parallel_requests=2` — peak `in_flight` ≤ 2, all 5 complete |
| Transport retry boundary | Async-mode: `429` stripped from transport retryable statuses for AIMD feedback; sync-mode: `429` kept for transport-level retry (no salvage queue). `strip_rate_limit_codes` kwarg on `create_retry_transport` controls behavior. `502` / `503` / `504` remain retried in both modes |
| Scheduler: one-way handoff for LLM tasks | LLM-wait acquired before submission released; submission never reacquired (no circular wait) |
| Scheduler: non-LLM tasks | Submission slot held during generator execution (no release/reacquire) |
| Scheduler: deadlock regression | `max_submitted_tasks=1`, `max_llm_wait_tasks=1`, two ready LLM tasks — completes without deadlock |
| Scheduler: cancellation safety | Scheduler cancelled while LLM-bound tasks are in-flight — semaphore slots are recovered (real cancellation via `task.cancel()`, not run-to-completion) |
| Scheduler: acquire timeout retryable | Throttle acquire timeout surfaces as `ModelTimeoutError` (via `ProviderError(kind=TIMEOUT)` → `@catch_llm_exceptions`), already in `_RETRYABLE_MODEL_ERRORS` |
| Scheduler: backward compat | Existing tests pass unchanged |
| Submission pool sizing | `MIN_SUBMITTED_TASKS` (unchanged true cap) |
| LLM-wait pool sizing | `max(MIN_SUBMITTED_TASKS, LLM_WAIT_POOL_MULTIPLIER * aggregate)` |
| AIMD: cascade dampening | Only first 429 in cascade reduces limit; subsequent cascade 429s release permits without reduction (`test_cascade_only_first_429_reduces_limit`) |
| AIMD: ceiling stabilization | After 429, additive increase caps at `ceiling + overshoot` instead of `effective_max` (`test_ceiling_stabilization_with_overshoot`) |
| AIMD: ceiling lowering | Repeated isolated 429s after recovery lower the ceiling (`test_ceiling_lowers_on_repeated_429_after_recovery`) |
| AIMD: ceiling at effective_max | Ceiling does not restrict recovery when effective_max is small (`test_ceiling_does_not_restrict_when_at_effective_max`) |
| AIMD: reduce_factor=0.75 | Default reduction is 25% (not 50%); assertions updated across all throttle tests |
| Scheduler: rate-limit exclusion | `ModelRateLimitError` does not count toward early shutdown error rate (`test_rate_limit_errors_do_not_trigger_early_shutdown`) |
| RunConfig → AIMD wiring | `RunConfig.throttle` (`ThrottleConfig`) forwarded to `ThrottleManager` constructor; `ThrottleConfig` is single source of truth for defaults |
| Factory wiring | `create_model_client` with `throttle_manager` returns `ThrottledModelClient` |
| Factory wiring: bridge scope | `LiteLLMBridgeClient` is wrapped (best-effort AIMD), with documented limitation |

## Design Rationale

Key points:

- **Why client-level, not scheduler-level throttling?** The scheduler
  sees one task = one throttle permit, but generators can make multiple
  HTTP calls (correction loops, tool calls, custom columns). The client
  wrapper sees every actual HTTP request.

- **Why the dual-semaphore model?** Without it, a throttled key holds
  its submission slot for the entire generator invocation (which may
  include multiple throttle waits inside the client wrapper).
  Releasing the submission slot before the generator runs prevents
  cross-key starvation. The separate LLM-wait semaphore bounds the
  number of coroutines executing LLM generators. The handoff is
  **one-way** (submission → LLM-wait, never reversed) to avoid
  circular wait. The `finally` bookkeeping is all in-memory state
  mutation that does not require holding either semaphore, so the
  LLM-bound path simply releases the LLM-wait slot and proceeds.
  Total live coroutine count is bounded by
  `max_submitted_tasks + max_llm_wait_tasks`.

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

- **Exact LLM-wait pool sizing** — replace the sum-based heuristic
  with a `get_effective_throttle_capacity()` method that deduplicates
  by `(provider_name, model_id)` and uses the real `min()` cap.

PR-7 picks up the config/CLI auth schema rollout that was originally
scoped for PR-6, adding typed provider-specific auth objects
(`AnthropicAuth`, `OpenAIApiKeyAuth`) to `ModelProvider` with
backward-compatible `api_key` fallback.

PR-8 flips the default backend to native while retaining the bridge path.

PR-9 removes the LiteLLM dependency after the soak window.
