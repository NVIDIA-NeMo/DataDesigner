---

## date: 2026-03-24
authors:
  - nmulepati

# Model Facade Overhaul PR-7 Architecture Notes

This document captures the architecture intent and implementation plan
for PR-7 from `plans/343/model-facade-overhaul-plan-step-1.md`.

PR-7 removes the LiteLLM dependency and bridge path entirely. With
PR-6 merged, all three predefined providers (NVIDIA, OpenAI,
OpenRouter) route to native HTTP adapters (`OpenAICompatibleClient`,
`AnthropicClient`) by default. The LiteLLM bridge was retained as a
fallback for unknown `provider_type` values and as an opt-in via the
`DATA_DESIGNER_MODEL_BACKEND=litellm_bridge` environment variable.
Neither path is needed: users with custom providers should configure
`provider_type` as `"openai"` or `"anthropic"` (the two API formats
the industry has converged on), and the env-var escape hatch was a
migration aid that is no longer necessary.

## Goal

Remove all LiteLLM runtime code, test code, and the `litellm` package
dependency. After this PR, `litellm` is not imported, installed, or
referenced anywhere in the runtime or test codebase.

## Problem

LiteLLM remains in the dependency tree and is eagerly imported at
startup via `apply_litellm_patches()` in `models/factory.py`, even
though no default code path uses it. This causes:

- **Startup cost:** `litellm` is a heavy import (~300ms+) that loads
  on every `create_model_registry()` call regardless of backend.
- **Dependency surface:** `litellm` pins a narrow version range
  (`>=1.77.0,<1.80.12`) that constrains upgrades and introduces
  transitive dependency conflicts.
- **Dead code:** The bridge adapter, LiteLLM overrides module,
  `CustomRouter`, `ThreadSafeCache`, and LiteLLM exception match arms
  in `errors.py` are unreachable in the default flow.
- **Test maintenance:** Bridge-specific tests and LiteLLM override
  tests exercise code that no production path uses.

## What Changes

### 1. Delete `litellm_bridge.py`

Remove `clients/adapters/litellm_bridge.py` — the bridge adapter that
wraps LiteLLM's router behind the `ModelClient` protocol.

### 2. Delete `litellm_overrides.py`

Remove `models/litellm_overrides.py` — the module containing
`ThreadSafeCache`, `CustomRouter`, `LiteLLMRouterDefaultKwargs`,
`patch_image_url_list_item`, and `apply_litellm_patches`. These were
LiteLLM-specific workarounds (thread-safe cache, exponential backoff
override, image URL schema patch) that are no longer needed.

### 3. Remove `apply_litellm_patches()` from `models/factory.py`

The `create_model_registry` factory unconditionally calls
`apply_litellm_patches()`. This import and call are removed. The
factory retains its existing structure — it just no longer has a
LiteLLM initialization step.

### 4. Remove LiteLLM fallback from `clients/factory.py`

The `create_model_client` function currently has four routing paths:

1. `DATA_DESIGNER_MODEL_BACKEND=litellm_bridge` → bridge (removed)
2. `provider_type == "openai"` → `OpenAICompatibleClient` (kept)
3. `provider_type == "anthropic"` → `AnthropicClient` (kept)
4. Unknown `provider_type` → bridge fallback (removed)

After this PR, unknown `provider_type` values raise a
`ValueError` with a clear message listing supported types. The
`DATA_DESIGNER_MODEL_BACKEND` env-var, `_BACKEND_ENV_VAR`,
`_BACKEND_BRIDGE` constants, and `_create_bridge_client` helper are
all removed.

The imports of `LiteLLMBridgeClient`, `CustomRouter`,
`LiteLLMRouterDefaultKwargs`, and `lazy` (for `lazy.litellm`) are
removed from the factory module.

### 5. Remove LiteLLM match arms from `models/errors.py`

The `handle_llm_exceptions` function has a `match` statement with
two groups of cases:

1. `ProviderError` — canonical errors from native adapters (kept)
2. `lazy.litellm.exceptions.*` — LiteLLM-specific errors labeled
   "safety net during bridge period" (removed)

After removal, the match statement handles: `ProviderError`,
`GenerationValidationFailureError`, `DataDesignerError`, and the
generic `case _:` fallback. The `DownstreamLLMExceptionMessageParser`
class is also removed — it only parsed LiteLLM exception types
(`BadRequestError`, `ContextWindowExceededError`, `APIError`).

The `import litellm` in the `TYPE_CHECKING` block and the
`import data_designer.lazy_heavy_imports as lazy` are removed from
this module.

**Ported behavior:** `DownstreamLLMExceptionMessageParser` had one
piece of logic that the native `_raise_from_provider_error` path
lacked: extracting the specific token-count detail from OpenAI-style
context window errors (e.g. "This model's maximum context length is
32768 tokens"). This is ported to a new private helper
`_extract_context_window_detail` called from the
`CONTEXT_WINDOW_EXCEEDED` branch of `_raise_from_provider_error`.
The `ProviderError.message` field carries the provider's response
body, which contains the same text the old LiteLLM exception did.

**403 behavioral delta:** The old LiteLLM path treated
`"Error code: 403"` in the exception string as
`ModelAuthenticationError`. The native path maps HTTP 403 to
`ProviderErrorKind.PERMISSION_DENIED` → `ModelPermissionDeniedError`.
This is more correct — 403 means "forbidden/permission denied", not
"bad credentials" (that's 401). The old behavior was a LiteLLM-era
workaround. No change is needed.

### 6. Remove `litellm` from `lazy_heavy_imports.py`

The `"litellm": "litellm"` entry is removed from `_LAZY_IMPORTS`.

### 7. Remove `litellm` from `pyproject.toml`

The `litellm>=1.77.0,<1.80.12` line is removed from the engine
package's runtime dependencies.

### 8. Clean up `adapters/__init__.py`

Remove `LiteLLMBridgeClient` and `LiteLLMRouter` from the
`__init__.py` exports.

### 9. Update `async_concurrency.py` docstring

The module docstring references LiteLLM as the reason for the
singleton event loop. The rationale is updated to reference
`httpx.AsyncClient` (the actual async-stateful dependency now).

### 10. Update `README.md` and docstrings

- `packages/data-designer-engine/README.md`: Remove "LLM integration
  via litellm" reference.
- `column_configs.py`: Update LLMTextColumnConfig docstring to remove
  "via LiteLLM" reference.

### 11. Remove `flatten_extra_body` from `TransportKwargs.from_request`

The `flatten_extra_body` parameter on `TransportKwargs.from_request`
existed solely for the LiteLLM bridge, which needed `extra_body`
preserved as a nested dict rather than merged into the top-level body.
All native adapters use the default (`True` — merge into top level).
The parameter, its `False` branch, and two tests exercising the
non-flatten path are removed.

### 12. Clean up stale LiteLLM references in docstrings and comments

All remaining LiteLLM references in docstrings and comments are
updated across the codebase:

- `models/factory.py`: "Heavy dependencies (litellm, httpx)" → httpx
- `resources/resource_provider.py`: "heavy dependencies like litellm"
  → httpx
- `clients/errors.py`: "stringified LiteLLM exception" → "stringified
  provider exception"
- `clients/types.py`: Remove LiteLLM references from
  `TransportKwargs` docstring
- `clients/parsing.py`: "LiteLLM-normalized fallback" → "legacy
  fallback"
- `async_concurrency.py`: "libraries (like LiteLLM)" →
  "httpx.AsyncClient"
- `AGENTS.md`: "via LiteLLM" → "via native HTTP adapters"

### 13. Update benchmark script

`scripts/benchmarks/benchmark_engine_v2.py` patches
`CustomRouter.completion` / `acompletion` for simulated LLM
responses. This is updated to patch the native
`OpenAICompatibleClient` instead.

### 14. Delete test files

- `tests/engine/models/clients/test_litellm_bridge.py` — bridge
  adapter tests
- `tests/engine/models/test_litellm_overrides.py` — override/patch
  tests

### 15. Update test files

- `tests/engine/models/clients/conftest.py` — remove `mock_router`
  and `bridge_client` fixtures (only used by bridge tests); retain
  shared HTTP mock helpers.
- `tests/engine/models/clients/test_factory.py` — remove bridge
  fallback tests (`test_unknown_provider_creates_bridge_client`,
  `test_bridge_env_override_forces_bridge_for_openai_provider`,
  `test_bridge_env_override_forces_bridge_for_anthropic_provider`,
  `test_bridge_client_is_wrapped_with_throttle_manager`); add test
  for unknown provider raising `ValueError`.
- `tests/engine/models/test_model_registry.py` — remove
  `apply_litellm_patches` mock from `test_create_model_registry`.
- `tests/engine/models/test_model_errors.py` — remove all test cases
  that construct LiteLLM exception types. Add parametrized test cases
  for every `ProviderErrorKind` value (`AUTHENTICATION`,
  `API_CONNECTION`, `TIMEOUT`, `NOT_FOUND`, `INTERNAL_SERVER`,
  `UNPROCESSABLE_ENTITY`, `API_ERROR`, multimodal `BAD_REQUEST`
  variant) to ensure full coverage of the native error path. Add
  dedicated tests for `_extract_context_window_detail` (with and
  without OpenAI-style detail in the error message).
- `tests/engine/models/clients/test_parsing.py` — remove two
  `flatten_extra_body=False` tests that exercised the removed
  parameter.
- `tests/engine/models/conftest.py` — update `stub_model_client`
  fixture docstring to remove "without a real LiteLLM router"
  reference.

## What Does NOT Change

1. **`ModelFacade`** — untouched. The facade delegates to
   `ModelClient` and is unaware of which adapter backs it.
2. **`ModelClient` protocol** — unchanged. Native adapters already
   implement it.
3. **`ProviderError` / `ProviderErrorKind`** — the canonical error
   model introduced in PR-2 is unchanged. It was always the target
   error type; the LiteLLM match arms were a bridge-period safety net.
4. **`ThrottledModelClient`** — unchanged. It wraps any `ModelClient`.
5. **`RetryTransport`** — unchanged.
6. **`ThrottleManager`** — unchanged.
7. **Native adapters** (`OpenAICompatibleClient`, `AnthropicClient`)
   — unchanged.
8. **`SecretResolver`** — unchanged.
9. **`RunConfig` / `ThrottleConfig`** — unchanged.

## Files Touched

| File | Change |
| --- | --- |
| `clients/adapters/litellm_bridge.py` | **Deleted** |
| `models/litellm_overrides.py` | **Deleted** |
| `clients/adapters/__init__.py` | Remove `LiteLLMBridgeClient`, `LiteLLMRouter` exports |
| `clients/factory.py` | Remove bridge fallback, env-var support, `_create_bridge_client`; raise `ValueError` for unknown `provider_type` |
| `models/factory.py` | Remove `apply_litellm_patches()` call and import |
| `models/errors.py` | Remove LiteLLM exception match arms and `DownstreamLLMExceptionMessageParser`; port context window detail extraction to `_extract_context_window_detail` |
| `clients/types.py` | Remove `flatten_extra_body` parameter from `TransportKwargs.from_request` |
| `clients/parsing.py` | Update "LiteLLM-normalized fallback" → "legacy fallback" in docstring |
| `clients/errors.py` | Update "stringified LiteLLM exception" → "stringified provider exception" in docstring |
| `resources/resource_provider.py` | Remove LiteLLM reference from docstring |
| `lazy_heavy_imports.py` | Remove `"litellm"` entry |
| `engine/pyproject.toml` | Remove `litellm` runtime dependency |
| `engine/README.md` | Remove LiteLLM reference |
| `config/column_configs.py` | Update docstring |
| `dataset_builders/utils/async_concurrency.py` | Update docstring |
| `scripts/benchmarks/benchmark_engine_v2.py` | Patch native client instead of `CustomRouter` |
| `tests/.../test_litellm_bridge.py` | **Deleted** |
| `tests/.../test_litellm_overrides.py` | **Deleted** |
| `tests/.../clients/conftest.py` | Remove bridge fixtures |
| `tests/.../clients/test_factory.py` | Remove bridge tests; add unknown-provider `ValueError` test |
| `tests/.../test_model_registry.py` | Remove `apply_litellm_patches` mock |
| `tests/.../test_model_errors.py` | Remove LiteLLM exception test cases; add full `ProviderErrorKind` coverage and context window detail tests |
| `tests/.../clients/test_parsing.py` | Remove `flatten_extra_body=False` tests |
| `tests/.../conftest.py` | Update docstring |
| `AGENTS.md` | Remove "via LiteLLM" references |

## Risk Assessment

**Low risk.** This PR only removes dead code paths. The native
adapters have been the default since PR-3/PR-4 and have been
exercised in production through PR-5 and PR-6. The bridge path was
a safety net that is no longer needed.

The only behavioral change is that unknown `provider_type` values
now raise a `ValueError` instead of silently falling back to the
LiteLLM bridge. This is intentional — users should explicitly
configure their provider type.

## Migration Impact

Users who were explicitly setting
`DATA_DESIGNER_MODEL_BACKEND=litellm_bridge` will see a startup
error. The fix is to remove the env var — native adapters are the
only path. Users with custom `provider_type` values that aren't
`"openai"` or `"anthropic"` will see a `ValueError` at client
creation time. The fix is to set `provider_type` to whichever API
format their provider uses (almost always `"openai"`).
