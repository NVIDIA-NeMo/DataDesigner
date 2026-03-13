---
date: 2026-03-13
authors:
  - nmulepati
---

# Model Facade Overhaul PR-4 Architecture Notes

This document captures the architecture intent for PR-4 from
`plans/343/model-facade-overhaul-plan-step-1.md`.

## Goal

Introduce a native HTTP adapter for the Anthropic Messages API
(`AnthropicClient`) with explicit capability gating.  After this PR, the
client factory routes `provider_type="anthropic"` to the native adapter.
Unknown provider types continue through `LiteLLMBridgeClient`.

## What Changes

### 1. Anthropic adapter (`clients/adapters/anthropic.py`)

`AnthropicClient` implements the `ModelClient` protocol using `httpx` with
`RetryTransport`, following the same structural patterns established by
`OpenAICompatibleClient` in PR-3.

Route:

- `POST /v1/messages` — chat completion and tool use

Capabilities:

| Capability | Supported |
|---|---|
| Chat completion | Yes |
| Tool calls | Yes |
| Embeddings | No — raises `ProviderError(kind=UNSUPPORTED_CAPABILITY)` |
| Image generation | No — raises `ProviderError(kind=UNSUPPORTED_CAPABILITY)` |

#### Request mapping (canonical -> Anthropic)

`_build_anthropic_payload()` converts a canonical `ChatCompletionRequest`
into an Anthropic Messages API payload:

- **System messages** — extracted from the `messages` array and joined into
  a top-level `system` string parameter (Anthropic does not accept system
  messages inline).  Multiple system messages are concatenated with double
  newlines.
- **`max_tokens`** — required by Anthropic; defaults to `4096` when the
  canonical request omits it.
- **`stop`** — mapped to `stop_sequences` (Anthropic's field name).  A bare
  string is wrapped in a single-element list.
- **`tools`** — forwarded directly (Anthropic's tool schema is compatible).
- **`extra_body`** — merged into the payload via `TransportKwargs`, with
  `stop` and `max_tokens` excluded from the forwarding set since they are
  handled explicitly.

#### Image content block translation

`MultiModalContext.get_contexts()` in the config layer emits image content
blocks in OpenAI format (`type: "image_url"`).  The adapter translates
these to Anthropic's `image` block format in `_translate_message_content()`,
which is called for every non-system message during payload building.

Three input shapes are handled:

| OpenAI shape | Anthropic output |
|---|---|
| `{"image_url": {"url": "data:image/png;base64,..."}}` | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}` |
| `{"image_url": "https://example.com/img.png"}` | `{"type": "image", "source": {"type": "url", "url": "https://..."}}` |
| `{"image_url": "data:image/jpeg;base64,..."}` | `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}` |

Non-image content blocks (text, custom types) pass through unchanged.
String content (non-list) also passes through unchanged.

This keeps the config layer provider-agnostic — it always emits OpenAI-format
blocks, and adapters own provider-specific translation.

#### Response parsing (Anthropic -> canonical)

`_parse_anthropic_response()` iterates over Anthropic's `content` blocks
and maps them to the canonical `AssistantMessage`:

| Anthropic block | Canonical field |
|---|---|
| `type=text` | `AssistantMessage.content` (joined with newlines) |
| `type=tool_use` | `AssistantMessage.tool_calls` — `id`, `name`, `input` serialized as `arguments_json` |
| `type=thinking` | `AssistantMessage.reasoning_content` (joined with newlines) |

Usage is extracted via the shared `extract_usage()` helper from
`parsing.py`, which handles the Anthropic field names (`input_tokens`,
`output_tokens`) directly.

### 2. Authentication headers

Anthropic uses a different auth scheme from OpenAI-compatible providers:

| Header | Value |
|---|---|
| `x-api-key` | Resolved API key (omitted when key is `None`) |
| `anthropic-version` | `2023-06-01` (always present) |
| `Content-Type` | `application/json` (always present) |

Notably, the `Authorization: Bearer` header used by OpenAI-compatible
providers is **not** sent.  This is implemented in `_build_headers()`.

### 3. Shared infrastructure reuse

The adapter reuses all shared infrastructure from PR-3:

- `RetryConfig` + `create_retry_transport` — same retry policy and transport
- `TransportKwargs` — same `extra_body`/`extra_headers`/timeout extraction
- `map_http_error_to_provider_error` — same HTTP status -> `ProviderError` mapping
- `infer_error_kind_from_exception` — same transport exception classification
- Lazy httpx client pattern with double-checked locking
- `close()`/`aclose()` with reference nulling (same lifecycle contract)
- Connection pool sizing policy (`max(32, 2*max_parallel)` / `max(16, max_parallel)`)

### 4. Client factory routing (`clients/factory.py`)

`create_model_client` now routes based on provider type with three
sequential early returns:

1. If `DATA_DESIGNER_MODEL_BACKEND=litellm_bridge` → always `LiteLLMBridgeClient`
2. If `provider_type == "openai"` → `OpenAICompatibleClient`
3. If `provider_type == "anthropic"` → `AnthropicClient`
4. Otherwise → `LiteLLMBridgeClient` (fallback for unknown providers)

The previous test `test_non_openai_provider_creates_bridge_client` used
an Anthropic provider fixture to verify bridge fallback.  This was replaced
with:

- `test_anthropic_provider_creates_native_client` — Anthropic now routes
  to `AnthropicClient`
- `test_anthropic_provider_type_case_insensitive` — case-insensitive matching
- `test_unknown_provider_creates_bridge_client` — unknown `provider_type`
  ("custom") still falls through to bridge

## What Does NOT Change

1. `ModelFacade` public method signatures — callers see the same API.
2. `OpenAICompatibleClient` — no changes to the OpenAI adapter.
3. `ThrottleManager` — standalone resource, unchanged.  Anthropic adapters
   will be throttled by the same `ThrottleManager` once the
   `AsyncTaskScheduler` (plan 346) is wired.
4. Error boundaries — `@catch_llm_exceptions` / `@acatch_llm_exceptions`
   decorators and `DataDesignerError` subclass hierarchy remain stable.
5. MCP tool-loop behavior — `ModelFacade` continues to orchestrate tool
   turns using canonical `ToolCall` objects regardless of provider.
6. Usage accounting — token tracking works unchanged via canonical `Usage`.
7. `LiteLLMBridgeClient` — retained for rollback and unknown providers.

## Design Decisions

### Why a separate adapter instead of extending OpenAICompatibleClient

Anthropic's API diverges from the OpenAI shape in several ways that would
make subclassing awkward:

1. **Auth header scheme** — `x-api-key` vs `Authorization: Bearer`
2. **System message handling** — top-level `system` param vs inline messages
3. **Response shape** — content blocks vs `choices[0].message`
4. **Required fields** — `max_tokens` is mandatory in Anthropic
5. **Field name differences** — `stop_sequences` vs `stop`

A separate adapter keeps each provider's concerns isolated and avoids
conditional branching in shared code paths.

### Why module-level helpers instead of methods

`_build_anthropic_payload()` and `_parse_anthropic_response()` are
module-level functions rather than class methods.  This follows the same
pattern as `OpenAICompatibleClient` and keeps the functions pure (no `self`
dependency), which simplifies testing and reasoning.

### Why `_parse_json_body` and `_wrap_transport_error` are duplicated

These two small functions are duplicated from `openai_compatible.py` rather
than extracted to a shared module.  They are three-line wrappers around
`ProviderError` construction, and sharing them would add an import
dependency between adapters or require a new shared module for minimal
benefit.  If a third adapter is added, extraction becomes justified.

## Files Touched

| File | Change |
|---|---|
| `clients/adapters/anthropic.py` | New — native Anthropic adapter |
| `clients/factory.py` | Route `provider_type=anthropic` to native adapter |
| `tests/.../test_anthropic.py` | New — 38 tests covering all adapter behavior |
| `tests/.../test_factory.py` | Updated — Anthropic routes to native; unknown provider routes to bridge |

## Test Coverage

| Category | Tests |
|---|---|
| Image block translation | data URI dict, URL string, data URI string, non-image passthrough, string content passthrough |
| Text content mapping | sync + async |
| Tool use block normalization | `tool_use` -> `ToolCall` with JSON-serialized arguments |
| Thinking block mapping | `thinking` -> `reasoning_content` |
| System message extraction | single and multiple system messages |
| `max_tokens` handling | default (4096) and explicit forwarding |
| `stop` -> `stop_sequences` | string and list variants |
| Tool forwarding | tools array passed through |
| Empty content | `None` content, no tool calls |
| Auth headers | `x-api-key` present/absent, `anthropic-version` always set, no `Authorization` |
| Extra headers | merged with adapter defaults |
| HTTP error mapping | 429, 401, 403, 404, 500 -> correct `ProviderErrorKind` |
| Transport errors | `TimeoutError` -> `ProviderErrorKind.TIMEOUT` |
| Unsupported capabilities | embeddings and image generation (sync + async) |
| Lifecycle | `close`/`aclose` delegation, no-op when no client |
| Capability flags | `supports_chat_completion=True`, others `False` |

## Planned Follow-On

PR-5 introduces the config/CLI auth schema rollout, adding typed
provider-specific auth objects (`AnthropicAuth`, `OpenAIApiKeyAuth`) to
`ModelProvider` with backward-compatible `api_key` fallback.

The `AsyncTaskScheduler` (plan 346) will call `ThrottleManager` for
Anthropic models using the same acquire/release pattern as OpenAI models,
since throttle management is provider-agnostic at the orchestration layer.
