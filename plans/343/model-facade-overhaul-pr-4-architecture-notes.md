---
date: 2026-03-13
updated: 2026-03-17
authors:
  - nmulepati
---

# Model Facade Overhaul PR-4 Architecture Notes

This document captures the architecture intent and current implementation
state for PR-4 from `plans/343/model-facade-overhaul-plan-step-1.md`.

The branch has evolved beyond the initial Anthropic adapter merge. These
notes reflect the current design after the follow-on refactors landed on the
branch.

## Goal

Introduce a native HTTP adapter for the Anthropic Messages API
(`AnthropicClient`) with explicit capability gating. After this PR, the
client factory routes `provider_type="anthropic"` to the native adapter.
Unknown provider types continue through `LiteLLMBridgeClient`.

The branch also now shares the common native `httpx` transport and lifecycle
machinery between `AnthropicClient` and `OpenAICompatibleClient` via a new
internal base class, `HttpModelClient`.

## What Changes

### 1. Anthropic adapter (`clients/adapters/anthropic.py`)

`AnthropicClient` is now a thin provider-specific adapter layered on top of
`HttpModelClient`. It owns Anthropic-specific capability flags, payload
construction, auth headers, and unsupported-operation behavior, while the
shared HTTP transport/lifecycle mechanics live in the shared base.

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

`build_anthropic_payload()` in `anthropic_translation.py` converts a canonical
`ChatCompletionRequest` into an Anthropic Messages API payload:

- **System messages** — extracted from the `messages` array and joined into
  a top-level `system` string parameter (Anthropic does not accept system
  messages inline). Multiple system messages are concatenated with double
  newlines. This extraction works for both plain string system content and
  normalized `ChatMessage.to_dict()` block content.
- **`max_tokens`** — required by Anthropic; defaults to `4096` when the
  canonical request omits it.
- **`stop`** — mapped to `stop_sequences` (Anthropic's field name). A bare
  string is wrapped in a single-element list.
- **`tools`** — translated into Anthropic's `input_schema` shape. The helper
  accepts either already-Anthropic tool definitions or OpenAI/MCP-style
  `{"type": "function", "function": ...}` tool schemas.
- **Assistant tool turns** — canonical assistant `tool_calls` are translated
  into Anthropic `tool_use` blocks.
- **Tool result turns** — canonical `role="tool"` messages are translated into
  Anthropic `tool_result` blocks. Consecutive tool results are merged into a
  single synthetic `user` turn to match Anthropic's conversation structure.
- **`extra_body`** — merged into the payload via `TransportKwargs`, with
  Anthropic-handled and OpenAI-only fields excluded from forwarding:
  `stop`, `max_tokens`, `tools`, `response_format`, `frequency_penalty`,
  `presence_penalty`, and `seed`.

Invalid Anthropic-specific payload shapes are surfaced as
`ProviderError(kind=BAD_REQUEST)` by `_build_payload_or_raise()`, which wraps
`ValueError` from the translation layer.

#### Image content block translation

`MultiModalContext.get_contexts()` in the config layer emits image content
blocks in OpenAI format (`type: "image_url"`). The adapter translates these to
Anthropic's `image` block format in `translate_content_blocks()` /
`translate_image_url_block()`.

Three input shapes are handled:

| OpenAI shape | Anthropic output |
|---|---|
| `{"image_url": {"url": "data:image/png;base64,..."}}` | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}` |
| `{"image_url": "https://example.com/img.png"}` | `{"type": "image", "source": {"type": "url", "url": "https://..."}}` |
| `{"image_url": "data:image/jpeg;base64,..."}` | `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}` |

Non-image content blocks (text, custom types) pass through unchanged.
String message content stays as plain string for regular user/assistant
messages. When block translation is required, non-list non-string content is
coerced into a text block.

This keeps the config layer provider-agnostic — it always emits OpenAI-format
blocks, and adapters own provider-specific translation.

#### Response parsing (Anthropic -> canonical)

`parse_anthropic_response()` iterates over Anthropic's `content` blocks and
maps them to the canonical `AssistantMessage`:

| Anthropic block | Canonical field |
|---|---|
| `type=text` | `AssistantMessage.content` (joined with newlines) |
| `type=tool_use` | `AssistantMessage.tool_calls` — `id`, `name`, `input` serialized as `arguments_json` |
| `type=thinking` | `AssistantMessage.reasoning_content` (joined with newlines) |

Usage is extracted via the shared `extract_usage()` helper from
`parsing.py`, which handles the Anthropic field names (`input_tokens`,
`output_tokens`) directly.

### 2. Public Anthropic translation module (`clients/adapters/anthropic_translation.py`)

The Anthropic-specific request/response translation logic now lives in a
separate public module rather than being embedded in `AnthropicClient`.

This module exposes pure functions that are directly testable without mocking
HTTP transport, including:

- `build_anthropic_payload()`
- `parse_anthropic_response()`
- `translate_request_messages()`
- `translate_tool_definition()`
- `translate_tool_call()`
- `translate_tool_result_message()`
- `translate_image_url_block()`
- `parse_tool_call_arguments()`

This separation keeps `AnthropicClient` focused on HTTP orchestration and
error mapping, while the module owns Anthropic protocol translation.

### 3. Authentication headers

Auth remains provider-specific and stays in the concrete adapter classes:

| Provider | Header | Value |
|---|---|---|
| Anthropic | `x-api-key` | Resolved API key (omitted when key is `None`) |
| Anthropic | `anthropic-version` | `2023-06-01` (always present) |
| Anthropic | `Content-Type` | `application/json` (always present) |
| OpenAI-compatible | `Authorization` | `Bearer <api_key>` when a key is configured |
| OpenAI-compatible | `Content-Type` | `application/json` (always present) |

Notably, the `Authorization: Bearer` header used by OpenAI-compatible
providers is **not** sent by the Anthropic adapter. This remains implemented
in provider-specific `_build_headers()` methods.

### 4. Shared HTTP helpers (`clients/adapters/http_helpers.py`)

The stateless HTTP-level wrapper logic has been extracted into a shared helper
module:

- `parse_json_body()` — wraps JSON decoding failures as canonical
  `ProviderError(kind=API_ERROR)`
- `wrap_transport_error()` — maps transport exceptions into canonical
  `ProviderError` values
- `resolve_timeout()` — converts adapter default and per-request timeout values
  into `httpx.Timeout`

These helpers are reused by both native adapters.

### 5. Shared native HTTP base (`clients/adapters/http_model_client.py`)

`HttpModelClient` is a new internal implementation base for native `httpx`
adapters. It centralizes the stateful behavior duplicated between
`OpenAICompatibleClient` and `AnthropicClient`:

- Shared adapter initialization (`provider_name`, `model_id`, `endpoint`,
  retry config, timeout, injected test clients)
- Retry transport creation via `RetryConfig` + `create_retry_transport`
- Shared connection pool sizing policy via named constants:
  `max(_MIN_MAX_CONNECTIONS, _POOL_MAX_MULTIPLIER * max_parallel_requests)` and
  `max(_MIN_KEEPALIVE_CONNECTIONS, max_parallel_requests)`
- Lazy sync/async `httpx` client creation
- Shared `_post_sync()` / `_apost()` request wrappers
- Shared `close()` / `aclose()` lifecycle management

`ModelClient` remains the public protocol. `HttpModelClient` is not a new
public semantic client type; it is only an internal implementation base for
the native adapters.

### 6. Lifecycle semantics

The lifecycle contract is stricter than the original PR-4 draft:

- `close()` / `aclose()` are terminal
- Once shutdown begins, the adapter sets `_closed = True`
- `_get_sync_client()` / `_get_async_client()` will not lazily recreate clients
  after shutdown; they raise `RuntimeError("Model client is closed.")`
- The shared `_post_sync()` / `_apost()` helpers acquire the client outside the
  transport exception wrapper so closed-adapter misuse is not misreported as a
  provider transport failure

This avoids the reopen-during-shutdown race that existed when clearing client
references to `None` without a separate closed sentinel.

### 7. OpenAI-compatible adapter refactor (`clients/adapters/openai_compatible.py`)

`OpenAICompatibleClient` still implements the same public behavior as before
for chat completion, embeddings, image generation, auth headers, and response
parsing. The implementation now inherits from `HttpModelClient`, so the
duplicated native HTTP lifecycle and request-wrapper code has been removed.

This is an internal refactor only; the OpenAI-compatible adapter's public
request/response semantics are unchanged.

### 8. Client factory routing (`clients/factory.py`)

`create_model_client` now routes based on provider type with three
sequential early returns:

1. If `DATA_DESIGNER_MODEL_BACKEND=litellm_bridge` → always `LiteLLMBridgeClient`
2. If `provider_type == "openai"` → `OpenAICompatibleClient`
3. If `provider_type == "anthropic"` → `AnthropicClient`
4. Otherwise → `LiteLLMBridgeClient` (fallback for unknown providers)

The tests now explicitly cover:

- `test_anthropic_provider_creates_native_client`
- `test_anthropic_provider_type_case_insensitive` — case-insensitive matching
- `test_unknown_provider_creates_bridge_client`
- Existing OpenAI native routing
- `DATA_DESIGNER_MODEL_BACKEND=litellm_bridge` override behavior

## What Does NOT Change

1. `ModelFacade` public method signatures — callers see the same API.
2. `ModelClient` — remains the public protocol used by the rest of the engine.
3. `ThrottleManager` — standalone resource, unchanged. Anthropic adapters
   will be throttled by the same `ThrottleManager` once the
   `AsyncTaskScheduler` (plan 346) is wired.
4. Error boundaries — `@catch_llm_exceptions` / `@acatch_llm_exceptions`
   decorators and `DataDesignerError` subclass hierarchy remain stable.
5. MCP tool-loop behavior — `ModelFacade` continues to orchestrate tool
   turns using canonical `ToolCall` objects regardless of provider.
6. Usage accounting — token tracking works unchanged via canonical `Usage`.
7. `LiteLLMBridgeClient` — retained for rollback and unknown providers, and it
   does not inherit from `HttpModelClient`.
8. Unknown provider types — still fall back to `LiteLLMBridgeClient`.

## Design Decisions

### Why Anthropic remains a separate concrete adapter

Anthropic's API diverges from the OpenAI shape in several ways that would
make subclassing awkward:

1. **Auth header scheme** — `x-api-key` vs `Authorization: Bearer`
2. **System message handling** — top-level `system` param vs inline messages
3. **Response shape** — content blocks vs `choices[0].message`
4. **Required fields** — `max_tokens` is mandatory in Anthropic
5. **Field name differences** — `stop_sequences` vs `stop`

A separate adapter keeps each provider's concerns isolated and avoids
conditional branching in shared code paths.

### Why translation lives in a separate public module

The Anthropic translation layer has enough branching logic to justify direct
unit tests without HTTP mocking. Public module-level helpers keep the code
pure and make the provider-specific protocol mapping easy to test and reason
about.

### Why `HttpModelClient` exists instead of a broader `BaseModelClient`

The duplication that justified extraction lives in the native `httpx`
transport/lifecycle implementation, not in the public model-client semantics.
`ModelClient` therefore remains the protocol, while `HttpModelClient` is an
internal implementation base used only by native adapters.

This keeps `LiteLLMBridgeClient` and the public client contract decoupled from
the native HTTP transport details.

### Why both `http_helpers.py` and `HttpModelClient` exist

The extracted shared code naturally splits into two kinds:

- **Stateless helpers** — JSON parsing, timeout resolution, and transport
  error wrapping (`http_helpers.py`)
- **Stateful adapter machinery** — retry transport creation, pooled client
  state, lifecycle, and shared request wrappers (`HttpModelClient`)

Keeping these separate avoids forcing small pure helpers into an inheritance
hierarchy while still removing the duplicated stateful code from the
concrete adapters.

## Files Touched

| File | Change |
|---|---|
| `clients/adapters/anthropic.py` | New native Anthropic adapter, now slimmed to provider-specific wrapper over `HttpModelClient` |
| `clients/adapters/anthropic_translation.py` | New — Anthropic request/response translation helpers |
| `clients/adapters/http_helpers.py` | New — shared stateless HTTP helpers |
| `clients/adapters/http_model_client.py` | New — shared native HTTP transport/lifecycle base |
| `clients/adapters/openai_compatible.py` | Refactored to inherit `HttpModelClient`; public behavior unchanged |
| `clients/factory.py` | Route `provider_type=anthropic` to native adapter |
| `tests/.../test_anthropic.py` | New/expanded — Anthropic adapter coverage, including lifecycle regression tests |
| `tests/.../test_anthropic_translation.py` | New — direct translation helper tests |
| `tests/.../test_openai_compatible.py` | Updated — shared lifecycle/base regression coverage |
| `tests/.../test_factory.py` | Updated — Anthropic routes to native; unknown providers route to bridge |
| `plans/343/model-facade-overhaul-pr-4-architecture-notes.md` | Updated to reflect the final branch architecture |

## Test Coverage

| Category | Tests |
|---|---|
| Anthropic payload building | system extraction from normalized messages, default/explicit `max_tokens`, `stop` -> `stop_sequences` |
| Tool schema and turn translation | OpenAI-style tools -> Anthropic `input_schema`, assistant `tool_use` turns, tool-result merging |
| Image block translation | data URI dict, URL string, data URI string, non-image passthrough, string-content passthrough |
| Anthropic response parsing | text blocks, `tool_use` blocks, `thinking` blocks, usage extraction |
| Error mapping | HTTP status mapping, timeout mapping, bad Anthropic payloads -> `ProviderError(kind=BAD_REQUEST)` |
| Unsupported capabilities | Anthropic embeddings and image generation (sync + async) |
| Lifecycle | `close()` / `aclose()` delegation, no-op when no client exists, post-close calls do not recreate clients |
| OpenAI adapter regression coverage | chat/embedding/image routes, auth headers, timeouts, post-close lifecycle guards |
| Factory routing | OpenAI native, Anthropic native, case-insensitive provider matching, bridge fallback, env override |

## Planned Follow-On

PR-5 introduces the config/CLI auth schema rollout, adding typed
provider-specific auth objects (`AnthropicAuth`, `OpenAIApiKeyAuth`) to
`ModelProvider` with backward-compatible `api_key` fallback.

The `AsyncTaskScheduler` (plan 346) will call `ThrottleManager` for
Anthropic models using the same acquire/release pattern as OpenAI models,
since throttle management is provider-agnostic at the orchestration layer.
