---
date: 2026-03-04
authors:
  - nmulepati
---

# Model Facade Overhaul PR-2 Architecture Notes

This document captures the architecture intent for PR-2 from
`plans/343/model-facade-overhaul-plan-step-1.md`.

## Goal

Switch `ModelFacade` from direct LiteLLM router usage to the `ModelClient` protocol
introduced in PR-1. After this PR, `ModelFacade` consumes only canonical types
(`ChatCompletionResponse`, `EmbeddingResponse`, `ImageGenerationResponse`) and has
no direct import or runtime dependency on LiteLLM response shapes.

## What Changes

### 1. ModelFacade internals rewired to ModelClient

`ModelFacade.__init__` currently constructs a `CustomRouter` and calls it directly:

```python
self._router = CustomRouter([self._litellm_deployment], ...)
# ...
response = self._router.completion(model=..., messages=..., **kwargs)
```

After PR-2, it receives a `ModelClient` (selected by factory) and builds canonical requests:

```python
self._client: ModelClient  # injected via factory
# ...
request = ChatCompletionRequest(model=..., messages=..., **consolidated)
response: ChatCompletionResponse = self._client.completion(request)
```

The same pattern applies to embeddings (`EmbeddingRequest` → `EmbeddingResponse`) and
image generation (`ImageGenerationRequest` → `ImageGenerationResponse`).

### 2. Client factory

New file: `clients/factory.py`

Responsible for selecting the right `ModelClient` adapter given a `ModelConfig` and
provider context. For PR-2, the only adapter is `LiteLLMBridgeClient`. The factory
encapsulates router construction and deployment config that currently lives in
`ModelFacade._get_litellm_deployment`.

`models/factory.py` (`create_model_registry`) is updated to use the client factory
when constructing each `ModelFacade`.

### 3. MCP compatibility update

`MCPFacade` methods (`has_tool_calls`, `tool_call_count`, `process_completion_response`,
`refuse_completion_response`) currently accept `Any` and traverse
`completion_response.choices[0].message` with `getattr` for LiteLLM shapes.

PR-2 updates these to accept `ChatCompletionResponse` and read from canonical fields:

- `response.message.tool_calls` → `list[ToolCall]` (id, name, arguments_json)
- `response.message.content` → `str | None`
- `response.message.reasoning_content` → `str | None`

`_extract_tool_calls` and `_normalize_tool_call` simplify significantly because
canonical `ToolCall` is already normalized (no nested `function` key, no dict vs
object polymorphism).

### 4. Usage tracking consolidation

The three existing methods:

- `_track_token_usage_from_completion`
- `_track_token_usage_from_embedding`
- `_track_token_usage_from_image_diffusion`

All read from provider-specific usage shapes (`litellm.types.utils.*`). PR-2 replaces
them with a single helper that reads from canonical `Usage`:

```python
def _track_usage(self, usage: Usage | None, *, is_request_successful: bool) -> None
```

### 5. Image extraction moves into adapter

`ModelFacade` currently does image extraction from raw LiteLLM responses
(`_try_extract_base64`, `_generate_image_chat_completion`, `_generate_image_diffusion`).

After PR-2, the adapter returns `ImageGenerationResponse.images: list[ImagePayload]`
with `b64_data` already resolved. `ModelFacade.generate_image` / `agenerate_image`
simply reads `response.images` and extracts `b64_data` values — no more format
detection, URL downloading, or data URI parsing at the facade level.

### 6. LiteLLM type removal from facade

After PR-2, `facade.py` no longer imports:

- `litellm` (the module, currently used for type hints)
- `CustomRouter`, `LiteLLMRouterDefaultKwargs`
- `litellm.types.utils.ModelResponse`, `EmbeddingResponse`, `ImageResponse`, `ImageUsage`

These remain internal to `LiteLLMBridgeClient` and `models/factory.py`.

### 7. Adapter lifecycle wiring

`ModelClient.close()` / `aclose()` are wired through `ModelRegistry` so adapter
resources (HTTP clients, connection pools) are torn down deterministically when
generation is complete.

- `ModelRegistry` gains `close()` / `aclose()` that iterate owned facades.
- `ModelFacade` gains `close()` / `aclose()` that delegate to `self._client`.
- `ResourceProvider` (or equivalent teardown hook) calls `ModelRegistry.close()`.

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
| `models/facade.py` | Rewire to `ModelClient`, canonical types, consolidated usage tracking |
| `models/factory.py` | Use client factory to inject `ModelClient` into `ModelFacade` |
| `models/registry.py` | Add `close` / `aclose` lifecycle methods |
| `clients/factory.py` | New — adapter selection by provider config |
| `mcp/facade.py` | Accept `ChatCompletionResponse` instead of raw LiteLLM response |

## Planned Follow-On

PR-3 introduces the OpenAI-compatible native adapter with shared retry/throttle
infrastructure. At that point, the client factory gains a second adapter option
alongside the LiteLLM bridge.
