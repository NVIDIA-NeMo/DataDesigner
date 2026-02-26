---
date: 2026-02-19
authors:
  - nmulepati
---

# Model Facade Overhaul Plan: Step 1 (Non-Bedrock)

Review Reference: `plans/343/_review-model-facade-overhaul-plan.md`

This document proposes a concrete migration plan to replace LiteLLM in Data Designer while keeping the public behavior of `ModelFacade` stable.

The short version:

1. Introduce a provider-agnostic client interface under `engine/models/clients/`.
2. Keep `ModelFacade` API and call sites unchanged.
3. Implement adapters for `openai`-compatible APIs first, then Anthropic.
4. Migrate error handling and retries into native code.
5. Remove LiteLLM only after contract-test parity is proven.

## Reviewer Snapshot

Reviewers should validate three things first:

1. `ModelFacade` public behavior does not regress (API shape, MCP loop behavior, usage accounting, user-facing errors).
2. Provider-specific concerns are isolated inside adapters (`openai_compatible`, `anthropic`) behind canonical request/response types.
3. Rollout is reversible with feature flag and bridge adapter until parity is proven.

## Feedback Closure Matrix

This section maps aggregated reviewer findings to the concrete plan updates.

| Reviewer finding | Resolution in this plan |
|---|---|
| MCP contract for `completion`/`acompletion` is ambiguous | `Explicit MCP Compatibility Contract` defines canonical response contract and required parity tests |
| Adaptive throttling contract had contradictions | `Adaptive Throttling` + `Sync/async throttle parity tests` now align on shared global cap + domain keys |
| Step 1 provider type hardening could block Step 2 Bedrock | `Config and CLI Evolution` keeps provider type extensible and reserves `bedrock` for Step 2 |
| Rollback safety conflicted with flip/removal sequencing | `PR slicing` and `Migration Phases` split default flip/soak from LiteLLM removal |
| Auth mapping for `401/403` ambiguous | `Auth error normalization` now defines deterministic default mapping |
| HTTP client lifecycle/pooling underspecified | `HTTP client lifecycle and pool policy` adds ownership/teardown/pool sizing contract |
| `extra_body`/`extra_headers` precedence unclear | `Request merge / precedence contract` defines explicit merge order |
| Anthropic capability statements inconsistent | `Capability Matrix` now matches Step 1 implementation scope (`No` for embeddings/image) |

## Locked Design Decisions (Step 1)

These are explicit decisions for Step 1 review and implementation.

1. `ModelFacade` public API remains unchanged.
2. Adapter boundary uses canonical request/response types; provider SDK/HTTP shapes do not leak upward.
3. Adaptive throttling uses:
   - global cap key: `(provider_name, model_identifier)`
   - domain key: `(provider_name, model_identifier, throttle_domain)`
4. If multiple model configs target the same global cap key, effective hard cap is:
   - `min(max_parallel_requests across matching model configs)`
5. Throttle domain is derived from actual backend route:
   - chat-completions-backed image generation shares `chat` domain.
6. Auth fallback compatibility is retained:
   - top-level `api_key` continues to work for `openai` and `anthropic`.
7. Streaming support is out of scope for Step 1.
8. Bedrock support is intentionally out of scope for Step 1.
9. `completion`/`acompletion` contract in Step 1 is canonical response shape.
10. MCP handling is adapted in Step 1 to consume canonical responses; no long-term LiteLLM-shaped response dependency.

## Reviewer Sign-Off Questions

These should be answered in review before implementation begins:

1. Is bridge-first migration (`LiteLLMBridgeClient`) acceptable as mandatory Phase 0?
2. Is the minimum-cap rule for shared provider/model concurrency acceptable?
3. Is the proposed feature-flag rollout (`litellm_bridge|native`) sufficient for rollback needs?

## Architecture Diagram

### 1. Structural view (boundaries and ownership)

```text
Callers
  - Column generators
  - ModelRegistry health checks
          |
          v
+---------------------------------------------------------------+
| ModelFacade (public surface; unchanged)                       |
| - generate/agenerate loops                                    |
| - MCP tool loop + correction/restart                          |
| - usage aggregation + user-facing error context               |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
| Model Client Layer (new)                                      |
|                                                               |
|  +----------------------+   +-------------------------------+ |
|  | Client Factory       |-->| Adapter selected by           | |
|  | - provider_type      |   | provider_type                | |
|  | - auth parsing       |   +---------------+--------------+ |
|  +----------+-----------+                   |                |
|             |                               |                |
|             v                               v                |
|  +----------------------+      +--------------------------+  |
|  | Throttle Manager     |<---->| Retry Engine             |  |
|  | - global cap key     |      | - jittered backoff       |  |
|  | - domain key         |      | - retry classifier       |  |
|  +----------+-----------+      +------------+-------------+  |
|             |                               |                |
|             +---------------+---------------+                |
|                             v                                |
|                  +-------------------------+                 |
|                  | Adapter implementation  |                 |
|                  | - OpenAI compatible     |                 |
|                  | - Anthropic             |                 |
|                  | - LiteLLM bridge (temp) |                 |
|                  +------------+------------+                 |
+-------------------------------|------------------------------+
                                v
                      Provider HTTP APIs
```

### 2. Runtime sequence (sync/async happy path)

```text
1) Caller -> ModelFacade.generate/agenerate(...)
2) ModelFacade builds canonical request (types.py)
3) ModelFacade -> ModelClient (adapter chosen by factory)
4) Throttle acquire using:
   - global key: (provider_name, model_identifier)
   - domain key: (provider_name, model_identifier, throttle_domain)
5) Adapter request with resolved auth context
6) Retry engine executes outbound call (httpx)
7) Provider returns response
8) Adapter normalizes response -> canonical response
9) Throttle release_success + additive recovery
10) ModelFacade applies parser/MCP logic, updates usage stats
11) Return parsed output + trace
```

### 3. Runtime sequence (429/throttling path)

```text
1) Provider returns 429 / throttling error
2) Retry engine classifies RATE_LIMIT and extracts Retry-After (if present)
3) Throttle release_rate_limited:
   - multiplicative decrease on domain current_limit
   - set blocked_until cooldown
4) Retry re-enters throttle acquire before next attempt
5) On recovery, additive increase restores capacity up to effective max
```

## Explicit MCP Compatibility Contract

Step 1 chooses this migration contract:

1. `ModelFacade` is refactored to consume canonical `ChatCompletionResponse` in PR-2.
2. MCP helpers (`has_tool_calls`, `tool_call_count`, processing/refusal path) are updated for canonical tool-call fields in the same PR.
3. Bridge adapter translates LiteLLM responses into canonical shape before `ModelFacade` sees them.
4. This contract supersedes any phrasing that MCP helpers stay interface-identical; behavior parity is preserved, interfaces are updated.

Required parity tests:

1. tool-call extraction count parity
2. refusal flow parity when tool budget is exceeded
3. reasoning content propagation parity
4. trace message shape parity for assistant/tool messages

## Concrete Implementation Plan

### File-level change map

New files (Step 1):

1. `packages/data-designer-engine/src/data_designer/engine/models/clients/base.py`
2. `packages/data-designer-engine/src/data_designer/engine/models/clients/types.py`
3. `packages/data-designer-engine/src/data_designer/engine/models/clients/errors.py`
4. `packages/data-designer-engine/src/data_designer/engine/models/clients/retry.py`
5. `packages/data-designer-engine/src/data_designer/engine/models/clients/throttle.py`
6. `packages/data-designer-engine/src/data_designer/engine/models/clients/factory.py`
7. `packages/data-designer-engine/src/data_designer/engine/models/clients/adapters/openai_compatible.py`
8. `packages/data-designer-engine/src/data_designer/engine/models/clients/adapters/anthropic.py`
9. `packages/data-designer-engine/src/data_designer/engine/models/clients/adapters/litellm_bridge.py`

Updated files (Step 1):

1. `packages/data-designer-engine/src/data_designer/engine/models/facade.py`
2. `packages/data-designer-engine/src/data_designer/engine/models/errors.py`
3. `packages/data-designer-engine/src/data_designer/engine/models/factory.py`
4. `packages/data-designer-engine/src/data_designer/engine/models/registry.py` (adapter lifecycle close/aclose ownership)
5. `packages/data-designer-engine/src/data_designer/engine/resources/resource_provider.py` (shutdown wiring if needed)
6. `packages/data-designer/src/data_designer/interface/data_designer.py` (invoke resource teardown hooks in generation entrypoints)
7. `packages/data-designer-config/src/data_designer/config/models.py` (auth schema extension)
8. `packages/data-designer/src/data_designer/cli/forms/provider_builder.py` (provider-specific auth input)
9. `packages/data-designer-config/src/data_designer/lazy_heavy_imports.py` (remove `litellm` after cutover)

### PR slicing (recommended)

1. PR-1: canonical types/interfaces/errors + bridge adapter + no behavior change.
   - files: `clients/base.py`, `clients/types.py`, `clients/errors.py`, `clients/adapters/litellm_bridge.py`
   - docs: add architecture notes for canonical adapter boundary and bridge purpose.
2. PR-2: `ModelFacade` switched to `ModelClient` + lifecycle wiring + parity tests on bridge.
   - files: `models/facade.py`, `models/errors.py`, `models/factory.py`, `clients/factory.py`, `models/registry.py`, `resources/resource_provider.py`, `interface/data_designer.py`
   - docs: update internal lifecycle/ownership docs for adapter teardown and resource shutdown behavior.
3. PR-3: OpenAI-compatible adapter + shared retry/throttle + auth integration.
   - files: `clients/retry.py`, `clients/throttle.py`, `clients/adapters/openai_compatible.py`
   - docs: add provider docs for openai-compatible routing, endpoint expectations, and retry/throttle semantics.
4. PR-4: Anthropic adapter + auth integration + capability gating.
   - files: `clients/adapters/anthropic.py`
   - docs: add Anthropic capability/limitations documentation for Step 1 scope.
5. PR-5: Config/CLI auth schema rollout + migration guards + docs.
   - files: `config/models.py`, `cli/forms/provider_builder.py`
   - docs: publish auth schema migration guide (legacy `api_key` fallback + typed `auth` objects) and CLI examples.
6. PR-6: Cutover flag default flip to native while retaining bridge path.
   - docs: update rollout runbook and env-flag guidance (`DATA_DESIGNER_MODEL_BACKEND`) for operators.
7. PR-7: Remove LiteLLM dependency/path after soak window.
   - files: `lazy_heavy_imports.py` and removal of legacy LiteLLM runtime path
   - docs: remove LiteLLM references and close out migration notes.

### PR coverage check (Step 1)

Every file listed in `File-level change map` must map to exactly one PR above. If a PR changes additional files, they should be explicitly scoped as tests/docs only.

### Reviewer checklist per PR

1. Are external method signatures unchanged on `ModelFacade`?
2. Are error classes unchanged at user-facing boundaries?
3. Are sync and async paths symmetric in behavior?
4. Does adaptive throttling honor global cap and domain key rules?
5. Is any secret material exposed in logs or reprs?
6. Is rollback possible via feature flag with bridge path retained during soak?
7. Are adapter lifecycle teardown hooks wired (`ModelRegistry`/`ResourceProvider`) with no leaked clients in tests?

## Why This Plan

Current usage is concentrated and replaceable:

1. `packages/data-designer-engine/src/data_designer/engine/models/facade.py`
2. `packages/data-designer-engine/src/data_designer/engine/models/errors.py`
3. `packages/data-designer-engine/src/data_designer/engine/models/litellm_overrides.py`
4. `packages/data-designer-engine/src/data_designer/engine/models/factory.py`

That makes this a good candidate for a strangler migration: preserve the outer behavior, replace internals incrementally.

## Current Responsibilities To Preserve

`ModelFacade` currently provides these behaviors and they must remain stable:

1. Sync and async methods:
   - `completion` / `acompletion`
   - `generate` / `agenerate`
   - `generate_text_embeddings` / `agenerate_text_embeddings`
   - `generate_image` / `agenerate_image`
2. Prompt/message conversion and multimodal context handling.
3. MCP tool-calling loop behavior, including tool-turn limits and refusal flow.
4. Usage tracking (`token_usage`, `request_usage`, `image_usage`, `tool_usage`).
5. Exception normalization into `DataDesignerError` subclasses.
6. Provider-level `extra_body` and `extra_headers` merge semantics.

## Target Architecture

### 1. New model client layer

Add a new package:

`packages/data-designer-engine/src/data_designer/engine/models/clients/`

Suggested files:

1. `base.py` - Protocols / interfaces
2. `types.py` - Canonical request/response objects
3. `errors.py` - Provider-agnostic transport/provider exceptions
4. `retry.py` - Backoff policy and retry decision logic
5. `throttle.py` - Adaptive concurrency state and AIMD controller
6. `factory.py` - Adapter selection by `provider_type`
7. `adapters/openai_compatible.py`
8. `adapters/anthropic.py`
9. `adapters/litellm_bridge.py` (temporary bridge for migration safety)

### 2. Keep `ModelFacade` as orchestrator

`ModelFacade` should continue to orchestrate:

1. Parser correction/restart loops
2. MCP tool loop
3. Usage aggregation
4. High-level convenience methods

`ModelFacade` should stop depending directly on LiteLLM response classes.

### 3. Transport stack

Use shared transport components:

1. `httpx.Client` / `httpx.AsyncClient` for HTTP adapters
2. Shared retry module to preserve current exponential backoff and jitter behavior

## Canonical Types (Adapter Contract)

Define provider-agnostic types so `ModelFacade` can consume one shape regardless of provider.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    generated_images: int | None = None


@dataclass
class ImagePayload:
    # canonical output to upper layers is base64 without data URI prefix
    b64_data: str
    mime_type: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass
class AssistantMessage:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    images: list[ImagePayload] = field(default_factory=list)


@dataclass
class ChatCompletionRequest:
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ChatCompletionResponse:
    message: AssistantMessage
    usage: Usage | None = None
    raw: Any | None = None


@dataclass
class EmbeddingRequest:
    model: str
    inputs: list[str]
    encoding_format: str | None = None
    dimensions: int | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None


@dataclass
class EmbeddingResponse:
    vectors: list[list[float]]
    usage: Usage | None = None
    raw: Any | None = None


@dataclass
class ImageGenerationRequest:
    model: str
    prompt: str
    messages: list[dict[str, Any]] | None = None
    n: int | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None


@dataclass
class ImageGenerationResponse:
    images: list[ImagePayload]
    usage: Usage | None = None
    raw: Any | None = None
```

Notes:

1. `raw` exists for diagnostics/logging only.
2. Canonical image output is always base64 payload.
3. Tool calls are normalized to `id/name/arguments_json`.
4. `Usage` includes non-token fields when providers expose them (for example `generated_images`).
5. For image generation, if provider usage does not include image counts, `ModelFacade` tracks `generated_images` from `len(images)` to preserve current `image_usage.total_images` behavior.

## Adapter Interfaces

Use explicit interfaces so capabilities are clear.

```python
from __future__ import annotations

from typing import Protocol


class ChatCompletionClient(Protocol):
    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...
    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...


class EmbeddingClient(Protocol):
    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


class ImageGenerationClient(Protocol):
    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse: ...
    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse: ...


class ModelClient(ChatCompletionClient, EmbeddingClient, ImageGenerationClient, Protocol):
    provider_name: str
    def supports_chat_completion(self) -> bool: ...
    def supports_embeddings(self) -> bool: ...
    def supports_image_generation(self) -> bool: ...
```

Capability checks are important because not all providers/models support all operations.

## Provider Adapter Shapes

### OpenAI-compatible adapter

File: `clients/adapters/openai_compatible.py`

Scope:

1. OpenAI REST-compatible endpoints
2. NVIDIA Integrate endpoints configured as `provider_type="openai"`
3. OpenRouter and similar gateways with compatible request/response shape

Implementation expectations:

1. Chat completion:
   - `POST /chat/completions`
   - map canonical `messages`, `tools`, and generation params directly
2. Embeddings:
   - `POST /embeddings`
3. Image generation:
   - Primary: `POST /images/generations`
   - Fallback mode: chat-completion image extraction for autoregressive models
4. Parse usage from provider response if present.
5. Normalize tool calls and reasoning fields.
6. Normalize image outputs from either `b64_json`, data URI, or URL download.

### Image routing ownership contract

1. `ModelFacade` remains responsible for deciding diffusion-vs-chat image generation using current model semantics (`is_image_diffusion_model(model_name)` and multimodal context).
2. `ModelFacade` constructs `ImageGenerationRequest` accordingly:
   - chat-based path: set `messages` and chat-oriented payload fields
   - diffusion path: omit `messages` and use prompt/image endpoint payload fields
3. Adapter routing is intentionally dumb and request-shape-based:
   - `request.messages is not None` -> chat route
   - otherwise -> image-generation route

### OpenAI-compatible image extraction waterfall

`parse_openai_image_response` must handle all currently observed formats:

1. `choices[0].message.images` entries as dicts containing nested `image_url`/data-URI data
2. `choices[0].message.images` entries as plain strings (base64 or data URI)
3. provider image objects with `b64_json`
4. provider image objects with `url` (requires outbound fetch + base64 normalization)
5. `choices[0].message.content` containing raw base64 image payloads

URL-to-base64 fetches happen in adapter helpers, using a dedicated outbound HTTP fetch client (separate from provider API transport), with timeouts and redacted logging.

### Anthropic adapter

File: `clients/adapters/anthropic.py`

Scope:

1. Anthropic Messages API and tool use
2. Streaming can be deferred for phase 1

Implementation expectations:

1. Chat completion:
   - map system + messages into Anthropic message schema
   - map tool schemas and tool-use response blocks to canonical `ToolCall`
2. Reasoning/thinking content:
   - map into `reasoning_content` if provider returns a separate channel
3. Unsupported capability handling:
   - if embeddings or images are not available/configured, raise canonical unsupported error

Bedrock adapter planning is intentionally deferred to Step 2:

1. `plans/343/model-facade-overhaul-plan-step-2-bedrock.md`

## Authentication and Credential Schema

Auth should be explicit and provider-specific. Today `ModelProvider` has one `api_key` field, but native adapters need different credential shapes.

### Auth design goals

1. Strongly typed auth config per provider.
2. Backward compatibility for existing `api_key` users.
3. Secret values resolved only at runtime via `SecretResolver`.
4. No secret material persisted in logs, traces, or exceptions.

### Proposed config model evolution

Keep current fields for compatibility:

1. `provider_type`
2. `endpoint`
3. `api_key`

Add optional provider-specific `auth` object:

```yaml
model_providers:
  - name: openai-prod
    provider_type: openai
    endpoint: https://api.openai.com/v1
    auth:
      mode: api_key
      api_key: OPENAI_API_KEY
      organization: org_abc123
      project: proj_abc123

  - name: anthropic-prod
    provider_type: anthropic
    endpoint: https://api.anthropic.com
    auth:
      mode: api_key
      api_key: ANTHROPIC_API_KEY
      anthropic_version: "2023-06-01"
```

Back-compat rule:

1. If `auth` is absent and `api_key` is present:
   - for `openai` and `anthropic`, treat as `auth.mode=api_key`
2. If both `auth` and `api_key` are absent for OpenAI-compatible providers:
   - treat as explicit no-auth mode and do not send `Authorization` header.

### Proposed typed auth schema

```python
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class OpenAIApiKeyAuth(BaseModel):
    mode: Literal["api_key"] = "api_key"
    api_key: str
    organization: str | None = None
    project: str | None = None


class OpenAINoAuth(BaseModel):
    mode: Literal["none"] = "none"


class AnthropicAuth(BaseModel):
    mode: Literal["api_key"] = "api_key"
    api_key: str
    anthropic_version: str = "2023-06-01"


OpenAIAuth = Annotated[OpenAIApiKeyAuth | OpenAINoAuth, Field(discriminator="mode")]
```

### Adapter auth behavior by provider

#### OpenAI-compatible

Headers:

1. `Authorization: Bearer <api_key>` when `auth.mode == "api_key"`
2. Optional `OpenAI-Organization: <organization>`
3. Optional `OpenAI-Project: <project>`

Credentials:

1. Resolve `api_key` through `SecretResolver`
2. Cache in-memory per client instance
3. Never log header values

#### Anthropic

Headers:

1. `x-api-key: <api_key>`
2. `anthropic-version: <anthropic_version>`
3. `content-type: application/json`

Credentials:

1. Resolve `api_key` through `SecretResolver`
2. `anthropic_version` can be defaulted in config model

### Auth resolution flow

At model client creation time:

1. Read `ModelProvider.provider_type`
2. Parse/validate `auth` object for that provider type
3. Resolve secret references with `SecretResolver`
4. Build adapter-specific auth context
5. Instantiate adapter client with immutable auth context

### Auth error normalization

Map provider auth failures into canonical errors:

1. Default mapping:
   - `401` -> `ProviderError(kind=AUTHENTICATION)`
   - `403` -> `ProviderError(kind=PERMISSION_DENIED)`
2. OpenAI-compatible: follow default mapping unless provider-specific payload indicates otherwise.
3. Anthropic: follow default mapping unless provider-specific payload indicates otherwise.

Then map canonical provider errors to existing Data Designer user-facing errors:

1. `ModelAuthenticationError`
2. `ModelPermissionDeniedError`

### Secret handling and logging rules

1. Never log resolved secret values.
2. Redact auth headers if request logging is enabled.
3. Redact any accidental credential-like substrings in exception messages.
4. Avoid storing secrets in dataclass `repr` by using custom `__repr__` or redaction wrappers.

### Migration plan for auth schema

#### Phase A

1. Add optional `auth` field to `ModelProvider`.
2. Keep `api_key` as fallback.
3. Add `OpenAINoAuth` support for auth-optional OpenAI-compatible providers.
4. Implement adapter builders that accept all supported forms.

#### Phase B

1. Update CLI provider flow to collect provider-specific auth fields.
2. Add validation messages tailored to provider type.

#### Phase C

1. Deprecate top-level `api_key` once migration is complete.
2. Keep a compatibility shim for one release cycle.

## Concrete Adapter Skeletons

These are intentionally close to implementation shape.

### Shared adapter base

```python
from __future__ import annotations

from typing import Any

import httpx

from data_designer.engine.models.clients.errors import ProviderError, map_http_error_to_provider_error
from data_designer.engine.models.clients.retry import RetryPolicy, run_with_retries
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)


class HTTPAdapterBase:
    ROUTES: dict[str, str] = {}

    def __init__(
        self,
        *,
        provider_name: str,
        endpoint: str,
        api_key: str | None,
        default_headers: dict[str, str] | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.provider_name = provider_name
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.default_headers = default_headers or {}
        self.retry_policy = retry_policy or RetryPolicy.default()
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=self.timeout_s)
        self._aclient = httpx.AsyncClient(timeout=self.timeout_s)

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(self.default_headers)
        headers.update(self._auth_headers())
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _resolve_url(self, route_key: str) -> str:
        try:
            route_path = self.ROUTES[route_key]
        except KeyError as exc:
            raise ValueError(f"Unknown route key {route_key!r} for provider {self.provider_name!r}") from exc
        return f"{self.endpoint}/{route_path.lstrip('/')}"

    def _post_json(
        self,
        route_key: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        response = self._client.post(
            self._resolve_url(route_key),
            json=payload,
            headers=self._headers(extra_headers),
        )
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(response=response, provider_name=self.provider_name)
        return response.json()

    async def _apost_json(
        self,
        route_key: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        response = await self._aclient.post(
            self._resolve_url(route_key),
            json=payload,
            headers=self._headers(extra_headers),
        )
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(response=response, provider_name=self.provider_name)
        return response.json()

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._aclient.aclose()
```

### HTTP client lifecycle and pool policy

Lifecycle contract:

1. Adapters are created by model client factory and owned by the model registry lifetime.
2. Model registry shutdown is responsible for invoking `close`/`aclose` on all adapter instances.
3. `ResourceProvider` exposes `close`/`aclose` and delegates to `ModelRegistry` teardown.
4. `DataDesigner` entrypoints (`create`, `preview`, `validate`) invoke resource teardown in `finally` blocks.
5. Tests must verify no leaked open HTTP clients after teardown.
6. MCP session lifecycle is explicitly owned per run: `ResourceProvider.close()` invokes MCP registry/session-pool cleanup rather than relying only on process-level `atexit`.

Pool sizing policy:

1. Configure `httpx` limits using effective concurrency with concrete defaults:
   - `max_connections = max(32, 2 * effective_max_parallel_requests)`
   - `max_keepalive_connections = max(16, effective_max_parallel_requests)`
2. Keep sync and async client limits aligned.
3. Revisit limits per provider if transport characteristics require overrides.
4. Pool limits are derived from the shared effective max cap at client creation time; AIMD adjusts request admission, not socket pool size.

### OpenAI-compatible adapter skeleton

```python
class OpenAICompatibleClient(HTTPAdapterBase, ModelClient):
    ROUTES = {
        "chat": "/chat/completions",
        "embedding": "/embeddings",
        "image": "/images/generations",
    }

    def supports_chat_completion(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return True

    def supports_image_generation(self) -> bool:
        return True

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = {
            "model": request.model,
            "messages": request.messages,
            "tools": request.tools,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if request.extra_body:
            payload.update(request.extra_body)

        response_json = run_with_retries(
            fn=lambda: self._post_json("chat", payload, request.extra_headers),
            policy=self.retry_policy,
        )
        return parse_openai_chat_response(response_json)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        ...

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        payload = {"model": request.model, "input": request.inputs}
        if request.extra_body:
            payload.update(request.extra_body)
        response_json = run_with_retries(
            fn=lambda: self._post_json("embedding", payload, request.extra_headers),
            policy=self.retry_policy,
        )
        return parse_openai_embedding_response(response_json)

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        ...

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        # Autoregressive image models may use chat route; diffusion-style models use image route.
        if request.messages:
            route_key = "chat"
            payload = openai_chat_image_payload_from_canonical(request)
        else:
            route_key = "image"
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "n": request.n,
            }
            payload = {k: v for k, v in payload.items() if v is not None}

        if request.extra_body:
            payload.update(request.extra_body)

        response_json = run_with_retries(
            fn=lambda: self._post_json(route_key, payload, request.extra_headers),
            policy=self.retry_policy,
        )
        return parse_openai_image_response(response_json)

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        ...
```

### Anthropic adapter skeleton

```python
class AnthropicClient(HTTPAdapterBase, ModelClient):
    ROUTES = {
        "chat": "/v1/messages",
    }

    def supports_chat_completion(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return False

    def supports_image_generation(self) -> bool:
        return False

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = anthropic_payload_from_canonical(request)
        response_json = run_with_retries(
            fn=lambda: self._post_json("chat", payload, request.extra_headers),
            policy=self.retry_policy,
        )
        return parse_anthropic_chat_response(response_json)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        ...

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="embeddings")

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="embeddings")

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="image-generation")

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="image-generation")
```

## Request and Response Mapping Details

### Request merge / precedence contract

To preserve current behavior, merge precedence is explicit:

1. Start from model inference params (`generate_kwargs`).
2. Overlay per-call kwargs.
3. Merge `extra_body` with provider extra body taking precedence on key conflicts.
4. Set `extra_headers` from provider extra headers (provider-level replacement semantics).
5. Drop non-provider params like `purpose` before outbound request.

Merge behavior note:

1. `extra_body` merge is shallow (top-level keys only), matching current behavior.

### Canonical -> OpenAI-compatible chat payload

| Canonical field | OpenAI payload field |
|---|---|
| `model` | `model` |
| `messages` | `messages` |
| `tools` | `tools` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `max_tokens` | `max_tokens` |
| `extra_body` | merged into payload |
| `extra_headers` | request headers |

OpenAI-compatible response parsing:

1. `choices[0].message.content` -> canonical `message.content`
2. `choices[0].message.tool_calls[*]` -> canonical `ToolCall`
3. `choices[0].message.reasoning_content` if present -> canonical `reasoning_content`
4. `usage.prompt_tokens/completion_tokens` -> canonical `Usage`

### Canonical -> Anthropic messages payload

| Canonical field | Anthropic field |
|---|---|
| system message in `messages` | top-level `system` |
| user/assistant/tool messages | `messages` blocks |
| `tools` | `tools` |
| `max_tokens` | `max_tokens` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `extra_body` | merged into payload |

Anthropic response parsing:

1. text blocks -> canonical `message.content`
2. tool_use blocks -> canonical `ToolCall`
3. thinking/reasoning blocks when available -> canonical `reasoning_content`
4. usage fields -> canonical `Usage`

## Client Factory and Provider Resolution

Add a dedicated factory:

`packages/data-designer-engine/src/data_designer/engine/models/clients/factory.py`

Factory inputs:

1. `ModelConfig`
2. `ModelProvider`
3. `SecretResolver`

Factory output:

1. A concrete `ModelClient` implementation

Routing logic:

1. `provider_type == "openai"` -> `OpenAICompatibleClient`
2. `provider_type == "anthropic"` -> `AnthropicClient`
3. `provider_type == "bedrock"` -> fail fast with:
   - `ValueError("provider_type='bedrock' is deferred to Step 2; see plans/343/model-facade-overhaul-plan-step-2-bedrock.md")`
4. unknown -> `ValueError` with supported provider types

Migration safety option:

1. `provider_type == "litellm-bridge"` or feature flag -> `LiteLLMBridgeClient`
2. lets us verify new abstraction without immediate provider rewrite

### Bridge coexistence with LiteLLM global patches

During mixed bridge/native rollout:

1. `apply_litellm_patches()` must run if any configured model resolves to `LiteLLMBridgeClient`.
2. Patch application must be idempotent and safe when called multiple times.
3. `ThreadSafeCache` + LiteLLM patch behavior remains in place until PR-7 removes bridge/LiteLLM path.
4. PR-7 is the cleanup point for removing `litellm_overrides.py` patch side effects.

## Error Model and Mapping

Current `errors.py` pattern-matches LiteLLM exception types. Replace this with canonical provider exceptions.

### Canonical provider exception

```python
from dataclasses import dataclass
from enum import Enum


class ProviderErrorKind(str, Enum):
    API_ERROR = "api_error"
    API_CONNECTION = "api_connection"
    AUTHENTICATION = "authentication"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    UNSUPPORTED_PARAMS = "unsupported_params"
    BAD_REQUEST = "bad_request"
    INTERNAL_SERVER = "internal_server"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNPROCESSABLE_ENTITY = "unprocessable_entity"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


@dataclass
class ProviderError(Exception):
    kind: ProviderErrorKind
    message: str
    status_code: int | None = None
    provider_name: str | None = None
    model_name: str | None = None
    cause: Exception | None = None
```

Then update `handle_llm_exceptions` to match on `ProviderError.kind` and raise existing user-facing `DataDesignerError` subclasses.

This preserves the public error model while removing LiteLLM-specific coupling.

## Retry and Backoff

Replicate current semantics from `LiteLLMRouterDefaultKwargs` and `CustomRouter`:

1. `initial_retry_after_s = 2.0`
2. `jitter_pct = 0.2`
3. retries at least for `rate_limit` and `timeout` (currently 3)
4. respect provider `Retry-After` when present and reasonable

Implement this in one shared module used by all adapters.

## Adaptive Throttling (429-Aware, Sync + Async)

In addition to retries, adapters should dynamically reduce concurrency when providers return `429`/throttling errors.

### Design goals

1. Respect configured `max_parallel_requests` as a hard upper bound.
2. Auto-throttle down on sustained throttling and recover gradually.
3. Use the same throttle state across sync and async code paths.
4. Share throttling across model configs that target the same provider + model identifier.

### Throttle key and scope

Use two related keys:

1. Global cap key: `(provider_name, model_identifier)`
2. Throttle domain key: `(provider_name, model_identifier, throttle_domain)`

`model_identifier` is the model id from model config (for example, `gpt-5`).

`throttle_domain` is derived from the actual backend route:

1. `chat` for chat/completions-backed traffic (including autoregressive image generation via chat)
2. `embedding` for embedding endpoint traffic
3. `image` for dedicated image generation endpoint traffic

This allows multimodal models to share budget when they use the same upstream route and to separate budgets when routes differ.

### Effective max concurrency across model configs

When multiple model configs map to the same global cap key:

1. Compute `effective_max_parallel_requests = min(max_parallel_requests across those model configs)`.
2. Use that effective max as the hard cap for all associated throttle domains.

This enforces the most conservative configured limit for shared upstream capacity.

### Shared throttle state

Store shared state at both levels:

1. `GlobalCapState` per `(provider_name, model_identifier)`:
   - `configured_limits_by_alias`
   - `effective_max_limit` (minimum of registered alias limits)
2. `DomainThrottleState` per `(provider_name, model_identifier, throttle_domain)`:
   - `current_limit` (`1..effective_max_limit`)
   - `in_flight`
   - `blocked_until_monotonic`
   - `success_streak`

Each domain state must clamp to the current global `effective_max_limit` if that value decreases.

State mutation must be thread-safe (`threading.Lock`) and use monotonic time.

### Execution-model agnostic core API

Core state methods should be non-blocking so both sync/async wrappers can reuse them:

1. `try_acquire(now_monotonic) -> float` where return is `wait_seconds` (`0` means acquired)
2. `release_success(now_monotonic) -> None`
3. `release_rate_limited(now_monotonic, retry_after_seconds) -> None`
4. `release_failure(now_monotonic) -> None`

### Sync and async wrappers

Use thin wrappers around the same state:

1. `acquire_sync()` loops with `time.sleep(wait_seconds)` until acquire.
2. `acquire_async()` loops with `await asyncio.sleep(wait_seconds)` until acquire.

This ensures sync and async traffic co-throttle against the same provider/model throttle domain budget.

### Adjustment policy

Use additive-increase / multiplicative-decrease (AIMD):

1. On `429`/throttling:
   - `current_limit = max(1, floor(current_limit * 0.5))`
   - set `blocked_until` using `Retry-After` when available, else default backoff
   - reset `success_streak`
2. On success:
   - increment `success_streak`
   - after `success_window` successes, increase `current_limit` by `+1` until `effective_max_limit`
3. On non-429 failures:
   - no immediate drop unless configured; still release in-flight slot

### Provider-specific throttling signals

1. OpenAI-compatible: HTTP `429`, parse `Retry-After` (seconds/date).
2. Anthropic: HTTP `429`, parse provider headers if present.

### Integration points

1. Acquire throttle slot immediately before outbound request attempt.
2. Release slot on completion in `finally`.
3. Apply `release_rate_limited(...)` in retry classifier when error kind is `RATE_LIMIT`.
4. Re-enter acquire path for each retry attempt (so retries also obey adaptive limits).
5. Register each `ModelFacade` limit contribution into `GlobalCapState` during initialization.

### Timeout and cancellation semantics

1. `inference_parameters.timeout` applies to outbound provider call duration per attempt (transport timeout), not queueing time before admission.
2. Throttle queue wait time is tracked separately in telemetry for observability.
3. Async throttle acquire loops must propagate `asyncio.CancelledError` immediately.
4. Retry loops must not swallow cancellation signals.

### Config knobs (optional, with safe defaults)

1. `adaptive_throttle_enabled: bool = true`
2. `adaptive_throttle_min_parallel: int = 1`
3. `adaptive_throttle_reduce_factor: float = 0.5`
4. `adaptive_throttle_success_window: int = 50`
5. `adaptive_throttle_default_block_seconds: float = 2.0`

### Compatibility expectation with existing config

1. `max_parallel_requests` remains user-facing and authoritative upper bound.
2. If multiple model configs use same provider+model, lower `max_parallel_requests` wins (minimum rule).
3. If adaptive throttling is disabled, behavior reverts to fixed concurrency at the effective max.
4. No public API changes required in `ModelFacade`.

## `ModelFacade` Refactor Plan

Minimal diff approach:

1. Keep public methods and signatures unchanged.
2. Replace `_router` with `_client: ModelClient`.
3. Convert `ChatMessage` list into canonical request.
4. Consume canonical response shape.
5. Preserve MCP generation semantics (tool budget, refusal behavior, corrections, trace behavior) while updating MCP helper interfaces to canonical response fields.
6. Move usage tracking methods to consume canonical `Usage`.
7. Keep `consolidate_kwargs` as the merge source of truth and add typed request builders:
   - `_build_chat_request(...)`
   - `_build_embedding_request(...)`
   - `_build_image_request(...)`
   This preserves dynamic inference-parameter sampling semantics while moving transport to canonical request types.
8. Preserve async MCP wrapping pattern (`asyncio.to_thread`) for MCP tool schema retrieval and completion processing where MCP services are sync/event-loop-isolated.

Expected code updates:

1. `facade.py`: swap transport layer calls and response parsing.
2. `factory.py`: initialize client factory instead of applying LiteLLM patches.
3. `errors.py`: map from canonical `ProviderError` instead of LiteLLM exception classes.
4. `lazy_heavy_imports.py`: remove `litellm` entry after complete cutover.
5. `registry.py` + `resource_provider.py` + `data_designer.py`: add deterministic client teardown hooks for sync/async lifecycles.

## Capability Matrix

Introduce explicit capability checks at adapter and model level.

| Capability | OpenAI-Compatible | Anthropic |
|---|---|---|
| Chat completion | Yes | Yes |
| Tool calls | Yes | Yes |
| Embeddings | Yes (endpoint/model dependent) | No (Step 1) |
| Image generation (diffusion endpoint) | Yes (provider dependent) | No (Step 1) |
| Image via chat completion | Some models | No (Step 1) |

If unsupported at runtime:

1. Return canonical `ProviderError(kind=UNSUPPORTED_CAPABILITY, ...)`
2. Surface as `ModelUnsupportedParamsError` or a dedicated model capability error

Bedrock capability planning is in Step 2:

1. `plans/343/model-facade-overhaul-plan-step-2-bedrock.md`

## Compatibility Matrix

This matrix defines expected parity between the current LiteLLM-backed implementation and the native adapter implementation.

### API behavior parity (`ModelFacade`)

| Surface | Current (LiteLLM) | Target (Native Adapters) | Compatibility expectation |
|---|---|---|---|
| `completion(messages, **kwargs)` | Router chat completion call | `ModelClient.completion(ChatCompletionRequest)` | Same public method signature, same return semantics consumed by `generate` |
| `acompletion(messages, **kwargs)` | Async router chat completion | `ModelClient.acompletion(ChatCompletionRequest)` | Same public method signature and async behavior |
| `generate(prompt, parser, ...)` | Completion + parser correction loop + MCP loop | Same orchestration, adapter-backed completion | Same correction/restart/tool-turn behavior and same trace shape |
| `agenerate(prompt, parser, ...)` | Async completion + async MCP handling | Same orchestration, adapter-backed async completion | Same behavior and error contracts as sync path |
| `generate_text_embeddings(input_texts, **kwargs)` | Router embedding | `ModelClient.embeddings(EmbeddingRequest)` | Same output type (`list[list[float]]`) and length checks |
| `agenerate_text_embeddings(input_texts, **kwargs)` | Async router embedding | `ModelClient.aembeddings(EmbeddingRequest)` | Same output type and error behavior |
| `generate_image(prompt, ...)` | Chat-completion image path or diffusion path | `ModelClient.generate_image(ImageGenerationRequest)` | Same output contract: list of base64 strings |
| `agenerate_image(prompt, ...)` | Async chat/diffusion path | `ModelClient.agenerate_image(ImageGenerationRequest)` | Same output contract and usage update behavior |
| `consolidate_kwargs(**kwargs)` | Merge model inference params + provider extra fields | Same logic before request conversion | No behavioral drift in precedence rules |

### Error mapping parity

| Current user-facing error | Current trigger source | Native trigger source | Compatibility expectation |
|---|---|---|---|
| `ModelAPIError` | LiteLLM API-level exceptions | `ProviderError(kind=API_ERROR)` | Same class, equivalent message quality |
| `ModelAPIConnectionError` | LiteLLM connection exceptions | `ProviderError(kind=API_CONNECTION)` | Same class and retryability semantics |
| `ModelAuthenticationError` | LiteLLM auth errors / some 403 API errors | `ProviderError(kind=AUTHENTICATION)` | Same class with provider-specific auth guidance |
| `ModelPermissionDeniedError` | LiteLLM permission denied | `ProviderError(kind=PERMISSION_DENIED)` | Same class and cause semantics |
| `ModelRateLimitError` | LiteLLM rate-limit errors | `ProviderError(kind=RATE_LIMIT)` | Same class and backoff behavior |
| `ModelTimeoutError` | LiteLLM timeout | `ProviderError(kind=TIMEOUT)` | Same class and retry policy |
| `ModelBadRequestError` | LiteLLM bad request | `ProviderError(kind=BAD_REQUEST)` | Same class and actionable remediation |
| `ModelUnsupportedParamsError` | LiteLLM unsupported params | `ProviderError(kind=UNSUPPORTED_PARAMS or UNSUPPORTED_CAPABILITY)` | Same class; message may include capability context |
| `ModelNotFoundError` | LiteLLM not found | `ProviderError(kind=NOT_FOUND)` | Same class |
| `ModelInternalServerError` | LiteLLM internal server | `ProviderError(kind=INTERNAL_SERVER)` | Same class |
| `ModelUnprocessableEntityError` | LiteLLM unprocessable entity | `ProviderError(kind=UNPROCESSABLE_ENTITY)` | Same class |
| `ModelContextWindowExceededError` | LiteLLM context overflow | `ProviderError(kind=CONTEXT_WINDOW_EXCEEDED)` | Same class with context-width hints |
| `ModelGenerationValidationFailureError` | Parser correction exhaustion | Same parser correction exhaustion path | Same class and retry/correction semantics |
| `ImageGenerationError` | Image extraction or empty image payload | Same canonical image extraction validations | Same class and failure conditions |

Provider error body normalization requirements:

1. OpenAI-compatible adapters parse structured error payload fields (`error.type`, `error.code`, `error.param`, `error.message`) into canonical `ProviderError.message`.
2. Anthropic adapters parse structured error payload fields (`error.type`, `error.message`) into canonical `ProviderError.message`.
3. Context-window and multimodal-capability hints should remain actionable in user-facing errors when provider payload includes equivalent context.

### Usage and telemetry parity

| Metric behavior | Current | Target | Compatibility expectation |
|---|---|---|---|
| Request success/failure tracking | Tracked in usage stats | Tracked from canonical response/error outcomes | Same counters and aggregation logic |
| Token usage for chat | From LiteLLM usage fields | From canonical `Usage` parsed by adapter | Same when provider reports usage; graceful fallback when omitted |
| Token usage for embeddings | From LiteLLM usage fields | From canonical `Usage` parsed by adapter | Same behavior |
| Token usage for image generation | From LiteLLM image usage (when present) | From canonical `Usage` parsed by adapter | Same behavior; request counts still update if token usage absent |
| Tool usage tracking | Managed in `generate/agenerate` loops | Unchanged in `ModelFacade` | Exact parity expected |
| Image count usage | Counted from returned images | Counted from `usage.generated_images` when provided, otherwise from canonical image payload count | Exact parity expected |

### Configuration compatibility

| Config surface | Current | Target | Compatibility expectation |
|---|---|---|---|
| `ModelProvider.provider_type` | Free-form string | Extensible string + known-values validation | Existing `"openai"` continues to work unchanged; `bedrock` remains reserved for Step 2 |
| `ModelProvider.api_key` | Top-level optional field | Supported as fallback for auth | Backward compatible during migration window |
| `ModelProvider.auth` | Not present | Optional provider-specific auth object | Additive, non-breaking introduction |
| `extra_headers` | Provider-level dict | Preserved and merged into adapter request | Same precedence behavior |
| `extra_body` | Provider/model kwargs passthrough | Preserved and merged into canonical request payload | Same precedence behavior |
| `inference_parameters.timeout` | Passed through to LiteLLM kwargs | Passed to adapter transport/request timeout | Same intent and default behavior |

### Non-goals for strict parity

These are allowed differences and should be documented when encountered:

1. Provider-specific raw response formats in debug logs.
2. Exact wording of low-level upstream exception strings.
3. Minor token accounting differences when providers omit or redefine usage fields.
4. Unsupported capability messages that are more explicit than current generic errors.
5. Cross-run persistence of adaptive throttle state. In Step 1, throttle state is per `ResourceProvider` lifetime and intentionally resets between independent `create`/`preview`/`validate` runs.

## Config and CLI Evolution

Current `ModelProvider.provider_type` is free-form string. Tighten this in phases:

### Phase A (non-breaking)

1. Keep `str` but validate known values in factory.
2. Emit warning for unknown values.

### Phase B (post-Step2 hardening)

1. Keep provider type extensible in Step 1 while validating known values (`openai`, `anthropic`, `bedrock` reserved).
2. Defer strict enum hardening until after Step 2 Bedrock delivery.
3. Update CLI provider form with controlled options and validation.

Related files:

1. `packages/data-designer-config/src/data_designer/config/models.py`
2. `packages/data-designer/src/data_designer/cli/forms/provider_builder.py`

## Testing Strategy

### 1. Contract tests for `ModelFacade`

Goal: prove behavior parity independent of backend.

1. Keep existing `test_facade.py` behavior assertions.
2. Parametrize backend selection (`litellm_bridge` vs native adapters).
3. Ensure MCP/tool-loop/correction behavior is unchanged.
4. Add explicit MCP parity tests for:
   - tool-call extraction count
   - refusal path
   - reasoning content propagation
   - trace message shape

### 2. Adapter unit tests

Per adapter:

1. request mapping tests
2. response parsing tests
3. error mapping tests
4. usage extraction tests
5. retry behavior tests
6. adaptive throttling behavior tests (drop on 429, gradual recovery)
7. auth status mapping tests (`401 -> AUTHENTICATION`, `403 -> PERMISSION_DENIED`)

Tools:

1. `pytest-httpx` for HTTP adapters

### 3. Integration smoke tests

Optional but recommended:

1. provider-backed smoke tests controlled by env vars
2. run outside CI by default

### 4. Health check tests

Ensure `ModelRegistry.run_health_check()` still behaves correctly for:

1. chat models
2. embedding models
3. image models

Health-check throttle contract:

1. Health checks go through the same adapter stack for realistic validation.
2. Health checks use a dedicated throttle domain (`healthcheck`) to reduce interference with workload traffic.
3. Health-check outcomes should not mutate adaptive AIMD state used by production generation traffic.

### 5. Sync/async throttle parity tests

1. Shared throttle state is enforced across mixed sync and async calls for same key.
2. `max_parallel_requests` is never exceeded under concurrent load.
3. `Retry-After` is respected by both sync and async wrappers.
4. Two aliases pointing to same provider/model share one global cap whose effective max is the lower configured limit.
5. Domain throttling remains route-aware (`chat`, `embedding`, `image`) under shared global cap.
6. Optional shared upstream pressure signal propagates cooldown correctly across aliases when enabled.

## Cutover Readiness Gates

Native backend becomes default only when all gates pass:

1. Contract tests:
   - zero regressions across sync and async `ModelFacade` behavior.
2. Error parity:
   - user-facing error classes unchanged for representative failure modes.
3. Throughput stability:
   - no sustained degradation in records/sec under standard load profile.
4. Throttling behavior:
   - 429 recovery stabilizes without oscillation under stress test profile.
5. Rollback safety:
   - feature flag rollback validated in a single release candidate.

## Migration Phases and Deliverables

### Phase 0: Baseline and abstraction setup

Deliverables:

1. `clients/types.py`, `clients/base.py`, `clients/errors.py`
2. `LiteLLMBridgeClient` adapter
3. no behavior changes expected

Exit criteria:

1. current tests pass with bridge client enabled

### Phase 1: OpenAI-compatible native adapter

Deliverables:

1. `OpenAICompatibleClient` sync/async methods
2. shared retry/transport modules
3. shared adaptive throttle manager with sync/async wrappers
4. `ModelFacade` consumes canonical responses

Exit criteria:

1. parity on facade contract tests
2. health checks pass with OpenAI-compatible providers
3. adaptive throttling tests pass for sync and async

### Phase 2: Anthropic adapter

Deliverables:

1. `AnthropicClient` chat + tool use
2. unsupported capability handling for unavailable operations

Exit criteria:

1. adapter unit tests pass
2. contract tests pass for Anthropic-configured chat workloads

### Phase 2b: Config and CLI auth schema rollout

Deliverables:

1. typed provider auth schema in `config/models.py` with backward-compatible `api_key` fallback
2. provider-specific auth input flow in `cli/forms/provider_builder.py`
3. migration docs and validation guards for legacy configs

Exit criteria:

1. config validation passes for legacy and typed auth examples
2. CLI form tests cover openai/anthropic auth input paths

### Phase 3: Native default flip + soak window

Deliverables:

1. flip default backend to native
2. retain bridge path for rollback
3. run soak window and monitor readiness gates for at least one release window

Exit criteria:

1. rollback switch validated in release candidate
2. soak window passes with no blocker regressions

### Phase 4: LiteLLM deprecation and removal (non-Bedrock paths)

Deliverables:

1. remove `litellm_overrides.py` usage path
2. remove LiteLLM dependency from engine package
3. clean up lazy imports and docs

Exit criteria:

1. no runtime import path to LiteLLM
2. full test suite green

## Operational Guardrails

1. Feature flag backend switch during migration:
   - `DATA_DESIGNER_MODEL_BACKEND=litellm_bridge|native`
2. Keep bridge path available for rollback until soak/release window completes.
3. Log provider, model, latency, retry_count per request for observability.
4. Keep raw provider response in debug logs only, with PII-safe handling.
5. Preserve timeout behavior and ensure async cancellation works cleanly.

## Risks and Mitigations

### Risk: subtle response-shape mismatch breaks MCP tool loop

Mitigation:

1. strict canonical response tests around `tool_calls` shape
2. reuse existing tool-loop tests unchanged

### Risk: usage reporting regressions

Mitigation:

1. make usage optional and track request success/failure even without tokens
2. add regression tests for `ModelUsageStats` updates per operation

### Risk: provider-specific capability confusion

Mitigation:

1. explicit adapter capability methods
2. actionable unsupported-capability errors that name provider/model/operation

### Risk: retry policy drift

Mitigation:

1. central retry module with deterministic tests
2. preserve current defaults from `LiteLLMRouterDefaultKwargs`

### Risk: throttle oscillation or starvation under bursty load

Mitigation:

1. bound decreases with minimum limit of `1`
2. additive recovery with tunable success window
3. optional smoothing and per-key metrics (`current_limit`, `429_rate`, queue wait time)

## Proposed Initial Task Breakdown (Implementation Tickets)

1. Create `clients/` package with canonical types, base protocols, canonical errors.
2. Implement `LiteLLMBridgeClient` and switch `ModelFacade` to use `ModelClient`.
3. Add provider-specific `auth` schema parsing with compatibility fallback from `api_key`.
4. Refactor `errors.py` to consume canonical provider errors.
5. Add `ModelRegistry.close/aclose` and `ResourceProvider.close/aclose`, and wire teardown in `DataDesigner` entrypoints.
6. Implement shared adaptive throttle manager keyed by `(provider_name, model_identifier, throttle_domain)` with sync/async wrappers.
7. Add optional shared upstream pressure signal keyed by `(provider_name, model_identifier)` with domain-aware cooldown propagation.
8. Implement `OpenAICompatibleClient` with sync/async, retry, adaptive throttle, auth headers, and image URL-to-base64 normalization.
9. Add adapter tests and contract parametrization by backend.
10. Implement Anthropic adapter (chat + tools + auth headers).
11. Update CLI provider forms for provider-specific auth input.
12. Flip default backend to native while retaining bridge rollback path.
13. Complete soak window against cutover readiness gates.
14. Remove LiteLLM dependency and legacy overrides for non-Bedrock paths.

## Definition of Done

The LiteLLM replacement is complete when all conditions are met:

1. `ModelFacade` no longer imports or types against LiteLLM.
2. `errors.py` no longer matches on LiteLLM exception classes.
3. Engine package does not depend on LiteLLM.
4. Existing model-facing behavior tests pass against native adapters.
5. OpenAI-compatible and Anthropic adapters are available with documented capability limits.
6. Bedrock work is explicitly deferred to `plans/343/model-facade-overhaul-plan-step-2-bedrock.md`.
7. Adapter teardown lifecycle is wired and validated (no leaked open HTTP clients after `create`/`preview`/`validate` flows).
8. Documentation and CLI provider guidance are updated.
