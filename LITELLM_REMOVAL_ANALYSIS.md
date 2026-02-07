# LiteLLM Removal: Impact Analysis & Implementation Plan

## Executive Summary

LiteLLM serves as Data Designer's abstraction layer between model configuration and HTTP API calls. It provides multi-provider routing, retry/backoff logic, error normalization, and response type unification. The dependency is well-contained — 12 files touch it directly, all within the `engine/models/` and `engine/models_v2/` layers. Removing it is feasible but non-trivial. The work breaks down into five areas: HTTP client, error handling, retry/resilience, response normalization, and provider abstraction.

**Dependency**: `litellm>=1.73.6,<1.80.12` (pinned in `packages/data-designer-engine/pyproject.toml`)

---

## What LiteLLM Provides Today

### 1. Multi-Provider Routing

LiteLLM's `Router` class accepts a deployment list and routes `completion()`/`embedding()` calls to the correct provider endpoint using the `{provider_type}/{model_name}` naming convention.

**DD's usage**: Each `ModelFacade` gets its own `CustomRouter` with a single-element deployment list — so DD does not actually use multi-deployment load balancing or failover. The Router is effectively a single-provider HTTP client with retry logic.

**Key takeaway**: DD uses the Router as a resilient HTTP client, not as a load balancer. This simplifies replacement significantly.

### 2. Provider Abstraction (`provider_type`)

`ModelProvider.provider_type` (default: `"openai"`) tells LiteLLM which API format to use. LiteLLM translates this into the correct HTTP request format, auth headers, and response parsing for each provider (OpenAI, Anthropic, Cohere, etc.).

**DD's usage**: The `provider_type` is combined with the model name (`f"{provider_type}/{model_name}"`) and passed to the Router. DD also passes `api_base` (endpoint URL) and `api_key` per deployment.

**Key question**: How many distinct `provider_type` values does DD actually use in production? If it's primarily `"openai"` (OpenAI-compatible APIs including NVIDIA NIM), the provider abstraction is low-value since most inference providers expose OpenAI-compatible endpoints. If it includes `"anthropic"`, `"cohere"`, etc., the translation layer is more valuable.

### 3. Retry & Exponential Backoff

DD's `CustomRouter` extends LiteLLM's `Router` with configurable exponential backoff:

- **Retry policy**: 3 retries for `RateLimitError`, 3 retries for `Timeout`
- **Backoff formula**: `initial_retry_after_s * 2^retry_count * (1 +/- jitter_pct)`
- **Defaults**: initial=2s, jitter=20%, timeout=60s
- **Server Retry-After**: Extracted from response headers, capped at 60s

This is the most custom piece of DD's LiteLLM integration. The `_time_to_sleep_before_retry()` and `_extract_retry_delay_from_headers()` overrides are well-tested.

### 4. Error Normalization (12 Exception Types)

`handle_llm_exceptions()` in `models/errors.py` pattern-matches 12 LiteLLM exception types and maps them to DD-specific error classes:

| LiteLLM Exception | DD Error | Notes |
|---|---|---|
| `APIError` | `ModelAPIError` | Special 403 detection |
| `APIConnectionError` | `ModelAPIConnectionError` | Network issues |
| `AuthenticationError` | `ModelAuthenticationError` | Invalid API key |
| `ContextWindowExceededError` | `ModelContextWindowExceededError` | Parses OpenAI token details |
| `UnsupportedParamsError` | `ModelUnsupportedParamsError` | |
| `BadRequestError` | `ModelBadRequestError` | Detects multimodal rejection |
| `InternalServerError` | `ModelInternalServerError` | |
| `NotFoundError` | `ModelNotFoundError` | |
| `PermissionDeniedError` | `ModelPermissionDeniedError` | |
| `RateLimitError` | `ModelRateLimitError` | |
| `Timeout` | `ModelTimeoutError` | |
| `UnprocessableEntityError` | `ModelUnprocessableEntityError` | |

The DD error types already exist and carry user-friendly `FormattedLLMErrorMessage(cause, solution)` payloads. The only LiteLLM-specific part is the `match` statement that catches LiteLLM's exception classes.

### 5. Response Type Normalization

DD accesses these fields from LiteLLM responses:

**Chat completions** (`ModelResponse`):
- `response.choices[0].message.content` — generated text
- `response.choices[0].message.reasoning_content` — extended thinking (via `getattr`)
- `response.choices[0].message.tool_calls` — MCP tool calls
- `response.usage.prompt_tokens` / `response.usage.completion_tokens` — token counts

**Embeddings** (`EmbeddingResponse`):
- `response.data[i]["embedding"]` — float arrays
- `response.usage.prompt_tokens` — input token count

These match the OpenAI API response format exactly. LiteLLM normalizes responses from non-OpenAI providers into this shape.

### 6. Global Patches

`apply_litellm_patches()` modifies LiteLLM's global state at startup:
- Replaces the in-memory client cache with a thread-safe version (`ThreadSafeCache`)
- Increases `LoggingCallbackManager.MAX_CALLBACKS` to 1000 (workaround for litellm#9792)
- Suppresses verbose logging from httpx, LiteLLM, and LiteLLM Router

These are workarounds for LiteLLM's rough edges — they disappear entirely if LiteLLM is removed.

---

## Arguments For Removal

### Reduced dependency weight
LiteLLM is a large dependency (~50+ transitive packages) with frequent releases and breaking changes. The version pin (`>=1.73.6,<1.80.12`) is already narrow, indicating past compatibility issues. Every LiteLLM upgrade is a risk surface.

### Simpler async story
The event loop issue we just fixed (LiteLLM's `LoggingWorker` queue binding to a stale loop) is symptomatic of LiteLLM's internal async state being opaque and hard to reason about. With a direct HTTP client, DD controls all async state.

### Thread-safety workarounds become unnecessary
`ThreadSafeCache`, the `LoggingCallbackManager.MAX_CALLBACKS` patch, and verbose logger suppression are all workarounds for LiteLLM behavior. They represent ongoing maintenance burden.

### Overfit abstraction
DD uses the Router as a single-deployment client with retry logic. The multi-model routing, caching, and callback infrastructure of LiteLLM's Router class is unused overhead.

### Import performance
LiteLLM's import time is significant (already lazy-loaded via `lazy_heavy_imports.py`). Removing it improves cold start time.

### Better error messages
DD already defines its own error types. Currently, LiteLLM's exceptions are caught and re-raised as DD errors. With a direct client, DD can produce better error messages without an intermediate translation layer.

---

## Arguments Against Removal

### Provider format translation
LiteLLM handles API format differences between providers. If DD only targets OpenAI-compatible endpoints, this is irrelevant. If it supports Anthropic, Cohere, etc., this is significant work to reimplement.

### Battle-tested retry logic
LiteLLM's Router has been hardened over many releases for rate limiting, retry-after headers, connection pooling, and edge cases. Reimplementing this from scratch risks regressions.

### Maintained by others
LiteLLM receives frequent updates for new models, API changes, and provider additions. DD's replacement would need to be maintained by the DD team.

### Feature velocity risk
If DD later needs streaming, function calling improvements, vision model support, or new provider integrations, LiteLLM provides these incrementally. A custom client requires explicit implementation for each.

---

## Blast Radius (by Phase)

All paths relative to `packages/data-designer-engine/src/data_designer/engine/` unless noted.

### Phase 1: Replace Router with ModelClient in `models_v2/`

**Response type strategy:** Keep the OpenAI response format (`choices[0].message.content`, `choices[0].message.tool_calls`, `usage.prompt_tokens`, etc.) as DD's canonical model response type. The `openai` SDK already returns typed objects in this shape — use them directly or define DD-owned dataclasses with the same structure. This means **zero changes** to response access sites in the facade, MCP facade, or anywhere else that reads model responses. Non-OpenAI providers (Phase 3) will be responsible for translating their native responses into this format within their adapter.

**New files:**
- `models_v2/client.py` — `ModelClient` protocol, `OpenAIModelClient` adapter wrapping the `openai` SDK

**Delete:**
- `models_v2/litellm_overrides.py` — `CustomRouter`, `ThreadSafeCache`, `apply_litellm_patches()` no longer needed in `models_v2/`

**Heavy modification:**
- `models_v2/facade.py` — Replace `self._router.completion()` / `self._router.acompletion()` with `ModelClient` calls. Response access patterns (`response.choices[0].message.content`, etc.) stay the same since we keep the OpenAI format. Replace `_get_litellm_deployment()` with adapter construction.
- `models_v2/errors.py` — Replace `litellm.exceptions.*` matching with `openai` SDK exception matching in `handle_llm_exceptions()`

**Light modification:**
- `models_v2/factory.py` — Remove `apply_litellm_patches()` call, remove litellm imports, construct `ModelClient` adapter instead of `CustomRouter`

**Tests (for `models_v2/` path):**
- `tests/engine/models/test_facade.py` — Medium rewrite: 26 tests, replace `CustomRouter` patches with `ModelClient` mocks. Response object construction in mocks can use `openai` SDK types directly (same shape as today's litellm types).
- `tests/engine/models/test_model_errors.py` — Medium rewrite: 7 tests, replace `litellm.exceptions.*` with `openai` SDK exceptions
- `tests/engine/models/test_model_registry.py` — Light: remove `apply_litellm_patches` mock
- `tests/engine/models/conftest.py` — Replace 2 fixtures that construct LiteLLM response objects (use `openai` SDK types instead — same shape)
- `scripts/benchmarks/benchmark_engine_v2.py` — Replace `CustomRouter` import/patches with `ModelClient` mocks

**Dependency:**
- `pyproject.toml` — Add `openai` as direct dependency (already transitive via litellm; no new weight)

**NOT touched:** `models/` directory is entirely unchanged. `engine/mcp/`, column generators, dataset builders, config layer, validators — all unchanged. No response format changes means no cross-layer ripple.

---

### Phase 2: Validate

**No code changes.** Validation only:
- Benchmark: `--mode compare` between `models/` (litellm, env var off) and `models_v2/` (direct SDK, env var on)
- Full test suite: `uv run pytest packages/data-designer-engine/tests -x -q`
- Real inference: pdf_qa recipe or equivalent with `DATA_DESIGNER_ASYNC_ENGINE=1`

---

### Phase 3: Additional provider adapters

**New files:**
- `models_v2/adapters/anthropic.py` — `AnthropicModelClient`
- `models_v2/adapters/bedrock.py` — `BedrockModelClient`

**Modification:**
- `config/models.py` — `ModelProvider.provider_type: str` → `ProviderType` enum with Pydantic string coercion
- `models_v2/factory.py` — Adapter selection: `match provider_type` → construct appropriate `ModelClient`
- `engine/mcp/facade.py` — If Anthropic's flat tool_use blocks need different extraction than OpenAI's nested format, the tool call normalization logic needs updating. **This is the highest-risk cross-layer change.**
- `models/utils.py` — `ChatMessage.to_dict()` may need to support Anthropic's message format for multi-turn conversations with tool calls

**Tests:**
- New test files for Anthropic and Bedrock adapters
- Update tool call extraction tests if MCP facade changes

**Dependency:**
- `pyproject.toml` — Add `anthropic`. Bedrock: add `boto3`/`aiobotocore` or use `asyncio.to_thread()` with sync boto3.

**NOT touched:** `models/` still unchanged — litellm fallback remains available.

---

### Phase 4: Consolidate and drop dependency

**Delete entirely:**
- `models/` directory (all files: `facade.py`, `errors.py`, `factory.py`, `litellm_overrides.py`, `registry.py`, `usage.py`, `telemetry.py`, `utils.py`, `parsers/`, `recipes/`)
- `tests/engine/models/test_litellm_overrides.py`

**Modification:**
- `models/__init__.py` — Remove the path redirect hack; `models_v2/` becomes the sole implementation (or rename to `models/`)
- `lazy_heavy_imports.py` — Remove `litellm` from lazy import registry
- `pyproject.toml` — Remove `litellm>=1.73.6,<1.80.12`
- `uv.lock` — Regenerate

**Cleanup (non-functional):**
- `config/column_configs.py` — Docstring mentions "via LiteLLM"
- `engine/resources/resource_provider.py` — Comment mentions "heavy dependencies like litellm"
- `engine/mcp/facade.py` — Type hint comment references `litellm.ModelResponse`
- `README.md`, `AGENTS.md` — Documentation references to LiteLLM
- `async_concurrency.py` — Comment mentions "libraries (like LiteLLM)"

---

## What Needs to Be Built

### 1. `ModelClient` Interface + Provider Adapters

Replace LiteLLM Router with a `ModelClient` protocol and thin adapter classes that wrap the official provider SDKs. **No raw HTTP** — the SDKs handle networking, connection pooling, retry/backoff, and rate limiting internally.

```python
class ModelClient(Protocol):
    def completion(self, messages: list[dict], **kwargs) -> CompletionResponse: ...
    async def acompletion(self, messages: list[dict], **kwargs) -> CompletionResponse: ...
    def embedding(self, input_texts: list[str], **kwargs) -> EmbeddingResponse: ...
    async def aembedding(self, input_texts: list[str], **kwargs) -> EmbeddingResponse: ...
```

Each adapter's job is purely **translation**:
- Translate DD's `ModelConfig` + `InferenceParams` → SDK-specific call parameters
- Call the SDK method (e.g., `await self._client.chat.completions.create(...)`)
- Translate the SDK response → DD's `CompletionResponse` / `EmbeddingResponse`
- Catch SDK-specific exceptions → DD error types

The SDK client is created once in the factory (same lifecycle as `CustomRouter` today) and reused for all calls. No need for a dedicated I/O service like `mcp/io.py` — the official SDKs already manage connection pools, event loops, and request lifecycle internally.

### 2. Response Types

Define lightweight response dataclasses replacing LiteLLM's `ModelResponse` and `EmbeddingResponse`:

```python
@dataclass
class CompletionResponse:
    content: str
    reasoning_content: str | None
    tool_calls: list[ToolCall] | None
    usage: UsageInfo | None

@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    usage: UsageInfo | None

@dataclass
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int
```

**Scope**: ~50 lines. The existing code already accesses these fields — the dataclass just formalizes the contract.

### 3. Error Handling

Each SDK has its own exception hierarchy. The adapter for each provider catches SDK-specific exceptions and maps them to DD's existing error types.

**OpenAI SDK** — exception types map almost 1:1 to DD errors (LiteLLM's exception hierarchy was modeled on OpenAI's):
`BadRequestError(400)`, `AuthenticationError(401)`, `PermissionDeniedError(403)`, `NotFoundError(404)`, `RateLimitError(429)`, `InternalServerError(5xx)`, `APIConnectionError`, `APITimeoutError`

**Anthropic SDK** — simpler hierarchy:
`APIStatusError` (with `.status_code`), `RateLimitError`, `APIConnectionError`, `APITimeoutError`. Adapter checks status code for finer-grained mapping.

**Bedrock** — all errors via `botocore.exceptions.ClientError`:
Check `response['Error']['Code']` for `ValidationException(400)`, `AccessDeniedException(403)`, `ThrottlingException(429)`, `InternalServerException(500)`, etc.

**Nuance**: Some providers encode context window errors as 400 with specific error messages. The existing `parse_context_window_exceeded_error()` and `parse_bad_request_error()` logic handles this — it would need to match on response body strings, same as today.

The DD error types and `FormattedLLMErrorMessage` formatting already exist. Only the matching logic changes per adapter.

### 4. Retry & Backoff

**This varies significantly by provider — the claim that "each SDK handles its own retries" is only partially true.**

**OpenAI SDK**: Built-in. Default 2 retries, exponential backoff + jitter (0.5s initial, 8s cap). Auto-retries 408, 409, 429, 5xx. Respects `Retry-After` headers (capped at 60s). Configurable via `max_retries` on client or per-request. DD's `CustomRouter` defaults (2s initial, 20% jitter, 3 retries) become client configuration.

**Anthropic SDK**: Built-in. Same defaults and behavior as OpenAI SDK (0.5s initial, 8s cap, 2 retries). Auto-retries connection errors, 408, 409, 429, 5xx. Same `max_retries` configuration.

**Bedrock**: **Retry is NOT fully handled.** boto3 has three retry modes (legacy, standard, adaptive), but `ThrottlingException` (429 rate limiting) is **not auto-retried in any mode**. Only `ModelNotReadyException` (also 429, for cold-start) is auto-retried. DD must implement its own retry logic for Bedrock throttling — exponential backoff with jitter, same as `CustomRouter` does today.

**Bottom line**: For OpenAI and Anthropic, DD can rely on the SDK's built-in retry. For Bedrock, DD needs a standalone retry utility (port of `CustomRouter`'s backoff logic) or a wrapper around the Bedrock adapter.

### 5. Provider Adapters (use official SDKs directly)

**Decision**: DD needs multi-provider support, but the set is bounded: OpenAI, Anthropic, and Bedrock. Use each provider's official SDK directly:

- **OpenAI-compatible** (`openai`): Covers OpenAI, NVIDIA NIM, Azure OpenAI, and any OpenAI-compatible endpoint. Native async via `AsyncOpenAI`. Built-in retry + rate limit handling. Response format is what DD already expects (`choices[0].message.content`). **Thinnest adapter — mostly passthrough.**
- **Anthropic** (`anthropic`): Native async via `AsyncAnthropic`. Built-in retry. But response format differs: content is an array of typed blocks (`text`, `tool_use`, `thinking`), not a single string. Extended thinking is native but structurally different. **Adapter must translate content blocks → DD's expected format.**
- **Bedrock** (`boto3` / `aiobotocore`): Sync-only in boto3; async requires `aiobotocore` as an additional dependency (or `asyncio.to_thread()` as a simpler fallback). No auto-retry for throttling. Response format is Bedrock-native (`output.message.content[].text`), not OpenAI-compatible. **Most adapter work: retry, async wrapper, and response translation.** AWS does offer an `/openai/v1` endpoint that returns OpenAI-compatible responses, which could reduce translation work.

Each adapter implements the `ModelClient` interface, translating DD's `ModelConfig` into the SDK-specific call and normalizing the response back to DD's `CompletionResponse` / `EmbeddingResponse` types.

**Key insight**: This approach also simplifies retry/backoff — each SDK handles its own retries natively. DD's `CustomRouter` backoff logic may reduce to just configuration on the underlying client, rather than a reimplementation.

---

## Recommended Approach

### Architecture: Parallel stack in `models_v2/`

The `engine/models/` and `engine/models_v2/` directories are near-complete copies, switchable at runtime via a `__init__.py` path redirect on `DATA_DESIGNER_ASYNC_ENGINE`. Only 2 of 14 files actually differ (`facade.py` adds async methods, `errors.py` adds async decorator). The other 12 are pure copy-paste duplicates.

**Strategy**: Build the new non-litellm implementation entirely within `models_v2/`. Leave `models/` untouched as the stable litellm-backed fallback. The existing env var switch (`DATA_DESIGNER_ASYNC_ENGINE=1`) already gates which module path is used. Once `models_v2/` is validated in production, consolidate by deleting `models/` and dropping the litellm dependency.

This approach:
- Avoids a risky big-bang swap — litellm remains available as fallback
- Contains all new work to `models_v2/` (6 files to modify, not 12)
- Reuses the existing runtime switching mechanism
- Defers consolidation and dep removal to a clean follow-up

### Phase 1: Replace Router with ModelClient in `models_v2/`
- Define `ModelClient` protocol and DD-owned response types in `models_v2/`
- Implement `OpenAIModelClient` using the `openai` SDK (already a transitive dep)
- Rewrite `models_v2/facade.py` to use `ModelClient` instead of `CustomRouter`
- Rewrite `models_v2/errors.py` to match on OpenAI SDK exceptions instead of litellm exceptions
- Remove `models_v2/litellm_overrides.py` and litellm imports from `models_v2/factory.py`
- Update response access sites within `models_v2/` (and any shared code that receives responses)
- **Result**: `models_v2/` is litellm-free, `models/` is unchanged

### Phase 2: Validate
- Run benchmark: `--mode compare` to verify identical output between `models/` (litellm) and `models_v2/` (direct SDK)
- Run full test suite
- Run real inference (pdf_qa recipe or equivalent) with `DATA_DESIGNER_ASYNC_ENGINE=1`
- **Result**: Confidence that the new stack is correct

### Phase 3: Additional provider adapters
- `AnthropicModelClient`: HIGH risk — content block → string translation, tool_use block → OpenAI tool_calls format, thinking block → reasoning_content. Requires changes to MCP facade tool extraction and ChatMessage serialization.
- `BedrockModelClient`: HIGH risk — manual throttle retry, async via `to_thread` or `aiobotocore`, response format translation from Converse API shape.
- `ProviderType` enum in config with Pydantic string coercion for backwards compatibility
- Each adapter raises explicit `UnsupportedFeatureError` for capabilities the provider doesn't support
- **Result**: Full provider coverage; `models/` (litellm) still available as fallback until all adapters are proven

### Phase 4: Consolidate and drop dependency
- Delete `models/` directory
- Remove `__init__.py` path redirect hack
- Remove `litellm` from `pyproject.toml`
- Remove `litellm` from `lazy_heavy_imports.py`
- Clean up `uv.lock`
- Update documentation references
- **Result**: Single implementation, litellm fully removed. Only after all provider adapters are validated in production.

---

## Design Decisions (Answered)

1. ~~**Which `provider_type` values are used in production?**~~ **Answered**: OpenAI (including NIM), Anthropic, and Bedrock. Bounded set — use official SDKs directly behind a `ModelClient` interface.

2. ~~**Is streaming on the roadmap?**~~ **Answered**: No. Streaming will not be supported. This simplifies the `ModelClient` interface — no need for `stream_completion()` or async iterator return types. Each method is a simple request/response call.

3. ~~**How does `ModelConfig.provider_type` map to adapter selection?**~~ **Answered**: `provider_type` should become an **enum**, not remain a free-form string. The enum values determine which `ModelClient` adapter is instantiated. This makes the mapping explicit and catches misconfiguration at validation time rather than at runtime.

4. ~~**What about provider-specific features?**~~ **Answered**: The `ModelClient` interface targets the **OpenAI feature superset** — completions, embeddings, tool calling, extended thinking, etc. OpenAI's adapter is the full implementation. Other providers implement what they can and **raise explicit incompatibility errors** for unsupported features (e.g., if a provider doesn't support tool calling, the adapter raises a clear error rather than silently degrading). This means:
   - `ModelClient` defines the full interface anchored to OpenAI's capabilities
   - `OpenAIModelClient` implements everything
   - `AnthropicModelClient` implements everything but translates response formats (content blocks → string, thinking blocks, tool_use blocks)
   - `BedrockModelClient` implements core completions/embeddings; raises `UnsupportedFeatureError` for anything Bedrock's Converse API doesn't support

---

## Risk Assessment

| Component | Risk | Why |
|---|---|---|
| `ModelClient` interface | Low | Well-understood contract; OpenAI response shape is the reference |
| OpenAI adapter | Low | Thinnest adapter — response format matches, retry built into SDK, `openai` already a transitive dep |
| Anthropic adapter | **High** | Content blocks vs strings, flat tool_use vs nested function format, thinking blocks vs reasoning_content field. Leaks into MCP facade and trace storage. |
| Bedrock adapter | **High** | Manual throttle retry, async requires extra dep, non-OpenAI response format |
| Error mapping (OpenAI) | Low | SDK exceptions map 1:1 to DD errors (LiteLLM modeled its hierarchy on OpenAI's) |
| Response type migration | Medium | Existing code accesses `response.choices[0].message.content` everywhere — either keep structural parity or coordinate refactor of all access sites |
| Test migration | Medium | ~56 test functions need modification; 26 require significant rework (all in `tests/engine/models/`) |
| Parallel stack validation | Low | Same env-var gating pattern as the async engine; benchmark already validates output correctness |

OpenAI adapter + validation (Phases 1-2) is low risk. The high-risk work is Anthropic and Bedrock adapters (Phase 4), which can be deferred and tackled independently.

---

## Review Findings (Moth Swarm)

10 independent reviewers examined this report against the actual codebase. Key corrections and additions:

### Blast radius is larger than stated

The original count of 2 test files is incomplete. The full test impact:

| File | Impact | Details |
|---|---|---|
| `test_facade.py` | Heavy rewrite | 26 test functions, imports `ModelResponse`/`EmbeddingResponse`/`Choices`/`Message` from litellm, 6 `CustomRouter` patches |
| `test_litellm_overrides.py` | Delete | 11 tests for `CustomRouter`, `ThreadSafeCache`, `apply_litellm_patches()` |
| `test_model_errors.py` | Medium rewrite | 7 tests, imports all 12 `litellm.exceptions.*` types for parametrized error mapping tests |
| `test_model_registry.py` | Light touch | 1 test patches `apply_litellm_patches()` |
| `conftest.py` (models) | Light touch | 2 fixtures construct `ModelResponse` and `EmbeddingResponse` objects |
| `benchmark_engine_v2.py` | Medium rewrite | Imports `CustomRouter`, patches `completion` and `acompletion` |

Total: ~56 test functions need modification, with 26 requiring significant rework. All contained within `tests/engine/models/`.

### Anthropic adapter is HIGH risk, not medium

The Anthropic SDK's response format is structurally incompatible with DD's assumptions:

- **Content is an array of typed blocks** (`text`, `thinking`, `tool_use`), not `choices[0].message.content` (string). DD accesses `response.choices[0].message.content` in facade.py, models_v2/facade.py, and mcp/facade.py.
- **Tool calls are flat content blocks** (`name`, `input`), not nested OpenAI format (`function.name`, `function.arguments`). The MCP facade's tool extraction logic (`mcp/facade.py:340-353`) assumes the nested structure.
- **Reasoning is a content block**, not a field. DD uses `getattr(message, "reasoning_content", None)` — Anthropic returns `ThinkingBlock(type="thinking", thinking="...")` in the content array.
- **This leaks beyond the adapter** into the MCP facade (tool extraction), the generation loop (content access), ChatMessage serialization (trace storage), and multimodal content formatting.

The Anthropic adapter requires refactoring core response handling, not just wrapping the SDK.

### ModelClient interface needs more specificity

The proposed 4-method interface is too thin:
- Missing explicit `model` parameter (currently passed per-request to Router)
- `ToolCall` type is undefined — needs `id`, `type`, `function.name`, `function.arguments`
- Response type structure decision: the proposed flat `CompletionResponse` breaks all existing `response.choices[0].message.content` access sites. Either keep structural parity with OpenAI's response shape (nested `choices[0].message`) or coordinate a refactor of every access site.
- `consolidate_kwargs()` merges `inference_parameters.generate_kwargs`, `extra_body`, and `extra_headers` before calling the Router — the adapter contract should document what's in `**kwargs` when it arrives.

### Retry-after header extraction is LiteLLM-specific

`CustomRouter._extract_retry_delay_from_headers()` uses:
- `exception.litellm_response_headers` (LiteLLM-specific attribute)
- `exception.response.headers` (httpx attribute, works with OpenAI/Anthropic SDKs)
- `litellm.utils._get_retry_after_from_exception_header()` (LiteLLM utility)

For OpenAI/Anthropic, the SDKs handle retry internally — this logic is only needed for Bedrock. But the Retry-After header parsing utility (`_get_retry_after_from_exception_header`) needs reimplementation as a standalone function.

### Dependency impact is lighter than expected

- `openai` is already a transitive dependency via litellm — promoting to direct dep adds zero new weight
- `httpx` stays regardless — DD depends on it directly for `engine/models/telemetry.py` and `engine/validators/remote.py`
- Net dependency change for OpenAI-only: a reduction (~10-15 packages removed with litellm)
- Adding Anthropic is lightweight (most deps already present via openai)
- Bedrock (boto3/botocore) is the heaviest new addition

### Config migration is clean

`provider_type` is only consumed in one place (`_get_litellm_deployment()` → `f"{provider_type}/{model_name}"`). Pydantic handles string → enum coercion automatically, so existing YAML/JSON configs and programmatic construction continue to work. The CLI text field for provider_type would become a select/choice field. All predefined providers and examples use `"openai"` — no existing users need migration.
