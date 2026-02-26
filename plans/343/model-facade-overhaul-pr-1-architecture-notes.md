---
date: 2026-02-25
authors:
  - nmulepati
---

# Model Facade Overhaul PR-1 Architecture Notes

This document captures the architecture intent for PR-1 from
`plans/343/model-facade-overhaul-plan-step-1.md`.

## Canonical Adapter Boundary

PR-1 introduces an internal `ModelClient` boundary under:

`packages/data-designer-engine/src/data_designer/engine/models/clients/`

Boundary contract:

1. `ModelFacade`-facing requests/responses use canonical dataclasses in `clients/types.py`.
2. Provider SDK and transport-specific response shapes do not leak above the adapter layer.
3. Provider failures normalize to canonical provider errors (`ProviderError`, `ProviderErrorKind`).

Canonical operation types in PR-1:

1. Chat completion (`ChatCompletionRequest` / `ChatCompletionResponse`)
2. Embeddings (`EmbeddingRequest` / `EmbeddingResponse`)
3. Image generation (`ImageGenerationRequest` / `ImageGenerationResponse`)

## LiteLLM Bridge Purpose

`LiteLLMBridgeClient` is a temporary adapter that preserves migration safety:

1. It wraps the existing LiteLLM router while emitting canonical response types.
2. It enables parity testing of canonical request/response contracts before native provider adapters are cut over.
3. It remains available as a rollback path during native adapter soak windows.

PR-1 is intentionally non-invasive:

1. No `ModelFacade` call-site behavior changes.
2. No provider routing changes.
3. No retry/throttle lifecycle migration yet.

## Planned Follow-On

In PR-2, `ModelFacade` will switch from direct router usage to `ModelClient` implementations,
consuming canonical responses from the bridge first, then native adapters.
