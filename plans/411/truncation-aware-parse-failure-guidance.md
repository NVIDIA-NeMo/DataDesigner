---
date: 2026-07-13
authors:
  - stepwise-ai-dev
issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/411
pull_request: https://github.com/NVIDIA-NeMo/DataDesigner/pull/754
---

# Plan: Truncation-aware parse failure guidance

## Problem

When structured model output cannot be parsed, DataDesigner reports a generic validation failure. That message is incomplete when the response ended because the model exhausted its output-token budget or context window: the parser is operating on truncated content, and schema changes or retries alone may not fix the problem.

PR #754 added useful guidance, but it was written before the legacy sync engine was removed. Its boolean truncation flag now crosses more layers than the current async error flow requires, and it only represents the final failed parse attempt. The current implementation also treats Anthropic `max_tokens` without distinguishing `model_context_window_exceeded`, even though the remedies differ.

## Current state

- OpenAI-compatible responses already normalize `choices[*].finish_reason` into `ChatCompletionChoice.finish_reason`.
- The Anthropic adapter preserves its response in `raw` but does not populate canonical choice finish reasons.
- `ModelFacade.generate()` and `agenerate()` own correction and conversation-restart loops.
- `GenerationValidationFailureError` is the internal validation boundary; `handle_llm_exceptions()` converts it into the public `ModelGenerationValidationFailureError`.
- The async scheduler already logs the formatted public exception before dropping the affected row. No scheduler-specific truncation metadata is needed.

## Goals

- Detect parse failures preceded by output-token or context-window termination.
- Preserve that reason across correction attempts and conversation restarts.
- Give users remediation specific to the detected reason.
- Normalize Anthropic termination metadata at the adapter boundary.
- Keep termination metadata on the internal validation error only.
- Let the existing async scheduler logging path surface the formatted guidance.
- Address the current maintainer feedback with a small current-main implementation.

## Non-goals

- Do not restore or modify the removed sync engine.
- Do not add truncation state to `ModelGenerationValidationFailureError`.
- Do not change retry counts, correction control flow, row-drop policy, or scheduler retryability.
- Do not build a general provider termination taxonomy beyond the two reasons needed by #411 and the cited Anthropic behavior.
- Do not refactor unrelated model-error messages or adapter response parsing.
- Do not add public APIs or configuration fields.

## Design

### Canonical adapter metadata

`parse_anthropic_response()` will place Anthropic's string `stop_reason` into `choices[0].finish_reason`, matching the canonical response representation already used by OpenAI-compatible adapters. Existing message, usage, and raw response fields remain unchanged.

Expected canonical values:

| Provider response | Canonical finish reason | Internal reason |
|---|---|---|
| OpenAI `finish_reason="length"` | `length` | `MAX_TOKENS` |
| Anthropic `stop_reason="max_tokens"` | `max_tokens` | `MAX_TOKENS` |
| Anthropic `stop_reason="model_context_window_exceeded"` | `model_context_window_exceeded` | `MODEL_CONTEXT_WINDOW_EXCEEDED` |
| Normal stop/tool use | provider value | none |

### Internal reason type

Add a small `GenerationTruncationReason` enum in `models/errors.py` with:

- `MAX_TOKENS`
- `MODEL_CONTEXT_WINDOW_EXCEEDED`

`GenerationValidationFailureError` gains an optional `truncation_reason`. The public `ModelGenerationValidationFailureError` remains unchanged; it continues to expose only normalized detail and failure kind.

This keeps provider-independent classification available to `handle_llm_exceptions()` without threading a boolean through the public error or the scheduler.

### Detection and compatibility fallback

`ModelFacade` will classify the canonical `choices[0].finish_reason` first. A short raw-response fallback remains for custom or future adapters that do not yet populate canonical choices.

The fallback will reuse `get_value_from()` and `get_first_value_or_none()` from `models/clients/parsing.py`. It will not introduce duplicate response-access helpers.

### Accumulation and precedence

The facade keeps one local `GenerationTruncationReason | None` for each `generate()` or `agenerate()` call. Whenever parsing fails, it merges the current response's reason into the accumulated reason before deciding whether to correct, restart, or raise.

Precedence is deterministic:

1. `MODEL_CONTEXT_WINDOW_EXCEEDED`
2. `MAX_TOKENS`
3. no classified reason

Context-window exhaustion wins because advising the user to increase `max_tokens` would be wrong after the model has already exhausted its total context window. A successful parse returns normally regardless of earlier failed attempts; accumulated state matters only if generation ultimately fails.

### User-facing messages

`handle_llm_exceptions()` formats the reason-specific message:

- `MAX_TOKENS`: explain that the response was cut off at the output-token limit and recommend increasing `inference_parameters.max_tokens`.
- `MODEL_CONTEXT_WINDOW_EXCEEDED`: explain that the model exhausted its context window and recommend reducing prompt/context/schema size or selecting a model with a larger context window.
- no classified reason: preserve the existing generic validation guidance.

The scheduler receives and logs the existing public exception normally. No `getattr()`, public boolean, or scheduler formatting branch is introduced.

## Files

Production changes:

- `packages/data-designer-engine/src/data_designer/engine/models/clients/adapters/anthropic_translation.py`
- `packages/data-designer-engine/src/data_designer/engine/models/errors.py`
- `packages/data-designer-engine/src/data_designer/engine/models/facade.py`

Focused tests:

- `packages/data-designer-engine/tests/engine/models/clients/test_anthropic_translation.py`
- `packages/data-designer-engine/tests/engine/models/test_facade.py`
- `packages/data-designer-engine/tests/engine/models/test_model_errors.py`
- `packages/data-designer-engine/tests/engine/dataset_builders/test_async_scheduler.py`

Explicitly excluded from the implementation:

- `packages/data-designer-engine/src/data_designer/engine/dataset_builders/dataset_builder.py`
- `packages/data-designer-engine/tests/engine/dataset_builders/test_dataset_builder.py`

## Test strategy

All new behavior is implemented test-first.

1. Adapter normalization:
   - Anthropic `end_turn`, `tool_use`, `max_tokens`, and `model_context_window_exceeded` become canonical finish reasons.
2. One-attempt behavior:
   - sync and async parse failures classify canonical OpenAI and Anthropic reasons;
   - raw fallback remains covered;
   - unclassified stops preserve the generic message.
3. Accumulation:
   - an early `max_tokens` response followed by an unclassified final failure still reports output-token truncation;
   - correction and restart paths are both covered across sync/async tests;
   - context-window exhaustion takes precedence when both reasons occur.
4. Error formatting:
   - each typed reason produces its specific remediation;
   - generic validation behavior and public error attributes remain unchanged.
5. Scheduler integration:
   - the existing non-retryable dropped-row warning includes the already-formatted reason-specific message without scheduler production changes.

Validation order:

1. Run each new focused test and observe the expected failure before implementation.
2. Run the four touched test modules.
3. Run `make check-engine`.
4. Run `make check-all-fix` and review any automatic changes.
5. Run the repository-required full `make test` suite before requesting review.
6. Record whether E2E coverage is applicable; run applicable E2E checks or explicitly report why they were not run.

## Delivery strategy

Delivery proceeds in reviewable stages from current `main`:

1. submit this plan on the existing PR and confirm the direction with maintainers;
2. add Anthropic normalization and its focused adapter tests;
3. add internal truncation classification, accumulation, formatting, and the scheduler regression;
4. run the focused tests and repository-required validation;
5. address review findings in additional commits that name the resolved concern.

Implementation began only after Andrea approved this direction on 2026-07-15, satisfying the `CONTRIBUTING.md` plan gate.
