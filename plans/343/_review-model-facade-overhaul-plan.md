---
date: 2026-02-20
reviewer: codex
branch: nm/overhaul-model-facade-guts
sources:
  - plans/343/model-facade-overhaul-plan-review-codex.md
  - plans/343/model-facade-overhaul-plan-review-opus.md
scope:
  - plans/343/model-facade-overhaul-plan-step-1.md
  - plans/343/model-facade-overhaul-plan-step-2-bedrock.md
---

# Aggregated Plan Review (Agreed Feedback)

## Verdict

Request changes. The migration direction is solid, but several contradictions and missing contracts should be resolved before implementation.

## Findings

### HIGH: `completion`/`acompletion` response contract is not explicit enough for MCP compatibility

Evidence:
- Step 1 says keep methods/signatures and preserve MCP loop unchanged (`plans/343/model-facade-overhaul-plan-step-1.md:924`, `plans/343/model-facade-overhaul-plan-step-1.md:928`).
- Step 1 also says ModelFacade will consume canonical response shapes (`plans/343/model-facade-overhaul-plan-step-1.md:927`) and maps LiteLLM fields into canonical message fields (`plans/343/model-facade-overhaul-plan-step-1.md:703`).
- Current MCP shape parity risk is explicitly called out (`plans/343/model-facade-overhaul-plan-step-1.md:1160`).

Recommendation:
- Pick one explicit migration contract:
  1. Refactor ModelFacade + MCP helpers to canonical response shape in PR-2, or
  2. Keep LiteLLM-compatible response shape through bridge phase.
- Add dedicated MCP parity tests (tool-call extraction/refusal/reasoning content) for the chosen contract.

### HIGH: Adaptive throttling contract is internally inconsistent

Evidence:
- Shared hard cap is defined as `min(max_parallel_requests...)` across aliases (`plans/343/model-facade-overhaul-plan-step-1.md:838`, `plans/343/model-facade-overhaul-plan-step-1.md:839`).
- Test plan also expects aliases on same provider/model to keep independent primary limits (`plans/343/model-facade-overhaul-plan-step-1.md:1093`).

Recommendation:
- Choose one contract and align tests:
  1. Shared hard cap across aliases (safest), or
  2. Per-alias limits with optional shared pressure signal.
- Remove contradictory assertions from the parity suite.

### HIGH: Step 1 provider-type hardening conflicts with Step 2 Bedrock routing

Evidence:
- Step 1 Phase B constrains provider type enum to `openai` and `anthropic` (`plans/343/model-facade-overhaul-plan-step-1.md:1038`).
- Step 2 requires `provider_type == "bedrock"` factory routing (`plans/343/model-facade-overhaul-plan-step-2-bedrock.md:32`).

Recommendation:
- Keep `provider_type` extensible through Step 1 (or include `bedrock` in planned enum values).
- Defer strict enum narrowing until after Bedrock support lands.

### HIGH: Rollback safety is inconsistent with default-flip/removal sequencing

Evidence:
- PR slicing combines cutover default flip and LiteLLM removal in PR-6 (`plans/343/model-facade-overhaul-plan-step-1.md:136`).
- Rollback guardrail promises backend flag toggle (`plans/343/model-facade-overhaul-plan-step-1.md:1152`).
- Phase 3 removes LiteLLM runtime path/dependency (`plans/343/model-facade-overhaul-plan-step-1.md:1141`, `plans/343/model-facade-overhaul-plan-step-1.md:1142`).

Recommendation:
- Separate default flip from dependency/path removal.
- Keep bridge rollback path for at least one soak/release window after native default.

### MEDIUM: Auth error mapping is ambiguous for `401/403`

Evidence:
- Step 1 maps `401/403` to `AUTHENTICATION | PERMISSION_DENIED` with no deterministic rule (`plans/343/model-facade-overhaul-plan-step-1.md:508`, `plans/343/model-facade-overhaul-plan-step-1.md:509`).
- Step 2 has same ambiguity for STS failures (`plans/343/model-facade-overhaul-plan-step-2-bedrock.md:116`).

Recommendation:
- Define deterministic status/code mapping (default `401 -> AUTHENTICATION`, `403 -> PERMISSION_DENIED`) and document provider-specific exceptions.
- Add explicit parity tests for this matrix.

### MEDIUM: HTTP client lifecycle and pool sizing are underspecified

Evidence:
- HTTP adapter skeleton instantiates both sync/async httpx clients and exposes `close`/`aclose` (`plans/343/model-facade-overhaul-plan-step-1.md:583`, `plans/343/model-facade-overhaul-plan-step-1.md:584`, `plans/343/model-facade-overhaul-plan-step-1.md:594`, `plans/343/model-facade-overhaul-plan-step-1.md:597`).
- Plan does not define owner/teardown integration or pool sizing policy.

Recommendation:
- Add lifecycle section specifying creation/ownership/teardown (factory, registry, or facade shutdown hook).
- Define connection pool sizing relative to concurrency settings.

### MEDIUM: `extra_body`/`extra_headers` precedence needs explicit contract

Evidence:
- Plan promises no precedence drift but does not state merge order (`plans/343/model-facade-overhaul-plan-step-1.md:975`, `plans/343/model-facade-overhaul-plan-step-1.md:1014`, `plans/343/model-facade-overhaul-plan-step-1.md:1015`).
- Existing implementation has specific precedence rules (provider overrides for `extra_body`, provider replacement for `extra_headers`).

Recommendation:
- Document exact merge precedence and add regression tests for it.

### MEDIUM: Anthropic capability statements conflict within Step 1

Evidence:
- Anthropic adapter skeleton marks embeddings and image generation unsupported (`plans/343/model-facade-overhaul-plan-step-1.md:656`, `plans/343/model-facade-overhaul-plan-step-1.md:660`, `plans/343/model-facade-overhaul-plan-step-1.md:674`, `plans/343/model-facade-overhaul-plan-step-1.md:680`).
- Capability matrix says Anthropic support is "Provider dependent" for those operations (`plans/343/model-facade-overhaul-plan-step-1.md:946`, `plans/343/model-facade-overhaul-plan-step-1.md:947`, `plans/343/model-facade-overhaul-plan-step-1.md:948`).

Recommendation:
- Align matrix with Step 1 implementation scope, or define exact conditions and gates for conditional support.
