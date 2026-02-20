---
date: 2026-02-19
authors:
  - nmulepati
---

# Model Facade Overhaul Plan: Step 2 (Bedrock)

This step adds native Bedrock support after Step 1 is complete.

Depends on:

1. `plans/343/model-facade-overhaul-plan-step-1.md`

## Reviewer Snapshot

Reviewers should verify:

1. Step 2 reuses Step 1 abstractions (client boundary, retry, throttling, error model) without forking patterns.
2. Bedrock-specific logic remains isolated to adapter/mappers/auth resolution.
3. Capability gating is explicit and fails early when an operation is unsupported for a model family.

## Entry Criteria

Step 2 starts only when these Step 1 conditions are met:

1. `ModelFacade` is fully backed by `ModelClient` abstraction.
2. Shared retry and adaptive throttle modules are in production.
3. OpenAI-compatible + Anthropic parity gates are complete.
4. Feature-flag-based rollback path is validated.

## Architecture Delta Diagram

Step 2 extends the Step 1 client layer by adding Bedrock-specific paths only.

```text
             Step 1 (existing)
ModelFacade -> ModelClient API -> [OpenAI Adapter | Anthropic Adapter | Bridge]

             Step 2 (new)
ModelFacade -> ModelClient API -> [OpenAI | Anthropic | Bedrock | Bridge?]
                                         |
                                         v
                                 Bedrock Mapper Layer
                           (claude/llama/nova/titan/stability)
                                         |
                                         v
                               AWS Bedrock Runtime API
```

Design intent:

1. No `ModelFacade` API changes in Step 2.
2. Bedrock complexity is contained under adapter + mapper + auth resolution.

## Scope

1. Add `BedrockClient` adapter under `engine/models/clients/adapters/bedrock.py`.
2. Add Bedrock auth schema and resolver support.
3. Add Bedrock request/response normalization to canonical model client types.
4. Integrate Bedrock into client factory routing.
5. Add Bedrock-specific tests (unit + integration stubs + optional smoke tests).

Out of scope:

1. Re-design of `ModelFacade` public API (already covered by Step 1).
2. Reworking shared retry/throttle abstractions except Bedrock-specific mappings.

## File-Level Change Map (Step 2)

New files:

1. `packages/data-designer-engine/src/data_designer/engine/models/clients/adapters/bedrock.py`
2. `packages/data-designer-engine/src/data_designer/engine/models/clients/bedrock_mappers/claude.py`
3. `packages/data-designer-engine/src/data_designer/engine/models/clients/bedrock_mappers/llama.py`
4. `packages/data-designer-engine/src/data_designer/engine/models/clients/bedrock_mappers/nova.py`
5. `packages/data-designer-engine/src/data_designer/engine/models/clients/bedrock_mappers/titan.py`
6. `packages/data-designer-engine/src/data_designer/engine/models/clients/bedrock_mappers/stability.py` (if enabled)

Updated files:

1. `packages/data-designer-engine/src/data_designer/engine/models/clients/factory.py` (route `provider_type=bedrock`)
2. `packages/data-designer-config/src/data_designer/config/models.py` (Bedrock auth schema)
3. `packages/data-designer/src/data_designer/cli/forms/provider_builder.py` (Bedrock auth input UX)
4. `packages/data-designer-engine/src/data_designer/engine/models/errors.py` (Bedrock-specific error normalization coverage)

## Adapter Design

### Factory routing

1. `provider_type == "bedrock"` -> `BedrockClient`
2. Unknown bedrock auth mode or invalid config -> `ValueError` with actionable message

### Operations

1. Chat completion: supported by model-family mapper.
2. Embeddings: supported by model-family mapper.
3. Image generation: supported by model-family mapper.
4. Unsupported operation for chosen model -> canonical `ProviderError(kind=UNSUPPORTED_CAPABILITY)`.

### Model family mappers

Implement Bedrock mappers per family:

1. `bedrock_mappers/claude.py`
2. `bedrock_mappers/llama.py`
3. `bedrock_mappers/nova.py`
4. `bedrock_mappers/titan.py`
5. `bedrock_mappers/stability.py` (if configured)

Each mapper provides:

1. canonical request -> Bedrock payload conversion
2. Bedrock response -> canonical response conversion
3. capability declaration per operation

## Authentication

Bedrock uses SigV4, not API key headers.

### Supported auth modes

1. `default_chain`: environment/profile/role chain
2. `profile`: named AWS profile
3. `access_key`: explicit key/secret/session token via `SecretResolver`
4. `assume_role`: STS role assumption

### Bedrock auth schema

```python
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class BedrockEnvAuth(BaseModel):
    mode: Literal["default_chain"] = "default_chain"
    region: str


class BedrockProfileAuth(BaseModel):
    mode: Literal["profile"] = "profile"
    region: str
    profile_name: str


class BedrockKeyAuth(BaseModel):
    mode: Literal["access_key"] = "access_key"
    region: str
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


class BedrockAssumeRoleAuth(BaseModel):
    mode: Literal["assume_role"] = "assume_role"
    region: str
    role_arn: str
    external_id: str | None = None
    session_name: str = "data-designer-bedrock"


BedrockAuth = Annotated[
    BedrockEnvAuth | BedrockProfileAuth | BedrockKeyAuth | BedrockAssumeRoleAuth,
    Field(discriminator="mode"),
]
```

### Auth error mapping

1. `UnrecognizedClientException` -> `ProviderError(kind=AUTHENTICATION)`
2. `AccessDeniedException` -> `ProviderError(kind=PERMISSION_DENIED)`
3. STS assume-role failures -> `ProviderError(kind=AUTHENTICATION | PERMISSION_DENIED)` based on error code

## Throttling and Concurrency

Use the same shared throttling framework introduced in Step 1.

### Keys

1. Global cap key: `(provider_name, model_identifier)`
2. Domain key: `(provider_name, model_identifier, throttle_domain)`

### Domain selection

1. Bedrock chat/converse calls -> `chat`
2. Bedrock embedding calls -> `embedding`
3. Bedrock image calls -> `image`

### 429/throttling signals

1. SDK throttling exceptions (`ThrottlingException` family)
2. retry hints when present in SDK metadata

## Testing

### Unit tests

1. mapper payload conversion tests
2. mapper response parsing tests
3. Bedrock auth mode resolution tests
4. Bedrock error-kind mapping tests
5. throttling signal mapping tests

### Integration-style tests

1. `botocore.stub.Stubber` for runtime responses
2. mixed sync/async throttling behavior with shared keys

### Optional smoke tests

1. gated by env vars/credentials
2. excluded from default CI

## Delivery Phases

### Phase A: Bedrock schema and factory wiring

1. Add Bedrock auth schema types and validation.
2. Add factory route for `provider_type == "bedrock"`.
3. Add placeholder adapter with unsupported-capability errors.

### Phase B: Chat completion support

1. Implement chat mapper(s).
2. Implement sync/async chat calls with canonical response conversion.
3. Add chat contract tests.

### Phase C: Embeddings and image support

1. Implement embedding and image mappers.
2. Add capability guards for unsupported model families.
3. Add usage and error parity tests.

### Phase D: Hardening and rollout

1. Add retry/throttle tuning for Bedrock exceptions.
2. Run staged rollout and soak tests.
3. Enable by default after parity gates pass.

## Definition of Done

1. `provider_type="bedrock"` resolves to `BedrockClient`.
2. Bedrock auth modes validate and resolve secrets correctly.
3. Chat/embedding/image operations are normalized to canonical response types.
4. Bedrock error mapping surfaces existing user-facing `DataDesignerError` classes.
5. Shared adaptive throttling works with Bedrock throttling signals.
6. Test coverage includes mapper logic, auth, errors, and throttling behavior.

## Risks and Mitigations

### Risk: model-family payload fragmentation

Mitigation:

1. strict per-family mapper contracts with canonical input/output tests
2. fail-fast unsupported-capability checks before outbound call

### Risk: AWS auth misconfiguration complexity

Mitigation:

1. explicit auth-mode validation and actionable errors
2. CI stub tests for all auth modes + optional smoke tests in controlled environment

### Risk: throttling differences vs HTTP providers

Mitigation:

1. explicit mapping from Bedrock throttling exceptions to canonical `RATE_LIMIT`
2. stress tests validating AIMD behavior under SDK exception patterns
