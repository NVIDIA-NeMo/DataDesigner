---
date: 2026-05-27
status: draft
authors:
  - mknepper
---

# Plan: Standalone Model & Tool Health Check on `DataDesigner`

## Problem

Today, the only way to verify that the models and MCP tools referenced by a
configuration are actually reachable is to start a workload. `DatasetBuilder.build()`
and `DatasetBuilder.build_preview()` both call
`_run_model_health_check_if_needed()` followed by `_run_mcp_tool_check_if_needed()`
as their first action
(`packages/data-designer-engine/src/data_designer/engine/dataset_builders/dataset_builder.py:258`,
`:592`). To find out a credential is wrong, an alias is unregistered, or an MCP
server is down, a user has to invoke `preview()` or `create()` and wait for it
to fail at the same gate.

`DataDesigner.validate(config_builder)` already covers the *internal* readiness
question — "is this configuration well-formed against my engine components?" —
but there is no symmetric *external* readiness method to ask "are the providers
this configuration depends on actually responsive?".

## Goals

1. Expose a public method on the `DataDesigner` interface that runs the same
   external-readiness checks as workload startup, with no other side effects
   (no artifact directory population, no batches, no profiling).
2. Cover both **models** and **MCP tools** — they are run together at workload
   startup, and a user asking "am I good to go?" expects the same coverage.
3. Mirror `validate()` in shape and error policy: takes a config builder,
   returns `None`, raises typed engine errors. Add a CLI command alongside the
   existing `validate` command.
4. Avoid drift between this method and the workload startup path by extracting
   the shared logic into the engine and calling it from both places.

## Non-goals

- Reworking the existing health-check probes themselves (still a tiny `"Hello!"`
  generation per model; no changes to `ModelRegistry.run_health_check` /
  `arun_health_check` semantics).
- Concurrency limits, partial-failure aggregation, or per-alias filtering. The
  method is a one-shot pass-through that fails fast on the first error, exactly
  like the startup path. These can be follow-ups if requested.
- Touching `skip_health_check` semantics. Models with `skip_health_check=True`
  remain skipped, just as they are at startup.

## Design

### Naming

Method: `DataDesigner.check_models(config_builder)`. CLI: `dd check-models`.

"Models" in this codebase already names the externally-hosted resources a
configuration depends on. MCP tools are coupled to model-using columns —
`_run_mcp_tool_check_if_needed` collects aliases from
`llm_generated_column_configs` (`dataset_builder.py:1354`), so any config with
tools necessarily has at least one model-using column; you cannot have a
"tools-only" config that this method would mis-name. Treating MCP tool
liveness as part of "checking the models" is consistent with how the codebase
already groups them at the startup gate.

This pairs cleanly with the existing surface: `validate` for internal
readiness, `check_models` for external readiness.

### Public method on `DataDesigner`

Add to `packages/data-designer/src/data_designer/interface/data_designer.py`,
adjacent to `validate` (line 533):

```python
def check_models(self, config_builder: DataDesignerConfigBuilder) -> None:
    """Probe every model and MCP tool referenced by the configuration.

    Runs the same readiness checks performed at the start of ``preview`` and
    ``create``: a tiny generation against each referenced model alias, plus
    a connectivity probe to each referenced MCP tool. Models whose
    ``ModelConfig`` has ``skip_health_check=True`` are skipped.

    Args:
        config_builder: The DataDesignerConfigBuilder whose column configs
            determine which model aliases and tool aliases are probed.

    Returns:
        None if every (non-skipped) probe succeeded.

    Raises:
        Typed model errors from ``data_designer.engine.models.errors``
            (e.g. ``ModelAuthenticationError``, ``ModelNotFoundError``,
            ``ModelAPIConnectionError``) for any failing model probe.
        DatasetGenerationError: If a tool alias is referenced but no
            ``MCPRegistry`` is configured.
        TimeoutError: If async health-check execution exceeds 180 seconds.
    """
    resource_provider = self._create_resource_provider("check-models", config_builder)
    config = config_builder.build()
    run_readiness_check(config.columns, resource_provider)
```

### Shared helper module

Add a new module
`packages/data-designer-engine/src/data_designer/engine/readiness.py`. It
hosts the shared logic so the standalone method and the workload startup
path can never drift:

```python
# data_designer/engine/readiness.py

def run_readiness_check(
    column_configs: Sequence[ColumnConfig],
    resource_provider: ResourceProvider,
) -> None:
    """Run model + MCP tool health checks for the given column configs.

    Used by both ``DatasetBuilder.build``/``build_preview`` (at workload start)
    and ``DataDesigner.check_models`` (standalone).
    """
    _run_model_health_check(column_configs, resource_provider)
    _run_mcp_tool_health_check(column_configs, resource_provider)


def _run_model_health_check(
    column_configs: Sequence[ColumnConfig],
    resource_provider: ResourceProvider,
) -> None:
    ...  # body lifted from DatasetBuilder._run_model_health_check_if_needed


def _run_mcp_tool_health_check(
    column_configs: Sequence[ColumnConfig],
    resource_provider: ResourceProvider,
) -> None:
    ...  # body lifted from DatasetBuilder._run_mcp_tool_check_if_needed
```

The bodies are essentially the existing
`_run_model_health_check_if_needed` (`dataset_builder.py:1330`) and
`_run_mcp_tool_check_if_needed` (`dataset_builder.py:1352`), rewritten to
take `column_configs` and `resource_provider` as arguments instead of reading
`self.single_column_configs` and `self._resource_provider`.

`run_readiness_check` is the only public symbol the module exports;
`_run_model_health_check` and `_run_mcp_tool_health_check` stay
module-private. This is achievable because the two `DatasetBuilder` call
sites that currently invoke the model check and tool check back-to-back
(`dataset_builder.py:258-259` and `:592-593`) have no logic between them and
both run unconditionally — they collapse cleanly to a single
`run_readiness_check(...)` call:

```python
# dataset_builder.py:258 and :592 (after the change)
run_readiness_check(self.single_column_configs, self._resource_provider)
```

The two `DatasetBuilder._run_*_if_needed` instance methods are removed
entirely (no longer needed as delegating wrappers). Builder tests that
currently patch `_run_mcp_tool_check_if_needed` to suppress MCP probing
during construction-focused tests
(`test_dataset_builder.py:2546, 2596, 2633, 2666`) migrate to patching
`run_readiness_check` at its import site in `dataset_builder.py`. In every
case those tests want the readiness gate as a whole bypassed, so widening
from "MCP-only" to "readiness as a whole" matches their actual intent.

A new module is preferred over keeping the helpers in `dataset_builder.py`:
the readiness pass is conceptually a separate phase of execution, callable
from outside the dataset builder, and putting it in its own file keeps that
boundary explicit.

### CLI command

Add `packages/data-designer/src/data_designer/cli/commands/check_models.py`,
modelled on `validate.py`. Wire it into the CLI in
`packages/data-designer/src/data_designer/cli/__init__.py` next to the
`validate` registration:

- `dd validate` — internal readiness (config well-formedness)
- `dd check-models` — external readiness (provider liveness)

### Async dispatch contract

`run_readiness_check` preserves the existing `DATA_DESIGNER_ASYNC_ENGINE`
switch and 180-second timeout for the model probes. MCP tool probes remain
sync because `MCPRegistry` has no async health-check method today
(`packages/data-designer-engine/src/data_designer/engine/mcp/registry.py:180`).
This matches current startup behavior exactly: the workload-startup pass
also runs MCP probes synchronously regardless of which engine is in use, so
nothing changes in observable behavior with this refactor. Adding async
parity for MCP is tracked as a follow-up (see Follow-ups).

### Logging

Both probes already produce informative logs from the registry layer (the
`🩺 Running health checks for models...` line at
`packages/data-designer-engine/src/data_designer/engine/models/registry.py:229`,
plus per-alias `✅ Passed!` / `❌ Failed!` lines). The new method needs only a
single high-level entry log, mirroring how `validate` currently emits one
log line via `compile_data_designer_config`. No additional verbosity.

---

## Files to change

### `packages/data-designer-engine/`

| File | Change |
|---|---|
| `src/data_designer/engine/readiness.py` | New module. Public surface: `run_readiness_check`. Module-private: `_run_model_health_check`, `_run_mcp_tool_health_check`. |
| `src/data_designer/engine/dataset_builders/dataset_builder.py` | Remove `_run_model_health_check_if_needed` (`:1330`) and `_run_mcp_tool_check_if_needed` (`:1352`). Replace each pair of call sites at `:258-259` and `:592-593` with a single `run_readiness_check(self.single_column_configs, self._resource_provider)` call. |
| `tests/engine/dataset_builders/test_dataset_builder.py` | The four `patch.object(builder, "_run_mcp_tool_check_if_needed")` sites (`:2546, :2596, :2633, :2666`) migrate to patching `run_readiness_check` at its import site in `dataset_builder`. Existing model-check tests at `:327, :362` migrate similarly. Coverage is preserved (and arguably improved — those tests want the readiness gate bypassed, which is now expressible in one patch instead of two). |
| `tests/engine/test_readiness.py` (new) | Direct tests for `run_readiness_check`: success, model auth failure surfaces verbatim, MCP missing-registry surfaces `DatasetGenerationError`, no-aliases short-circuit, async-engine timeout, `skip_health_check=True` honored. Component functions are exercised through `run_readiness_check` rather than directly, since they're module-private. |

### `packages/data-designer/`

| File | Change |
|---|---|
| `src/data_designer/interface/data_designer.py` | Add `check_models` method next to `validate`. Import `run_readiness_check` from `data_designer.engine.readiness`. |
| `src/data_designer/cli/commands/check_models.py` | New file. Mirrors `validate.py`. |
| `src/data_designer/cli/main.py` | Register the new command in the lazy Typer group. |
| `tests/interface/test_data_designer.py` | New tests for `check_models`: success path; surfaces `ModelAuthenticationError`; surfaces `DatasetGenerationError` for missing MCP registry; no-op when no models or tools are referenced; respects `skip_health_check=True`. |
| `tests/cli/commands/test_check_models_command.py` | New file mirroring `test_validate_command.py`. |

### Docs (handled in a follow-up via the `datadesigner-docs` skill)

- `docs/concepts/models/model-configs.md` — cross-reference the new method
  next to the existing `skip_health_check` discussion (`:112`).
- Data Designer API reference — add `check_models` next to `validate`.
- A short Fern page or section explaining the `validate` / `check_models`
  pair as the "pre-flight" surface.

Documentation is intentionally scoped out of this plan's first PR to keep the
change set focused; the implementation PR will note the docs follow-up.

---

## Validation

Engine-level tests already cover the registry probes themselves
(`packages/data-designer-engine/tests/engine/models/test_model_registry.py:329-449`).
The work here adds:

1. **Refactor safety** — existing `DatasetBuilder` health-check tests
   (`test_dataset_builder.py:327, 362, 2546, 2596, 2633, 2666`) keep passing
   without modification.
2. **New free-function tests** — direct coverage of `run_readiness_check`
   and the two component functions against fakes for both registries.
3. **Interface tests** — `DataDesigner.check_models` end-to-end with a
   mocked `ResourceProvider`/registries:
   - Success path returns `None`.
   - Model auth failure surfaces the typed error verbatim.
   - Missing MCP registry surfaces `DatasetGenerationError`.
   - Configuration with no model-using columns and no tools is a no-op.
   - `skip_health_check=True` aliases are not probed.
4. **CLI test** — invocation delegates to the controller, mirroring
   `test_validate_command.py`.

Manual verification before merge:

```bash
make check-all-fix
make test
```

Plus a smoke run with a real (or recorded) provider:

```bash
dd check-models path/to/config.yaml      # exits 0 on success
dd check-models path/to/bad-creds.yaml   # exits non-zero with typed error
```

---

## Follow-ups

- **Async parity for MCP tool probes.** Today, MCP probes go through
  `MCPRegistry.run_health_check` regardless of which engine is in use; there
  is no `arun_health_check`. This is a stylistic asymmetry rather than a
  correctness gap (the probe runs serially before any concurrent work
  begins), but adding async parity would let the readiness pass complete
  fully on the async event loop. Worth a tracking issue if/when MCP usage
  grows.
- **Per-alias filtering / partial reporting.** The current method is a
  one-shot fail-fast pass-through. If users want "tell me everything that's
  broken in one go" or "only check this subset of models," that's an
  additive API on top of the same engine helper.
