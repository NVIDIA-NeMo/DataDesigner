# Plan: Enrich `schema_text()` Output for Agent CLI

Created: 2026-03-15
Status: Complete

Issue: [#418](https://github.com/NVIDIA-NeMo/DataDesigner/issues/418)

## Goal

Make `data-designer agent schema` output self-contained so agents never need fallback
introspection calls (`help(Score)`, `print(list(CodeLang))`) to discover enum values
or nested type structures.

## Context

Benchmark analysis of 30 agent runs shows the `agent schema` output lacks nested type
details and enum values. Every run needing judge or code columns fell back to Python
introspection, costing 2-3 extra tool calls each. The fix is contained: enhance
`ConfigBase.schema_text()` so it renders complete type information in a single call.

## Success Criteria

- [x] Discriminator fields (`column_type: Literal[...]`) suppressed from output
- [x] Internal fields (`repr=False`) suppressed from output
- [x] Enum values shown inline after field description
- [x] Nested `ConfigBase` models expanded 1 level deep
- [x] Instantiation example appended at depth 0
- [x] Multi-member unions (discriminated unions) NOT expanded inline
- [x] All existing tests pass, new tests cover each behavior
- [x] `make check-all` clean

## Implementation Steps

### Step 1: Extract rendering logic into `base_utils.py`

Single public function `generate_schema_text()` in `config/base_utils.py` (same import
constraint as `base.py` — no `data_designer.*` imports except pydantic). Private helpers:
`_render_model`, `_render_field`, `_format_default`, `_format_annotation`,
`_unwrap_annotation`, `_is_discriminator_field`, `_find_expandable_types`,
`_get_docstring_summary`.

- [x] `ConfigBase.schema_text()` delegates to `generate_schema_text(cls, base_cls=ConfigBase)`

### Step 2: Skip discriminator and internal fields

- [x] `_is_discriminator_field()` detects single-value `Literal[x]` with matching default
- [x] Fields with `repr=False` skipped via `field_info.repr is False`
- [x] `SingleColumnConfig.allow_resize` marked with `repr=False`

### Step 3: Show enum values and expand nested models

- [x] `_find_expandable_types()` walks annotation tree for Enum / ConfigBase leaves
- [x] Enum subclass → append `values: bash, c, python, ...`
- [x] ConfigBase subclass (depth < 1) → append indented `_render_model()` output
- [x] Multi-member unions (discriminated unions) NOT expanded

### Step 4: Append instantiation example

- [x] After all fields (depth 0 only), append `Example: dd.ClassName(required=..., fields=...)`
- [x] Auto-generated from required non-discriminator field names

### Step 5: Supporting changes

- [x] `SingleColumnConfig.drop` field description added
- [x] `DropColumnsProcessorConfig` docstring updated to lead with `drop=True` preference
- [x] `get_schema()` error message improved with usage suggestions
- [x] PyArrow `DeprecationWarning` / `FutureWarning` suppressed in CLI `main.py`

### Step 6: Tests

- [x] 33 unit tests in `test_schema_text.py` covering all field variants, discriminator
  suppression, repr=False suppression, enum values, nested expansion, depth limiting,
  multi-member union non-expansion, all-fields-hidden, docstring truncation, examples
- [x] Updated `test_agent_introspection.py` for new format
- [x] Added real-world smoke tests for LLMJudgeColumnConfig and LLMCodeColumnConfig

## Verification

```bash
.venv/bin/pytest packages/data-designer-config/tests/config/test_schema_text.py -v
.venv/bin/pytest packages/data-designer/tests/cli/ -v

.venv/bin/python -m data_designer.cli.main agent schema columns llm-judge
.venv/bin/python -m data_designer.cli.main agent schema columns llm-code
.venv/bin/python -m data_designer.cli.main agent schema columns sampler

make check-all
```
