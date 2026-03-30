---
date: 2026-03-30
authors:
  - nmulepati
issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/479
---

# Plan: `skip_when` — Conditional Column Generation

## Problem

DataDesigner's DAG executes every column for every row unconditionally. In multi-stage synthesis pipelines, expensive downstream generation (LLM calls, segmentation, etc.) runs even when an earlier gate column indicates the row should be filtered out.

Today the only workarounds are:

1. **Generate all columns unconditionally and post-filter** — wastes LLM calls on rows that will be discarded.
2. **Split into multiple `DataDesigner.create()` calls** with intermediate filtering — loses single-pipeline ergonomics and forces the user to manage seed-dataset hand-offs.

## Proposed Solution

Add a `skip_when` field to `SingleColumnConfig`. When its Jinja2 expression evaluates truthy for a row, the cell is set to `None` and the generator is never called. Skips auto-propagate through the DAG: downstream columns whose `required_columns` include a skipped cell also skip automatically.

Example: a pipeline that generates product reviews only for items in stock. The `sentiment_analysis` and `review` columns are expensive LLM calls that should be skipped for out-of-stock items:

```python
config_builder.add_column(
    name="in_stock", column_type="sampler",
    sampler_type="bernoulli", params=BernoulliSamplerParams(p=0.7),
)
config_builder.add_column(
    name="sentiment_analysis",
    column_type="llm-structured",
    skip_when="{{ in_stock == 0 }}",
    prompt="Analyze the sentiment of reviews for {{ product_name }}...",
    ...
)
# review depends on sentiment_analysis — auto-skips when sentiment_analysis is skipped
config_builder.add_column(
    name="review",
    column_type="llm-text",
    prompt="Write a {{ sentiment_analysis.tone }} review for {{ product_name }}...",
)
```

Skipped rows stay in the output (row count is preserved). Skipped cells contain `None`.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Where does `skip_when` live? | `SingleColumnConfig` base class | Cross-cutting; applies to all column types |
| What happens to skipped cells? | Set to `None`, row stays in output | Rows are not dropped — users can post-filter or inspect |
| Do downstream columns auto-skip? | Yes, always | If upstream data is missing, generating downstream is wasteful and error-prone |
| How are `skip_when` columns ordered in the DAG? | `skip_when_columns` extracted from the expression become DAG edges | Ensures the gate column is generated before the guarded column |
| How does this interact with `_records_to_drop`? | Independently — skip does not drop rows | Skip produces `None`; drop removes the row entirely |

---

## Implementation

### 1. Config: `SingleColumnConfig` — add field + property

**File:** `packages/data-designer-config/src/data_designer/config/base.py`

Add to `SingleColumnConfig` (after `allow_resize`):

```python
skip_when: str | None = Field(
    default=None,
    description="Jinja2 expression; when truthy, skip generation for this row (cell set to None).",
)
```

Add a `@field_validator("skip_when")` that validates Jinja2 syntax. **Critical constraint:** `base.py` line 4 prohibits `data_designer.*` imports, so use `jinja2` directly:

```python
@field_validator("skip_when")
@classmethod
def _validate_skip_when(cls, v: str | None) -> str | None:
    if v is not None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        ImmutableSandboxedEnvironment().parse(v)
    return v
```

Add a concrete property `skip_when_columns` (not abstract — base provides default):

```python
@property
def skip_when_columns(self) -> list[str]:
    if self.skip_when is None:
        return []
    from jinja2 import meta
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment()
    ast = env.parse(self.skip_when)
    return list(meta.find_undeclared_variables(ast))
```

### 2. DAG: add `skip_when_columns` edges

#### 2a. `dag.py` — `topologically_sort_column_configs()`

**File:** `packages/data-designer-engine/src/data_designer/engine/dataset_builders/utils/dag.py`

After the `for req_col_name in col.required_columns:` block (line 35-47), add a matching block for `col.skip_when_columns` that adds edges using the same pattern (direct column match + side-effect resolution).

#### 2b. `execution_graph.py` — `ExecutionGraph.create()`

**File:** `packages/data-designer-engine/src/data_designer/engine/dataset_builders/utils/execution_graph.py`

In the second pass (line 78-88), after the `for req in sub.required_columns:` edge loop, add:

```python
for skip_col in sub.skip_when_columns:
    resolved = graph.resolve_side_effect(skip_col)
    if resolved not in known_columns:
        continue  # seed/sampler columns not in graph
    if resolved == name:
        continue
    graph.add_edge(upstream=resolved, downstream=name)
```

Store skip metadata on the graph for runtime access:

- Add `_skip_when: dict[str, str]` to `__init__`
- Populate during first pass: `graph._skip_when[name] = sub.skip_when` (when not None)
- Add accessor: `get_skip_when(column) -> str | None`

### 3. New utility: `skip_evaluator.py`

**New file:** `packages/data-designer-engine/src/data_designer/engine/dataset_builders/utils/skip_evaluator.py`

Two pure functions, no engine state dependencies:

```python
def evaluate_skip_when(expression: str, record: dict) -> bool:
    """Render expression against record; return True if truthy."""

def should_skip_by_propagation(
    required_columns: list[str],
    skipped_columns_for_row: set[str],
) -> bool:
    """Return True if any required column was skipped."""
```

`evaluate_skip_when` wraps expression in `{{ }}`, renders via `ImmutableSandboxedEnvironment`, checks truthiness (result not in `("", "false", "0", "none", "null")`).

`should_skip_by_propagation` returns `True` if the intersection of `required_columns` and `skipped_columns_for_row` is non-empty.

### 4. Sync engine: `DatasetBuilder`

**File:** `packages/data-designer-engine/src/data_designer/engine/dataset_builders/dataset_builder.py`

#### 4a. Add state
- `self._skipped_cells: dict[int, set[str]] = {}` — buffer index to skipped column names
- Clear at start of `_run_batch()` (line ~428)

#### 4b. Add helper methods
- `_should_skip_cell(config, record_index, record) -> bool` — checks propagation (any upstream skipped?) then evaluates `skip_when` expression
- `_mark_cell_skipped(record_index, column_name, side_effect_columns, record)` — records skip, writes `None` to buffer

#### 4c. Modify `_fan_out_with_threads()` (line 638)
Before `executor.submit()`, check `_should_skip_cell()`. If skip: write `None` for column + side effects, record success on progress tracker, `continue`.

#### 4d. Modify `_fan_out_with_async()` (line 621)
Convert list comprehension to explicit loop with same skip check.

#### 4e. Modify `_run_full_column_generator()` (line 503)
After `generator.generate()` returns, iterate records. For each row where `_should_skip_cell()` is true, overwrite that column + side effects with `None` and record in `_skipped_cells`. Replace buffer with updated records.

### 5. Async engine: `AsyncTaskScheduler`

#### 5a. `CompletionTracker` — add skip tracking

**File:** `packages/data-designer-engine/src/data_designer/engine/dataset_builders/utils/completion_tracker.py`

- Add `_skipped: dict[int, dict[int, set[str]]]` (rg -> ri -> column names)
- `mark_cell_skipped(column, row_group, row_index)`
- `get_skipped_columns_for_row(row_group, row_index) -> set[str]`

#### 5b. Modify `_run_cell()` (line 767 of `async_scheduler.py`)

After the `is_dropped` guard (line 772), add skip evaluation:

1. Get `skipped_cols` from tracker for this row
2. Check `should_skip_by_propagation` using `config.required_columns` and `skipped_cols`
3. If not propagation-skipped, check `evaluate_skip_when` using `self._graph.get_skip_when(task.column)` against `row_data`
4. If skip: write `None` to buffer for all output cols, call `tracker.mark_cell_skipped()`, return `None`

The caller (`_execute_task_inner_impl`) still marks the task complete — skipped cells ARE complete (they produced `None`). Downstream tasks get unblocked and will themselves check propagation.

#### 5c. Modify `_run_batch()` (line 792 of `async_scheduler.py`)

After generation, iterate rows. For skipped rows, overwrite with `None` and mark in tracker. Same pattern as sync path Step 4e.

### 6. Expression generator: defensive `None` guard

**File:** `packages/data-designer-engine/src/data_designer/engine/column_generators/generators/expression.py`

In `generate()` (line 24), inside the per-record loop: if any `required_columns` value is `None`, set `record[self.config.name] = None` instead of rendering the Jinja2 template (which would crash on `None` arithmetic like `{{ price * 1.1 }}`).

### 7. Validation: check `skip_when` references

**File:** `packages/data-designer-engine/src/data_designer/engine/validation.py`

- Add `SKIP_WHEN_REFERENCE_MISSING` to `ViolationType` enum
- Add `validate_skip_when_references(columns, allowed_references)` — iterates columns with `skip_when`, checks each `skip_when_columns` entry exists in `allowed_references`
- Wire into `validate_data_designer_config()`

---

## Files Modified

| File | Change |
|---|---|
| `config/base.py` | `skip_when` field + validator + `skip_when_columns` property |
| `engine/.../dag.py` | Add `skip_when_columns` edges in topological sort |
| `engine/.../execution_graph.py` | Add `skip_when_columns` edges + skip metadata storage + accessor |
| `engine/.../skip_evaluator.py` | **NEW** — `evaluate_skip_when()`, `should_skip_by_propagation()` |
| `engine/.../dataset_builder.py` | `_skipped_cells` state, `_should_skip_cell()`, modify 3 fan-out/generation methods |
| `engine/.../async_scheduler.py` | Skip checks in `_run_cell()` and `_run_batch()` |
| `engine/.../completion_tracker.py` | `_skipped` dict + `mark_cell_skipped` + `get_skipped_columns_for_row` |
| `engine/.../expression.py` | Defensive `None` guard when upstream is null |
| `engine/validation.py` | `validate_skip_when_references()` |

---

## Open Questions

### 1. What value should skipped cells contain?

The current plan sets skipped cells to `None` (which becomes `NaN`/`pd.NA` in the DataFrame). Alternatives:

- **`None`** — simple, standard pandas null. Downside: indistinguishable from a legitimate `None` produced by a generator (e.g., an LLM that returns no output).
- **Sentinel value** (e.g., `SKIPPED = "__SKIPPED__"` or a dedicated `SkippedValue` type) — distinguishable from real nulls. Downside: leaks into user-facing DataFrames unless stripped at output time; complicates type handling.
- **`pd.NA` with metadata** — store skip status in a sidecar structure (the `_skipped_cells` / `CompletionTracker._skipped` dicts already track this) and write `None` to the cell. Users who need to distinguish skip-null from real-null can inspect the metadata.

Recommendation: Use `None` in the cell, track skip provenance in engine-internal state. If users need to distinguish, expose a `results.load_skip_mask()` or similar accessor.

### 2. Should there be an option to auto-remove skipped rows from the final output?

Many pipelines want to discard rows where a gate column failed — they don't need the skipped rows in the output at all. Options:

- **Post-hoc filtering by the user** — `df = df.dropna(subset=["categories"])`. Simple but manual.
- **`drop_skipped_rows` option on `DataDesigner.create()`** — automatically remove any row where at least one column was skipped before writing to disk. Clean UX but may surprise users who want to inspect skipped rows.
- **A built-in `DropSkippedRowsProcessorConfig` processor** — runs as a post-generation processor that removes rows with any skipped cells. Fits the existing processor model and is opt-in.
- **`drop_when_skipped=True` on individual column configs** — drop the row if *this specific column* was skipped. More granular than a global flag.

Recommendation: Start with a `DropSkippedRowsProcessorConfig` processor — it's opt-in, composable with other processors, and doesn't require new parameters on `create()` or column configs.

---

## Verification

1. **Unit tests:** Config field defaults, Jinja2 validation, `skip_when_columns` extraction, DAG edge creation, skip evaluator truthiness, CompletionTracker skip tracking
2. **Integration tests (sync):** Column with `skip_when` produces `None` for matching rows; downstream auto-skips; row count preserved (no drops)
3. **Integration tests (async):** Same scenarios under `DATA_DESIGNER_ASYNC_ENGINE=1`
4. **Validation tests:** Unknown column in `skip_when` produces ERROR violation
5. **Run:** `make check-all-fix` + `make test` + `make update-license-headers`
