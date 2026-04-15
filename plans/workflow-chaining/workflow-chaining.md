---
date: 2026-04-15
authors:
  - amanoel
---

# Plan: Workflow chaining and `allow_resize` removal

## Problem

DataDesigner workflows are self-contained: one config, one `create()` call, one output. There is no first-class way to combine workflows in sequence, where the output of one feeds the input of the next. Users who need this must manually wire `DataFrameSeedSource` between calls.

Separately, the `allow_resize` flag on column configs lets a generator change the row count mid-generation. This works in the sync engine via in-place buffer replacement, but is fundamentally incompatible with the async engine's fixed-size `CompletionTracker` grid. The async engine currently rejects `allow_resize=True` with a validation error. Pre-batch processors that resize have a similar problem: the async path handles shrinking accidentally (via drop-marking), but expansion is silently ignored.

These are the same problem viewed from different angles: the need to change row counts between generation steps.

## Proposed solution

Replace the in-place resize mechanism with **workflow chaining**: a thin orchestration layer that sequences multiple generation stages, passing each stage's output as the next stage's seed dataset.

This is a three-part change:

1. **Remove `allow_resize`** from the column config and all engine code that supports it.
2. **Disallow row-count changes in pre-batch processors** (fail-fast if the processor returns a different number of rows).
3. **Add a `Pipeline` class** in the interface layer that auto-chains stages, with support for explicit multi-stage configs.

### Why chaining instead of fixing async resize

The async scheduler's `CompletionTracker` pre-allocates a (row_group x row_index x column) task grid. Supporting mid-run resize requires either rebuilding the tracker (complex, error-prone) or pausing execution at resize boundaries (loses parallelism). Chaining sidesteps this entirely: each stage gets a fresh tracker sized to its actual input. The engine stays simple - always fixed-size - and resize becomes a between-stage concern.

## Design

### Part 1: Remove `allow_resize`

**Config changes** (`data-designer-config`):

- Remove `allow_resize: bool = False` from `SingleColumnConfig` (or its base class `ColumnConfigBase`).
- Deprecation: keep the field for one release cycle with a deprecation warning, then remove.

**Engine changes** (`data-designer-engine`):

- Remove `_cell_resize_mode`, `_cell_resize_results`, and the resize branch in `_finalize_fan_out()` from `DatasetBuilder`.
- Remove `allow_resize` parameter from `DatasetBatchManager.replace_buffer()`.
- Remove `_validate_async_compatibility()` (no longer needed - nothing to reject).
- Simplify `_run_full_column_generator()` to always enforce row-count invariance.

**Migration path**: Users with `allow_resize=True` columns split their config into a pipeline with a stage boundary at the resize column. The resize column becomes the last column of its stage, and downstream columns move to the next stage.

### Part 2: Fail-fast on pre-batch processor resize

In `ProcessorRunner.run_pre_batch()` and `run_pre_batch_on_df()`, raise `DatasetProcessingError` if the returned DataFrame has a different row count than the input.

This applies to both sync and async paths. Users who need to filter or expand between seeds and generation use the pipeline's between-stage callback instead.

For users who need programmatic filtering at the seed boundary, a seed reader plugin is the escape hatch (the seed reader can filter/transform before the engine ever sees the data).

### Part 3: Pipeline class

A new `Pipeline` class in `data_designer.interface` that orchestrates multi-stage generation.

#### User-facing API

**Explicit multi-stage pipeline:**

```python
pipeline = dd.pipeline()
pipeline.add_stage("personas", config_personas, num_records=100)
pipeline.add_stage("conversations", config_convos, num_records=1000)  # explode: 100 -> 1000
pipeline.add_stage("judged", config_judge)  # defaults to previous stage's output size

results = pipeline.run()

results["personas"].load_dataset()       # stage 1 output
results["conversations"].load_dataset()  # stage 2 output
results["judged"].load_dataset()         # final output
```

**Auto-chaining from a single config (future):**

The engine detects columns that were previously `allow_resize=True` (or a new marker like `stage_boundary=True`) and auto-splits the DAG into stages. This is a convenience layer on top of the explicit API - not required for v1.

#### Between-stage callbacks

Users may need to transform data between stages. The pipeline supports an optional callback:

```python
def filter_high_quality(stage_output_path: Path) -> Path:
    df = pd.read_parquet(stage_output_path / "data")
    df = df[df["quality_score"] > 0.8]
    out = stage_output_path.parent / "filtered"
    df.to_parquet(out / "data.parquet")
    return out

pipeline.add_stage("generated", config_gen, num_records=1000)
pipeline.add_stage(
    "enriched",
    config_enrich,
    after=filter_high_quality,  # runs on stage output before next stage seeds from it
)
```

The callback receives the path to the completed stage's artifacts and returns a path to the (possibly modified) artifacts. This keeps large DataFrames on disk and gives users full control.

The callback signature is `(Path) -> Path`. If the user returns the same path, no copy is made. If they return a new path, the next stage seeds from that.

#### `num_records` behavior

- If `num_records` is explicitly set on a stage, that value is used.
- If omitted, defaults to the previous stage's output row count (after any between-stage callback).
- The seed reader's existing cycling behavior handles the explode case: requesting 1000 records from a 100-row seed cycles through the seed 10 times.

#### Artifact management

Each stage writes to its own subdirectory under the pipeline's artifact path:

```
artifacts/
  pipeline-name/
    stage-1-personas/
      parquet-files/
      metadata.json
    stage-2-conversations/
      parquet-files/
      metadata.json
    stage-3-judged/
      parquet-files/
      metadata.json
    pipeline-metadata.json  # stage order, configs, lineage
```

#### Checkpointing and resume

Each stage produces durable parquet output before the next stage starts. This provides natural checkpoint boundaries:

- If stage 3 of 4 fails, stages 1 and 2 are already on disk.
- A `resume=True` flag on `pipeline.run()` skips completed stages (detected via `pipeline-metadata.json`).
- Within a stage, batch-level resume (#525) can further reduce re-work.

The connection to #525: chaining gives coarse (stage-level) checkpointing for free. #525 gives fine (batch-level) checkpointing within a stage. They are complementary.

#### Provenance

`pipeline-metadata.json` records:
- Stage order, names, and configs used
- `num_records` requested vs actual per stage
- Which stage's output seeded the next
- Timestamp and duration per stage

### Where it fits in the architecture

| Layer | Changes |
|-------|---------|
| `data-designer-config` | Remove `allow_resize` field. No new config models needed for v1 (pipeline is imperative, not declarative). |
| `data-designer-engine` | Remove resize code paths. Add fail-fast guard in `ProcessorRunner`. No new engine features. |
| `data-designer` (interface) | New `Pipeline` class. Thin orchestration: calls `DataDesigner.create()` per stage, wires `DataFrameSeedSource` between stages for in-memory handoff or `LocalFileSeedSource` for on-disk handoff. |

The engine does not know about pipelines. Each stage is a regular `DatasetBuilder.build()` call.

## Implementation phases

### Phase 1: Pipeline class (can ship independently)

- Add `Pipeline` class with `add_stage()`, `run()`, between-stage callbacks.
- Add `pipeline-metadata.json` writing.
- Add `dd.pipeline()` factory method on `DataDesigner`.
- Tests: multi-stage runs, explode/filter via callbacks, num_records defaulting, artifact layout.

### Phase 2: Remove `allow_resize`

- Deprecate `allow_resize` with a warning pointing to pipelines.
- Remove resize code from sync engine (`_cell_resize_mode`, `_finalize_fan_out` resize branch, `replace_buffer` `allow_resize` param).
- Remove `_validate_async_compatibility()` from async engine.
- Add fail-fast guard in `ProcessorRunner` for pre-batch row-count changes.
- Tests: verify rejection, migration path examples.

### Phase 3: Stage-level resume

- Add `resume=True` to `pipeline.run()`.
- Read `pipeline-metadata.json` to detect completed stages.
- Skip completed stages, seed next stage from last completed output.
- Depends on artifact layout from phase 1.

### Phase 4 (future): Auto-chaining from single config

- Detect stage boundaries in the DAG (via a new config marker or heuristic).
- Auto-split into pipeline stages internally.
- User sees a single `dd.create(config)` call but gets multi-stage execution.

## Open questions

1. **In-memory vs on-disk handoff between stages**: For small datasets, `DataFrameSeedSource` avoids disk I/O. For large datasets, writing parquet between stages is safer. Should the pipeline auto-detect based on row count, or always go through disk for consistency?

2. **Preview support**: Should `pipeline.preview()` run all stages with small `num_records`? Or just preview the last stage seeded from a prior full run?

3. **Config serialization**: A pipeline config can't be serialized to YAML if stages use `DataFrameSeedSource`. For persistence, stages would need symbolic references ("seed from stage X's output"). This is needed for auto-chaining (phase 4) but not for the explicit API (phases 1-3).

4. **Naming**: `Pipeline` vs `Chain` vs `WorkflowChain`. `Pipeline` is the most intuitive and aligns with ML pipeline terminology.

## Related issues

- #447 - AsyncRunController refactor (partially superseded: pre-batch resize handling moves to pipeline level instead of controller level)
- #525 - Resume interrupted runs (complementary: stage-level resume from pipeline, batch-level resume from #525)
- #462 - Progress bar and scheduler polish (independent)
- #464 - Custom column retryable errors (independent)
