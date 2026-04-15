---
date: 2026-04-15
authors:
  - amanoel
---

# Plan: Workflow chaining

## Problem

DataDesigner workflows are self-contained: one config, one `create()` call, one output. There is no first-class way to combine workflows in sequence, where the output of one feeds the input of the next. Users who need this must manually wire `DataFrameSeedSource` between calls.

This matters for several use cases:

- **Filter-then-enrich**: Generate candidates, filter to high-quality rows, then generate detailed content from survivors. The second stage's row count depends on the first stage's filter output.
- **Explode**: Generate a small set of seed entities (e.g., 100 personas), then generate many records from each (e.g., 1000 conversations). The seed reader's cycling handles the expansion, but the user must manually wire stages.
- **Generate-then-judge**: Generate a dataset, then run a separate LLM-as-judge pass with different models or stricter prompts. Iterating on the judging config shouldn't require re-generating the base data.
- **Multi-turn construction**: Each conversation turn has a different prompt structure and possibly a different model. Composing these as sequential stages is more natural than a single flat config.

## Proposed solution

Add **workflow chaining**: a thin orchestration layer that sequences multiple generation stages, passing each stage's output as the next stage's seed dataset. This is the primary deliverable.

As a secondary benefit, chaining also enables the removal of `allow_resize` and simplification of the engine's resize handling.

### Secondary benefit: `allow_resize` removal and sync/async convergence

The `allow_resize` flag on column configs lets a generator change the row count mid-generation. This works in the sync engine but is fundamentally incompatible with the async engine's fixed-size `CompletionTracker` grid (currently rejected with a validation error). Pre-batch processors that resize have a similar problem.

`allow_resize` is one of the remaining divergences between sync and async. Since the long-term direction is to remove the sync engine entirely, maintaining a sync-only feature is counterproductive. With chaining in place, resize becomes a between-stage concern rather than a mid-generation concern. This lets us remove `allow_resize` and the associated engine complexity, and disallow row-count changes in pre-batch processors. Users who need resize use a pipeline with a stage boundary at the resize point.

Note: `allow_resize` is currently documented in custom columns, plugin examples, and agent rollout ingestion docs. Removal requires a deprecation cycle and doc updates.

### Why chaining instead of fixing async resize

The async scheduler's `CompletionTracker` pre-allocates a (row_group x row_index x column) task grid. Supporting mid-run resize requires either rebuilding the tracker (complex, error-prone) or pausing execution at resize boundaries (loses parallelism). Chaining sidesteps this entirely: each stage gets a fresh tracker sized to its actual input. The engine stays simple - always fixed-size - and resize becomes a between-stage concern.

## Design

### Part 1: Pipeline class

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

**Convenience method on results (lightweight, for notebooks):**

For interactive use where a full pipeline is overkill, a `to_config_builder()` method on `DatasetCreationResults` returns a pre-seeded `DataDesignerConfigBuilder`:

```python
# Stage 1
result = dd.create(config_personas, num_records=100)

# Stage 2 - just grab the result and keep going
config_convos = (
    result.to_config_builder(columns=["name", "age", "background"])  # optional column selection
    .add_column(name="conversation", column_type="llm_text", prompt="...")
)
result_2 = dd.create(config_convos, num_records=1000)
```

This is a thin wrapper: loads the dataset into memory, optionally filters columns, wraps in `DataFrameSeedSource`, returns a new config builder. No tracking, no provenance, no callbacks - just a quick bridge for iteration. Not suitable for large datasets (loads full DataFrame into memory) or serializable configs (`DataFrameSeedSource` can't be written to YAML). For production pipelines, use the `Pipeline` class.

**Auto-chaining from a single config (future):**

The engine detects columns that were previously `allow_resize=True` (or a new marker like `stage_boundary=True`) and auto-splits the DAG into stages. This is a convenience layer on top of the explicit API - not required for v1.

#### Stage data contract

Each stage seeds from the **previous stage's final dataset** - the post-processor output with dropped columns excluded. This is the same DataFrame returned by `DatasetCreationResults.load_dataset()`.

Processor outputs (named processor artifacts) and media assets (images stored on disk with relative paths in the DataFrame) are NOT automatically forwarded. If a downstream stage needs image columns from an upstream stage, the pipeline must resolve image paths relative to the upstream stage's artifact directory. This needs explicit handling - TBD in implementation.

#### Between-stage callbacks

Users may need to transform data between stages. The pipeline supports an optional callback:

```python
def filter_high_quality(stage_output_path: Path) -> Path:
    df = pd.read_parquet(stage_output_path / "parquet-files")
    df = df[df["quality_score"] > 0.8]
    out = stage_output_path.parent / "filtered"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "data.parquet")
    return out

pipeline.add_stage("generated", config_gen, num_records=1000)
pipeline.add_stage(
    "enriched",
    config_enrich,
    after=filter_high_quality,  # runs on stage output before next stage seeds from it
)
```

The callback receives the path to the completed stage's artifact directory (containing `parquet-files/`, `metadata.json`, etc.) and returns a path that the next stage will seed from. This keeps large DataFrames on disk and gives users full control.

**Empty stage policy**: If a callback filters all rows (or a stage produces zero rows), the pipeline raises `DataDesignerPipelineError` by default. Stages can opt in to empty output with `allow_empty=True` on `add_stage()`, in which case the pipeline short-circuits and skips subsequent stages.

#### `num_records` and seed behavior

- If `num_records` is explicitly set on a stage, that value is used.
- If omitted, defaults to the previous stage's output row count (after any between-stage callback).
- The seed reader's existing cycling behavior handles the explode case: requesting 1000 records from a 100-row seed cycles through the seed 10 times.
- `add_stage()` accepts optional `sampling_strategy` (ordered/shuffle) and `selection_strategy` (IndexRange/PartitionBlock) to control how the previous stage's output is sampled. Defaults to ordered.

#### Artifact management

The pipeline owns its directory layout directly, bypassing `ArtifactStorage`'s default auto-rename behavior (which appends timestamps to non-empty directories). Stage directories use stable, deterministic names based on stage index and name:

```
artifacts/
  pipeline-name/
    stage-0-personas/
      parquet-files/
      metadata.json
    stage-1-conversations/
      parquet-files/
      metadata.json
    stage-2-judged/
      parquet-files/
      metadata.json
    pipeline-metadata.json
```

The pipeline creates each stage's `ArtifactStorage` with the stage directory as `dataset_name`, ensuring stable paths across reruns.

#### Checkpointing and resume

Each stage produces durable parquet output before the next stage starts. This provides natural checkpoint boundaries:

- If stage 3 of 4 fails, stages 1 and 2 are already on disk.
- A `resume=True` flag on `pipeline.run()` skips completed stages.
- Within a stage, batch-level resume (#525) can further reduce re-work.

**Resume safety**: Naive "skip if directory exists" is not sufficient. Configs, model settings, callbacks, or DD version may have changed between runs. Resume must compare a fingerprint of each stage's inputs (config hash, num_records, DD version, upstream stage fingerprint) against what's recorded in `pipeline-metadata.json`. If any input changed, that stage and all downstream stages must re-run. This is a phase 3 concern but the metadata format in phase 1 should record enough information to support it.

The connection to #525: chaining gives coarse (stage-level) checkpointing for free. #525 gives fine (batch-level) checkpointing within a stage. They are complementary.

#### Provenance

`pipeline-metadata.json` records:
- Stage order, names, and configs used
- Config fingerprint (hash) per stage for resume invalidation
- `num_records` requested vs actual per stage
- Which stage's output seeded the next
- Timestamp, duration, and DD version per stage

### Part 2: Remove `allow_resize`

With the pipeline in place, `allow_resize` is no longer needed as an engine-internal mechanism. Resize becomes a between-stage concern.

**Config changes** (`data-designer-config`):

- Remove `allow_resize: bool = False` from `SingleColumnConfig` (or its base class `ColumnConfigBase`).
- Deprecation: keep the field for one release cycle with a deprecation warning, then remove.

**Engine changes** (`data-designer-engine`):

- Remove `_cell_resize_mode`, `_cell_resize_results`, and the resize branch in `_finalize_fan_out()` from `DatasetBuilder`.
- Remove `allow_resize` parameter from `DatasetBatchManager.replace_buffer()`.
- Remove `_validate_async_compatibility()` (no longer needed - nothing to reject).
- Simplify `_run_full_column_generator()` to always enforce row-count invariance.

**Migration path**: Users with `allow_resize=True` columns split their config into a pipeline with a stage boundary at the resize column. The resize column becomes the last column of its stage, and downstream columns move to the next stage.

### Part 3: Fail-fast on pre-batch processor resize

In `ProcessorRunner.run_pre_batch()` and `run_pre_batch_on_df()`, raise `DatasetProcessingError` if the returned DataFrame has a different row count than the input.

This applies to both sync and async paths. Users who need to filter or expand between seeds and generation use the pipeline's between-stage callback instead. Note that a seed reader plugin is NOT an equivalent escape hatch: seed readers run before any columns are generated (including samplers), so they can't filter on generated column values.

### Where it fits in the architecture

| Layer | Changes |
|-------|---------|
| `data-designer-config` | Remove `allow_resize` field. No new config models needed for v1 (pipeline is imperative, not declarative). |
| `data-designer-engine` | Remove resize code paths. Add fail-fast guard in `ProcessorRunner`. No new engine features. |
| `data-designer` (interface) | New `Pipeline` class. Thin orchestration: calls `DataDesigner.create()` per stage, wires `DataFrameSeedSource` between stages for in-memory handoff or `LocalFileSeedSource` for on-disk handoff. |

The engine does not know about pipelines. Each stage is a regular `DatasetBuilder.build()` call.

## Use cases for implementation and testing

These should guide the implementation and serve as the basis for tutorial notebooks.

### 1. Explode: personas to conversations

Generate a small, high-quality set of personas, then produce many conversations from each.

```python
# Stage 1: 100 diverse personas
config_personas = (
    DataDesignerConfigBuilder()
    .add_column(name="name", column_type="sampler", sampler_type="person_name")
    .add_column(name="age", column_type="sampler", sampler_type="uniform_int", params=...)
    .add_column(name="background", column_type="llm_text", prompt="Write a short background for {{ name }}, age {{ age }}.")
)

# Stage 2: 1000 conversations (each persona used ~10 times via seed cycling)
config_convos = (
    DataDesignerConfigBuilder()
    .add_column(name="topic", column_type="llm_text", prompt="Generate a conversation topic for {{ name }}...")
    .add_column(name="conversation", column_type="llm_text", prompt="Write a conversation between {{ name }} and an assistant about {{ topic }}...")
)

pipeline = dd.pipeline()
pipeline.add_stage("personas", config_personas, num_records=100)
pipeline.add_stage("conversations", config_convos, num_records=1000)
results = pipeline.run()
```

### 2. Filter-then-enrich

Generate candidates, use a between-stage callback to filter, then enrich survivors.

```python
config_gen = ...  # generates rows with a quality_score column
config_enrich = ...  # adds detailed analysis columns

def keep_high_quality(stage_output_path: Path) -> Path:
    df = pd.read_parquet(stage_output_path / "parquet-files")
    df = df[df["quality_score"] > 0.8]
    out = stage_output_path.parent / "filtered"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "data.parquet")
    return out

pipeline = dd.pipeline()
pipeline.add_stage("candidates", config_gen, num_records=5000)
pipeline.add_stage("enriched", config_enrich, after=keep_high_quality)
results = pipeline.run()
```

### 3. Generate-then-judge with different models

Iterate on the judging config without re-generating the base data.

```python
# Stage 1: generate with a fast model
config_gen = DataDesignerConfigBuilder(model_configs=[fast_model])...

# Stage 2: judge with a stronger model
config_judge = DataDesignerConfigBuilder(model_configs=[strong_model])...

pipeline = dd.pipeline()
pipeline.add_stage("generated", config_gen, num_records=1000)
pipeline.add_stage("judged", config_judge)
results = pipeline.run()

# Later: tweak judging config, resume from stage 1 output
pipeline_v2 = dd.pipeline()
pipeline_v2.add_stage("generated", config_gen, num_records=1000)
pipeline_v2.add_stage("judged", config_judge_v2)
results_v2 = pipeline_v2.run(resume=True)  # skips stage 1
```

### 4. Interactive notebook chaining (lightweight, no pipeline)

Quick iteration using `to_config_builder()`:

```python
result = dd.create(config_personas, num_records=50)
result.load_dataset()  # inspect, looks good

# Chain into next step
config_2 = (
    result.to_config_builder(columns=["name", "background"])
    .add_column(name="question", column_type="llm_text", prompt="...")
)
result_2 = dd.create(config_2, num_records=200)  # explode: 50 -> 200
```

## Implementation phases

### Phase 1: Pipeline class and `to_config_builder()` (can ship independently)

- Add `to_config_builder()` on `DatasetCreationResults` and `PreviewResults`.
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

1. **In-memory vs on-disk handoff between stages**: For small datasets, `DataFrameSeedSource` avoids disk I/O. For large datasets, writing parquet between stages is safer. Should the pipeline auto-detect based on row count, or always go through disk for consistency? (Leaning toward always-on-disk for simplicity and resume support.)

2. **Preview support**: Should `pipeline.preview()` run all stages with small `num_records`? Or just preview the last stage seeded from a prior full run?

3. **Config serialization**: A pipeline config can't be serialized to YAML if stages use `DataFrameSeedSource`. For persistence, stages would need symbolic references ("seed from stage X's output"). This is needed for auto-chaining (phase 4) but not for the explicit API (phases 1-3).

4. **Naming**: `Pipeline` vs `Chain` vs `WorkflowChain`. `Pipeline` is the most intuitive and aligns with ML pipeline terminology.

5. **Image/media column forwarding**: Images in create mode are stored as relative file paths. If a downstream stage seeds from an upstream stage that produced images, the relative paths break. Options: (a) resolve to absolute paths at stage boundary, (b) copy media assets into downstream stage's directory, (c) document as unsupported in v1.

6. **Branch/fan-out semantics**: Linear chaining covers the common cases. But "generate once, judge several ways" (fan-out) currently requires building multiple pipelines that repeat stage 1. Should the pipeline support DAG-shaped stage graphs, or is that future work?

7. **Downstream seeding scope**: Should downstream stages only seed from the final dataset, or should they also be able to access dropped columns or named processor outputs from upstream stages?

## Related issues

- #447 - AsyncRunController refactor (partially superseded: pre-batch resize handling moves to pipeline level instead of controller level)
- #525 - Resume interrupted runs (complementary: stage-level resume from pipeline, batch-level resume from #525)
- #462 - Progress bar and scheduler polish (independent)
- #464 - Custom column retryable errors (independent)
