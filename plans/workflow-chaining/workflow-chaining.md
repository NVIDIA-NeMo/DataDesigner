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

The `allow_resize` flag on column configs lets a generator change the row count mid-generation. This works in the sync engine but is fundamentally incompatible with the async engine's fixed-size `CompletionTracker` grid. As of #553, an `allow_resize=True` config in async mode logs a `DeprecationWarning` and silently falls back to the sync engine for that run; it is no longer hard-rejected.

`allow_resize` is one of the remaining divergences between sync and async. The async engine is the default execution path as of #592; sync remains only as a fallback for `allow_resize` runs. Maintaining a sync-only feature to keep one fallback path alive is counterproductive. With chaining in place, resize becomes a between-stage concern rather than a mid-generation concern. This lets us remove `allow_resize` and the associated engine complexity, and disallow row-count changes in pre-batch processors. Users who need resize use a pipeline with a stage boundary at the resize point.

Note: `allow_resize` is documented in custom columns, plugin examples, and agent rollout ingestion docs (verified post-Fern migration in #581). The deprecation warning has shipped in #553; full removal still requires doc updates and the migration of any in-tree usage.

### Why chaining instead of fixing async resize

The async scheduler's `CompletionTracker` pre-allocates a (row_group x row_index x column) task grid. Supporting mid-run resize requires either rebuilding the tracker (complex, error-prone) or pausing execution at resize boundaries (loses parallelism). Chaining sidesteps this entirely: each stage gets a fresh tracker sized to its actual input. The engine stays simple - always fixed-size - and resize becomes a between-stage concern.

## Design

### Part 1: Pipeline class

A new `Pipeline` class in `data_designer.interface` that orchestrates multi-stage generation.

#### User-facing API

**Explicit multi-stage pipeline:**

```python
pipeline = dd.pipeline(name="persona-conversations")
pipeline.add_stage("personas", config_personas, num_records=100)
pipeline.add_stage("conversations", config_convos, num_records=1000)  # explode: 100 -> 1000
pipeline.add_stage("judged", config_judge)  # defaults to previous stage's output size

results = pipeline.run()

results["personas"].load_dataset()       # stage 1 output
results["conversations"].load_dataset()  # stage 2 output
results["judged"].load_dataset()         # final output
```

`name` is required and is the durable identity for artifact lookup and resume. Reusing the same name across Python sessions lets `pipeline.run(resume=ResumeMode.IF_POSSIBLE)` find the previous `pipeline-metadata.json`.

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

This is a thin wrapper: loads the dataset into memory, optionally filters columns, wraps in `DataFrameSeedSource`, returns a new config builder. No tracking, no provenance, no callbacks - just a quick bridge for iteration. Not suitable for large datasets (loads full DataFrame into memory) or serializable configs (`DataFrameSeedSource` can't be written to YAML).

This is the *only* place in the chaining surface that uses an in-memory handoff. `Pipeline` itself always hands off between stages on disk - see "Composability and the throttle invariant" below. For production pipelines, use the `Pipeline` class.

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

pipeline = dd.pipeline(name="filter-enrich")
pipeline.add_stage("generated", config_gen, num_records=1000)
pipeline.add_stage(
    "enriched",
    config_enrich,
    after=filter_high_quality,  # runs on stage output before next stage seeds from it
    after_version="quality-filter-v1",
)
```

The callback receives the path to the completed stage's artifact directory (containing `parquet-files/`, `metadata.json`, etc.) and returns a path that the next stage will seed from. This keeps large DataFrames on disk and gives users full control.

**Callback resume policy**: The pipeline does not hash arbitrary Python source or bytecode in v1. `after_version` is the explicit callback identity recorded in `pipeline-metadata.json` and included in the next stage's fingerprint. If `after` is set without `after_version`, that stage is treated as dirty on every resume so a changed callback cannot silently reuse stale transformed data. The resolved path returned by the callback is also recorded as the dependent stage's seed path; a stage seeded from callback output is skippable only if that recorded path still exists and is readable by `LocalFileSeedSource`.

**Empty stage policy**: If a callback filters all rows (or a stage produces zero rows), the pipeline raises `DataDesignerPipelineError` by default. Stages can opt in to empty output with `allow_empty=True` on `add_stage()`, in which case the pipeline short-circuits and skips subsequent stages.

#### `num_records` and seed behavior

- If `num_records` is explicitly set on a stage, that value is used.
- If omitted, defaults to the previous stage's output row count (after any between-stage callback).
- The seed reader's existing cycling behavior handles the explode case: requesting 1000 records from a 100-row seed cycles through the seed 10 times.
- `add_stage()` accepts optional `sampling_strategy` (ordered/shuffle) and `selection_strategy` (IndexRange/PartitionBlock) to control how the previous stage's output is sampled. Defaults to ordered.

#### Artifact management

The pipeline owns its directory layout directly, bypassing `ArtifactStorage`'s default auto-rename behavior (which appends timestamps to non-empty directories). `dd.pipeline(name=...)` maps to `artifacts/<name>/`; no timestamp, UUID, or object-derived default is used for resumable pipelines. Stage directories use stable, deterministic names based on stage index and name:

```
artifacts/
  <pipeline-name>/
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

The pipeline creates each stage's `ArtifactStorage` with the stage directory as `dataset_name`, ensuring stable paths across reruns. A fresh `dd.pipeline(name="gen-judge")` finds the same `artifacts/gen-judge/pipeline-metadata.json` path as the original run.

#### Checkpointing and resume

Each stage produces durable parquet output before the next stage starts. This provides natural checkpoint boundaries:

- If stage 3 of 4 fails, stages 1 and 2 are already on disk.
- `pipeline.run(resume=ResumeMode.IF_POSSIBLE)` skips compatible completed stages and resumes compatible partial stages.
- Within a stage, batch/row-group resume from #526 can further reduce re-work.

**Relationship to #526**: #526 is the fine-grained single-stage resume primitive. It resumes one `DataDesigner.create()` call from completed batches (sync) or row groups (async), but its compatibility check applies to the whole `DataDesignerConfig`. Pipeline resume adds a coarser stage graph above that primitive. A downstream config change invalidates only that stage and its descendants, so upstream generation can be reused instead of failing the whole-config compatibility check.

Pipeline resume should decide stage compatibility before calling `DataDesigner.create()`. If a stage fingerprint matches and the stage is partial, the pipeline delegates to `create(..., resume=ResumeMode.ALWAYS)` for that stage. If a stage fingerprint changed, the pipeline invalidates that stage directory and descendants, then starts them fresh. It should not blindly pass `ResumeMode.IF_POSSIBLE` through to stage `create()`, because pipeline stage directories must remain deterministic under `artifacts/<pipeline-name>/`.

If a stage's seed came from an `after` callback, a fingerprint match is necessary but not sufficient to skip it. Resume must also verify the recorded callback output path exists. If the path is missing, the dependent stage and its descendants are invalidated and the callback/stage boundary is re-run from the upstream stage output.

**Resume safety**: Naive "skip if directory exists" is not sufficient. Configs, model settings, callbacks, or DD version may have changed between runs. Resume must compare a fingerprint of each stage's inputs against what's recorded in `pipeline-metadata.json`. The per-stage fingerprint composes:

- `DataDesignerConfig.fingerprint()` (introduced in #587) - content-addressable sha256 over the data-relevant portion of the config
- `num_records` (requested)
- `sampling_strategy`, `selection_strategy`, and `allow_empty`
- `after_version` when `after` is configured; if omitted, the stage is always dirty on resume
- DD version
- Upstream stage fingerprint (the directly preceding stage's recorded fingerprint, so a change anywhere in the chain invalidates downstream stages)

If any component changed, that stage and all downstream stages must re-run. This is a phase 3 concern but the metadata format in phase 1 should record enough information to support it.

The connection to #526/#525: chaining gives workflow-level checkpointing and smaller invalidation boundaries. #526 gives fine-grained crash recovery within each stage. They are complementary, and Phase 3 should use #526 rather than inventing another intra-stage resume mechanism.

#### Provenance

`pipeline-metadata.json` records:
- Pipeline name
- Stage order, names, and configs used
- Per-stage fingerprint for resume invalidation: `DataDesignerConfig.fingerprint()` (#587) combined with `num_records`, seed sampling/selection controls, `after_version`, DD version, and the upstream stage fingerprint
- `num_records` requested vs actual per stage
- Which stage's output seeded the next
- Resolved seed path per stage, including callback output paths returned by `after`
- Timestamp, duration, and DD version per stage

#### Composability and the throttle invariant

The `Pipeline` is constructed via `dd.pipeline(name=...)` and holds a reference to the parent `DataDesigner`. Every stage runs `dd.create()` (or `dd.acreate()` once available - see Engine API surface below) on that same instance. This is a load-bearing API contract for two reasons.

**Throttle coordination across stages.** A `DataDesigner` owns one `ModelRegistry`, which owns one `ThrottleManager`. AIMD rate-limit state is per-instance. If the pipeline constructed a fresh `DataDesigner` per stage, each stage would adapt independently and the aggregate request rate against a provider could exceed the configured cap by a multiple of the stage count. The same hazard applies to parallel branches in Phase 4: branches sharing one `DataDesigner` automatically share throttling; branches each holding their own `DataDesigner` silently fragment it. Reusing one instance is the simple, correct default.

**Door open for external orchestration.** The pipeline's choice to reuse one `DataDesigner` is the in-process strategy: shared throttling across stages, branches gathered in the orchestrator process. A cross-process strategy is a separate but compatible model - see Future considerations. v1 only needs to avoid encoding assumptions that would prevent it.

**On-disk handoffs for the same reason.** Stage handoffs go through parquet on disk via `LocalFileSeedSource`, never through an in-memory `DataFrameSeedSource`. This composes with any future orchestration model (in-process, cross-process, distributed) without per-environment branching. The cost is one parquet round-trip per stage boundary, which is negligible compared to LLM call time at any realistic scale. The notebook ergonomic `to_config_builder()` is the in-memory escape hatch and is explicitly not a Pipeline.

**Internal stage model is a graph, not a list.** v1 exposes a linear `add_stage()` API and runs stages sequentially. Internally the pipeline represents stages as a DAG with the linear case being the default chain. This lets Phase 4 add parallel branches as an additive API change without restructuring orchestration.

#### Engine API surface: `acreate()`

`Pipeline` v1 calls `DataDesigner.create()` synchronously per stage and runs them in order. Sequential execution doesn't need an async API. *In-process* parallel execution does, and the engine doesn't expose one today. Cross-process orchestration is not the same problem: each worker runs sync `create()` in its own process and doesn't need an async surface.

Adding `async def acreate(...)` on `DataDesigner` is a small, additive change. The underlying `_build_async` already runs on a singleton background event loop and submits work via a `concurrent.futures.Future`; `acreate()` bridges it into the caller's loop via `asyncio.wrap_future`. The sync `create()` becomes a one-line wrapper. No breaking changes.

`acreate()` enables two things without touching `Pipeline`:

- **Parallel-independent workflows.** Users can `asyncio.gather(dd.acreate(c1), dd.acreate(c2))` for unrelated configs and get coordinated throttling automatically through the shared `ThrottleManager`.
- **Pipeline DAG branches (Phase 4).** When the pipeline graduates to a DAG, parallel branches are a pure orchestration change - `asyncio.gather` over `acreate()` calls inside `pipeline.run()` - with no further engine work required.

`acreate()` is *not* part of chaining v1. It ships as its own small piece of work that can land before, alongside, or after Phase 1; the dependency only becomes hard for Phase 4. Listed as a sidecar under Implementation phases.

### Part 2: Remove `allow_resize`

With the pipeline in place, `allow_resize` is no longer needed as an engine-internal mechanism. Resize becomes a between-stage concern.

**Config changes** (`data-designer-config`):

- Remove `allow_resize: bool = False` from `SingleColumnConfig` (or its base class `ColumnConfigBase`).
- The deprecation warning has already shipped in #553. After one release cycle from that point, remove the field.

**Engine changes** (`data-designer-engine`):

- Remove `_cell_resize_mode`, `_cell_resize_results`, and the resize branch in `_finalize_fan_out()` from `DatasetBuilder`.
- Remove `allow_resize` parameter from `DatasetBatchManager.replace_buffer()`.
- Remove `_resolve_async_compatibility()` and the sync-fallback branch in `_build_async()` (no longer needed - nothing to fall back for).
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
| `data-designer` (interface) | New `Pipeline` class. Thin orchestration: holds a reference to the parent `DataDesigner`, calls `DataDesigner.create()` per stage, hands off between stages on disk via `LocalFileSeedSource`. All stages share the same `ModelRegistry` and `ThrottleManager`. Optionally consumes `DataDesigner.acreate()` (sidecar) once available, for Phase 4 parallel branches. |

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

pipeline = dd.pipeline(name="persona-conversations")
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

pipeline = dd.pipeline(name="filter-enrich")
pipeline.add_stage("candidates", config_gen, num_records=5000)
pipeline.add_stage("enriched", config_enrich, after=keep_high_quality, after_version="quality-filter-v1")
results = pipeline.run()
```

### 3. Generate-then-judge with different models

Iterate on the judging config without re-generating the base data.

```python
# Stage 1: generate with a fast model
config_gen = DataDesignerConfigBuilder(model_configs=[fast_model])...

# Stage 2: judge with a stronger model
config_judge = DataDesignerConfigBuilder(model_configs=[strong_model])...

pipeline = dd.pipeline(name="gen-judge")
pipeline.add_stage("generated", config_gen, num_records=1000)
pipeline.add_stage("judged", config_judge)
results = pipeline.run()

# Later: tweak judging config, resume from stage 1 output
pipeline_v2 = dd.pipeline(name="gen-judge")
pipeline_v2.add_stage("generated", config_gen, num_records=1000)
pipeline_v2.add_stage("judged", config_judge_v2)
results_v2 = pipeline_v2.run(resume=ResumeMode.IF_POSSIBLE)  # skips stage 1
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
- Add `Pipeline` class with `add_stage()`, `run()`, between-stage callbacks. Pipeline holds a reference to the parent `DataDesigner` and reuses it across stages.
- Stage handoff is always on disk via `LocalFileSeedSource`; no in-memory handoff path inside `Pipeline`.
- Internal stage representation is a DAG (linear-only inputs in v1).
- Add `pipeline-metadata.json` writing.
- Add `dd.pipeline(name: str)` factory method on `DataDesigner`.
- Tests: multi-stage runs, explode/filter via callbacks, num_records defaulting, artifact layout, throttle reuse across stages.

### Sidecar: `acreate()` on `DataDesigner` (independent of chaining v1)

- Add `async def acreate(...)` mirroring `create()` but returning the awaitable instead of blocking.
- `create()` becomes a one-line wrapper around `acreate()` (or both share a common builder helper).
- Tests: parallel-independent workflows via `asyncio.gather`; verify shared `ThrottleManager` keeps aggregate request rate within configured caps.
- Can ship before, alongside, or after Phase 1. Hard dependency for Phase 4.

### Phase 2: Remove `allow_resize`

- (Done in #553) `allow_resize=True` in async mode emits a `DeprecationWarning` and falls back to sync.
- Update docs that still reference `allow_resize` (`docs/concepts/custom_columns.md`, `docs/plugins/example.md`, `docs/concepts/agent-rollout-ingestion.md`) to point at pipelines.
- Remove resize code from sync engine (`_cell_resize_mode`, `_finalize_fan_out` resize branch, `replace_buffer` `allow_resize` param).
- Remove `_resolve_async_compatibility()` and its sync-fallback branch from `_build_async()`.
- Remove the `allow_resize` field from the config schema.
- Add fail-fast guard in `ProcessorRunner` for pre-batch row-count changes.
- Tests: verify rejection, migration path examples.

### Phase 3: Stage-level resume

- Add `resume: ResumeMode` to `pipeline.run()`, reusing the enum introduced by #526.
- Read `pipeline-metadata.json` to detect completed stages.
- Resolve the metadata path from the explicit pipeline name.
- Compute each stage's fingerprint via `DataDesignerConfig.fingerprint()` (#587) combined with `num_records`, seed sampling/selection controls, `after_version`, DD version, and upstream stage fingerprint; invalidate the stage and everything downstream on any mismatch.
- Skip stages whose fingerprints match and are complete; for matching partial stages, call `DataDesigner.create(..., resume=ResumeMode.ALWAYS)` to use #526's batch/row-group resume.
- Before skipping or resuming a stage seeded by `after`, validate the recorded callback output path exists and can seed `LocalFileSeedSource`; if missing, invalidate that stage and descendants.
- For invalidated stages, clear or replace the deterministic stage directory before starting fresh so `ArtifactStorage` does not timestamp away from the pipeline layout.
- Depends on artifact layout from phase 1.

### Phase 4: DAG-shaped stages with parallel branches

- Extend `add_stage()` with an optional `depends_on=[stage_name, ...]` argument; default keeps the linear behavior.
- `pipeline.run()` walks the resulting DAG, gathering independent branches via `asyncio.gather` over `dd.acreate()` calls.
- Per-stage fingerprint composition (Phase 3) generalizes naturally: a stage's upstream fingerprint becomes the hash of all parent fingerprints sorted by stage name, making joins stable regardless of `depends_on` declaration order.
- Throttle coordination relies on the existing invariant: all branches run on the same parent `DataDesigner`, so `ThrottleManager` is shared.
- Hard dependency on the `acreate()` sidecar.
- **Scope: branch parallelism, not stage pipelining.** Stages still wait for their dependencies to fully complete before starting; pipelined execution of dependent stages is a separate direction sketched in Future considerations.
- Tests: fan-out (one upstream, multiple parallel children); join (multiple upstreams, one child); resume invalidation when one branch's fingerprint changes; throttle behavior under N parallel branches.

### Phase 5 (future): Auto-chaining from single config

- Detect stage boundaries in the DAG (via a new config marker or heuristic).
- Auto-split into pipeline stages internally.
- User sees a single `dd.create(config)` call but gets multi-stage execution.

## Future considerations

Items not on the current roadmap but worth flagging so they don't get accidentally precluded by v1-v5 design choices.

**External orchestration for cross-process / distributed execution.** There is interest in eventually running DataDesigner workloads across processes or nodes - self-hosted serving, multi-host fan-out, scheduling against external clusters. The specific shape of that orchestration is still under discussion and is not committed to here. The chaining plan's design choices (parent `DataDesigner` reuse, on-disk handoffs, no new engine surface) compose naturally with such a system: an external orchestrator could dispatch independent `DataDesigner.create()` calls against partitioned slices and per-replica endpoints without the pipeline class needing to change. v1-v5 do not depend on this materializing.

**Pipelined execution of dependent stages.** Today the stage data contract is "final dataset" - a downstream stage waits for its upstream to fully complete. A future direction is to let downstream stages consume upstream batches as they're produced, overlapping execution across the dependency edge. Required changes: streaming seed sources, an explicit "stage done" sentinel rather than file-completion checks, and resume semantics for partially-consumed upstreams. Most useful when stage bottlenecks are heterogeneous (LLM-bound stage feeding a CPU-bound validator); little gain when both stages are LLM-bound since they share provider capacity. Not designed here; flagged so the stage contract isn't quietly closed off.

## Resolved decisions

These were open in earlier drafts; recording the resolutions here so the design is unambiguous.

1. **In-memory vs on-disk handoff between stages** -> Always on-disk inside `Pipeline`. The in-memory `DataFrameSeedSource` mode is reserved for the lightweight `to_config_builder()` notebook ergonomic, which is explicitly *not* a `Pipeline`. Reasons: single execution model, simpler resume story, and composability with any future external orchestration that can't share an in-memory DataFrame across process boundaries. Cost is one parquet round-trip per stage, negligible relative to LLM call time.

2. **Branch/fan-out semantics (DAG)** -> Designed-in but not v1. The internal stage representation is a DAG; v1 only accepts linear inputs through `add_stage()`. Phase 4 ships parallel branches via `asyncio.gather` over `acreate()`. v1 stays sequential.

3. **Pipeline construction** -> `Pipeline` is created via `dd.pipeline(name=...)` and reuses the parent `DataDesigner`'s `ModelRegistry` and `ThrottleManager` across all stages. The explicit name is the durable artifact identity used for resume, and the pipeline does not construct its own `DataDesigner` instances. This is the throttle-coordination invariant (see Composability section).

## Open questions

1. **Preview support**: Should `pipeline.preview()` run all stages with small `num_records`? Or just preview the last stage seeded from a prior full run?

2. **Config serialization**: For persistence, pipeline configs would need symbolic stage references ("seed from stage X's output"). With the on-disk handoff decision above, the `DataFrameSeedSource` blocker is no longer relevant; the remaining question is how to encode stage dependencies in YAML. Needed for auto-chaining (Phase 5) but not for the explicit API (phases 1-4).

3. **Naming**: `Pipeline` vs `Chain` vs `WorkflowChain`. `Pipeline` is the most intuitive and aligns with ML pipeline terminology.

4. **Image/media column forwarding**: Images in create mode are stored as relative file paths. If a downstream stage seeds from an upstream stage that produced images, the relative paths break. Options: (a) resolve to absolute paths at stage boundary, (b) copy media assets into downstream stage's directory, (c) document as unsupported in v1.

5. **Downstream seeding scope**: Should downstream stages only seed from the final dataset, or should they also be able to access dropped columns or named processor outputs from upstream stages?

## Related issues

- #447 - AsyncRunController refactor (partially superseded: pre-batch resize handling moves to pipeline level instead of controller level)
- #526 / #525 - Resume interrupted runs (single-stage batch/row-group resume primitive used by pipeline stage resume)
- #462 - Progress bar and scheduler polish (independent)
- #464 - Custom column retryable errors (independent)
