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

Add **composite workflows**: a thin orchestration layer that sequences multiple independent DataDesigner workflows, passing each stage's output as the next stage's seed dataset. This is the primary deliverable.

A `CompositeWorkflow` composes independent `DataDesigner.create()` runs. Each stage is still a normal DataDesigner workflow with its own config; the composition layer manages ordering/dependencies, deterministic on-disk handoffs, shared throttling, provenance, and resume. v1 exposes linear stage composition, but the internal representation is DAG-shaped so future branches and joins do not require renaming the abstraction.

As a secondary benefit, chaining also enables the removal of `allow_resize` and simplification of the engine's resize handling.

### Secondary benefit: `allow_resize` removal and sync/async convergence

The `allow_resize` flag on column configs lets a generator change the row count mid-generation. This works in the sync engine but is fundamentally incompatible with the async engine's fixed-size `CompletionTracker` grid. As of #553, an `allow_resize=True` config in async mode logs a `DeprecationWarning` and silently falls back to the sync engine for that run; it is no longer hard-rejected.

`allow_resize` is one of the remaining divergences between sync and async. The async engine is the default execution path as of #592; sync remains only as a fallback for `allow_resize` runs. Maintaining a sync-only feature to keep one fallback path alive is counterproductive. With composite workflows in place, resize becomes a between-stage concern rather than a mid-generation concern. This lets us remove `allow_resize` and the associated engine complexity, and disallow row-count changes in pre-batch processors. Users who need resize use a composite workflow with a stage boundary at the resize point.

Note: `allow_resize` is documented in custom columns, plugin examples, and agent rollout ingestion docs (verified post-Fern migration in #581). The deprecation warning has shipped in #553; full removal still requires doc updates and the migration of any in-tree usage.

### Why chaining instead of fixing async resize

The async scheduler's `CompletionTracker` pre-allocates a (row_group x row_index x column) task grid. Supporting mid-run resize requires either rebuilding the tracker (complex, error-prone) or pausing execution at resize boundaries (loses parallelism). Chaining sidesteps this entirely: each stage gets a fresh tracker sized to its actual input. The engine stays simple - always fixed-size - and resize becomes a between-stage concern.

## Design

### Part 1: `CompositeWorkflow` class

A new `CompositeWorkflow` class in `data_designer.interface` that orchestrates multi-stage generation. It is created with `DataDesigner.compose_workflow(name=...)`: the factory name describes the action of composing workflows, while the returned object names the resulting abstraction.

#### User-facing API

**Explicit multi-stage composite workflow:**

```python
data_designer = DataDesigner()
workflow = data_designer.compose_workflow(name="persona-conversations")
workflow.add_stage("personas", config_personas, num_records=100)
workflow.add_stage("conversations", config_convos, num_records=1000)  # explode: 100 -> 1000
workflow.add_stage("judged", config_judge)  # defaults to previous stage's output size

results = workflow.run()

results["personas"].load_dataset()       # stage 1 output
results["conversations"].load_dataset()  # stage 2 output
results["judged"].load_dataset()         # final output
```

`name` is required and is the durable identity for artifact lookup and resume. Reusing the same name across Python sessions lets `workflow.run(resume=ResumeMode.IF_POSSIBLE)` find the previous `workflow-metadata.json`.

Each stage can use a different `DataDesignerConfig`, including different model configs. v1 still uses the parent `DataDesigner` instance and run settings for all stages; per-stage `RunConfig` and compute placement are future work.

Stage names must be unique within a `CompositeWorkflow`. `add_stage()` validates this immediately and raises `DataDesignerWorkflowError` on duplicates before any artifact directory is created. This keeps `CompositeWorkflowResults`, `workflow-metadata.json`, and Phase 3 fingerprint lookup unambiguous.

**Convenience method on results (lightweight, for notebooks):**

For interactive use where a full composite workflow is overkill, a `to_config_builder()` method on `DatasetCreationResults` returns a pre-seeded `DataDesignerConfigBuilder`:

```python
# Stage 1
result = data_designer.create(config_personas, num_records=100)

# Stage 2 - just grab the result and keep going
config_convos = (
    result.to_config_builder(columns=["name", "age", "background"])  # optional column selection
    .add_column(name="conversation", column_type="llm_text", prompt="...")
)
result_2 = data_designer.create(config_convos, num_records=1000)
```

This is a thin wrapper: loads the dataset into memory, optionally filters columns, wraps in `DataFrameSeedSource`, returns a new config builder. It starts a new config seeded from existing results; it does not mutate, continue, or resume the original config. No tracking, no provenance, no callbacks - just a quick bridge for iteration. Not suitable for large datasets (loads full DataFrame into memory) or serializable configs (`DataFrameSeedSource` can't be written to YAML).

This is the *only* place in the chaining surface that uses an in-memory handoff. `CompositeWorkflow` itself always hands off between stages on disk - see "Composability and the throttle invariant" below. For production workflows, use the `CompositeWorkflow` class.

#### Stage data contract

Each stage seeds from the **previous stage's final dataset** - the post-processor output with dropped columns excluded. This is the same DataFrame returned by `DatasetCreationResults.load_dataset()`.

Processor outputs (named processor artifacts), schema-transform artifacts, dropped columns, and media assets (images stored on disk with relative paths in the DataFrame) are NOT automatically forwarded in v1. If a downstream stage needs one of these artifacts, use an `on_success` callback to read the upstream stage artifact directory and write a `LocalFileSeedSource`-compatible dataset for the next stage. First-class seeding from named processor outputs is a future extension. Related near-term extensions are seeded processor-only configs and structured stage output processors, so processor-expressible transforms can remain normal Data Designer processor configs rather than workflow-specific callback code.

#### Between-stage output processors and callbacks

Users may need to transform data between stages. The preferred future API is an `output_processors` argument on `CompositeWorkflow.add_stage()`:

```python
workflow.add_stage(
    "generated",
    config_gen,
    num_records=1000,
    output_processors=[
        DropColumnsProcessorConfig(
            name="drop_scratch",
            column_names=["scratch_*"],
        )
    ],
)
workflow.add_stage("enriched", config_enrich)
```

Postprocessors run after the stage's `DataDesigner.create()` completes and before child stages seed from that stage. They operate on the stage's final dataset, write a deterministic output-processed artifact, and make that artifact the stage output for downstream seeding. They should accept existing built-in or plugin `ProcessorConfig` objects and be fingerprinted from those configs, so no callback version string is needed.

Arbitrary Python data transforms should come through the `CustomProcessor` work tracked in #159, rather than a workflow-only callable type. Until that exists, `on_success` remains the escape hatch for arbitrary data-changing code:

```python
def filter_high_quality(stage_output_path: Path) -> Path:
    df = pd.read_parquet(stage_output_path / "parquet-files")
    df = df[df["quality_score"] > 0.8]
    out = stage_output_path / "callback-outputs" / "quality-filter-v1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "data.parquet")
    return out

workflow = data_designer.compose_workflow(name="filter-enrich")
workflow.add_stage(
    "generated",
    config_gen,
    num_records=1000,
    on_success=filter_high_quality,
    on_success_version="quality-filter-v1",
)
workflow.add_stage("enriched", config_enrich)
```

The callback receives the path to the completed stage's artifact directory (containing `parquet-files/`, `metadata.json`, etc.) and returns a path that child stages will seed from. This keeps large DataFrames on disk and gives users full control. Callback outputs should live under a managed subdirectory such as `<stage-dir>/callback-outputs/<callback-name>/` so they stay deterministic and easy to clean. Once output processors and seeded processor-only configs are valid, processor-expressible transforms should use those structured APIs instead of raw callbacks:

```python
filter_config = DataDesignerConfigBuilder().add_processor(
    DropColumnsProcessorConfig(
        name="drop_scratch",
        column_names=["scratch_*"],
    ),
)
workflow.add_stage("generated", config_gen, num_records=1000)
workflow.add_stage("filtered", filter_config)
workflow.add_stage("enriched", config_enrich)
```

**Callback resume policy**: The composite workflow does not hash arbitrary Python source or bytecode in v1. `on_success_version` is the explicit callback identity recorded in `workflow-metadata.json` and included in the stage's exposed output fingerprint. If `on_success` is set without `on_success_version`, that stage's transformed output is treated as dirty on every resume so a changed callback cannot silently reuse stale transformed data. The resolved path returned by the callback is also recorded as the dependent stage's seed path; a stage seeded from callback output is skippable only if that recorded path still exists and is readable by `LocalFileSeedSource`. Structured output processors should use their processor config fingerprint instead.

**Empty stage policy**: If a callback filters all rows (or a stage produces zero rows), the composite workflow raises `DataDesignerWorkflowError` by default. Stages can opt in to empty output with `allow_empty=True` on `add_stage()`, in which case the workflow marks that stage as `completed_empty` and all downstream stages as `skipped_empty_upstream`. `CompositeWorkflowResults` still contains every declared stage name: executed stages map to `DatasetCreationResults`, while skipped downstream stages map to `SkippedStageResult` with `status="skipped_empty_upstream"` and `upstream_stage=<stage_name>`. This avoids `KeyError`/`None` ambiguity and gives resume a durable state distinct from normal completion.

**Stage output selection**: The default stage output is the final dataset returned by `DatasetCreationResults.load_dataset()`. A future additive option can select a named processor artifact as the stage output:

```python
workflow.add_stage("drafts", config_drafts, num_records=1000)
workflow.add_stage("sft", sft_config, output="processor:sft")
```

When `output="processor:<name>"` is set, child stages seed from `processors-files/<name>/` instead of `parquet-files/`, and workflow metadata records the selected output source. This covers processors such as `SchemaTransformProcessorConfig`, which write side datasets while leaving the main dataset unchanged.

**Export/push policy**: `CompositeWorkflowResults` export and push helpers default to the final stage's dataset. Exporting selected stages or the full composite workflow artifact bundle is explicit future work.

#### `num_records` and seed behavior

- If `num_records` is explicitly set on a stage, that value is used.
- If omitted, defaults to the previous stage's output row count (after any between-stage callback).
- The seed reader's existing cycling behavior handles the explode case: requesting 1000 records from a 100-row seed cycles through the seed 10 times.
- `add_stage()` accepts optional `sampling_strategy` (ordered/shuffle) and `selection_strategy` (IndexRange/PartitionBlock) to control how the previous stage's output is sampled. Defaults to ordered.

#### Artifact management

The composite workflow owns its directory layout directly, bypassing `ArtifactStorage`'s default auto-rename behavior (which appends timestamps to non-empty directories). `data_designer.compose_workflow(name=...)` maps to `artifacts/<name>/`; no timestamp, UUID, or object-derived default is used for resumable workflows. Stage directories use stable, deterministic names based on stage index and name:

```
artifacts/
  <workflow-name>/
    stage-0-personas/
      parquet-files/
      metadata.json
    stage-1-conversations/
      parquet-files/
      metadata.json
    stage-2-judged/
      parquet-files/
      metadata.json
    workflow-metadata.json
```

The composite workflow creates each stage's `ArtifactStorage` with the stage directory as `dataset_name`, ensuring stable paths across reruns. A fresh `data_designer.compose_workflow(name="gen-judge")` finds the same `artifacts/gen-judge/workflow-metadata.json` path as the original run.

#### Checkpointing and resume

Each stage produces durable parquet output before the next stage starts. This provides natural checkpoint boundaries:

- If stage 3 of 4 fails, stages 1 and 2 are already on disk.
- `workflow.run(resume=ResumeMode.IF_POSSIBLE)` skips compatible completed stages and resumes compatible partial stages.
- Within a stage, batch/row-group resume from #526 can further reduce re-work.

**Relationship to #526**: #526 is the fine-grained single-stage resume primitive. It resumes one `DataDesigner.create()` call from completed batches (sync) or row groups (async), but its compatibility check applies to the whole `DataDesignerConfig`. Composite workflow resume adds a coarser stage graph above that primitive. A downstream config change invalidates only that stage and its descendants, so upstream generation can be reused instead of failing the whole-config compatibility check.

Composite workflow resume should decide stage compatibility before calling `DataDesigner.create()`. If a stage fingerprint matches and the stage is partial, the workflow delegates to `create(..., resume=ResumeMode.ALWAYS)` for that stage. If a stage fingerprint changed, the workflow invalidates that stage directory and descendants, then starts them fresh. It should not blindly pass `ResumeMode.IF_POSSIBLE` through to stage `create()`, because workflow stage directories must remain deterministic under `artifacts/<workflow-name>/`.

If a stage's seed came from an `on_success` callback, a fingerprint match is necessary but not sufficient to skip it. Resume must also verify the recorded callback output path exists. If the path is missing, the dependent stage and its descendants are invalidated and the callback/stage boundary is re-run from the upstream stage output.

**Resume safety**: Naive "skip if directory exists" is not sufficient. Configs, model settings, callbacks, or DD version may have changed between runs. Resume must compare a fingerprint of each stage's inputs against what's recorded in `workflow-metadata.json`. The per-stage fingerprint composes:

- `DataDesignerConfig.fingerprint()` (introduced in #587) - content-addressable sha256 over the data-relevant portion of the config
- `num_records` (requested)
- `sampling_strategy`, `selection_strategy`, and `allow_empty`
- `on_success_version` when `on_success` is configured; if omitted, the stage's transformed output is always dirty on resume
- DD version
- Upstream stage fingerprint (the directly preceding stage's recorded fingerprint, so a change anywhere in the chain invalidates downstream stages)

If any component changed, that stage and all downstream stages must re-run. This is a phase 3 concern but the metadata format in phase 1 should record enough information to support it.

The connection to #526/#525: chaining gives workflow-level checkpointing and smaller invalidation boundaries. #526 gives fine-grained crash recovery within each stage. They are complementary, and Phase 3 should use #526 rather than inventing another intra-stage resume mechanism.

#### Provenance

`workflow-metadata.json` records:
- Composite workflow name
- Stage order, names, and configs used
- Per-stage status: `completed`, `completed_empty`, `skipped_empty_upstream`, or `failed`
- Per-stage fingerprint for resume invalidation: `DataDesignerConfig.fingerprint()` (#587) combined with `num_records`, seed sampling/selection controls, `on_success_version`, DD version, and the upstream stage fingerprint
- `num_records` requested vs actual per stage
- Which stage's output seeded the next
- Resolved seed path per stage, including callback output paths returned by `on_success`
- Timestamp, duration, and DD version per stage

#### Composability and the throttle invariant

The `CompositeWorkflow` is constructed via `DataDesigner.compose_workflow(name=...)` and holds a reference to that parent `DataDesigner`. Every stage runs `data_designer.create()` (or `data_designer.acreate()` once available - see Engine API surface below) on that same instance. This is a load-bearing API contract for two reasons.

**Throttle coordination across stages.** A `DataDesigner` owns one `ModelRegistry`, which owns one `ThrottleManager`. AIMD rate-limit state is per-instance. If the workflow constructed a fresh `DataDesigner` per stage, each stage would adapt independently and the aggregate request rate against a provider could exceed the configured cap by a multiple of the stage count. The same hazard applies to parallel branches in Phase 4: branches sharing one `DataDesigner` automatically share throttling; branches each holding their own `DataDesigner` silently fragment it. Reusing one instance is the simple, correct default.

**Door open for external orchestration.** The workflow's choice to reuse one `DataDesigner` is the in-process strategy: shared throttling across stages, branches gathered in the orchestrator process. A cross-process strategy is a separate but compatible model - see Future considerations. v1 only needs to avoid encoding assumptions that would prevent it.

**On-disk handoffs for the same reason.** Stage handoffs go through parquet on disk via `LocalFileSeedSource`, never through an in-memory `DataFrameSeedSource`. This composes with any future orchestration model (in-process, cross-process, distributed) without per-environment branching. The cost is one parquet round-trip per stage boundary, which is negligible compared to LLM call time at any realistic scale. The notebook ergonomic `to_config_builder()` is the in-memory escape hatch and is explicitly not a `CompositeWorkflow`.

**Internal stage model is a graph, not a list.** v1 exposes a linear `add_stage()` API and runs stages sequentially. Internally the composite workflow represents stages as a DAG with the linear case being the default chain. This lets Phase 4 add parallel branches as an additive API change without restructuring orchestration.

#### Engine API surface: `acreate()`

`CompositeWorkflow` v1 calls `DataDesigner.create()` synchronously per stage and runs them in order. Sequential execution doesn't need an async API. *In-process* parallel execution does, and the engine doesn't expose one today. Cross-process orchestration is not the same problem: each worker runs sync `create()` in its own process and doesn't need an async surface.

Adding `async def acreate(...)` on `DataDesigner` is a small, additive change. The underlying `_build_async` already runs on a singleton background event loop and submits work via a `concurrent.futures.Future`; `acreate()` bridges it into the caller's loop via `asyncio.wrap_future`. The sync `create()` becomes a one-line wrapper. No breaking changes.

`acreate()` enables two things without touching `CompositeWorkflow`:

- **Parallel-independent workflows.** Users can `asyncio.gather(data_designer.acreate(c1), data_designer.acreate(c2))` for unrelated configs and get coordinated throttling automatically through the shared `ThrottleManager`.
- **CompositeWorkflow DAG branches (Phase 4).** When the workflow accepts non-linear dependencies, parallel branches are a pure orchestration change - `asyncio.gather` over `acreate()` calls inside `workflow.run()` - with no further engine work required.

`acreate()` is *not* part of chaining v1. It ships as its own small piece of work that can land before, alongside, or after Phase 1; the dependency only becomes hard for Phase 4. Listed as a sidecar under Implementation phases.

### Part 2: Remove `allow_resize`

With composite workflows in place, `allow_resize` is no longer needed as an engine-internal mechanism. Resize becomes a between-stage concern.

**Config changes** (`data-designer-config`):

- Remove `allow_resize: bool = False` from `SingleColumnConfig` (or its base class `ColumnConfigBase`).
- The deprecation warning has already shipped in #553. After one release cycle from that point, remove the field.

**Engine changes** (`data-designer-engine`):

- Remove `_cell_resize_mode`, `_cell_resize_results`, and the resize branch in `_finalize_fan_out()` from `DatasetBuilder`.
- Remove `allow_resize` parameter from `DatasetBatchManager.replace_buffer()`.
- Remove `_resolve_async_compatibility()` and the sync-fallback branch in `_build_async()` (no longer needed - nothing to fall back for).
- Simplify `_run_full_column_generator()` to always enforce row-count invariance.

**Migration path**: Users with `allow_resize=True` columns split their config into a composite workflow with a stage boundary at the resize column. The resize column becomes the last column of its stage, and downstream columns move to the next stage.

### Part 3: Fail-fast on pre-batch processor resize

In `ProcessorRunner.run_pre_batch()` and `run_pre_batch_on_df()`, raise `DatasetProcessingError` if the returned DataFrame has a different row count than the input.

This applies to both sync and async paths. Users who need to filter or expand between seeds and generation use the composite workflow's between-stage callback instead. Note that a seed reader plugin is NOT an equivalent escape hatch: seed readers run before any columns are generated (including samplers), so they can't filter on generated column values.

### Where it fits in the architecture

| Layer | Changes |
|-------|---------|
| `data-designer-config` | Remove `allow_resize` field. No new config models needed for v1 (composite workflow is imperative, not declarative). |
| `data-designer-engine` | Remove resize code paths. Add fail-fast guard in `ProcessorRunner`. No new engine features. |
| `data-designer` (interface) | New `CompositeWorkflow` class. Thin orchestration: holds a reference to the parent `DataDesigner`, calls `DataDesigner.create()` per stage, hands off between stages on disk via `LocalFileSeedSource`. All stages share the same `ModelRegistry` and `ThrottleManager`. Optionally consumes `DataDesigner.acreate()` (sidecar) once available, for Phase 4 parallel branches. |

The engine does not know about composite workflows. Each stage is a regular `DatasetBuilder.build()` call.

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

data_designer = DataDesigner()
workflow = data_designer.compose_workflow(name="persona-conversations")
workflow.add_stage("personas", config_personas, num_records=100)
workflow.add_stage("conversations", config_convos, num_records=1000)
results = workflow.run()
```

### 2. Filter-then-enrich

Generate candidates, use a between-stage callback to filter, then enrich survivors.

```python
config_gen = ...  # generates rows with a quality_score column
config_enrich = ...  # adds detailed analysis columns

def keep_high_quality(stage_output_path: Path) -> Path:
    df = pd.read_parquet(stage_output_path / "parquet-files")
    df = df[df["quality_score"] > 0.8]
    out = stage_output_path / "callback-outputs" / "quality-filter-v1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "data.parquet")
    return out

workflow = data_designer.compose_workflow(name="filter-enrich")
workflow.add_stage(
    "candidates",
    config_gen,
    num_records=5000,
    on_success=keep_high_quality,
    on_success_version="quality-filter-v1",
)
workflow.add_stage("enriched", config_enrich)
results = workflow.run()
```

### 3. Generate-then-judge with different models

Iterate on the judging config without re-generating the base data.

```python
# Stage 1: generate with a fast model
config_gen = DataDesignerConfigBuilder(model_configs=[fast_model])...

# Stage 2: judge with a stronger model
config_judge = DataDesignerConfigBuilder(model_configs=[strong_model])...

workflow = data_designer.compose_workflow(name="gen-judge")
workflow.add_stage("generated", config_gen, num_records=1000)
workflow.add_stage("judged", config_judge)
results = workflow.run()

# Later: tweak judging config, resume from stage 1 output
workflow_v2 = data_designer.compose_workflow(name="gen-judge")
workflow_v2.add_stage("generated", config_gen, num_records=1000)
workflow_v2.add_stage("judged", config_judge_v2)
results_v2 = workflow_v2.run(resume=ResumeMode.IF_POSSIBLE)  # skips stage 1
```

### 4. Interactive notebook chaining (lightweight, no composite workflow)

Quick iteration using `to_config_builder()`:

```python
result = data_designer.create(config_personas, num_records=50)
result.load_dataset()  # inspect, looks good

# Chain into next step
config_2 = (
    result.to_config_builder(columns=["name", "background"])
    .add_column(name="question", column_type="llm_text", prompt="...")
)
result_2 = data_designer.create(config_2, num_records=200)  # explode: 50 -> 200
```

## Implementation phases

### Phase 1: `CompositeWorkflow` class and `to_config_builder()` (can ship independently)

- Add `to_config_builder()` on `DatasetCreationResults` and `PreviewResults`.
- Add `CompositeWorkflow` class with `add_stage()`, `run()`, and `on_success` callbacks. `CompositeWorkflow` holds a reference to the parent `DataDesigner` and reuses it across stages.
- Stage handoff is always on disk via `LocalFileSeedSource`; no in-memory handoff path inside `CompositeWorkflow`.
- Internal stage representation is a DAG (linear-only inputs in v1).
- Add `workflow-metadata.json` writing.
- Add `compose_workflow(name: str)` factory method on `DataDesigner`.
- Tests: multi-stage runs, explode/filter via callbacks, num_records defaulting, duplicate stage-name rejection, artifact layout, throttle reuse across stages.

**Status after PR #636:** Implemented `CompositeWorkflow`, `compose_workflow()`, `to_config_builder()`, disk handoff, stage metadata, `acreate()`, shared throttle manager reuse, explicit stage artifact roots, cloned stage builders, concurrent-safe seed reader/resource-provider handling, seeded processor-only configs, stage output processors, and stage output selection. Still deferred after #636: stage-level resume, DAG branches, `allow_resize` removal, config bundles, and broader first-class artifact seeding.

### Sidecar: `acreate()` on `DataDesigner` (independent of chaining v1)

- Add `async def acreate(...)` mirroring `create()` but returning the awaitable instead of blocking.
- `create()` becomes a one-line wrapper around `acreate()` (or both share a common builder helper).
- Tests: parallel-independent workflows via `asyncio.gather`; verify shared `ThrottleManager` keeps aggregate request rate within configured caps.
- Can ship before, alongside, or after Phase 1. Hard dependency for Phase 4.

### Phase 2: Remove `allow_resize`

- (Done in #553) `allow_resize=True` in async mode emits a `DeprecationWarning` and falls back to sync.
- Update docs that still reference `allow_resize` (`docs/concepts/custom_columns.md`, `docs/plugins/example.md`, `docs/concepts/agent-rollout-ingestion.md`) to point at composite workflows.
- Remove resize code from sync engine (`_cell_resize_mode`, `_finalize_fan_out` resize branch, `replace_buffer` `allow_resize` param).
- Remove `_resolve_async_compatibility()` and its sync-fallback branch from `_build_async()`.
- Remove the `allow_resize` field from the config schema.
- Add fail-fast guard in `ProcessorRunner` for pre-batch row-count changes.
- Tests: verify rejection, migration path examples.
- Migration examples should prefer `output_processors` for inline processor-expressible transforms, or seeded processor-only configs when the transform deserves its own named stage. Raw `on_success` remains the escape hatch for arbitrary custom transforms.

### Phase 3: Stage-level resume

- Add `resume: ResumeMode` to `workflow.run()`, reusing the enum introduced by #526.
- Read `workflow-metadata.json` to detect completed stages.
- Resolve the metadata path from the explicit workflow name.
- Compute each stage's fingerprint via `DataDesignerConfig.fingerprint()` (#587) combined with `num_records`, seed sampling/selection controls, `on_success_version`, DD version, and upstream stage fingerprint; invalidate the stage and everything downstream on any mismatch.
- Skip stages whose fingerprints match and are complete; for matching partial stages, call `DataDesigner.create(..., resume=ResumeMode.ALWAYS)` to use #526's batch/row-group resume.
- Preserve `completed_empty` and `skipped_empty_upstream` statuses on resume when fingerprints still match; rerun from the first changed or missing upstream stage otherwise.
- Before skipping or resuming a stage seeded by `on_success`, validate the recorded callback output path exists and can seed `LocalFileSeedSource`; if missing, invalidate that stage and descendants.
- For invalidated stages, clear or replace the deterministic stage directory before starting fresh so `ArtifactStorage` does not timestamp away from the workflow layout.
- Depends on artifact layout from phase 1.

**Status after stage-level resume slice:** Implemented `workflow.run(resume=...)`, compatible completed-stage reuse, matching partial-stage delegation to `DataDesigner.create(..., resume=ResumeMode.ALWAYS)`, downstream invalidation after changed or missing stages, callback output path checks, and docs for `ResumeMode.IF_POSSIBLE` / `ResumeMode.ALWAYS`. Still deferred: DAG branches, `allow_resize` removal, config bundles, and broader first-class artifact seeding.

### Phase 4: DAG-shaped stages with parallel branches

- Extend `add_stage()` with an optional `depends_on=[stage_name, ...]` argument; default keeps the linear behavior.
- `workflow.run()` walks the resulting DAG, gathering independent branches via `asyncio.gather` over `data_designer.acreate()` calls.
- Per-stage fingerprint composition (Phase 3) generalizes naturally: a stage's upstream fingerprint becomes the hash of all parent fingerprints sorted by stage name, making joins stable regardless of `depends_on` declaration order.
- Join data contract is not automatic dataset fan-in. A multi-parent child still needs one concrete seed dataset; Phase 4 can require a single declared seed parent or an explicit merge stage/callback that writes one `LocalFileSeedSource`-compatible dataset. First-class multi-parent callbacks keyed by parent name are future work.
- Throttle coordination relies on the existing invariant: all branches run on the same parent `DataDesigner`, so `ThrottleManager` is shared.
- Hard dependency on the `acreate()` sidecar.
- **Scope: branch parallelism, not streaming dependent-stage execution.** Stages still wait for their dependencies to fully complete before starting; overlapped execution across a dependency edge is a separate direction sketched in Future considerations.
- Tests: fan-out (one upstream, multiple parallel children); join (multiple upstreams, one child); resume invalidation when one branch's fingerprint changes; throttle behavior under N parallel branches.

## Future considerations

Items not on the current roadmap but worth flagging so they don't get accidentally precluded by the current design choices.

**External orchestration for cross-process / distributed execution.** There is interest in eventually running DataDesigner workloads across processes or nodes - self-hosted serving, multi-host fan-out, scheduling against external clusters. The specific shape of that orchestration is still under discussion and is not committed to here. The chaining plan's design choices (parent `DataDesigner` reuse, on-disk handoffs, no new engine surface) compose naturally with such a system: an external orchestrator could dispatch independent `DataDesigner.create()` calls against partitioned slices and per-replica endpoints without the `CompositeWorkflow` class needing to change. Phases 1-4 do not depend on this materializing.

**Streaming dependent-stage execution.** Today the stage data contract is "final dataset" - a downstream stage waits for its upstream to fully complete. A future direction is to let downstream stages consume upstream batches as they're produced, overlapping execution across the dependency edge. Required changes: streaming seed sources, an explicit "stage done" sentinel rather than file-completion checks, and resume semantics for partially-consumed upstreams. Most useful when stage bottlenecks are heterogeneous (LLM-bound stage feeding a CPU-bound validator); little gain when both stages are LLM-bound since they share provider capacity. Not designed here; flagged so the stage contract isn't quietly closed off.

**Stage-level run config and compute placement.** v1 stages can have different `DataDesignerConfig` values, including different model configs, but reuse the parent `DataDesigner` and run settings for shared throttle coordination. A future extension can allow per-stage `RunConfig`, provider, or compute placement, but it needs explicit rules for throttling, artifact ownership, and cross-process resume.

**Failure hooks.** v1 should raise on failure by default. A future `on_failure` callback could support cleanup, custom recovery, or explicit handling of all-rows-filtered cases, but it should not obscure the default failure semantics.

**Seeded processor-only configs (implemented in #636).** Do not introduce a separate public `stages` wrapper module or a workflow-specific `processors=[...]` stage argument. Keep `CompositeWorkflow.add_stage()` as the orchestration surface, and make processor-only work expressible as a normal seeded config:

```python
workflow.add_stage("drafts", config_drafts, num_records=1000)
workflow.add_stage("filtered", DataDesignerConfigBuilder().add_processor(...))
workflow.add_stage("enriched", config_enrich)
```

The same config should also work outside workflows when the user supplies a seed directly:

```python
config = (
    DataDesignerConfigBuilder()
    .with_seed_dataset(LocalFileSeedSource(path="data/*.parquet"))
    .add_processor(...)
)
result = data_designer.create(config)
```

Validation allows a seed-only config when it has processors and a seed dataset, while still rejecting processor-only configs with no seed and configs that drop all output columns.

**Stage output processors (implemented in #636).** `CompositeWorkflow.add_stage()` accepts `output_processors=[ProcessorConfig, ...]` for compact inline transforms at a stage boundary. Internally this reuses the seeded processor-only config path above: seed a temporary processor config from the completed stage output, run it under a deterministic subdirectory, and use its final dataset as the stage output. This keeps row-count-changing 1:N and N:1 operations out of the async generation batch path while making the common case concise. Custom arbitrary transforms should use `CustomProcessor` when #159 lands; until then, keep raw `on_success` as the escape hatch.

**Config bundles.** Built-in or plugin-provided bundles should be reusable config fragments/factories, not execution stages. A bundle can return or extend a `DataDesignerConfigBuilder`, and `add_stage()` can accept either a builder or a bundle by normalizing the bundle to a builder:

```python
workflow.add_stage("draft_qa", bundles.document_qa(model_alias="fast"))
workflow.add_stage("quality", bundles.quality_judge(model_alias="judge"))
workflow.add_stage("sft", bundles.sft_export(), output="processor:sft")
```

Bundles can cover DataArc-like document QA, refinement, distillation, SFT export, or GRPO export shapes without introducing another workflow abstraction. Plugins can contribute bundles alongside existing column, seed-reader, and processor plugins.

**First-class artifact seeding.** v1 can select named processor outputs with `output="processor:<name>"`. Broader artifact seeding, such as media assets or arbitrary artifact folders, still requires `on_success` callbacks. A later API could support `seed_from_artifact(...)` references directly.

**Auto-composition from a single config.** Auto-detecting stage boundaries in one config is possible, but likely overkill for the initial roadmap. If it returns later, it should be a convenience layer on top of `CompositeWorkflow`, not a separate execution model.

## Resolved decisions

These were open in earlier drafts; recording the resolutions here so the design is unambiguous.

1. **Naming** -> The public abstraction is `CompositeWorkflow`, created through `data_designer.compose_workflow(name=...)`. This makes composition of multiple independent workflows explicit without implying a broader builder abstraction or a strictly linear linked list.

2. **In-memory vs on-disk handoff between stages** -> Always on-disk inside `CompositeWorkflow`. The in-memory `DataFrameSeedSource` mode is reserved for the lightweight `to_config_builder()` notebook ergonomic, which is explicitly *not* a `CompositeWorkflow`. Reasons: single execution model, simpler resume story, and composability with any future external orchestration that can't share an in-memory DataFrame across process boundaries. Cost is one parquet round-trip per stage, negligible relative to LLM call time.

3. **Branch/fan-out semantics (DAG)** -> Designed-in but not v1. The internal stage representation is a DAG; v1 only accepts linear inputs through `add_stage()`. Phase 4 ships parallel branches via `asyncio.gather` over `acreate()`. v1 stays sequential.

4. **CompositeWorkflow construction** -> `CompositeWorkflow` is created via `data_designer.compose_workflow(name=...)` and reuses the parent `DataDesigner`'s `ModelRegistry` and `ThrottleManager` across all stages. The explicit name is the durable artifact identity used for resume, and the workflow does not construct its own `DataDesigner` instances. This is the throttle-coordination invariant (see Composability section).

5. **Downstream seeding scope** -> v1 seeds downstream stages from the upstream stage's final dataset by default. Named processor outputs can be selected with `output="processor:<name>"`. Dropped columns, media forwarding, and arbitrary artifacts require an `on_success` callback in v1 and can become first-class APIs later.

6. **Export and push semantics** -> v1 export/push defaults to the final stage's dataset only. Exporting selected stages or a full composite workflow artifact bundle is future work.

7. **Stage extension model** -> The public surface remains `CompositeWorkflow`, not a separate `stages` module. Near-term extensibility is seeded processor-only configs, output processors, and stage output selection. Shared built-in/plugin units are config bundles, not a new executor.

## Open questions

1. **Preview support**: Should `workflow.preview()` run all stages with small `num_records`? Or just preview the last stage seeded from a prior full run?

2. **Config serialization**: For persistence, composite workflow configs would need symbolic stage references ("seed from stage X's output"). With the on-disk handoff decision above, the `DataFrameSeedSource` blocker is no longer relevant; the remaining question is how to encode stage dependencies in YAML. Needed for future declarative workflows or auto-composition, but not for the explicit API in phases 1-4.

3. **Image/media column forwarding**: Images in create mode are stored as relative file paths. If a downstream stage seeds from an upstream stage that produced images, the relative paths break. v1 can bridge this with an `on_success` callback or expression columns that rewrite paths relative to the upstream stage's artifact directory. First-class options remain: (a) resolve to absolute paths at stage boundary, (b) copy media assets into downstream stage's directory, (c) document as unsupported in v1.

## Related issues

- #447 - AsyncRunController refactor (partially superseded: pre-batch resize handling moves to composite workflow level instead of controller level)
- #526 / #525 - Resume interrupted runs (single-stage batch/row-group resume primitive used by composite workflow stage resume)
- #159 - Custom processors (needed for arbitrary Python output processors without workflow-only callback types)
- #462 - Progress bar and scheduler polish (independent)
- #464 - Custom column retryable errors (independent)
