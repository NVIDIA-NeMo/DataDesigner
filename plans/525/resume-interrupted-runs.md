---
date: 2026-04-13
authors:
  - pboruta
issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/525
---

# Plan: Resume interrupted dataset generation runs

## Problem

When a long-running `DataDesigner.create()` call is interrupted (machine crash, OOM kill, preemption), the user has to restart generation from scratch — even though completed batches are already durably written to disk and `metadata.json` tracks exactly how many finished.

The situation is made worse by an existing safeguard that fires at the wrong time: `ArtifactStorage.resolved_dataset_name` detects the existing folder on the next run and silently creates a new timestamped directory, orphaning the previous partial results instead of resuming from them.

## Proposed Solution

Add `resume: bool = False` to `DataDesigner.create()`. When `resume=True` the engine reads `metadata.json` from the existing dataset directory, validates that the run parameters are compatible, and starts the batch loop from the first incomplete batch rather than from zero.

Expected usage:

```python
dd = DataDesigner(...)
dd.add_column(...)

# First run — interrupted at batch 7 of 20
results = dd.create(config_builder, num_records=10_000)

# After restart — picks up from batch 8
results = dd.create(config_builder, num_records=10_000, resume=True)
```

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| API surface | `resume: bool = False` on `DataDesigner.create()` | Opt-in flag keeps default behaviour unchanged. Users who want a clean re-run keep getting the timestamped-folder behaviour. |
| Resume state source | Read `metadata.json` written after each completed batch | Already contains `num_completed_batches`, `target_num_records`, `buffer_size`, `actual_num_records`. No new persistence needed. |
| Partial batch at crash time | Clear `tmp-partial-parquet-files/` at resume start | Simpler and safer than merging an incomplete parquet; losing one batch is acceptable since the user is already recovering from a crash. |
| Compatibility validation | Raise `DatasetGenerationError` if `num_records` or `buffer_size` changed | Different `num_records` changes which rows land in which batch file, breaking the numbering invariant. `buffer_size` changes the file-per-batch mapping. Both must match. |
| Async engine | Raise `DatasetGenerationError` if `DATA_DESIGNER_ASYNC_ENGINE=1` with `resume=True` | The async path uses a row-group scheduler rather than an indexed batch loop; resume would require a different strategy. Out of scope for v1. |
| Already-complete runs | Detect and warn, return existing path | If `num_completed_batches == total_num_batches` the dataset is already complete; the user may have re-run by mistake. |
| No metadata → error | Raise `DatasetGenerationError` | Resuming without a checkpoint is impossible; a clear error is better than silent fallback to a fresh run. |

## Affected Files

| File | Change |
|---|---|
| `packages/data-designer-engine/src/data_designer/engine/storage/artifact_storage.py` | Add `resume: bool = False` field; modify `resolved_dataset_name` to skip timestamping when `resume=True`; add `clear_partial_results()` helper |
| `packages/data-designer-engine/src/data_designer/engine/dataset_builders/utils/dataset_batch_manager.py` | Add `start_batch: int = 0` and `initial_actual_num_records: int = 0` to `start()` |
| `packages/data-designer-engine/src/data_designer/engine/dataset_builders/dataset_builder.py` | Add `resume: bool = False` to `build()`; add `_load_resume_state()` private method; implement validation and batch-skip logic |
| `packages/data-designer/src/data_designer/interface/data_designer.py` | Add `resume: bool = False` to `create()` and `_create_resource_provider()`; pass through to `ArtifactStorage` and `builder.build()` |
| `packages/data-designer-engine/tests/engine/storage/test_artifact_storage.py` | Tests for resume flag on `resolved_dataset_name` and `clear_partial_results()` |
| `packages/data-designer-engine/tests/engine/dataset_builders/utils/test_dataset_batch_manager.py` | Tests for `start_batch` and `initial_actual_num_records` parameters |
| `packages/data-designer-engine/tests/engine/dataset_builders/test_dataset_builder.py` | Tests for resume validation, batch skipping, async engine error, already-complete detection |

## Implementation Sketch

### `ArtifactStorage`

```python
class ArtifactStorage(BaseModel):
    ...
    resume: bool = False

    @cached_property
    def resolved_dataset_name(self) -> str:
        dataset_path = self.artifact_path / self.dataset_name
        if dataset_path.exists() and len(list(dataset_path.iterdir())) > 0:
            if self.resume:
                return self.dataset_name  # use existing folder as-is
            # existing behaviour: create timestamped copy
            new_dataset_name = f"{self.dataset_name}_{datetime.now().strftime(...)}"
            ...
            return new_dataset_name
        if self.resume:
            raise ArtifactStorageError(
                f"Cannot resume: no existing dataset found at {dataset_path!r}."
            )
        return self.dataset_name

    def clear_partial_results(self) -> None:
        """Remove any in-flight partial results left over from an interrupted run."""
        if self.partial_results_path.exists():
            shutil.rmtree(self.partial_results_path)
```

### `DatasetBatchManager.start()`

```python
def start(
    self,
    *,
    num_records: int,
    buffer_size: int,
    start_batch: int = 0,
    initial_actual_num_records: int = 0,
) -> None:
    ...
    self.reset()
    self._current_batch_number = start_batch
    self._actual_num_records = initial_actual_num_records
```

### `DatasetBuilder.build()` — resume path

```python
@dataclass
class _ResumeState:
    num_completed_batches: int
    actual_num_records: int
    buffer_size: int

def _load_resume_state(self, num_records: int, buffer_size: int) -> _ResumeState:
    try:
        metadata = self.artifact_storage.read_metadata()
    except FileNotFoundError:
        raise DatasetGenerationError("Cannot resume: metadata.json not found. ...")

    target = metadata.get("target_num_records")
    if target != num_records:
        raise DatasetGenerationError(
            f"Cannot resume: num_records={num_records} does not match "
            f"the original run's target_num_records={target}. ..."
        )

    meta_buffer_size = metadata.get("buffer_size")
    if meta_buffer_size != buffer_size:
        raise DatasetGenerationError(
            f"Cannot resume: buffer_size={buffer_size} does not match "
            f"the original run's buffer_size={meta_buffer_size}. ..."
        )

    return _ResumeState(
        num_completed_batches=metadata["num_completed_batches"],
        actual_num_records=metadata["actual_num_records"],
        buffer_size=buffer_size,
    )

def build(self, *, num_records, on_batch_complete=None, save_multimedia_to_disk=True, resume=False):
    ...
    if resume and DATA_DESIGNER_ASYNC_ENGINE:
        raise DatasetGenerationError("resume=True is not supported with DATA_DESIGNER_ASYNC_ENGINE.")

    buffer_size = self._resource_provider.run_config.buffer_size

    if resume:
        state = self._load_resume_state(num_records, buffer_size)
        if state.num_completed_batches * buffer_size >= num_records:
            logger.warning("Dataset already complete — nothing to resume.")
            return self.artifact_storage.final_dataset_path
        self.artifact_storage.clear_partial_results()
        self.batch_manager.start(
            num_records=num_records,
            buffer_size=buffer_size,
            start_batch=state.num_completed_batches,
            initial_actual_num_records=state.actual_num_records,
        )
        for batch_idx in range(state.num_completed_batches, self.batch_manager.num_batches):
            ...
    else:
        # existing path unchanged
        self.batch_manager.start(num_records=num_records, buffer_size=buffer_size)
        for batch_idx in range(self.batch_manager.num_batches):
            ...
```

## Trade-offs Considered

- **Automatic resume detection** (no flag, detect existing folder automatically): rejected — removes user intent. A user re-running a pipeline from scratch would be surprised by silent resumption.
- **Resume support for async engine**: deferred to a follow-up. The async scheduler's row-group model doesn't map 1:1 to batch indices; implementing it would require a separate mechanism.
- **Per-column resume** (resume from column N within an interrupted batch): out of scope. Requires per-column checkpointing and state reconstruction, significantly higher complexity.

## Delivery

Single PR implementing all changes listed in the affected-files table plus tests. No backwards-incompatible changes — `resume` defaults to `False` and all existing call sites are unaffected.
