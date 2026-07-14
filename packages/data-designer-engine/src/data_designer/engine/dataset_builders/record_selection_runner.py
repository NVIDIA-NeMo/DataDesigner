# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.record_selection import RecordSelectionConfig, RecordSelectionExhaustion
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.column_generators.generators.base import ColumnGenerator
from data_designer.engine.dataset_builders.acceptance import (
    AcceptanceController,
    CandidateBatch,
    SelectionBatchMarker,
    SelectionDecision,
)
from data_designer.engine.dataset_builders.async_scheduler import AsyncTaskScheduler
from data_designer.engine.dataset_builders.errors import (
    DatasetGenerationError,
    RecordSelectionEarlyShutdownError,
    RecordSelectionExhaustedError,
)
from data_designer.engine.dataset_builders.row_group_plan import ExplicitRowGroupPlan
from data_designer.engine.dataset_builders.utils.async_concurrency import (
    await_async_scheduler_result,
    ensure_async_engine_loop,
    is_async_trace_enabled,
)
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.observability import JsonlSchedulerEventSink, SchedulerAdmissionEventSink
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.storage.artifact_storage import (
    METADATA_FILENAME,
    SDG_CONFIG_FILENAME,
    ResumeMode,
)
from data_designer.engine.storage.media_storage import StorageMode

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.run_config import RunConfig
    from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)


class _SelectionPostCommitError(DatasetGenerationError):
    """A transient failure after a candidate checkpoint reached durable storage."""


@dataclass(slots=True)
class _SelectionCandidateBatchRuntime:
    """Mutable state shared by one candidate batch and its scheduler callbacks."""

    controller: AcceptanceController
    batch: CandidateBatch
    on_batch_complete: Callable[[Path], None] | None
    media_staged: bool
    decision: SelectionDecision | None = None
    buffer_manager: RowGroupBufferManager | None = None
    scheduler: AsyncTaskScheduler | None = None
    checkpoint_committed: bool = False


class _SelectionMetadataState(StrEnum):
    MISSING = "missing"
    READABLE = "readable"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class _SelectionResumeProbe:
    """Unresolved on-disk state used by selection resume compatibility and cleanup."""

    metadata_state: _SelectionMetadataState
    has_selection_metadata: bool
    stored_target_num_records: Any
    stored_buffer_size: Any
    checkpoint_directory_exists: bool
    checkpoint_marker_exists: bool

    @property
    def is_selection_run(self) -> bool:
        """Whether cleanup may safely treat the unresolved directory as selection-owned."""
        return self.has_selection_metadata or self.checkpoint_directory_exists


class RecordSelectionRunner:
    """Generate, checkpoint, resume, and publish one record-selection build."""

    def __init__(self, builder: DatasetBuilder) -> None:
        self._builder = builder

    def _selection_resume_probe(self) -> _SelectionResumeProbe:
        """Read selection state without resolving or caching an ArtifactStorage dataset path."""
        storage = self._builder.artifact_storage
        dataset_dir = Path(storage.artifact_path) / storage.dataset_name
        metadata_path = dataset_dir / METADATA_FILENAME
        checkpoints_path = dataset_dir / "selection-checkpoints"
        checkpoint_directory_exists = checkpoints_path.exists()
        checkpoint_marker_exists = checkpoint_directory_exists and any(checkpoints_path.glob("batch_*.json"))

        if not metadata_path.exists():
            return _SelectionResumeProbe(
                metadata_state=_SelectionMetadataState.MISSING,
                has_selection_metadata=False,
                stored_target_num_records=None,
                stored_buffer_size=None,
                checkpoint_directory_exists=checkpoint_directory_exists,
                checkpoint_marker_exists=checkpoint_marker_exists,
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return _SelectionResumeProbe(
                metadata_state=_SelectionMetadataState.UNREADABLE,
                has_selection_metadata=False,
                stored_target_num_records=None,
                stored_buffer_size=None,
                checkpoint_directory_exists=checkpoint_directory_exists,
                checkpoint_marker_exists=checkpoint_marker_exists,
            )

        selection = metadata.get("record_selection") if isinstance(metadata, dict) else None
        has_selection_metadata = isinstance(selection, dict)
        return _SelectionResumeProbe(
            metadata_state=_SelectionMetadataState.READABLE,
            has_selection_metadata=has_selection_metadata,
            stored_target_num_records=metadata.get("target_num_records") if isinstance(metadata, dict) else None,
            stored_buffer_size=selection.get("run_buffer_size") if has_selection_metadata else None,
            checkpoint_directory_exists=checkpoint_directory_exists,
            checkpoint_marker_exists=checkpoint_marker_exists,
        )

    def runtime_inputs_are_compatible(self, target_num_records: int, buffer_size: int) -> bool:
        """Check selection-only resume inputs without resolving a new dataset path."""
        probe = self._selection_resume_probe()
        if probe.metadata_state == _SelectionMetadataState.MISSING:
            return not probe.checkpoint_marker_exists
        if probe.metadata_state == _SelectionMetadataState.UNREADABLE or not probe.has_selection_metadata:
            return False
        return probe.stored_target_num_records == target_num_records and probe.stored_buffer_size == buffer_size

    def clear_incompatible_artifacts(self) -> None:
        """Clear engine-managed artifacts when IF_POSSIBLE restarts an existing selection run."""
        if self._builder._data_designer_config.record_selection is None:
            return
        storage = self._builder.artifact_storage
        dataset_dir = Path(storage.artifact_path) / storage.dataset_name
        if not self._selection_resume_probe().is_selection_run:
            return

        managed_directories = {
            storage.final_dataset_folder_name,
            storage.partial_results_folder_name,
            storage.dropped_columns_folder_name,
            storage.processors_outputs_folder_name,
            storage.media_storage.images_subdir,
            "selection-accepted",
            "selection-checkpoints",
            "selection-media-staging",
            "selection-publication-staging",
        }
        for name in managed_directories:
            path = dataset_dir / name
            if path.exists():
                shutil.rmtree(path)
        for name in (
            METADATA_FILENAME,
            SDG_CONFIG_FILENAME,
            "scheduler_events.jsonl",
        ):
            (dataset_dir / name).unlink(missing_ok=True)

    def run(
        self,
        generators: list[ColumnGenerator],
        *,
        target_num_records: int,
        buffer_size: int,
        on_batch_complete: Callable[[Path], None] | None,
        resume: ResumeMode,
    ) -> None:
        """Generate immutable candidate batches until the accepted-row target or cap is reached."""
        config = self._builder._data_designer_config.record_selection
        if config is None:
            raise RuntimeError("Record-selection build path requires a record-selection config.")

        self._builder.artifact_storage.configure_selection_batch_file_width(
            max_candidate_records=config.max_candidate_records,
            candidate_batch_size=min(buffer_size, target_num_records),
        )
        controller, completed_publication = self._restore_selection_run(
            config=config,
            target_num_records=target_num_records,
            buffer_size=buffer_size,
            resume=resume,
        )
        if self._reuse_completed_selection_publication(
            controller,
            completed_publication=completed_publication,
        ):
            return

        self._write_selection_metadata(controller)
        self._run_selection_candidate_batches(
            generators,
            controller=controller,
            on_batch_complete=on_batch_complete,
        )
        self._finalize_selection_build(
            controller,
            target_num_records=target_num_records,
            buffer_size=buffer_size,
        )

    def _restore_selection_run(
        self,
        *,
        config: RecordSelectionConfig,
        target_num_records: int,
        buffer_size: int,
        resume: ResumeMode,
    ) -> tuple[AcceptanceController, bool]:
        """Restore durable selection progress and prepare it for generation or publication."""
        markers = self._load_selection_markers() if resume == ResumeMode.ALWAYS else ()
        try:
            controller = AcceptanceController(
                config=config,
                target_records=target_num_records,
                buffer_size=buffer_size,
                markers=markers,
            )
        except ValueError as exc:
            raise DatasetGenerationError(f"🛑 Cannot resume record selection: {exc}") from exc

        existing_metadata: dict[str, Any] = {}
        if self._builder.artifact_storage.metadata_file_path.exists():
            try:
                existing_metadata = self._builder.artifact_storage.read_metadata()
            except (OSError, json.JSONDecodeError):
                existing_metadata = {}

        self._builder.artifact_storage.clear_selection_transient_artifacts()
        self._builder.artifact_storage.clean_uncommitted_selection_batch(controller.candidate_batches_completed)
        self._validate_selection_partitions(controller.markers)
        self._restore_committed_selection_schema(controller)
        self._builder._actual_num_records = controller.accepted_records

        if resume == ResumeMode.ALWAYS:
            if controller.last_batch_stopped_early and controller.is_exhausted:
                raise RecordSelectionEarlyShutdownError(candidate_budget_remaining=False)
            self._raise_nonretryable_empty_partial(controller)

        completed_publication = (
            resume == ResumeMode.ALWAYS and existing_metadata.get("post_generation_state") == "complete"
        )
        return controller, completed_publication

    def _reuse_completed_selection_publication(
        self,
        controller: AcceptanceController,
        *,
        completed_publication: bool,
    ) -> bool:
        """Reuse a valid completed publication or mark it incomplete for rebuilding."""
        if not completed_publication:
            return False

        terminal_selection = controller.has_reached_target or controller.is_exhausted
        if terminal_selection:
            published_files = list(self._builder.artifact_storage.final_dataset_path.glob("batch_*.parquet"))
            try:
                final_count: int | None = sum(lazy.pq.read_metadata(path).num_rows for path in published_files)
            except (OSError, lazy.pa.ArrowInvalid):
                final_count = None
            publication_reusable = bool(published_files) and final_count == controller.accepted_records
            if publication_reusable:
                logger.warning("▶️ Record-selection dataset is already complete; nothing to resume.")
                return True
            logger.warning(
                "⚠️ Completed record-selection publication contains %s rows, but committed markers contain %d; "
                "rebuilding the published view from immutable accepted partitions.",
                final_count if final_count is not None else "unreadable",
                controller.accepted_records,
            )

        self._builder.artifact_storage.update_metadata(
            {"post_generation_state": "started", "post_generation_processed": False}
        )
        return False

    def _run_selection_candidate_batches(
        self,
        generators: list[ColumnGenerator],
        *,
        controller: AcceptanceController,
        on_batch_complete: Callable[[Path], None] | None,
    ) -> None:
        """Generate and durably commit candidate batches until selection terminates."""
        while not controller.has_reached_target and controller.has_candidate_budget:
            batch = controller.next_candidate_batch()
            self._run_candidate_batch(
                generators,
                controller=controller,
                batch=batch,
                on_batch_complete=on_batch_complete,
            )
            self._raise_if_selection_stopped_early(controller)

    def _raise_if_selection_stopped_early(
        self,
        controller: AcceptanceController,
    ) -> None:
        """Raise the structured error for scheduler early shutdown."""
        if not self._builder._early_shutdown or controller.has_reached_target:
            return
        raise RecordSelectionEarlyShutdownError(
            candidate_budget_remaining=controller.has_candidate_budget,
        )

    def _finalize_selection_build(
        self,
        controller: AcceptanceController,
        *,
        target_num_records: int,
        buffer_size: int,
    ) -> None:
        """Apply terminal policy and publish the immutable accepted partitions."""
        self._builder._actual_num_records = controller.accepted_records
        if controller.is_exhausted and controller.config.on_exhausted == RecordSelectionExhaustion.RAISE:
            self._write_selection_metadata(controller)
            raise RecordSelectionExhaustedError(
                target_records=target_num_records,
                accepted_records=controller.accepted_records,
                candidate_records=controller.candidate_records,
                max_candidate_records=controller.config.max_candidate_records,
            )

        self._raise_nonretryable_empty_partial(controller)
        self._ensure_selection_schema(controller)
        self._publish_selection_result(controller, buffer_size)

    def _run_candidate_batch(
        self,
        generators: list[ColumnGenerator],
        *,
        controller: AcceptanceController,
        batch: CandidateBatch,
        on_batch_complete: Callable[[Path], None] | None,
    ) -> None:
        """Run and durably commit one candidate batch."""
        settings = self._builder._resource_provider.run_config
        trace_enabled = is_async_trace_enabled(settings)
        media_staged = (
            self._builder._has_image_columns() and self._builder.artifact_storage.media_storage.mode == StorageMode.DISK
        )
        runtime = _SelectionCandidateBatchRuntime(
            controller=controller,
            batch=batch,
            on_batch_complete=on_batch_complete,
            media_staged=media_staged,
        )

        if media_staged:
            self._builder.artifact_storage.begin_selection_media_batch(batch.candidate_batch_id)

        pre_batch_snapshot = self._builder._resource_provider.model_registry.get_model_usage_snapshot()
        group_id = uuid.uuid4().hex
        event_sink_context = (
            JsonlSchedulerEventSink(self._builder.artifact_storage.base_dataset_path / "scheduler_events.jsonl")
            if settings.write_scheduler_events
            else contextlib.nullcontext()
        )
        run_error: BaseException | None = None
        try:
            with event_sink_context as scheduler_event_sink:
                self._execute_selection_candidate_scheduler(
                    generators,
                    runtime=runtime,
                    settings=settings,
                    trace_enabled=trace_enabled,
                    scheduler_event_sink=scheduler_event_sink,
                )
        except BaseException as exc:
            run_error = exc
            raise
        finally:
            self._finish_selection_candidate_batch(runtime, run_error=run_error)

        try:
            usage_deltas = self._builder._resource_provider.model_registry.get_usage_deltas(pre_batch_snapshot)
            self._builder._emit_batch_inference_events("selection_batch", usage_deltas, group_id)
        except Exception:
            logger.debug("Failed to emit batch telemetry for record selection", exc_info=True)

    def _select_selection_candidate_dataframe(
        self,
        runtime: _SelectionCandidateBatchRuntime,
        _row_group_id: int,
        row_group_size: int,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply the selection predicate before post-batch processing."""
        runtime.decision = runtime.controller.select(
            dataframe,
            batch=runtime.batch,
            failed_generation_records=row_group_size - len(dataframe),
        )
        selected = dataframe.iloc[list(runtime.decision.accepted_indices)].reset_index(drop=True)
        if runtime.media_staged:
            # Commit accepted media before post-batch processors run. Processors may
            # persist transformed or dropped-column side artifacts, and every such
            # artifact must contain the same durable paths as the accepted partition.
            selected = self._builder.artifact_storage.promote_selection_media(
                selected,
                runtime.batch.candidate_batch_id,
            )
        return selected

    def _finalize_selection_candidate_row_group(
        self,
        runtime: _SelectionCandidateBatchRuntime,
        row_group_id: int,
    ) -> None:
        """Commit a selected row group, then mirror progress and notify the caller."""
        buffer_manager = runtime.buffer_manager
        decision = runtime.decision
        if buffer_manager is None or decision is None:
            raise DatasetGenerationError("🛑 Record selection reached finalization without a selection decision.")
        try:
            partition_path = self._commit_selection_candidate_row_group(
                runtime,
                row_group_id=row_group_id,
                buffer_manager=buffer_manager,
                decision=decision,
            )
            self._mirror_committed_selection_batch(
                runtime,
                row_group_id=row_group_id,
                buffer_manager=buffer_manager,
            )
            if partition_path is not None and runtime.on_batch_complete is not None:
                try:
                    runtime.on_batch_complete(partition_path)
                except Exception as exc:
                    raise _SelectionPostCommitError(str(exc)) from exc
        except DatasetGenerationError:
            raise
        except Exception as exc:
            raise DatasetGenerationError(
                f"🛑 Failed to commit record-selection candidate batch {runtime.batch.candidate_batch_id}: {exc}"
            ) from exc

    def _commit_selection_candidate_row_group(
        self,
        runtime: _SelectionCandidateBatchRuntime,
        *,
        row_group_id: int,
        buffer_manager: RowGroupBufferManager,
        decision: SelectionDecision,
    ) -> Path | None:
        """Persist selection artifacts and the authoritative candidate checkpoint."""
        dataframe = buffer_manager.get_dataframe(row_group_id)
        if len(dataframe) != decision.accepted_records:
            raise DatasetGenerationError(
                "🛑 Post-batch processing changed the selected row count from "
                f"{decision.accepted_records} to {len(dataframe)}."
            )
        schema_materialized = len(dataframe.columns) > 0
        if schema_materialized and not self._builder.artifact_storage.selection_schema_path.is_file():
            self._builder.artifact_storage.write_selection_schema(dataframe)
        partition_path: Path | None = None
        partition_relative: str | None = None
        if len(dataframe) > 0:
            partition_path = self._builder.artifact_storage.write_selection_partition(
                runtime.batch.candidate_batch_id,
                dataframe,
            )
            partition_relative = str(partition_path.relative_to(self._builder.artifact_storage.base_dataset_path))

        non_retryable_error, stopped_early = self._selection_candidate_diagnostics(runtime, decision=decision)
        marker = runtime.controller.record_checkpoint(
            batch=runtime.batch,
            decision=decision,
            accepted_partition=partition_relative,
            schema_materialized=schema_materialized,
            non_retryable_error=non_retryable_error,
            stopped_early=stopped_early,
        )
        self._builder.artifact_storage.write_selection_checkpoint(
            runtime.batch.candidate_batch_id,
            marker.to_dict(),
        )
        runtime.checkpoint_committed = True
        return partition_path

    @staticmethod
    def _selection_candidate_diagnostics(
        runtime: _SelectionCandidateBatchRuntime,
        *,
        decision: SelectionDecision,
    ) -> tuple[str | None, bool]:
        """Capture scheduler diagnostics in the same transaction as the candidate marker."""
        scheduler = runtime.scheduler
        non_retryable = scheduler.first_non_retryable_error if scheduler is not None else None
        non_retryable_error = f"{type(non_retryable).__name__}: {non_retryable}" if non_retryable is not None else None
        stopped_early = bool(
            scheduler is not None
            and scheduler.early_shutdown
            and runtime.controller.accepted_records + decision.accepted_records < runtime.controller.target_records
        )
        return non_retryable_error, stopped_early

    def _mirror_committed_selection_batch(
        self,
        runtime: _SelectionCandidateBatchRuntime,
        *,
        row_group_id: int,
        buffer_manager: RowGroupBufferManager,
    ) -> None:
        """Mirror a durable checkpoint into in-memory state and metadata."""
        controller = runtime.controller
        try:
            buffer_manager.free_row_group(row_group_id)
            self._builder._actual_num_records = controller.accepted_records
            self._write_selection_metadata(controller)
            rate = controller.accepted_records / controller.candidate_records
            logger.info(
                "🎯 Record selection: accepted %d / %d; candidates generated %d / %d; acceptance rate %.1f%%.",
                controller.accepted_records,
                controller.target_records,
                controller.candidate_records,
                controller.config.max_candidate_records,
                rate * 100,
            )
        except Exception as exc:
            raise _SelectionPostCommitError(
                f"🛑 Failed after committing record-selection candidate batch {runtime.batch.candidate_batch_id}: {exc}"
            ) from exc

    def _execute_selection_candidate_scheduler(
        self,
        generators: list[ColumnGenerator],
        *,
        runtime: _SelectionCandidateBatchRuntime,
        settings: RunConfig,
        trace_enabled: bool,
        scheduler_event_sink: SchedulerAdmissionEventSink | None,
    ) -> None:
        """Wire and run the async scheduler for one explicit candidate row group."""
        scheduler, buffer_manager = self._builder._prepare_async_run(
            generators,
            runtime.batch.size,
            runtime.batch.size,
            on_finalize_row_group=functools.partial(self._finalize_selection_candidate_row_group, runtime),
            finalize_empty_row_groups=True,
            shutdown_error_rate=settings.shutdown_error_rate,
            shutdown_error_window=settings.shutdown_error_window,
            disable_early_shutdown=settings.disable_early_shutdown,
            trace=trace_enabled,
            precomputed_row_groups=ExplicitRowGroupPlan(
                ((runtime.batch.row_group_id, runtime.batch.size),),
                base_offset=runtime.batch.start_offset,
            ),
            scheduler_event_sink=scheduler_event_sink,
            select_dataframe=functools.partial(self._select_selection_candidate_dataframe, runtime),
            log_pre_generation=False,
        )
        runtime.scheduler = scheduler
        runtime.buffer_manager = buffer_manager
        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(scheduler.run(), loop)
        await_async_scheduler_result(future, scheduler)

    def _finish_selection_candidate_batch(
        self,
        runtime: _SelectionCandidateBatchRuntime,
        *,
        run_error: BaseException | None,
    ) -> None:
        """Clean staging and merge scheduler state after a candidate attempt."""
        if runtime.media_staged:
            try:
                self._builder.artifact_storage.finish_selection_media_batch(runtime.batch.candidate_batch_id)
            except Exception as exc:
                if run_error is not None:
                    logger.warning("⚠️ Failed to clean record-selection media staging.", exc_info=True)
                elif runtime.checkpoint_committed:
                    raise _SelectionPostCommitError(
                        f"🛑 Failed after committing record-selection candidate batch "
                        f"{runtime.batch.candidate_batch_id}: {exc}"
                    ) from exc
                else:
                    raise

        scheduler = runtime.scheduler
        if scheduler is None:
            return
        self._builder._task_traces.extend(scheduler.traces)
        self._builder._early_shutdown = self._builder._early_shutdown or scheduler.early_shutdown
        self._builder._partial_row_groups = tuple(
            sorted(set(self._builder._partial_row_groups).union(scheduler.partial_row_groups))
        )
        if self._builder._first_non_retryable_error is None:
            self._builder._first_non_retryable_error = scheduler.first_non_retryable_error

    def _load_selection_markers(self) -> tuple[SelectionBatchMarker, ...]:
        try:
            raw_markers = self._builder.artifact_storage.read_selection_checkpoints()
            return tuple(SelectionBatchMarker.from_dict(value) for value in raw_markers)
        except (ValueError, TypeError) as exc:
            raise DatasetGenerationError(f"🛑 Cannot resume record selection: {exc}") from exc

    def _validate_selection_partitions(self, markers: tuple[SelectionBatchMarker, ...]) -> None:
        referenced: set[Path] = set()
        for marker in markers:
            if marker.accepted_partition is None:
                continue
            expected = self._builder.artifact_storage.selection_partition_path(marker.candidate_batch_id)
            actual = self._builder.artifact_storage.base_dataset_path / marker.accepted_partition
            if actual != expected:
                raise DatasetGenerationError(
                    f"🛑 Selection checkpoint {marker.candidate_batch_id} references unexpected partition "
                    f"{marker.accepted_partition!r}."
                )
            try:
                rows = lazy.pq.read_metadata(actual).num_rows
            except (OSError, lazy.pa.ArrowInvalid) as exc:
                raise DatasetGenerationError(
                    f"🛑 Selection checkpoint {marker.candidate_batch_id} references a missing or unreadable "
                    f"accepted partition: {actual}."
                ) from exc
            if rows != marker.accepted_records:
                raise DatasetGenerationError(
                    f"🛑 Selection checkpoint {marker.candidate_batch_id} records {marker.accepted_records} "
                    f"accepted rows, but its partition contains {rows}."
                )
            referenced.add(actual)

        if self._builder.artifact_storage.selection_accepted_path.exists():
            for partition in self._builder.artifact_storage.selection_accepted_path.glob("batch_*.parquet"):
                if partition not in referenced:
                    partition.unlink()

    @staticmethod
    def _raise_nonretryable_empty_partial(
        controller: AcceptanceController,
    ) -> None:
        """Surface the scheduler diagnostic when every candidate failed generation."""
        if (
            not controller.is_exhausted
            or controller.accepted_records != 0
            or RecordSelectionExhaustion(controller.config.on_exhausted) != RecordSelectionExhaustion.RETURN_PARTIAL
        ):
            return
        diagnostic = controller.first_non_retryable_error
        if diagnostic is not None:
            raise DatasetGenerationError(diagnostic)

    def _derive_empty_selection_schema(self) -> pd.DataFrame:
        """Build a name-bearing fallback schema when every candidate slot failed generation."""
        columns: dict[str, None] = {}
        for config in self._builder.single_column_configs:
            columns[config.name] = None
            columns.update(dict.fromkeys(config.side_effect_columns))

        drop_patterns = [
            pattern
            for processor in self._builder._processor_runner.processors
            if isinstance(processor, DropColumnsProcessor)
            for pattern in processor.config.column_names
        ]
        output_columns = [
            column for column in columns if not any(fnmatch(column, pattern) for pattern in drop_patterns)
        ]
        return lazy.pd.DataFrame(columns=output_columns)

    def _restore_committed_selection_schema(self, controller: AcceptanceController) -> None:
        """Discard an orphan schema or restore the schema owned by committed work."""
        schema_path = self._builder.artifact_storage.selection_schema_path
        schema_materialized = any(
            marker.schema_materialized or marker.accepted_partition is not None for marker in controller.markers
        )
        if schema_path.is_file():
            if not schema_materialized:
                schema_path.unlink()
            return

        accepted_marker = next((marker for marker in controller.markers if marker.accepted_partition is not None), None)
        if accepted_marker is not None:
            partition = self._builder.artifact_storage.base_dataset_path / accepted_marker.accepted_partition
            self._builder.artifact_storage.write_selection_schema(lazy.pd.read_parquet(partition))
        elif schema_materialized:
            raise DatasetGenerationError("🛑 Cannot resume record selection: the committed output schema is missing.")

    def _ensure_selection_schema(self, controller: AcceptanceController) -> None:
        """Ensure empty or all-failed selections still publish a schema-bearing dataset."""
        self._restore_committed_selection_schema(controller)
        if self._builder.artifact_storage.selection_schema_path.is_file():
            return
        self._builder.artifact_storage.write_selection_schema(self._derive_empty_selection_schema())

    def _write_selection_metadata(
        self,
        controller: AcceptanceController,
        *,
        post_generation_state: str | None = None,
    ) -> None:
        selection_summary: dict[str, Any] = controller.summary()
        updates: dict[str, Any] = {
            "target_num_records": controller.target_records,
            "original_target_num_records": controller.target_records,
            "actual_num_records": controller.accepted_records,
            "total_num_batches": controller.accepted_partitions,
            "buffer_size": controller.buffer_size,
            "dataset_name": self._builder.artifact_storage.dataset_name,
            "num_completed_batches": controller.candidate_batches_completed,
            "record_selection": selection_summary,
        }
        if post_generation_state == "complete":
            # Candidate checkpoints are already represented by their immutable marker and
            # accepted partition. Refreshing the full processor-artifact manifest after
            # every candidate batch makes selection quadratic as that tree grows.
            updates["file_paths"] = self._builder.artifact_storage.get_file_paths()
        if post_generation_state is not None:
            updates["post_generation_state"] = post_generation_state
            updates["post_generation_processed"] = post_generation_state == "complete"
        self._builder.artifact_storage.update_metadata(updates)

    def _publish_selection_result(self, controller: AcceptanceController, buffer_size: int) -> None:
        self._write_selection_metadata(controller, post_generation_state="started")
        self._builder.artifact_storage.materialize_selection_dataset()
        self._builder._processor_runner.run_after_generation(buffer_size, selection_publication=True)
        actual_records = self._count_published_selection_records()
        if actual_records != controller.accepted_records:
            raise DatasetGenerationError(
                "🛑 After-generation processing changed the record-selection output count from "
                f"{controller.accepted_records} to {actual_records}. Row-count-changing after-generation "
                "processors are not supported with record selection."
            )
        self._builder.artifact_storage.clear_selection_transient_artifacts()
        self._write_selection_metadata(controller, post_generation_state="complete")

    def _count_published_selection_records(self) -> int:
        return sum(
            lazy.pq.read_metadata(path).num_rows
            for path in self._builder.artifact_storage.final_dataset_path.glob("batch_*.parquet")
        )
