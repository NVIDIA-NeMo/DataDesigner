# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType, is_plugin_column_type
from data_designer.config.config_builder import BuilderConfig
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorType,
)
from data_designer.config.record_selection import RecordSelectionConfig, RecordSelectionExhaustion
from data_designer.config.sampler_params import CategorySamplerParams, SubcategorySamplerParams
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.config.version import get_library_version
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
)
from data_designer.engine.compiler import compile_data_designer_config
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
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.row_group_plan import (
    CompactRowGroupPlan,
    ExplicitRowGroupPlan,
    RowGroupInput,
    RowGroupPlanLike,
    normalize_row_group_plan,
)
from data_designer.engine.dataset_builders.scheduling.completion import CompletionTracker, FrontierDelta
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.engine.dataset_builders.utils.config_compiler import compile_dataset_builder_column_configs
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.processor_runner import ProcessorRunner, ProcessorStage
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum, TelemetryHandler
from data_designer.engine.observability import (
    JsonlSchedulerEventSink,
    SchedulerAdmissionEventSink,
    fanout_scheduler_event_sinks,
)
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.readiness import run_readiness_check
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.storage.artifact_storage import (
    METADATA_FILENAME,
    SDG_CONFIG_FILENAME,
    SELECTION_ARTIFACT_MIGRATION_FILENAME,
    ArtifactStorage,
    ResumeMode,
)
from data_designer.engine.storage.media_storage import StorageMode

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.run_config import RunConfig
    from data_designer.engine.dataset_builders.scheduling.task_model import TaskTrace
    from data_designer.engine.models.usage import ModelUsageStats

logger = logging.getLogger(__name__)


_CLIENT_VERSION: str = get_library_version()
PRESERVE_DROPPED_COLUMNS_METADATA_KEY = "preserve_dropped_columns"
_SELECTION_TERMINAL_ERROR_KEY = "terminal_error"
_SELECTION_EARLY_SHUTDOWN_KIND = "early_shutdown"
_SELECTION_GENERATION_ERROR_KIND = "generation_error"
_SELECTION_ARTIFACT_MIGRATION_METADATA_KEY = "record_selection_artifact_migration"
_SELECTION_ARTIFACT_MIGRATION_DEFER_SIDE_KEY = "side_artifacts_deferred"


def _is_async_trace_enabled(settings: RunConfig) -> bool:
    return settings.async_trace or os.environ.get("DATA_DESIGNER_ASYNC_TRACE", "0") == "1"


def _await_async_scheduler_result(future: concurrent.futures.Future[Any], scheduler: AsyncTaskScheduler) -> None:
    try:
        future.result()
    except KeyboardInterrupt:
        scheduler.request_cancel()
        try:
            future.result()
        except concurrent.futures.CancelledError:
            pass
        except Exception:
            logger.debug("Async scheduler raised while cancelling after KeyboardInterrupt", exc_info=True)
        raise


class _ConfigCompatibility(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    NO_PRIOR_DATASET = "no_prior_dataset"


class _SelectionPostCommitError(DatasetGenerationError):
    """A transient failure after a candidate checkpoint reached durable storage."""


class _SelectionCallbackError(_SelectionPostCommitError):
    """A post-commit callback error that must remain resumable."""


@dataclass(slots=True)
class _SelectionSchemaState:
    """Track whether the durable selection schema is a fallback or materialized anchor."""

    is_written: bool
    is_materialized: bool


@dataclass(frozen=True, slots=True)
class _SelectionArtifactMigrationState:
    """Describe how legacy artifact migration affects an existing publication."""

    previously_completed_publication: bool
    side_artifacts_deferred: bool
    publication_refresh_required: bool


@dataclass(slots=True)
class _SelectionCandidateBatchRuntime:
    """Mutable state shared by one candidate batch and its scheduler callbacks."""

    controller: AcceptanceController
    batch: CandidateBatch
    schema_state: _SelectionSchemaState
    on_batch_complete: Callable[[Path], None] | None
    media_staged: bool
    decision: SelectionDecision | None = None
    buffer_manager: RowGroupBufferManager | None = None
    scheduler: AsyncTaskScheduler | None = None
    checkpoint_committed: bool = False


@dataclass
class _ResumeState:
    num_completed_batches: int
    actual_num_records: int
    buffer_size: int
    target_num_records: int
    original_target_num_records: int
    completed_row_groups: dict[int, int]


@dataclass(frozen=True, slots=True)
class RowGroupResumePlan:
    """Plan describing the row groups left to generate when resuming an async run.

    Attributes:
        total_row_groups: Total row group count for the full target (original + extension).
        remaining_row_groups: lazy plan of ``(rg_id, rg_size)`` for groups not yet on disk, in id order.
    """

    total_row_groups: int
    remaining_row_groups: CompactRowGroupPlan


def build_row_group_resume_plan(
    *,
    original_target: int,
    num_records: int,
    buffer_size: int,
    completed_ids: set[int],
) -> RowGroupResumePlan:
    """Compute the remaining row-group plan for an async resume.

    Original groups are immutable: their per-group sizes were fixed by the first
    run's ``original_target_num_records`` and ``buffer_size``. Any extension
    (``num_records > original_target``) always adds new groups beyond the
    original count — ``ceil(num_records/buffer_size)`` would give the wrong
    total when the original run was non-aligned and the extension fits in the
    last original group's slack.

    Args:
        original_target: Target record count from the first run (immutable).
        num_records: Current target record count (may extend ``original_target``).
        buffer_size: Records per row group.
        completed_ids: Row-group IDs already persisted on disk.

    Returns:
        A ``RowGroupResumePlan`` whose remaining row-group plan preserves full
        original offsets, so the offset for ``rg_id`` is the same whether or not
        earlier groups have completed. This is what lets ordered seed generators
        seek to the correct row when resuming with holes.
    """
    remaining_row_groups = CompactRowGroupPlan.resume(
        original_target=original_target,
        num_records=num_records,
        buffer_size=buffer_size,
        completed_ids=completed_ids,
    )
    return RowGroupResumePlan(
        total_row_groups=remaining_row_groups.total_row_groups,
        remaining_row_groups=remaining_row_groups,
    )


class DatasetBuilder:
    def __init__(
        self,
        data_designer_config: DataDesignerConfig,
        resource_provider: ResourceProvider,
        registry: DataDesignerRegistry | None = None,
    ):
        self._resource_provider = resource_provider
        self._task_traces: list[TaskTrace] = []
        self._registry = registry or DataDesignerRegistry()
        self._graph: ExecutionGraph | None = None
        # Structured signal: set by _build_async if the scheduler hit early shutdown.
        # Reset at the start of each public run path so reused builder instances
        # don't leak state across runs.
        self._early_shutdown: bool = False
        self._partial_row_groups: tuple[int, ...] = ()
        # Number of records actually written by the most recent async run.
        # ``-1`` means "no async run has executed yet" so callers can
        # distinguish "0 records produced" from "never ran".
        self._actual_num_records: int = -1
        # First non-retryable error captured by the scheduler in the most recent
        # async run, if any. Used by the interface to surface the original cause
        # when a run produces 0 records due to deterministic failures.
        self._first_non_retryable_error: Exception | None = None

        self._data_designer_config = compile_data_designer_config(data_designer_config, resource_provider)
        self._column_configs = compile_dataset_builder_column_configs(self._data_designer_config)
        processors = self._initialize_processors(self._data_designer_config.processors or [])
        self._processor_runner = ProcessorRunner(
            processors=processors,
            artifact_storage=resource_provider.artifact_storage,
        )
        self._validate_column_configs()
        self._validate_record_selection_config()

    @property
    def artifact_storage(self) -> ArtifactStorage:
        return self._resource_provider.artifact_storage

    @property
    def data_designer_config(self) -> DataDesignerConfig:
        return self._data_designer_config

    @property
    def processors(self) -> tuple[Processor, ...]:
        return self._processor_runner.processors

    @property
    def task_traces(self) -> list[TaskTrace]:
        return self._task_traces

    @property
    def early_shutdown(self) -> bool:
        """True if the most recent async run terminated via the early-shutdown gate."""
        return self._early_shutdown

    @property
    def partial_row_groups(self) -> tuple[int, ...]:
        """Row group ids that were partially salvaged after early shutdown (most recent run)."""
        return self._partial_row_groups

    @property
    def actual_num_records(self) -> int:
        """Records actually written by the most recent async run (-1 if no run yet)."""
        return self._actual_num_records

    @property
    def first_non_retryable_error(self) -> Exception | None:
        """First non-retryable error captured by the scheduler in the most recent run."""
        return self._first_non_retryable_error

    @functools.cached_property
    def single_column_configs(self) -> list[ColumnConfigT]:
        configs = []
        for config in self._column_configs:
            if isinstance(config, MultiColumnConfig):
                configs.extend(config.columns)
            else:
                configs.append(config)
        return configs

    def build(
        self,
        *,
        num_records: int,
        on_batch_complete: Callable[[Path], None] | None = None,
        save_multimedia_to_disk: bool = True,
        resume: ResumeMode = ResumeMode.NEVER,
    ) -> Path:
        """Build the dataset.

        Args:
            num_records: Number of output records to generate. When record selection is configured,
                this is the exact accepted-row target rather than the number of candidate attempts.
            on_batch_complete: Optional callback function called when each batch completes.
            save_multimedia_to_disk: Whether to save generated multimedia (images, audio, video) to disk.
                If False, multimedia is stored directly in the DataFrame (e.g., images as base64).
                Default is True.
            resume: Controls how interrupted runs are handled.

                - ``ResumeMode.NEVER`` (default): always start a fresh generation run.
                - ``ResumeMode.ALWAYS``: resume from the last completed row group. ``buffer_size``
                  must match the original run. ``num_records`` may be equal to or greater than what
                  was already generated (you can extend the dataset); ``num_records`` less than
                  actual records so far raises ``DatasetGenerationError``. If no checkpoint exists
                  yet (interrupted before the first row group finished), silently restarts from the
                  beginning. Raises if the stored config is incompatible.
                - ``ResumeMode.IF_POSSIBLE``: like ``ALWAYS`` when the current config fingerprint
                  matches the stored config; otherwise starts a fresh run without raising an error.

                In all resume modes, in-flight partial results from the interrupted run are
                discarded before generation continues.

                Record-selection runs additionally require the same ``num_records`` and
                ``buffer_size`` used by the original run. ``IF_POSSIBLE`` clears engine-managed
                selection artifacts and starts fresh when either runtime input changes.

        Returns:
            Path to the generated dataset directory.
        """
        self._reset_run_state()
        self._validate_record_selection_request(num_records)
        requested_resume = resume

        run_readiness_check(
            self.single_column_configs,
            self._resource_provider,
        )

        # For IF_POSSIBLE and ALWAYS: check config compatibility before touching the artifact
        # directory. _check_resume_config_compatibility() must NOT access base_dataset_path
        # (which would cache resolved_dataset_name prematurely). After the decision, sync
        # artifact_storage.resume so that resolved_dataset_name picks up the right semantics
        # on its first real access.
        #
        # Also invalidate any stale resolved_dataset_name cache: ArtifactStorage's Pydantic
        # validator accesses base_dataset_path at construction time, which caches resolved_dataset_name
        # under the original resume mode semantics. Popping it forces a fresh resolution.
        if resume in (ResumeMode.IF_POSSIBLE, ResumeMode.ALWAYS):
            compat = self._check_resume_config_compatibility()
            if resume == ResumeMode.ALWAYS and compat == _ConfigCompatibility.INCOMPATIBLE:
                raise DatasetGenerationError(
                    "🛑 Cannot resume: the current config or dropped-column artifact policy does not match the "
                    "config used in the interrupted run. "
                    "Use resume=ResumeMode.IF_POSSIBLE to start fresh automatically, or "
                    "resume=ResumeMode.NEVER to force a new run."
                )
            if resume == ResumeMode.IF_POSSIBLE:
                if compat != _ConfigCompatibility.COMPATIBLE:
                    if compat == _ConfigCompatibility.INCOMPATIBLE:
                        logger.info(
                            "▶️ Config has changed since the last run — starting a fresh generation (resume=IF_POSSIBLE)."
                        )
                    self._clear_incompatible_selection_artifacts()
                    resume = ResumeMode.NEVER
                    self.artifact_storage.resume = ResumeMode.NEVER
                    self.artifact_storage.__dict__.pop("resolved_dataset_name", None)
                    self.artifact_storage.refresh_media_storage_path()
                else:
                    resume = ResumeMode.ALWAYS
                    self.artifact_storage.resume = ResumeMode.ALWAYS
                    self.artifact_storage.__dict__.pop("resolved_dataset_name", None)

        buffer_size = self._resource_provider.run_config.buffer_size
        if self._data_designer_config.record_selection is not None and resume in (
            ResumeMode.IF_POSSIBLE,
            ResumeMode.ALWAYS,
        ):
            runtime_compatible = self._selection_runtime_inputs_are_compatible(num_records, buffer_size)
            if not runtime_compatible:
                if requested_resume == ResumeMode.ALWAYS:
                    raise DatasetGenerationError(
                        "🛑 Cannot resume record selection: num_records and buffer_size must exactly match "
                        "the interrupted run. Use resume=ResumeMode.IF_POSSIBLE to start fresh automatically, "
                        "or resume=ResumeMode.NEVER to force a new run."
                    )
                logger.info(
                    "▶️ Record-selection runtime inputs changed — starting a fresh generation (resume=IF_POSSIBLE)."
                )
                self._clear_incompatible_selection_artifacts()
                resume = ResumeMode.NEVER
                self.artifact_storage.resume = ResumeMode.NEVER
                self.artifact_storage.__dict__.pop("resolved_dataset_name", None)
                self.artifact_storage.refresh_media_storage_path()

        self._set_metadata_defaults()

        if (
            self._data_designer_config.record_selection is None
            and self._post_generation_processed_resume_result(resume, num_records) is not None
        ):
            return self.artifact_storage.final_dataset_path

        self._write_builder_config()

        # Set media storage mode based on parameters
        if self._has_image_columns():
            mode = StorageMode.DISK if save_multimedia_to_disk else StorageMode.DATAFRAME
            self.artifact_storage.set_media_storage_mode(mode)

        generators, self._graph = self._initialize_generators_and_graph()
        start_time = time.perf_counter()

        if (
            self._data_designer_config.record_selection is None
            and resume == ResumeMode.ALWAYS
            and not self.artifact_storage.metadata_file_path.exists()
        ):
            # No metadata.json means the previous run was interrupted before any
            # row group completed. Nothing to resume, so discard leftover
            # partial results and start fresh.
            logger.info(
                "▶️ No metadata.json found — the previous run was interrupted before any row group "
                "completed. Starting generation from the beginning."
            )
            self.artifact_storage.clear_partial_results()
            resume = ResumeMode.NEVER
            self.artifact_storage.resume = ResumeMode.NEVER

        if self._data_designer_config.record_selection is not None:
            for generator in generators:
                generator.log_pre_generation()
            try:
                self._build_with_record_selection(
                    generators,
                    target_num_records=num_records,
                    buffer_size=buffer_size,
                    on_batch_complete=on_batch_complete,
                    resume=resume,
                )
            finally:
                self._resource_provider.model_registry.log_model_usage(time.perf_counter() - start_time)
            return self.artifact_storage.final_dataset_path

        self._build_async(generators, num_records, buffer_size, on_batch_complete, resume=resume)

        # After-generation processors run unconditionally on the on-disk dataset
        # (not gated on ``generated``). When resume sees every row group already
        # on disk, ``_build_*`` returns ``False`` without writing the "started"
        # marker; gating after-generation on ``generated`` would then leave a
        # complete dataset with after-generation processors permanently unrun if
        # the original process crashed in the narrow window between the final
        # parquet write and the "started" marker write.
        #
        # The short-circuits inside ``_post_generation_processed_resume_result``
        # cover the already-processed cases (``post_generation_processed`` /
        # ``post_generation_state == "complete"`` → return early;
        # ``post_generation_state == "started"`` → raise as ambiguous), so by
        # the time we reach this point after-generation has demonstrably not
        # been applied to the dataset on disk.
        has_after_generation_processors = self._processor_runner.has_processors_for(ProcessorStage.AFTER_GENERATION)
        if has_after_generation_processors:
            self.artifact_storage.update_metadata(
                {"post_generation_state": "started", "post_generation_processed": False}
            )
            self._processor_runner.run_after_generation(buffer_size)
            self.artifact_storage.update_metadata(
                {"post_generation_state": "complete", "post_generation_processed": True}
            )
        self._resource_provider.model_registry.log_model_usage(time.perf_counter() - start_time)

        return self.artifact_storage.final_dataset_path

    def _set_metadata_defaults(self) -> None:
        """Attach config identity fields to every metadata write in this build."""
        self.artifact_storage.set_metadata_defaults(
            {
                **self._data_designer_config.fingerprint(),
                PRESERVE_DROPPED_COLUMNS_METADATA_KEY: self._resource_provider.run_config.preserve_dropped_columns,
            }
        )

    def _post_generation_processed_resume_result(self, resume: ResumeMode, num_records: int) -> Path | None:
        """Decide whether to short-circuit resume based on after-generation processor state.

        Returns:
            * ``None`` if normal resume should proceed (no metadata, not in resume mode, or
              after-generation processors have not run yet).
            * ``final_dataset_path`` for the no-op case (dataset is already complete and
              post-processed and the caller asked for the same target).

        Raises:
            DatasetGenerationError: If after-generation processing started but did not
                complete (parquet files may already be rewritten), if the terminal
                metadata is missing required fields (``target_num_records``), or if the
                caller asked for a different target than the one this terminal dataset
                was built for.
        """
        if resume != ResumeMode.ALWAYS or not self.artifact_storage.metadata_file_path.exists():
            return None

        try:
            metadata = self.artifact_storage.read_metadata()
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        post_generation_state = metadata.get("post_generation_state")
        if post_generation_state == "started":
            raise DatasetGenerationError(
                "🛑 Cannot resume: process_after_generation started but did not complete for this dataset. "
                "The final parquet files may already have been rewritten, so resuming would risk mixing pre- "
                "and post-processor records. Use resume=ResumeMode.NEVER to start a new generation run."
            )

        if not metadata.get("post_generation_processed", False) and post_generation_state != "complete":
            return None

        prior_target = metadata.get("target_num_records")
        if prior_target is None:
            raise DatasetGenerationError(
                "🛑 Cannot resume: metadata.json is missing required field 'target_num_records'. "
                "Start a fresh run with resume=ResumeMode.NEVER, or restore a valid metadata.json."
            )
        if num_records == prior_target:
            logger.warning("▶️ Dataset is already complete and post-processed; nothing to resume.")
            return self.artifact_storage.final_dataset_path

        if num_records < prior_target:
            raise DatasetGenerationError(
                f"🛑 Cannot resume: num_records={num_records} is less than the {prior_target} records "
                "already generated and post-processed for this dataset. Use num_records >= "
                f"{prior_target}, or resume=ResumeMode.NEVER to start a new generation run."
            )

        raise DatasetGenerationError(
            "🛑 Cannot resume: process_after_generation has already been applied to this dataset "
            f"(original target {prior_target}, requested {num_records}). Extending would mix pre- and "
            "post-processor records. Use resume=ResumeMode.NEVER to start a new generation run."
        )

    def _load_resume_state(self, num_records: int, buffer_size: int) -> _ResumeState:
        """Read and validate resume state from metadata + the filesystem.

        ``metadata.json`` is the source of truth for the run *configuration*
        (``buffer_size``, ``target_num_records``, ``original_target_num_records``,
        config fingerprint). The filesystem (``parquet-files/batch_*.parquet``) is
        the source of truth for run *progress* (``num_completed_batches``,
        ``actual_num_records``). Splitting the two sources is what lets resume
        survive a crash between writing a batch and updating metadata: the
        filesystem reflects the durable state even when metadata lags by a step.

        ``num_records`` must be >= the number of records already on disk (you may
        extend a dataset, but cannot shrink it below what has been written).
        ``buffer_size`` must match the original run because it determines row-group
        boundaries. Resume tolerates holes from out-of-order row-group completion.

        Raises:
            DatasetGenerationError: If metadata is missing or incompatible, or if
                the filesystem state is inconsistent.
        """
        try:
            metadata = self.artifact_storage.read_metadata()
        except FileNotFoundError as exc:
            raise DatasetGenerationError(
                "🛑 Cannot resume: metadata.json not found in the existing dataset directory. "
                "Run without resume=ResumeMode.ALWAYS to start a new generation."
            ) from exc
        except json.JSONDecodeError as exc:
            raise DatasetGenerationError(
                "🛑 Cannot resume: metadata.json is corrupt or partially written. "
                "Start a fresh run with resume=ResumeMode.NEVER, or restore a valid metadata.json."
            ) from exc

        num_completed_batches, actual_num_records, completed_row_groups = self._recover_progress_from_disk(
            allow_holes=True,
        )

        if num_records < actual_num_records:
            raise DatasetGenerationError(
                f"🛑 Cannot resume: num_records={num_records} is less than the {actual_num_records} "
                "records already generated. Use num_records >= actual_num_records, "
                "or start a new run without resume=ResumeMode.ALWAYS."
            )

        target_num_records = metadata.get("target_num_records")
        if target_num_records is None:
            raise DatasetGenerationError(
                "🛑 Cannot resume: metadata.json is missing required field 'target_num_records'. "
                "Start a fresh run with resume=ResumeMode.NEVER, or restore a valid metadata.json."
            )
        if num_records < target_num_records:
            raise DatasetGenerationError(
                f"🛑 Cannot resume: num_records={num_records} is less than the original target "
                f"({target_num_records}). To resume, use num_records >= {target_num_records} "
                "(you may extend the dataset beyond the original target). "
                "Use resume=ResumeMode.NEVER to start a new run."
            )
        original_target_num_records = metadata.get("original_target_num_records", target_num_records)
        if original_target_num_records > target_num_records:
            raise DatasetGenerationError(
                "🛑 Cannot resume: metadata.json has original_target_num_records="
                f"{original_target_num_records}, which is greater than target_num_records={target_num_records}. "
                "Start a fresh run with resume=ResumeMode.NEVER, or restore a valid metadata.json."
            )

        meta_buffer_size = metadata.get("buffer_size")
        if meta_buffer_size != buffer_size:
            raise DatasetGenerationError(
                f"🛑 Cannot resume: buffer_size={buffer_size} does not match the original run's "
                f"buffer_size={meta_buffer_size}. Use the same buffer_size as the interrupted run, "
                "or start a new run without resume=ResumeMode.ALWAYS."
            )

        if not self._dropped_column_artifact_policy_matches(metadata):
            raise DatasetGenerationError(
                "🛑 Cannot resume: preserve_dropped_columns="
                f"{self._resource_provider.run_config.preserve_dropped_columns} does not match the original "
                "run's dropped-column artifact policy. Start a fresh run with resume=ResumeMode.NEVER, or "
                "use resume=ResumeMode.IF_POSSIBLE to start fresh automatically when the policy differs."
            )

        return _ResumeState(
            num_completed_batches=num_completed_batches,
            actual_num_records=actual_num_records,
            buffer_size=buffer_size,
            target_num_records=target_num_records,
            original_target_num_records=original_target_num_records,
            completed_row_groups=completed_row_groups,
        )

    def build_preview(self, *, num_records: int) -> pd.DataFrame:
        self._reset_run_state()
        if self._data_designer_config.record_selection is not None:
            raise DatasetGenerationError(
                "🛑 preview() does not support record selection because preview has no accepted-row retry or "
                "checkpoint contract. Use create(), or preview the same config with record selection disabled."
            )
        run_readiness_check(
            self.single_column_configs,
            self._resource_provider,
        )

        # Set media storage to DATAFRAME mode for preview - base64 stored directly in DataFrame
        if self._has_image_columns():
            self.artifact_storage.set_media_storage_mode(StorageMode.DATAFRAME)

        generators, self._graph = self._initialize_generators_and_graph()
        start_time = time.perf_counter()

        dataset = self._build_async_preview(generators, num_records)

        self._resource_provider.model_registry.log_model_usage(time.perf_counter() - start_time)

        return dataset

    def _reset_run_state(self) -> None:
        """Clear per-run signals so reused builder instances don't leak state across runs."""
        self._early_shutdown = False
        self._partial_row_groups = ()
        self._actual_num_records = -1
        self._first_non_retryable_error = None
        self._task_traces = []

    def _build_async_preview(self, generators: list[ColumnGenerator], num_records: int) -> pd.DataFrame:
        """Async preview path - single row group, no disk writes, returns in-memory DataFrame."""
        logger.info("⚡ Using async task-queue preview")

        settings = self._resource_provider.run_config
        trace_enabled = _is_async_trace_enabled(settings)

        scheduler, buffer_manager = self._prepare_async_run(
            generators,
            num_records,
            buffer_size=num_records,
            run_post_batch_in_scheduler=False,
            trace=trace_enabled,
        )

        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(scheduler.run(), loop)
        try:
            _await_async_scheduler_result(future, scheduler)
        finally:
            self._task_traces = scheduler.traces
            self._early_shutdown = scheduler.early_shutdown
            self._partial_row_groups = scheduler.partial_row_groups
            self._actual_num_records = buffer_manager.actual_num_records
            self._first_non_retryable_error = scheduler.first_non_retryable_error

        if not buffer_manager.has_row_group(0):
            return lazy.pd.DataFrame()

        dataset = buffer_manager.get_dataframe(0)
        buffer_manager.free_row_group(0)
        return dataset

    def _find_completed_row_groups(self) -> dict[int, int]:
        """Scan final parquet files and return row-group IDs with persisted row counts.

        Returns:
            Mapping of row-group ID (batch number) to actual parquet row count.
        """
        final_path = self.artifact_storage.final_dataset_path
        if not final_path.exists():
            return {}
        row_groups: dict[int, int] = {}
        for p in final_path.glob("batch_*.parquet"):
            try:
                row_group_id = int(p.stem.split("_", 1)[1])
                row_groups[row_group_id] = lazy.pq.read_metadata(p).num_rows
            except (ValueError, IndexError, OSError):
                logger.warning("⚠️ Ignoring unreadable row-group file during resume: %s", p)
                continue
        return row_groups

    def _recover_progress_from_disk(self, *, allow_holes: bool) -> tuple[int, int, dict[int, int]]:
        """Derive resume progress counters from completed parquet files on disk.

        The filesystem is the source of truth for ``num_completed_batches`` and
        ``actual_num_records`` because a crash between
        ``move_partial_result_to_final_file_path`` and the metadata write that follows
        can leave parquet files on disk while metadata still reports stale counters.
        Args:
            allow_holes: Whether to tolerate non-contiguous row-group IDs. Async
                scheduling may complete row groups out of order.

        Returns:
            ``(num_completed_batches, actual_num_records, completed_row_groups)``.
        """
        completed_row_groups = self._find_completed_row_groups()
        if completed_row_groups and not allow_holes:
            ids = sorted(completed_row_groups)
            if ids != list(range(len(ids))):
                raise DatasetGenerationError(
                    "🛑 Cannot resume: completed batch files on disk are non-contiguous "
                    f"(found row group IDs {ids}). The dataset directory may have been "
                    "written by an incompatible engine or modified externally. Use "
                    "resume=ResumeMode.NEVER to start a new run."
                )
        return len(completed_row_groups), sum(completed_row_groups.values()), completed_row_groups

    def _check_resume_config_compatibility(self) -> _ConfigCompatibility:
        """Compare the current config fingerprint against stored resume identity.

        Returns:
            NO_PRIOR_DATASET  — directory absent or empty (no prior run to resume from).
            COMPATIBLE        — fingerprints match.
            INCOMPATIBLE      — fingerprints differ; continuing would mix records from two configs.

        Uses artifact_path / dataset_name directly — NOT base_dataset_path — to avoid
        prematurely triggering the resolved_dataset_name cached_property before the
        caller has had a chance to decide whether to resume or start fresh.
        """
        dataset_dir = Path(self.artifact_storage.artifact_path) / self.artifact_storage.dataset_name
        if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
            return _ConfigCompatibility.NO_PRIOR_DATASET
        current_fp = self._data_designer_config.fingerprint()
        metadata_path = dataset_dir / METADATA_FILENAME
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError as exc:
                raise DatasetGenerationError(
                    "🛑 Cannot resume: metadata.json is corrupt or partially written. "
                    "Start a fresh run with resume=ResumeMode.NEVER, or restore a valid metadata.json."
                ) from exc
            except OSError:
                logger.warning(
                    "⚠️ Could not read metadata at %s for config compatibility check — treating as incompatible.",
                    metadata_path,
                )
                return _ConfigCompatibility.INCOMPATIBLE

            if not self._dropped_column_artifact_policy_matches(metadata):
                return _ConfigCompatibility.INCOMPATIBLE

            stored_hash = metadata.get("config_hash")
            stored_version = metadata.get("config_hash_version")
            if stored_hash is not None:
                if stored_version != current_fp["config_hash_version"]:
                    logger.warning(
                        "⚠️ Stored config_hash_version=%s does not match current version=%s.",
                        stored_version,
                        current_fp["config_hash_version"],
                    )
                    return _ConfigCompatibility.INCOMPATIBLE
                return (
                    _ConfigCompatibility.COMPATIBLE
                    if stored_hash == current_fp["config_hash"]
                    else _ConfigCompatibility.INCOMPATIBLE
                )

        if not metadata_path.exists():
            return _ConfigCompatibility.COMPATIBLE

        config_path = dataset_dir / SDG_CONFIG_FILENAME
        if not config_path.exists():
            logger.warning(
                "⚠️ No builder_config.json found in %s — skipping config compatibility check on resume.",
                dataset_dir,
            )
            return _ConfigCompatibility.COMPATIBLE
        try:
            stored_data = json.loads(config_path.read_text())
            stored_config = BuilderConfig.model_validate(stored_data)
            stored_fp = stored_config.data_designer.fingerprint()["config_hash"]
            return (
                _ConfigCompatibility.COMPATIBLE
                if current_fp["config_hash"] == stored_fp
                else _ConfigCompatibility.INCOMPATIBLE
            )
        except (OSError, json.JSONDecodeError, ValidationError):
            logger.warning(
                "⚠️ Could not read stored config at %s for compatibility check — treating as incompatible.",
                config_path,
            )
            return _ConfigCompatibility.INCOMPATIBLE

    def _dropped_column_artifact_policy_matches(self, metadata: dict[str, Any]) -> bool:
        """Return whether stored dropped-column artifact behavior matches this run.

        Metadata written before this RunConfig option existed implicitly used the
        historical behavior, which preserved dropped-column artifacts.
        """
        stored = metadata.get(PRESERVE_DROPPED_COLUMNS_METADATA_KEY, True)
        current = self._resource_provider.run_config.preserve_dropped_columns
        if stored != current:
            logger.warning(
                "⚠️ preserve_dropped_columns changed from %s to %s; treating the existing dataset as "
                "incompatible for resume because dropped-column parquet artifacts would be inconsistent.",
                stored,
                current,
            )
            return False
        return True

    def _selection_runtime_inputs_are_compatible(self, target_num_records: int, buffer_size: int) -> bool:
        """Check selection-only resume inputs without resolving a new dataset path."""
        dataset_dir = Path(self.artifact_storage.artifact_path) / self.artifact_storage.dataset_name
        metadata_path = dataset_dir / METADATA_FILENAME
        checkpoints_path = dataset_dir / "selection-checkpoints"
        if not metadata_path.exists():
            return not checkpoints_path.exists() or not any(checkpoints_path.glob("batch_*.json"))
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        selection = metadata.get("record_selection")
        if not isinstance(selection, dict):
            return False
        return (
            metadata.get("target_num_records") == target_num_records and selection.get("run_buffer_size") == buffer_size
        )

    def _clear_incompatible_selection_artifacts(self) -> None:
        """Clear engine-managed artifacts when IF_POSSIBLE restarts an existing selection run."""
        if self._data_designer_config.record_selection is None:
            return
        dataset_dir = Path(self.artifact_storage.artifact_path) / self.artifact_storage.dataset_name
        metadata_path = dataset_dir / METADATA_FILENAME
        checkpoints_path = dataset_dir / "selection-checkpoints"
        is_selection_run = checkpoints_path.exists()
        if metadata_path.exists():
            try:
                is_selection_run = is_selection_run or isinstance(
                    json.loads(metadata_path.read_text(encoding="utf-8")).get("record_selection"),
                    dict,
                )
            except (OSError, json.JSONDecodeError):
                is_selection_run = is_selection_run or checkpoints_path.exists()
        if not is_selection_run:
            return

        managed_directories = {
            self.artifact_storage.final_dataset_folder_name,
            self.artifact_storage.partial_results_folder_name,
            self.artifact_storage.dropped_columns_folder_name,
            self.artifact_storage.processors_outputs_folder_name,
            "images",
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
            SELECTION_ARTIFACT_MIGRATION_FILENAME,
            "scheduler_events.jsonl",
        ):
            (dataset_dir / name).unlink(missing_ok=True)

    def _build_with_record_selection(
        self,
        generators: list[ColumnGenerator],
        *,
        target_num_records: int,
        buffer_size: int,
        on_batch_complete: Callable[[Path], None] | None,
        resume: ResumeMode,
    ) -> None:
        """Generate immutable candidate batches until the accepted-row target or cap is reached."""
        config = self._data_designer_config.record_selection
        if config is None:
            raise RuntimeError("Record-selection build path requires a record-selection config.")

        self.artifact_storage.configure_selection_batch_file_width(
            max_candidate_records=config.max_candidate_records,
            candidate_batch_size=min(buffer_size, target_num_records),
        )
        migration = self._prepare_selection_artifact_migration(resume=resume)
        controller, schema_state, stored_selection = self._restore_selection_run(
            config=config,
            target_num_records=target_num_records,
            buffer_size=buffer_size,
            resume=resume,
        )
        if self._reconcile_completed_selection_publication(
            controller,
            migration=migration,
            stored_selection=stored_selection,
        ):
            return

        self._write_selection_metadata(controller)
        self._run_selection_candidate_batches(
            generators,
            controller=controller,
            schema_state=schema_state,
            on_batch_complete=on_batch_complete,
        )
        self._finalize_selection_build(
            controller,
            target_num_records=target_num_records,
            buffer_size=buffer_size,
        )

    def _prepare_selection_artifact_migration(
        self,
        *,
        resume: ResumeMode,
    ) -> _SelectionArtifactMigrationState:
        """Normalize legacy candidate artifact names without exposing a mixed publication."""
        migration_metadata: dict[str, Any] = {}
        if resume == ResumeMode.ALWAYS and self.artifact_storage.metadata_file_path.is_file():
            migration_metadata = self.artifact_storage.read_metadata()
        persisted_migration = migration_metadata.get(_SELECTION_ARTIFACT_MIGRATION_METADATA_KEY)
        persisted_completed_publication = (
            isinstance(persisted_migration, dict) and persisted_migration.get("previously_complete") is True
        )
        migrated_completed_publication = (
            persisted_completed_publication or migration_metadata.get("post_generation_state") == "complete"
        )
        has_after_generation_processors = self._processor_runner.has_processors_for(ProcessorStage.AFTER_GENERATION)
        if (
            isinstance(persisted_migration, dict)
            and _SELECTION_ARTIFACT_MIGRATION_DEFER_SIDE_KEY in persisted_migration
        ):
            deferred_value = persisted_migration[_SELECTION_ARTIFACT_MIGRATION_DEFER_SIDE_KEY]
            if not isinstance(deferred_value, bool):
                raise DatasetGenerationError("🛑 Record-selection artifact migration metadata is invalid.")
            side_artifacts_deferred = deferred_value
        else:
            side_artifacts_deferred = migrated_completed_publication and has_after_generation_processors
        include_side_artifacts = not side_artifacts_deferred
        migration_needed = (
            resume == ResumeMode.ALWAYS
            and self.artifact_storage.requires_selection_candidate_artifact_migration(
                include_side_artifacts=include_side_artifacts,
            )
        )
        publication_refresh_required = persisted_completed_publication
        if migration_needed:
            if self.artifact_storage.metadata_file_path.is_file():
                # Invalidate the old publication before the first rename so a
                # concurrent uploader cannot accept a mixed-width snapshot.
                migration_updates: dict[str, Any] = {
                    "post_generation_state": "started",
                    "post_generation_processed": False,
                }
                if migrated_completed_publication:
                    migration_updates[_SELECTION_ARTIFACT_MIGRATION_METADATA_KEY] = {
                        "previously_complete": True,
                        _SELECTION_ARTIFACT_MIGRATION_DEFER_SIDE_KEY: side_artifacts_deferred,
                    }
                self.artifact_storage.update_metadata(migration_updates)
            publication_refresh_required = (
                self.artifact_storage.normalize_selection_candidate_artifact_width(
                    include_side_artifacts=include_side_artifacts,
                )
                or publication_refresh_required
            )

        return _SelectionArtifactMigrationState(
            previously_completed_publication=migrated_completed_publication,
            side_artifacts_deferred=side_artifacts_deferred,
            publication_refresh_required=publication_refresh_required,
        )

    def _restore_selection_run(
        self,
        *,
        config: RecordSelectionConfig,
        target_num_records: int,
        buffer_size: int,
        resume: ResumeMode,
    ) -> tuple[AcceptanceController, _SelectionSchemaState, dict[str, Any] | None]:
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
        if self.artifact_storage.metadata_file_path.exists():
            try:
                existing_metadata = self.artifact_storage.read_metadata()
            except (OSError, json.JSONDecodeError):
                existing_metadata = {}
        stored_selection_value = existing_metadata.get("record_selection")
        stored_selection = stored_selection_value if isinstance(stored_selection_value, dict) else None
        terminal_error: object | None = None
        stored_batches = 0
        if stored_selection is not None:
            stored_batches = stored_selection.get("candidate_batches_completed", 0)
            if isinstance(stored_batches, int) and stored_batches > controller.candidate_batches_completed:
                raise DatasetGenerationError(
                    "🛑 Cannot resume record selection: metadata reports more completed candidate batches "
                    "than the durable checkpoint directory. Restore the missing checkpoints or start fresh."
                )
            terminal_error = stored_selection.get(_SELECTION_TERMINAL_ERROR_KEY)

        # Checkpoints are the source of truth for committed candidate work. A
        # marker-level failure survives the crash window before metadata catches up.
        # When checkpoints are ahead, the newest marker's explicit lack of a
        # terminal error must also supersede stale terminal metadata.
        if controller.terminal_error is not None:
            terminal_error = controller.terminal_error
        elif isinstance(stored_batches, int) and controller.candidate_batches_completed > stored_batches:
            terminal_error = None

        if terminal_error is None:
            terminal_error = self._derive_nonretryable_selection_terminal_error(controller)

        self.artifact_storage.clear_selection_transient_artifacts()
        self.artifact_storage.clean_uncommitted_selection_batch(controller.candidate_batches_completed)
        self._validate_selection_partitions(controller.markers)
        schema_state = self._hydrate_selection_schema_state(controller)
        self._actual_num_records = controller.accepted_records

        if resume == ResumeMode.ALWAYS:
            self._raise_unrecoverable_selection_terminal_error(controller, terminal_error)

        return controller, schema_state, stored_selection

    def _reconcile_completed_selection_publication(
        self,
        controller: AcceptanceController,
        *,
        migration: _SelectionArtifactMigrationState,
        stored_selection: dict[str, Any] | None,
    ) -> bool:
        """Reuse a valid completed publication or invalidate it before rebuilding."""
        if not migration.previously_completed_publication:
            return False

        terminal_selection = controller.has_reached_target or controller.is_exhausted
        if terminal_selection:
            published_files = list(self.artifact_storage.final_dataset_path.glob("batch_*.parquet"))
            try:
                final_count: int | None = sum(lazy.pq.read_metadata(path).num_rows for path in published_files)
            except (OSError, lazy.pa.ArrowInvalid):
                final_count = None
            publication_reusable = bool(published_files) and final_count == controller.accepted_records
            stored_publication_id = stored_selection.get("publication_id") if stored_selection is not None else None
            if publication_reusable and migration.publication_refresh_required:
                self._write_selection_metadata(
                    controller,
                    post_generation_state="complete",
                    publication_id=uuid.uuid4().hex,
                )
                self._clear_selection_artifact_migration_metadata()
                logger.warning(
                    "▶️ Migrated legacy record-selection artifact names and refreshed the completed publication."
                )
                return True
            if (
                publication_reusable
                and not migration.publication_refresh_required
                and isinstance(stored_publication_id, str)
                and stored_publication_id.strip()
            ):
                logger.warning("▶️ Record-selection dataset is already complete; nothing to resume.")
                return True
            if publication_reusable:
                logger.warning(
                    "⚠️ Completed record-selection publication has no publication ID; rebuilding the "
                    "published view to migrate it to the current atomic-upload protocol."
                )
            else:
                logger.warning(
                    "⚠️ Completed record-selection publication contains %s rows, but committed markers contain %d; "
                    "rebuilding the published view from immutable accepted partitions.",
                    final_count if final_count is not None else "unreadable",
                    controller.accepted_records,
                )

        self._invalidate_completed_selection_publication(migration)
        return False

    def _invalidate_completed_selection_publication(
        self,
        migration: _SelectionArtifactMigrationState,
    ) -> None:
        """Move a completed publication back to an explicitly incomplete rebuild state."""
        migration_updates: dict[str, Any] = {
            "post_generation_state": "started",
            "post_generation_processed": False,
        }
        if migration.side_artifacts_deferred:
            migration_updates[_SELECTION_ARTIFACT_MIGRATION_METADATA_KEY] = {
                "previously_complete": True,
                _SELECTION_ARTIFACT_MIGRATION_DEFER_SIDE_KEY: False,
            }
        self.artifact_storage.update_metadata(migration_updates)
        if migration.side_artifacts_deferred:
            self.artifact_storage.normalize_selection_candidate_artifact_width(
                include_side_artifacts=True,
            )
        # The old completed publication is not reusable. Drop migration
        # provenance before rebuilding so a crash during processors cannot
        # be mistaken for the untouched pre-migration publication.
        self._clear_selection_artifact_migration_metadata()

    def _run_selection_candidate_batches(
        self,
        generators: list[ColumnGenerator],
        *,
        controller: AcceptanceController,
        schema_state: _SelectionSchemaState,
        on_batch_complete: Callable[[Path], None] | None,
    ) -> None:
        """Generate and durably commit candidate batches until selection terminates."""
        while not controller.has_reached_target and controller.has_candidate_budget:
            batch = controller.next_candidate_batch()
            try:
                self._run_candidate_batch(
                    generators,
                    controller=controller,
                    batch=batch,
                    on_batch_complete=on_batch_complete,
                    schema_state=schema_state,
                )
            except _SelectionPostCommitError:
                raise
            except Exception as exc:
                self._persist_selection_generation_error(controller, batch=batch, error=exc)
                raise
            self._raise_if_selection_stopped_early(controller, batch=batch)

    def _persist_selection_generation_error(
        self,
        controller: AcceptanceController,
        *,
        batch: CandidateBatch,
        error: Exception,
    ) -> None:
        """Attach a fatal post-checkpoint generation error to the durable marker."""
        if not self.artifact_storage.selection_checkpoint_path(batch.candidate_batch_id).is_file():
            return
        terminal_error = {
            "kind": _SELECTION_GENERATION_ERROR_KIND,
            "message": str(error),
        }
        marker = controller.replace_last_marker_terminal_error(terminal_error)
        self.artifact_storage.write_selection_checkpoint(batch.candidate_batch_id, marker.to_dict())
        try:
            self._write_selection_metadata(controller, terminal_error=terminal_error)
        except Exception:
            logger.warning(
                "⚠️ Failed to mirror a durable record-selection terminal error into metadata.",
                exc_info=True,
            )

    def _raise_if_selection_stopped_early(
        self,
        controller: AcceptanceController,
        *,
        batch: CandidateBatch,
    ) -> None:
        """Persist and raise the structured error for scheduler early shutdown."""
        if not self._early_shutdown or controller.has_reached_target:
            return
        error = RecordSelectionEarlyShutdownError(
            candidate_budget_remaining=controller.has_candidate_budget,
        )
        terminal_error = {
            "kind": _SELECTION_EARLY_SHUTDOWN_KIND,
            "message": str(error),
        }
        if controller.terminal_error is None and controller.candidate_batches_completed > batch.candidate_batch_id:
            marker = controller.replace_last_marker_terminal_error(terminal_error)
            self.artifact_storage.write_selection_checkpoint(marker.candidate_batch_id, marker.to_dict())
        self._write_selection_metadata(
            controller,
            terminal_error=terminal_error,
        )
        raise error

    def _finalize_selection_build(
        self,
        controller: AcceptanceController,
        *,
        target_num_records: int,
        buffer_size: int,
    ) -> None:
        """Apply terminal policy and publish the immutable accepted partitions."""
        self._actual_num_records = controller.accepted_records
        if controller.is_exhausted and controller.config.on_exhausted == RecordSelectionExhaustion.RAISE:
            self._write_selection_metadata(controller)
            raise RecordSelectionExhaustedError(
                target_records=target_num_records,
                accepted_records=controller.accepted_records,
                candidate_records=controller.candidate_records,
                max_candidate_records=controller.config.max_candidate_records,
            )

        terminal_error = self._derive_nonretryable_selection_terminal_error(controller)
        if terminal_error is not None:
            marker = controller.replace_last_marker_terminal_error(terminal_error)
            self.artifact_storage.write_selection_checkpoint(marker.candidate_batch_id, marker.to_dict())
            self._write_selection_metadata(controller, terminal_error=terminal_error)
            raise DatasetGenerationError(terminal_error["message"])

        self._publish_selection_result(controller, buffer_size)

    def _run_candidate_batch(
        self,
        generators: list[ColumnGenerator],
        *,
        controller: AcceptanceController,
        batch: CandidateBatch,
        on_batch_complete: Callable[[Path], None] | None,
        schema_state: _SelectionSchemaState | None = None,
    ) -> None:
        """Run and durably commit one candidate batch."""
        settings = self._resource_provider.run_config
        trace_enabled = _is_async_trace_enabled(settings)
        if schema_state is None:
            schema_state = self._hydrate_selection_schema_state(controller)
        media_staged = self._has_image_columns() and self.artifact_storage.media_storage.mode == StorageMode.DISK
        runtime = _SelectionCandidateBatchRuntime(
            controller=controller,
            batch=batch,
            schema_state=schema_state,
            on_batch_complete=on_batch_complete,
            media_staged=media_staged,
        )

        if media_staged:
            self.artifact_storage.begin_selection_media_batch(batch.candidate_batch_id)

        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()
        group_id = uuid.uuid4().hex
        event_sink_context = (
            JsonlSchedulerEventSink(self.artifact_storage.base_dataset_path / "scheduler_events.jsonl")
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
            usage_deltas = self._resource_provider.model_registry.get_usage_deltas(pre_batch_snapshot)
            self._emit_batch_inference_events("selection_batch", usage_deltas, group_id)
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
            selected = self.artifact_storage.promote_selection_media(
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
                    raise _SelectionCallbackError(str(exc)) from exc
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
        self._update_selection_schema(runtime.schema_state, dataframe=dataframe)

        partition_path: Path | None = None
        partition_relative: str | None = None
        if len(dataframe) > 0:
            partition_path = self.artifact_storage.write_selection_partition(
                runtime.batch.candidate_batch_id,
                dataframe,
            )
            partition_relative = str(partition_path.relative_to(self.artifact_storage.base_dataset_path))

        non_retryable_error, terminal_error = self._selection_candidate_diagnostics(runtime, decision=decision)
        marker = runtime.controller.record_checkpoint(
            batch=runtime.batch,
            decision=decision,
            accepted_partition=partition_relative,
            schema_materialized=runtime.schema_state.is_materialized,
            non_retryable_error=non_retryable_error,
            terminal_error=terminal_error,
        )
        self.artifact_storage.write_selection_checkpoint(
            runtime.batch.candidate_batch_id,
            marker.to_dict(),
        )
        runtime.checkpoint_committed = True
        return partition_path

    def _update_selection_schema(
        self,
        schema_state: _SelectionSchemaState,
        *,
        dataframe: pd.DataFrame,
    ) -> None:
        """Create or upgrade the durable schema anchor for accepted partitions."""
        if len(dataframe.columns) > 0 and not schema_state.is_materialized:
            # A later materialized batch upgrades any name-only fallback
            # written for an earlier completely failed batch.
            self.artifact_storage.write_selection_schema(dataframe)
            schema_state.is_written = True
            schema_state.is_materialized = True
        elif not schema_state.is_written:
            # Preserve an existing materialized schema when a later batch
            # completely fails and therefore has no columns of its own.
            self.artifact_storage.write_selection_schema(self._derive_empty_selection_schema())
            schema_state.is_written = True

    @staticmethod
    def _selection_candidate_diagnostics(
        runtime: _SelectionCandidateBatchRuntime,
        *,
        decision: SelectionDecision,
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        """Capture scheduler diagnostics in the same transaction as the candidate marker."""
        scheduler = runtime.scheduler
        non_retryable = scheduler.first_non_retryable_error if scheduler is not None else None
        non_retryable_error = (
            {
                "type": type(non_retryable).__name__,
                "message": str(non_retryable),
            }
            if non_retryable is not None
            else None
        )
        terminal_error: dict[str, str] | None = None
        if (
            scheduler is not None
            and scheduler.early_shutdown
            and runtime.controller.accepted_records + decision.accepted_records < runtime.controller.target_records
        ):
            terminal_error = {
                "kind": _SELECTION_EARLY_SHUTDOWN_KIND,
                "message": str(
                    RecordSelectionEarlyShutdownError(
                        candidate_budget_remaining=(
                            runtime.batch.start_offset + decision.candidate_records
                            < runtime.controller.config.max_candidate_records
                        ),
                    )
                ),
            }
        return non_retryable_error, terminal_error

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
            self._actual_num_records = controller.accepted_records
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
        scheduler, buffer_manager = self._prepare_async_run(
            generators,
            runtime.batch.size,
            runtime.batch.size,
            on_finalize_row_group=functools.partial(self._finalize_selection_candidate_row_group, runtime),
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
        _await_async_scheduler_result(future, scheduler)

    def _finish_selection_candidate_batch(
        self,
        runtime: _SelectionCandidateBatchRuntime,
        *,
        run_error: BaseException | None,
    ) -> None:
        """Clean staging and merge scheduler state after a candidate attempt."""
        if runtime.media_staged:
            try:
                self.artifact_storage.finish_selection_media_batch(runtime.batch.candidate_batch_id)
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
        self._task_traces.extend(scheduler.traces)
        self._early_shutdown = self._early_shutdown or scheduler.early_shutdown
        self._partial_row_groups = tuple(sorted(set(self._partial_row_groups).union(scheduler.partial_row_groups)))
        if self._first_non_retryable_error is None:
            self._first_non_retryable_error = scheduler.first_non_retryable_error

    def _load_selection_markers(self) -> tuple[SelectionBatchMarker, ...]:
        try:
            raw_markers = self.artifact_storage.read_selection_checkpoints()
            return tuple(SelectionBatchMarker.from_dict(value) for value in raw_markers)
        except (ValueError, TypeError) as exc:
            raise DatasetGenerationError(f"🛑 Cannot resume record selection: {exc}") from exc

    def _validate_selection_partitions(self, markers: tuple[SelectionBatchMarker, ...]) -> None:
        referenced: set[Path] = set()
        for marker in markers:
            if marker.accepted_partition is None:
                continue
            expected = self.artifact_storage.selection_partition_path(marker.candidate_batch_id)
            actual = self.artifact_storage.base_dataset_path / marker.accepted_partition
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

        if self.artifact_storage.selection_accepted_path.exists():
            for partition in self.artifact_storage.selection_accepted_path.glob("batch_*.parquet"):
                if partition not in referenced:
                    partition.unlink()

    def _raise_unrecoverable_selection_terminal_error(
        self,
        controller: AcceptanceController,
        terminal_error: object | None,
    ) -> None:
        """Replay a durable terminal failure when resume cannot safely supersede it."""
        if terminal_error is None:
            return
        if not isinstance(terminal_error, dict):
            raise DatasetGenerationError("🛑 Cannot resume record selection: terminal error metadata is invalid.")
        kind = terminal_error.get("kind")
        message = terminal_error.get("message")
        if not isinstance(kind, str) or not isinstance(message, str):
            raise DatasetGenerationError("🛑 Cannot resume record selection: terminal error metadata is invalid.")
        if kind == _SELECTION_GENERATION_ERROR_KIND:
            raise DatasetGenerationError(message)
        if kind == _SELECTION_EARLY_SHUTDOWN_KIND:
            if not controller.has_reached_target and not controller.has_candidate_budget:
                raise RecordSelectionEarlyShutdownError(candidate_budget_remaining=False)
            return
        raise DatasetGenerationError(f"🛑 Cannot resume record selection: unknown terminal error kind {kind!r}.")

    @staticmethod
    def _derive_nonretryable_selection_terminal_error(
        controller: AcceptanceController,
    ) -> dict[str, str] | None:
        """Derive an authoritative zero-row failure from durable marker diagnostics."""
        if (
            not controller.is_exhausted
            or controller.accepted_records != 0
            or RecordSelectionExhaustion(controller.config.on_exhausted) != RecordSelectionExhaustion.RETURN_PARTIAL
        ):
            return None
        diagnostic = controller.first_non_retryable_error
        if diagnostic is None:
            return None
        return {
            "kind": _SELECTION_GENERATION_ERROR_KIND,
            "message": f"{diagnostic['type']}: {diagnostic['message']}",
        }

    def _derive_empty_selection_schema(self) -> pd.DataFrame:
        """Build a name-bearing fallback schema when every candidate slot failed generation."""
        columns: dict[str, None] = {}
        for config in self.single_column_configs:
            columns[config.name] = None
            columns.update(dict.fromkeys(config.side_effect_columns))

        drop_patterns = [
            pattern
            for processor in self._processor_runner.processors
            if isinstance(processor, DropColumnsProcessor)
            for pattern in processor.config.column_names
        ]
        output_columns = [
            column for column in columns if not any(fnmatch(column, pattern) for pattern in drop_patterns)
        ]
        return lazy.pd.DataFrame(columns=output_columns)

    def _hydrate_selection_schema_state(self, controller: AcceptanceController) -> _SelectionSchemaState:
        """Restore schema-anchor state from durable markers, repairing a missing reconstructable anchor."""
        schema_path = self.artifact_storage.selection_schema_path
        materialized = any(
            marker.schema_materialized or marker.accepted_partition is not None for marker in controller.markers
        )
        if schema_path.is_file():
            return _SelectionSchemaState(is_written=True, is_materialized=materialized)
        if not controller.markers:
            return _SelectionSchemaState(is_written=False, is_materialized=False)

        accepted_marker = next((marker for marker in controller.markers if marker.accepted_partition is not None), None)
        if accepted_marker is not None:
            partition = self.artifact_storage.base_dataset_path / accepted_marker.accepted_partition
            self.artifact_storage.write_selection_schema(lazy.pd.read_parquet(partition))
            return _SelectionSchemaState(is_written=True, is_materialized=True)
        if materialized:
            raise DatasetGenerationError(
                "🛑 Cannot resume record selection: the materialized schema anchor is missing and no "
                "accepted partition is available to reconstruct it."
            )

        self.artifact_storage.write_selection_schema(self._derive_empty_selection_schema())
        return _SelectionSchemaState(is_written=True, is_materialized=False)

    def _write_selection_metadata(
        self,
        controller: AcceptanceController,
        *,
        post_generation_state: str | None = None,
        terminal_error: dict[str, str] | None = None,
        publication_id: str | None = None,
    ) -> None:
        selection_summary: dict[str, Any] = controller.summary()
        if terminal_error is None:
            terminal_error = controller.terminal_error
        if terminal_error is not None:
            selection_summary[_SELECTION_TERMINAL_ERROR_KEY] = terminal_error
        if publication_id is not None:
            selection_summary["publication_id"] = publication_id
        updates: dict[str, Any] = {
            "target_num_records": controller.target_records,
            "original_target_num_records": controller.target_records,
            "actual_num_records": controller.accepted_records,
            "total_num_batches": controller.accepted_partitions,
            "buffer_size": controller.buffer_size,
            "dataset_name": self.artifact_storage.dataset_name,
            "num_completed_batches": controller.candidate_batches_completed,
            "record_selection": selection_summary,
        }
        if post_generation_state == "complete":
            # Candidate checkpoints are already represented by their immutable marker and
            # accepted partition. Refreshing the full processor-artifact manifest after
            # every candidate batch makes selection quadratic as that tree grows.
            updates["file_paths"] = self.artifact_storage.get_file_paths()
        if post_generation_state is not None:
            updates["post_generation_state"] = post_generation_state
            updates["post_generation_processed"] = post_generation_state == "complete"
        self.artifact_storage.update_metadata(updates)

    def _publish_selection_result(self, controller: AcceptanceController, buffer_size: int) -> None:
        publication_id = uuid.uuid4().hex
        self._write_selection_metadata(
            controller,
            post_generation_state="pending",
            publication_id=publication_id,
        )
        self._write_selection_metadata(
            controller,
            post_generation_state="started",
            publication_id=publication_id,
        )
        self.artifact_storage.materialize_selection_dataset()
        self._processor_runner.run_after_generation(buffer_size, selection_publication=True)
        actual_records = self._count_published_selection_records()
        if actual_records != controller.accepted_records:
            raise DatasetGenerationError(
                "🛑 After-generation processing changed the record-selection output count from "
                f"{controller.accepted_records} to {actual_records}. Row-count-changing after-generation "
                "processors are not supported with record selection."
            )
        self.artifact_storage.clear_selection_transient_artifacts()
        processor_names = self.artifact_storage.list_processor_names()
        self.artifact_storage.update_metadata(
            {
                "publication": {
                    "managed_local_prefixes": [
                        "parquet-files",
                        "images",
                        *(f"processors-files/{name}" for name in processor_names),
                    ],
                    "managed_hub_prefixes": ["data", "images", *processor_names],
                }
            }
        )
        self._write_selection_metadata(
            controller,
            post_generation_state="complete",
            publication_id=publication_id,
        )
        self._clear_selection_artifact_migration_metadata()

    def _clear_selection_artifact_migration_metadata(self) -> None:
        if not self.artifact_storage.metadata_file_path.is_file():
            return
        metadata = self.artifact_storage.read_metadata()
        if _SELECTION_ARTIFACT_MIGRATION_METADATA_KEY not in metadata:
            return
        metadata.pop(_SELECTION_ARTIFACT_MIGRATION_METADATA_KEY)
        self.artifact_storage.write_metadata(metadata)

    def _count_published_selection_records(self) -> int:
        return sum(
            lazy.pq.read_metadata(path).num_rows
            for path in self.artifact_storage.final_dataset_path.glob("batch_*.parquet")
        )

    def _build_async(
        self,
        generators: list[ColumnGenerator],
        num_records: int,
        buffer_size: int,
        on_batch_complete: Callable[[Path], None] | None = None,
        *,
        resume: ResumeMode = ResumeMode.NEVER,
    ) -> bool:
        """Async task-queue builder path - dispatches tasks based on dependency readiness.

        Returns:
            False if the dataset was already complete (no new records generated),
            True after successfully running the scheduler.
        """
        logger.info("⚡ Using async task-queue builder")

        settings = self._resource_provider.run_config
        trace_enabled = _is_async_trace_enabled(settings)

        precomputed_row_groups: RowGroupInput | None = None
        initial_actual_num_records = 0
        initial_total_num_batches = 0
        original_target = num_records  # immutable original target; overridden on resume

        if resume == ResumeMode.ALWAYS:
            state = self._load_resume_state(num_records, buffer_size)
            # _load_resume_state already scans the filesystem for completed row groups
            # and exposes them via state.completed_row_groups. The filesystem is the
            # source of truth for progress (metadata may lag by one row group between
            # move_partial_result_to_final_file_path and write_metadata).
            completed_row_groups = state.completed_row_groups
            completed_ids = set(completed_row_groups)
            initial_total_num_batches = state.num_completed_batches
            initial_actual_num_records = state.actual_num_records
            # Use the original target (not the new num_records) so the last row group of a
            # non-aligned run gets its true size, not buffer_size.
            original_target = state.original_target_num_records

            self.artifact_storage.clear_partial_results()

            resume_plan = build_row_group_resume_plan(
                original_target=original_target,
                num_records=num_records,
                buffer_size=buffer_size,
                completed_ids=completed_ids,
            )
            remaining_row_group_count = len(resume_plan.remaining_row_groups)
            completed_row_group_count = resume_plan.total_row_groups - remaining_row_group_count
            if remaining_row_group_count == 0:
                logger.warning(
                    "⚠️ Dataset is already complete — all row groups were found in the existing artifact "
                    "directory. Nothing to resume. Use resume=ResumeMode.NEVER if you want to generate a new dataset."
                )
                return False

            logger.info(
                f"▶️ Resuming async run: {completed_row_group_count} of {resume_plan.total_row_groups} row group(s) "
                f"already complete ({initial_actual_num_records} records), skipping them."
            )

            precomputed_row_groups = resume_plan.remaining_row_groups

        def finalize_row_group(rg_id: int) -> None:
            def on_complete(final_path: Path | str | None) -> None:
                if final_path is not None and on_batch_complete:
                    on_batch_complete(final_path)

            buffer_manager.checkpoint_row_group(rg_id, on_complete=on_complete)
            # Write incremental metadata after each row group so interrupted runs can be resumed.
            buffer_manager.write_metadata(
                target_num_records=num_records,
                original_target_num_records=original_target,
                buffer_size=buffer_size,
            )

        # Telemetry snapshot
        group_id = uuid.uuid4().hex
        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()

        event_sink_context = (
            JsonlSchedulerEventSink(self.artifact_storage.base_dataset_path / "scheduler_events.jsonl")
            if settings.write_scheduler_events
            else contextlib.nullcontext()
        )
        with event_sink_context as scheduler_event_sink:
            scheduler, buffer_manager = self._prepare_async_run(
                generators,
                num_records,
                buffer_size,
                on_finalize_row_group=finalize_row_group,
                shutdown_error_rate=settings.shutdown_error_rate,
                shutdown_error_window=settings.shutdown_error_window,
                disable_early_shutdown=settings.disable_early_shutdown,
                trace=trace_enabled,
                precomputed_row_groups=precomputed_row_groups,
                initial_actual_num_records=initial_actual_num_records,
                initial_total_num_batches=initial_total_num_batches,
                scheduler_event_sink=scheduler_event_sink,
            )

            # Run on background event loop. Capture scheduler state in `finally`
            # so the structured signal is preserved even if `scheduler.run()`
            # raises during the salvage path - otherwise callers see a generic
            # error and lose the early-shutdown context.
            loop = ensure_async_engine_loop()
            future = asyncio.run_coroutine_threadsafe(scheduler.run(), loop)
            try:
                _await_async_scheduler_result(future, scheduler)
            finally:
                self._task_traces = scheduler.traces
                self._early_shutdown = scheduler.early_shutdown
                self._partial_row_groups = scheduler.partial_row_groups
                self._actual_num_records = buffer_manager.actual_num_records
                self._first_non_retryable_error = scheduler.first_non_retryable_error

        # Emit telemetry
        try:
            usage_deltas = self._resource_provider.model_registry.get_usage_deltas(pre_batch_snapshot)
            self._emit_batch_inference_events("batch", usage_deltas, group_id)
        except Exception:
            logger.debug("Failed to emit batch telemetry for async run", exc_info=True)

        # Write final metadata (overwrites the last incremental write with identical content).
        buffer_manager.write_metadata(
            target_num_records=num_records,
            original_target_num_records=original_target,
            buffer_size=buffer_size,
        )

        # Surface partial completion
        actual = self._actual_num_records
        if actual < num_records:
            pct = actual / num_records * 100 if num_records > 0 else 0
            base = f"⚠️ Generated {actual} of {num_records} requested records ({pct:.0f}%). "
            if scheduler.early_shutdown:
                partial = scheduler.partial_row_groups
                detail = (
                    f"Early shutdown was triggered (non-retryable error rate exceeded threshold); "
                    f"{len(partial)} row group(s) salvaged with partial rows."
                    if partial
                    else "Early shutdown was triggered (non-retryable error rate exceeded threshold)."
                )
                logger.warning(base + detail)
            else:
                logger.warning(base + "The dataset may be incomplete due to dropped rows.")

        return True

    def _prepare_async_run(
        self,
        generators: list[ColumnGenerator],
        num_records: int,
        buffer_size: int,
        *,
        on_finalize_row_group: Callable[[int], None] | None = None,
        run_post_batch_in_scheduler: bool = True,
        shutdown_error_rate: float = 0.5,
        shutdown_error_window: int = 10,
        disable_early_shutdown: bool = False,
        trace: bool = False,
        precomputed_row_groups: RowGroupInput | None = None,
        initial_actual_num_records: int = 0,
        initial_total_num_batches: int = 0,
        scheduler_event_sink: SchedulerAdmissionEventSink | None = None,
        select_dataframe: Callable[[int, int, pd.DataFrame], pd.DataFrame] | None = None,
        log_pre_generation: bool = True,
    ) -> tuple[AsyncTaskScheduler, RowGroupBufferManager]:
        """Build a fully-wired scheduler and buffer manager for async generation.

        Shared setup for both build and preview paths. Processor hooks are always
        wired when the config has processors, so callers cannot accidentally omit them.
        """
        strategies: dict[str, GenerationStrategy] = {}
        gen_map: dict[str, ColumnGenerator] = {}
        for gen in generators:
            if isinstance(gen.config, MultiColumnConfig):
                for sub in gen.config.columns:
                    strategies[sub.name] = gen.get_generation_strategy()
                    gen_map[sub.name] = gen
            else:
                strategies[gen.config.name] = gen.get_generation_strategy()
                gen_map[gen.config.name] = gen

        graph = ExecutionGraph.create(self._column_configs, strategies)

        if log_pre_generation:
            for generator in generators:
                generator.log_pre_generation()

        if precomputed_row_groups is not None:
            row_groups: RowGroupPlanLike = normalize_row_group_plan(precomputed_row_groups)
        else:
            row_groups = CompactRowGroupPlan.fresh(num_records=num_records, buffer_size=buffer_size)

        tracker = CompletionTracker.with_graph(graph, row_groups)
        buffer_manager = RowGroupBufferManager(
            self.artifact_storage,
            initial_actual_num_records=initial_actual_num_records,
            initial_total_num_batches=initial_total_num_batches,
        )

        # Pre-batch processor callback: runs after seed tasks complete for a row group.
        # If it raises, the scheduler propagates the error as DatasetGenerationError (fail-fast).
        def on_seeds_complete(rg_id: int, rg_size: int) -> FrontierDelta:
            df = buffer_manager.get_dataframe(rg_id)
            df = self._processor_runner.run_pre_batch_on_df(df)
            buffer_manager.replace_dataframe(rg_id, df)
            deltas: list[FrontierDelta] = []
            for ri in range(rg_size):
                if buffer_manager.is_dropped(rg_id, ri) and not tracker.is_dropped(rg_id, ri):
                    deltas.append(tracker.drop_row(rg_id, ri))
            return FrontierDelta(
                added=tuple(task for delta in deltas for task in delta.added),
                removed=tuple(task for delta in deltas for task in delta.removed),
            )

        # Post-batch processor callback: runs after all columns, before finalization.
        def on_before_checkpoint(rg_id: int, rg_size: int) -> None:
            df = buffer_manager.get_dataframe(rg_id)
            df = self._processor_runner.run_post_batch(df, current_batch_number=rg_id, strict_row_count=True)
            buffer_manager.replace_dataframe(rg_id, df)

        def on_select_before_checkpoint(rg_id: int, rg_size: int) -> None:
            if select_dataframe is None:
                return
            dataframe = buffer_manager.get_dataframe(rg_id)
            dataframe = select_dataframe(rg_id, rg_size, dataframe)
            buffer_manager.replace_dataframe(rg_id, dataframe)
            for row_index in range(rg_size):
                if buffer_manager.is_dropped(rg_id, row_index) and not tracker.is_dropped(rg_id, row_index):
                    tracker.drop_row(rg_id, row_index)

        max_in_flight_tasks = self._resource_provider.run_config.max_in_flight_tasks
        max_model_task_admission = max_in_flight_tasks

        scheduler = AsyncTaskScheduler(
            generators=gen_map,
            graph=graph,
            tracker=tracker,
            row_groups=row_groups,
            buffer_manager=buffer_manager,
            max_concurrent_row_groups=self._resource_provider.run_config.max_concurrent_row_groups,
            max_in_flight_tasks=max_in_flight_tasks,
            max_model_task_admission=max_model_task_admission,
            on_finalize_row_group=on_finalize_row_group,
            on_select_before_checkpoint=on_select_before_checkpoint if select_dataframe is not None else None,
            on_seeds_complete=(
                on_seeds_complete if self._processor_runner.has_processors_for(ProcessorStage.PRE_BATCH) else None
            ),
            on_before_checkpoint=(
                on_before_checkpoint
                if run_post_batch_in_scheduler and self._processor_runner.has_processors_for(ProcessorStage.POST_BATCH)
                else None
            ),
            shutdown_error_rate=shutdown_error_rate,
            shutdown_error_window=shutdown_error_window,
            disable_early_shutdown=disable_early_shutdown,
            trace=trace,
            num_records=num_records,
            buffer_size=buffer_size,
            initial_completed_records=initial_actual_num_records,
            progress_interval=self._resource_provider.run_config.progress_interval,
            display_tui=self._resource_provider.run_config.display_tui,
            scheduler_event_sink=fanout_scheduler_event_sinks(
                scheduler_event_sink,
                self._resource_provider.scheduler_event_sink,
            ),
            request_pressure_provider=self._resource_provider.model_registry.request_admission,
            request_pressure_advisory=True,
        )
        return scheduler, buffer_manager

    def process_preview(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = self._processor_runner.run_post_batch(dataset.copy(), current_batch_number=None)
        return self._processor_runner.run_after_generation_on_df(df)

    def _has_image_columns(self) -> bool:
        """Check if config has any image generation columns."""
        return any(col.column_type == DataDesignerColumnType.IMAGE for col in self.single_column_configs)

    def _initialize_generators_and_graph(self) -> tuple[list[ColumnGenerator], ExecutionGraph]:
        generators = [
            self._registry.column_generators.get_for_config_type(type(config))(
                config=config, resource_provider=self._resource_provider
            )
            for config in self._column_configs
        ]
        strategies: dict[str, GenerationStrategy] = {}
        for gen in generators:
            strategy = gen.get_generation_strategy()
            if isinstance(gen.config, MultiColumnConfig):
                for sub in gen.config.columns:
                    strategies[sub.name] = strategy
            else:
                strategies[gen.config.name] = strategy
        graph = ExecutionGraph.create(self._column_configs, strategies)
        return generators, graph

    def _write_builder_config(self) -> None:
        self.artifact_storage.mkdir_if_needed(self.artifact_storage.base_dataset_path)
        BuilderConfig(data_designer=self._data_designer_config).to_json(
            self.artifact_storage.base_dataset_path / SDG_CONFIG_FILENAME
        )

    def _validate_column_configs(self) -> None:
        if len(self._column_configs) == 0:
            raise DatasetGenerationError("🛑 No column configs provided.")

        if not self._registry.column_generators.get_for_config_type(
            type(self._column_configs[0])
        ).can_generate_from_scratch:
            raise DatasetGenerationError("🛑 The first column config must be a from-scratch column generator.")

    def _validate_record_selection_config(self) -> None:
        config = self._data_designer_config.record_selection
        if config is None:
            return
        predicate_config: ColumnConfigT | None = None
        predicate_owner: ColumnConfigT | None = None
        available_columns: set[str] = set()
        for column_config in self.single_column_configs:
            available_columns.add(column_config.name)
            available_columns.update(column_config.side_effect_columns)
            if column_config.name == config.predicate_column:
                predicate_config = column_config
                predicate_owner = column_config
            elif config.predicate_column in column_config.side_effect_columns:
                predicate_owner = column_config
        if config.predicate_column not in available_columns:
            raise DatasetGenerationError(
                f"🛑 Record-selection predicate column {config.predicate_column!r} does not exist in the "
                "compiled dataset columns."
            )
        if predicate_owner is None:
            raise DatasetGenerationError(
                f"🛑 Record-selection predicate column {config.predicate_column!r} has no owning column config."
            )

        column_type = predicate_owner.column_type
        if column_type == DataDesignerColumnType.EXPRESSION:
            if predicate_config is not None and getattr(predicate_config, "dtype", None) == "bool":
                return
            raise DatasetGenerationError(
                f"🛑 Record-selection predicate expression {config.predicate_column!r} must use dtype='bool'."
            )

        if isinstance(predicate_config, SamplerColumnConfig) and self._sampler_predicate_is_known_boolean(
            predicate_config
        ):
            return

        # Seed, custom, and plugin output types are intentionally runtime-defined.
        # Their values remain subject to AcceptanceController's strict bool/null check.
        if column_type in (DataDesignerColumnType.SEED_DATASET, DataDesignerColumnType.CUSTOM) or is_plugin_column_type(
            column_type
        ):
            return

        raise DatasetGenerationError(
            f"🛑 Record-selection predicate column {config.predicate_column!r} must be boolean; "
            f"built-in column type {str(column_type)!r} does not produce boolean values."
        )

    @staticmethod
    def _sampler_predicate_is_known_boolean(config: SamplerColumnConfig) -> bool:
        if config.convert_to is not None:
            return False
        params = (config.params, *config.conditional_params.values())
        if all(isinstance(item, CategorySamplerParams) for item in params):
            return all(isinstance(value, bool) for item in params for value in item.values)
        if all(isinstance(item, SubcategorySamplerParams) for item in params):
            values = [value for item in params for group in item.values.values() for value in group]
            return bool(values) and all(isinstance(value, bool) for value in values)
        return False

    def _validate_record_selection_request(self, num_records: int) -> None:
        config = self._data_designer_config.record_selection
        if config is None:
            return
        if num_records <= 0:
            raise DatasetGenerationError("🛑 num_records must be positive when record selection is configured.")
        if config.max_candidate_records < num_records:
            raise DatasetGenerationError(
                "🛑 record_selection.max_candidate_records must be greater than or equal to "
                f"num_records ({config.max_candidate_records} < {num_records})."
            )

    def _initialize_processors(self, processor_configs: list[ProcessorConfig]) -> list[Processor]:
        # Check columns marked for drop
        columns_to_drop = [config.name for config in self.single_column_configs if config.drop]

        processors: list[Processor] = []
        for config in processor_configs:
            processors.append(
                self._registry.processors.get_for_config_type(type(config))(
                    config=config,
                    resource_provider=self._resource_provider,
                )
            )

            # Manually included "drop columns" processor takes precedence
            if config.processor_type == ProcessorType.DROP_COLUMNS:
                for column in config.column_names:
                    if column in columns_to_drop:
                        columns_to_drop.remove(column)

        # If there are still columns marked for drop, add the "drop columns" processor to drop them
        if len(columns_to_drop) > 0:
            processors.append(
                DropColumnsProcessor(
                    config=DropColumnsProcessorConfig(
                        name="default_drop_columns_processor",
                        column_names=columns_to_drop,
                    ),
                    resource_provider=self._resource_provider,
                )
            )

        return processors

    def _emit_batch_inference_events(
        self, batch_mode: str, usage_deltas: dict[str, ModelUsageStats], group_id: str
    ) -> None:
        if not usage_deltas:
            return

        events = [
            InferenceEvent(
                nemo_source=NemoSourceEnum.DATADESIGNER,
                task=batch_mode,
                task_status=TaskStatusEnum.SUCCESS,
                model=model_name,
                input_tokens=delta.token_usage.input_tokens,
                output_tokens=delta.token_usage.output_tokens,
            )
            for model_name, delta in usage_deltas.items()
        ]

        with TelemetryHandler(source_client_version=_CLIENT_VERSION, session_id=group_id) as telemetry_handler:
            for event in events:
                telemetry_handler.enqueue(event)
