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
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType
from data_designer.config.config_builder import BuilderConfig
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorType,
)
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.config.version import get_library_version
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
)
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.dataset_builders.async_scheduler import AsyncTaskScheduler
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.row_group_plan import (
    CompactRowGroupPlan,
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
from data_designer.engine.observability import JsonlSchedulerEventSink, SchedulerAdmissionEventSink
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.readiness import run_readiness_check
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.storage.artifact_storage import (
    METADATA_FILENAME,
    SDG_CONFIG_FILENAME,
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
            num_records: Number of records to generate.
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

        Returns:
            Path to the generated dataset directory.
        """
        self._reset_run_state()

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
                    resume = ResumeMode.NEVER
                    self.artifact_storage.resume = ResumeMode.NEVER
                    self.artifact_storage.__dict__.pop("resolved_dataset_name", None)
                    self.artifact_storage.refresh_media_storage_path()
                else:
                    resume = ResumeMode.ALWAYS
                    self.artifact_storage.resume = ResumeMode.ALWAYS
                    self.artifact_storage.__dict__.pop("resolved_dataset_name", None)

        self._set_metadata_defaults()

        if self._post_generation_processed_resume_result(resume, num_records) is not None:
            return self.artifact_storage.final_dataset_path

        self._write_builder_config()

        # Set media storage mode based on parameters
        if self._has_image_columns():
            mode = StorageMode.DISK if save_multimedia_to_disk else StorageMode.DATAFRAME
            self.artifact_storage.set_media_storage_mode(mode)

        generators, self._graph = self._initialize_generators_and_graph()
        start_time = time.perf_counter()
        buffer_size = self._resource_provider.run_config.buffer_size

        if resume == ResumeMode.ALWAYS and not self.artifact_storage.metadata_file_path.exists():
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

        for gen in generators:
            gen.log_pre_generation()

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

        max_in_flight_tasks = self._resource_provider.run_config.max_in_flight_tasks
        max_model_task_admission = max_in_flight_tasks

        scheduler = AsyncTaskScheduler(
            generators=gen_map,
            graph=graph,
            tracker=tracker,
            row_groups=row_groups,
            buffer_manager=buffer_manager,
            max_in_flight_tasks=max_in_flight_tasks,
            max_model_task_admission=max_model_task_admission,
            on_finalize_row_group=on_finalize_row_group,
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
            scheduler_event_sink=scheduler_event_sink,
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
