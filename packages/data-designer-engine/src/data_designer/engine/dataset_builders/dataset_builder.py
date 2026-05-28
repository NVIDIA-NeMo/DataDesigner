# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from data_designer.config.utils.warning_helpers import warn_at_caller
from data_designer.config.version import get_library_version
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorWithModel,
    GenerationStrategy,
)
from data_designer.engine.column_generators.utils.generator_classification import column_type_is_model_generated
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.context import current_row_group, current_row_group_start_offset
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.engine.dataset_builders.utils.config_compiler import compile_dataset_builder_column_configs
from data_designer.engine.dataset_builders.utils.dataset_batch_manager import DatasetBatchManager
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.processor_runner import ProcessorRunner, ProcessorStage
from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.engine.dataset_builders.utils.skip_evaluator import should_skip_column_for_record
from data_designer.engine.dataset_builders.utils.skip_tracker import (
    SKIPPED_COLUMNS_RECORD_KEY,
    apply_skip_to_record,
    prepare_records_for_skip_metadata_round_trip,
    restore_skip_metadata,
    strip_skip_metadata_from_records,
)
from data_designer.engine.dataset_builders.utils.sticky_progress_bar import StickyProgressBar
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum, TelemetryHandler
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
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
    from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry
    from data_designer.engine.dataset_builders.scheduling.task_model import TaskTrace
    from data_designer.engine.models.usage import ModelUsageStats

logger = logging.getLogger(__name__)

# Async engine is the default execution path. Set ``DATA_DESIGNER_ASYNC_ENGINE=0``
# to opt back into the legacy sync engine for one transitional release; the sync
# path is scheduled for removal afterwards.
DATA_DESIGNER_ASYNC_ENGINE = os.environ.get("DATA_DESIGNER_ASYNC_ENGINE", "1") == "1"

if DATA_DESIGNER_ASYNC_ENGINE:
    import asyncio

    from data_designer.engine.dataset_builders.async_scheduler import (
        AsyncTaskScheduler,
    )
    from data_designer.engine.dataset_builders.scheduling.completion import CompletionTracker, FrontierDelta
    from data_designer.engine.dataset_builders.utils.async_concurrency import (
        AsyncConcurrentExecutor,
        ensure_async_engine_loop,
    )
    from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager


_CLIENT_VERSION: str = get_library_version()
PRESERVE_DROPPED_COLUMNS_METADATA_KEY = "preserve_dropped_columns"


def _is_async_trace_enabled(settings: RunConfig) -> bool:
    return settings.async_trace or os.environ.get("DATA_DESIGNER_ASYNC_TRACE", "0") == "1"


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
        remaining_row_groups: ``(rg_id, rg_size)`` for groups not yet on disk, in id order.
        row_group_start_offsets: ``rg_id -> planned start offset`` for the remaining
            groups, computed from the original plan so completed-group sizes are
            preserved (offsets are not recomputed from the remaining list, which
            would shift them when there are holes).
    """

    total_row_groups: int
    remaining_row_groups: list[tuple[int, int]]
    row_group_start_offsets: dict[int, int]


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
        A ``RowGroupResumePlan`` whose ``row_group_start_offsets`` are taken from
        the full original plan, so the offset for ``rg_id`` is the same whether
        or not earlier groups have completed. This is what lets ordered seed
        generators seek to the correct row when resuming with holes.
    """
    num_original_groups = -(-original_target // buffer_size)
    extension_records = num_records - original_target
    total_row_groups = num_original_groups + -(-extension_records // buffer_size)

    def _rg_size(rg_id: int) -> int:
        if rg_id < num_original_groups:
            return min(buffer_size, original_target - rg_id * buffer_size)
        ext_group_idx = rg_id - num_original_groups
        return min(buffer_size, extension_records - ext_group_idx * buffer_size)

    all_start_offsets: dict[int, int] = {}
    next_offset = 0
    for rg_id in range(total_row_groups):
        all_start_offsets[rg_id] = next_offset
        next_offset += _rg_size(rg_id)

    remaining_row_groups = [(rg_id, _rg_size(rg_id)) for rg_id in range(total_row_groups) if rg_id not in completed_ids]
    return RowGroupResumePlan(
        total_row_groups=total_row_groups,
        remaining_row_groups=remaining_row_groups,
        row_group_start_offsets={rg_id: all_start_offsets[rg_id] for rg_id, _ in remaining_row_groups},
    )


class DatasetBuilder:
    def __init__(
        self,
        data_designer_config: DataDesignerConfig,
        resource_provider: ResourceProvider,
        registry: DataDesignerRegistry | None = None,
    ):
        self.batch_manager = DatasetBatchManager(resource_provider.artifact_storage)
        self._resource_provider = resource_provider
        self._records_to_drop: set[int] = set()
        self._cell_resize_results: list[dict | list[dict] | None] = []
        self._cell_resize_mode = False
        self._task_traces: list[TaskTrace] = []
        self._registry = registry or DataDesignerRegistry()
        self._graph: ExecutionGraph | None = None
        self._use_async: bool = DATA_DESIGNER_ASYNC_ENGINE
        # Structured signal: set by _build_async if the scheduler hit early shutdown.
        # Stays at defaults for sync-engine and successful async runs. Reset at
        # the start of each public run path so reused builder instances don't
        # leak state across runs.
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

    def set_processor_runner(self, processors: list[Processor]) -> None:
        """Replace the processor runner with a new one using the given processors."""
        self._processor_runner = ProcessorRunner(
            processors=processors,
            artifact_storage=self.artifact_storage,
        )

    @functools.cached_property
    def single_column_configs(self) -> list[ColumnConfigT]:
        configs = []
        for config in self._column_configs:
            if isinstance(config, MultiColumnConfig):
                configs.extend(config.columns)
            else:
                configs.append(config)
        return configs

    @functools.cached_property
    def single_column_config_by_name(self) -> dict[str, ColumnConfigT]:
        return {config.name: config for config in self.single_column_configs}

    @functools.cached_property
    def llm_generated_column_configs(self) -> list[ColumnConfigT]:
        return [config for config in self.single_column_configs if column_type_is_model_generated(config.column_type)]

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
                - ``ResumeMode.ALWAYS``: resume from the last completed batch (sync) or row group
                  (async). ``buffer_size`` must match the original run. ``num_records`` may be
                  equal to or greater than what was already generated (you can extend the dataset);
                  ``num_records`` less than actual records so far raises ``DatasetGenerationError``.
                  If no checkpoint exists yet (interrupted before the first batch finished), silently
                  restarts from the beginning. Raises if the stored config is incompatible.
                - ``ResumeMode.IF_POSSIBLE``: like ``ALWAYS`` when the current config fingerprint
                  matches the stored config; otherwise starts a fresh run without raising an error.

                In all resume modes, in-flight partial results from the interrupted run are
                discarded before generation continues.

        Returns:
            Path to the generated dataset directory.
        """
        self._reset_run_state()

        self._run_model_health_check_if_needed()
        self._run_mcp_tool_check_if_needed()

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
            # No metadata.json means the previous run was interrupted before any batch (sync) or
            # row group (async) completed.  Nothing to resume — discard any leftover partial
            # results and start fresh.
            logger.info(
                "▶️ No metadata.json found — the previous run was interrupted before any batch "
                "completed. Starting generation from the beginning."
            )
            self.artifact_storage.clear_partial_results()
            resume = ResumeMode.NEVER
            self.artifact_storage.resume = ResumeMode.NEVER

        if resume == ResumeMode.ALWAYS and self._has_allow_resize_columns():
            raise DatasetGenerationError(
                "🛑 Cannot resume when any column has allow_resize=True. Resized batches change row boundaries, "
                "so the original batch plan cannot be reconstructed safely. Use resume=ResumeMode.NEVER to "
                "start a new generation run."
            )

        self._use_async = DATA_DESIGNER_ASYNC_ENGINE and self._resolve_async_compatibility()
        if self._use_async:
            self._build_async(generators, num_records, buffer_size, on_batch_complete, resume=resume)
        elif resume == ResumeMode.ALWAYS:
            self._build_with_resume(generators, num_records, buffer_size, on_batch_complete)
        else:
            group_id = uuid.uuid4().hex
            self.batch_manager.start(num_records=num_records, buffer_size=buffer_size)
            for batch_idx in range(self.batch_manager.num_batches):
                logger.info(f"⏳ Processing batch {batch_idx + 1} of {self.batch_manager.num_batches}")
                self._run_batch(
                    generators,
                    batch_mode="batch",
                    group_id=group_id,
                    current_batch_number=batch_idx,
                    on_batch_complete=on_batch_complete,
                )
            self.batch_manager.finish()

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

    def _has_allow_resize_columns(self) -> bool:
        return any(getattr(config, "allow_resize", False) for config in self.single_column_configs)

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
        boundaries. The sync engine additionally requires contiguous batch IDs;
        the async engine tolerates holes from out-of-order completion.

        Raises:
            DatasetGenerationError: If metadata is missing or incompatible, or if
                the filesystem state is inconsistent with the engine in use.
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
            allow_holes=self._use_async,
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
            original_target_num_records=metadata.get("original_target_num_records", target_num_records),
            completed_row_groups=completed_row_groups,
        )

    def _build_with_resume(
        self,
        generators: list[ColumnGenerator],
        num_records: int,
        buffer_size: int,
        on_batch_complete: Callable[[Path], None] | None,
    ) -> bool:
        """Resume generation from the last completed batch.

        Returns:
            False if the dataset was already complete (no new records generated),
            True after successfully generating the remaining batches.
        """
        state = self._load_resume_state(num_records, buffer_size)

        # Compute the correct per-batch sizes. ceil(num_records/bs) is wrong for a
        # non-aligned extension: original groups are immutable, so any extension always
        # adds new groups beyond num_original_batches.
        original_target = state.original_target_num_records
        num_original_batches = -(-original_target // buffer_size)
        extension_records = num_records - original_target
        num_extension_batches = -(-extension_records // buffer_size)
        original_sizes = [min(buffer_size, original_target - i * buffer_size) for i in range(num_original_batches)]
        extension_sizes = [min(buffer_size, extension_records - i * buffer_size) for i in range(num_extension_batches)]

        self.batch_manager.start(
            num_records=num_records,
            buffer_size=buffer_size,
            start_batch=state.num_completed_batches,
            initial_actual_num_records=state.actual_num_records,
            num_records_list=original_sizes + extension_sizes,
            original_target_num_records=original_target,
        )

        if state.num_completed_batches >= self.batch_manager.num_batches:
            logger.warning(
                "⚠️ Dataset is already complete — all batches were found in the existing artifact directory. "
                "Nothing to resume. Use resume=ResumeMode.NEVER if you want to generate a new dataset."
            )
            return False

        logger.info(
            f"▶️ Resuming from batch {state.num_completed_batches + 1} of {self.batch_manager.num_batches} "
            f"({state.actual_num_records} records already generated)."
        )

        self.artifact_storage.clear_partial_results()

        group_id = uuid.uuid4().hex
        for batch_idx in range(state.num_completed_batches, self.batch_manager.num_batches):
            logger.info(f"⏳ Processing batch {batch_idx + 1} of {self.batch_manager.num_batches}")
            self._run_batch(
                generators,
                batch_mode="batch",
                group_id=group_id,
                current_batch_number=batch_idx,
                on_batch_complete=on_batch_complete,
                row_group_start_offset=sum(self.batch_manager.num_records_list[:batch_idx]),
            )
        self.batch_manager.finish()
        return True

    def build_preview(self, *, num_records: int) -> pd.DataFrame:
        self._reset_run_state()
        self._run_model_health_check_if_needed()
        self._run_mcp_tool_check_if_needed()

        # Set media storage to DATAFRAME mode for preview - base64 stored directly in DataFrame
        if self._has_image_columns():
            self.artifact_storage.set_media_storage_mode(StorageMode.DATAFRAME)

        generators, self._graph = self._initialize_generators_and_graph()
        start_time = time.perf_counter()

        self._use_async = DATA_DESIGNER_ASYNC_ENGINE and self._resolve_async_compatibility()
        if self._use_async:
            dataset = self._build_async_preview(generators, num_records)
        else:
            group_id = uuid.uuid4().hex
            self.batch_manager.start(num_records=num_records, buffer_size=num_records)
            self._run_batch(generators, batch_mode="preview", save_partial_results=False, group_id=group_id)
            dataset = self.batch_manager.get_current_batch(as_dataframe=True)
            self.batch_manager.reset()

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
        logger.info("⚡ DATA_DESIGNER_ASYNC_ENGINE is enabled - using async task-queue preview")

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
            future.result()
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

    def _resolve_async_compatibility(self) -> bool:
        """Check if the async engine can be used; auto-fallback to sync if not.

        Returns True if async is usable, False if allow_resize forces sync fallback.
        """
        offending = [config.name for config in self.single_column_configs if getattr(config, "allow_resize", False)]
        if offending:
            msg = (
                f"allow_resize=True detected on column(s) {offending}. "
                "Falling back to sync engine for this run. "
                "allow_resize is deprecated and will be removed in a future release; "
                "use workflow chaining instead (see issue #552)."
            )
            logger.warning(f"⚠️ {msg}")
            # ``warn_at_caller`` rather than ``warnings.warn(stacklevel=N)`` so
            # attribution lands on the user's call site instead of an internal
            # ``DatasetBuilder.build`` / ``data_designer.interface`` frame.
            # The exact internal-frame depth from this method up to user code
            # depends on which entry point invoked the builder (build vs.
            # build_preview, sync vs. async wrapping), so a hard-coded
            # ``stacklevel`` is brittle; ``warn_at_caller`` walks past every
            # ``data_designer.*`` frame regardless of chain shape. Library
            # attribution would also be silenced under Python's default
            # ``ignore::DeprecationWarning`` filter. See PR #594 review.
            warn_at_caller(msg, DeprecationWarning)
            return False
        return True

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
        Both engines use the same scan so resume semantics stay consistent.

        Args:
            allow_holes: ``True`` for the async engine, which schedules row groups
                concurrently and may complete them out of order. ``False`` for the sync
                engine, which writes batches sequentially — a non-contiguous set of IDs
                indicates external mutation or a directory written by an incompatible
                engine and is rejected with ``DatasetGenerationError``.

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
                "⚠️ Could not read stored config at %s for compatibility check — assuming compatible.",
                config_path,
            )
            return _ConfigCompatibility.COMPATIBLE

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
        logger.info("⚡ DATA_DESIGNER_ASYNC_ENGINE is enabled - using async task-queue builder")

        settings = self._resource_provider.run_config
        trace_enabled = _is_async_trace_enabled(settings)

        precomputed_row_groups: list[tuple[int, int]] | None = None
        row_group_start_offsets: dict[int, int] | None = None
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
            if len(completed_ids) >= resume_plan.total_row_groups:
                logger.warning(
                    "⚠️ Dataset is already complete — all row groups were found in the existing artifact "
                    "directory. Nothing to resume. Use resume=ResumeMode.NEVER if you want to generate a new dataset."
                )
                return False

            logger.info(
                f"▶️ Resuming async run: {len(completed_ids)} of {resume_plan.total_row_groups} row group(s) already "
                f"complete ({initial_actual_num_records} records), skipping them."
            )

            precomputed_row_groups = resume_plan.remaining_row_groups
            row_group_start_offsets = resume_plan.row_group_start_offsets

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
            row_group_start_offsets=row_group_start_offsets,
            initial_actual_num_records=initial_actual_num_records,
            initial_total_num_batches=initial_total_num_batches,
        )

        # Telemetry snapshot
        group_id = uuid.uuid4().hex
        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()

        # Run on background event loop. Capture scheduler state in `finally`
        # so the structured signal is preserved even if `scheduler.run()`
        # raises during the salvage path - otherwise callers see a generic
        # error and lose the early-shutdown context.
        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(scheduler.run(), loop)
        try:
            future.result()
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
        precomputed_row_groups: list[tuple[int, int]] | None = None,
        row_group_start_offsets: dict[int, int] | None = None,
        initial_actual_num_records: int = 0,
        initial_total_num_batches: int = 0,
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
            row_groups = precomputed_row_groups
        else:
            row_groups = []
            remaining = num_records
            rg_id = 0
            while remaining > 0:
                size = min(buffer_size, remaining)
                row_groups.append((rg_id, size))
                remaining -= size
                rg_id += 1

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
            df = self._processor_runner.run_pre_batch_on_df(df, strict_row_count=True)
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
            row_group_start_offsets=row_group_start_offsets,
            progress_interval=self._resource_provider.run_config.progress_interval,
            progress_bar=self._resource_provider.run_config.progress_bar,
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

    def _run_batch(
        self,
        generators: list[ColumnGenerator],
        *,
        batch_mode: str,
        save_partial_results: bool = True,
        group_id: str,
        current_batch_number: int | None = None,
        on_batch_complete: Callable[[Path], None] | None = None,
        row_group_start_offset: int | None = None,
    ) -> None:
        """Run one batch of generators in the sync engine.

        Sets two ContextVars for the duration of the batch so order-dependent
        generators (e.g. seed dataset under ORDERED sampling) and log helpers
        can observe the row group's place in the run:

        - ``current_row_group`` is set whenever ``current_batch_number`` is known
          (both fresh and resumed sync runs), so ``format_row_group_tag()``
          produces a consistent ``(x/X)`` log prefix in either path.
        - ``current_row_group_start_offset`` is set only when the caller supplies
          the planned start offset (sync resume passes it; fresh sync and preview
          do not), so generators can seek into the correct seed slice without
          replaying already-consumed rows.
        """
        row_group_token = None
        start_offset_token = None
        if current_batch_number is not None:
            row_group_token = current_row_group.set((current_batch_number, self.batch_manager.num_batches))
        if row_group_start_offset is not None:
            start_offset_token = current_row_group_start_offset.set(row_group_start_offset)

        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()
        ran_pre_batch = False
        try:
            for generator in generators:
                generator.log_pre_generation()
                try:
                    generation_strategy = generator.get_generation_strategy()
                    if generator.can_generate_from_scratch and self.batch_manager.buffer_is_empty:
                        self._run_from_scratch_column_generator(generator)
                        # Run PRE_BATCH after seed generator, before other columns
                        if not ran_pre_batch:
                            self._processor_runner.run_pre_batch(self.batch_manager)
                            ran_pre_batch = True
                    elif generation_strategy == GenerationStrategy.CELL_BY_CELL:
                        self._run_cell_by_cell_generator(generator)
                    elif generation_strategy == GenerationStrategy.FULL_COLUMN:
                        self._run_full_column_generator(generator)
                    else:
                        logger.error(f"❌ Unknown generation strategy: {generation_strategy}")
                        raise DatasetGenerationError(f"🛑 Unknown generation strategy: {generation_strategy}")
                    if save_partial_results:
                        self.batch_manager.write()
                except Exception as e:
                    column_error_str = (
                        f"columns {generator.config.column_names}"
                        if hasattr(generator.config, "column_names")
                        else f"column {generator.config.name!r}"
                    )
                    raise DatasetGenerationError(f"🛑 Failed to process {column_error_str}:\n{e}")

            try:
                usage_deltas = self._resource_provider.model_registry.get_usage_deltas(pre_batch_snapshot)
                self._emit_batch_inference_events(batch_mode, usage_deltas, group_id)
            except Exception:
                pass

            if current_batch_number is not None:
                df_batch = self.batch_manager.get_current_batch(as_dataframe=True)
                df_batch = self._processor_runner.run_post_batch(df_batch, current_batch_number=current_batch_number)
                self._write_processed_batch(df_batch)
                self.batch_manager.finish_batch(on_batch_complete)
        finally:
            if start_offset_token is not None:
                current_row_group_start_offset.reset(start_offset_token)
            if row_group_token is not None:
                current_row_group.reset(row_group_token)

    def _run_from_scratch_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate_from_scratch(self.batch_manager.num_records_batch)
        self.batch_manager.add_records(df.to_dict(orient="records"))

    def _run_cell_by_cell_generator(self, generator: ColumnGenerator) -> None:
        max_workers = self._resource_provider.run_config.non_inference_max_parallel_workers
        if isinstance(generator, ColumnGeneratorWithModel):
            max_workers = generator.inference_parameters.max_parallel_requests
        if self._use_async:
            logger.info("⚡ Using async engine for concurrent execution")
            self._fan_out_with_async(generator, max_workers=max_workers)
        else:
            self._fan_out_with_threads(generator, max_workers=max_workers)

    def _column_display_name(self, config: ColumnConfigT) -> str:
        return f"columns {config.column_names}" if hasattr(config, "column_names") else config.name

    def _log_resize_if_changed(self, column_name: str, original_count: int, new_count: int, allow_resize: bool) -> None:
        if not allow_resize or new_count == original_count:
            return
        if new_count == 0:
            logger.warning(f"⚠️ Column '{column_name}' reduced batch to 0 records. This batch will be skipped.")
        else:
            emoji = "💥" if new_count > original_count else "✂️"
            logger.info(f"{emoji} Column '{column_name}' resized batch: {original_count} -> {new_count} records.")

    def _require_graph(self) -> ExecutionGraph:
        """Return the initialized execution graph for the current run."""
        graph = self._graph
        if graph is None:
            raise DatasetGenerationError("Execution graph accessed before generator initialization.")
        return graph

    def _column_can_skip(self, column_name: str) -> bool:
        """Fast check: can *column_name* ever be skipped (expression gate or propagation)?

        Returns ``False`` for ``allow_resize=True`` columns because 1:N generators
        change the row count — the skip-aware merge path assumes a 1:1 mapping
        between input and output rows and would raise on the row-count check.
        """
        if self._graph is None:
            return False
        config = self.single_column_config_by_name.get(column_name)
        if config is not None and config.allow_resize:
            return False
        if self._graph.get_skip_config(column_name) is not None:
            return True
        return self._graph.should_propagate_skip(column_name) and bool(self._graph.get_required_columns(column_name))

    def _should_skip_cell(self, column_name: str, record: dict) -> bool:
        """Decide whether a single cell should be skipped (propagation or expression gate)."""
        skip_config = self._graph.get_skip_config(column_name)
        return should_skip_column_for_record(
            record,
            propagate_skip=self._graph.should_propagate_skip(column_name),
            required_columns=self._graph.get_required_columns(column_name),
            skip_config_when=skip_config.when if skip_config is not None else None,
        )

    def _write_skip_to_record(self, column_name: str, record: dict) -> None:
        """Write skip metadata and the skip value into *record* in-place."""
        skip_config = self._graph.get_skip_config(column_name)
        skip_value = skip_config.value if skip_config is not None else None
        apply_skip_to_record(
            record,
            column_name=column_name,
            cell_value=skip_value,
            side_effect_columns=self._graph.get_side_effect_columns(column_name),
        )

    def _run_full_column_generator(self, generator: ColumnGenerator) -> None:
        column_name = generator.config.name if not isinstance(generator.config, MultiColumnConfig) else None

        if column_name is not None and self._column_can_skip(column_name):
            self._run_full_column_generator_with_skip(generator, column_name)
        else:
            self._run_full_column_generator_without_skip(generator)

    def _run_full_column_generator_without_skip(self, generator: ColumnGenerator) -> None:
        """Run the generator on the full batch, preserving skip metadata across the replace."""
        original_count = self.batch_manager.num_records_in_buffer
        allow_resize = generator.config.allow_resize if not isinstance(generator.config, MultiColumnConfig) else False
        old_records = [record for _, record in self.batch_manager.iter_current_batch()]
        input_records, restore_context = prepare_records_for_skip_metadata_round_trip(old_records)

        df = generator.generate(lazy.pd.DataFrame(input_records))
        self._log_resize_if_changed(self._column_display_name(generator.config), original_count, len(df), allow_resize)
        new_records = df.to_dict(orient="records")
        if restore_context is not None:
            try:
                restore_skip_metadata(new_records, context=restore_context, allow_resize=allow_resize)
            except ValueError as exc:
                raise DatasetGenerationError(
                    f"Unable to restore skip provenance after FULL_COLUMN generation for "
                    f"{self._column_display_name(generator.config)}: {exc}"
                ) from exc
        self.batch_manager.replace_buffer(new_records, allow_resize=allow_resize)

    def _run_full_column_generator_with_skip(self, generator: ColumnGenerator, column_name: str) -> None:
        """Run a FULL_COLUMN generator with per-row skip evaluation and merge-back.

        Only reachable when ``_column_can_skip`` is True, which excludes
        ``allow_resize=True`` columns, so resize handling is not needed here.
        """
        active_records: list[dict] = []
        records_with_skip_status: list[tuple[bool, dict]] = []
        has_skipped = False
        for _, record in self.batch_manager.iter_current_batch():
            skipped = self._should_skip_cell(column_name, record)
            if skipped:
                has_skipped = True
                self._write_skip_to_record(column_name, record)
            else:
                active_records.append(record)
            records_with_skip_status.append((skipped, record))

        if not has_skipped:
            # No rows were actually skipped — use the normal path to avoid the
            # overhead of stripping metadata, building a separate active DataFrame,
            # and merging results back.
            self._run_full_column_generator_without_skip(generator)
            return

        batch = self._merge_skipped_and_generated(generator, column_name, active_records, records_with_skip_status)
        self.batch_manager.replace_buffer(batch, allow_resize=False)

    def _merge_skipped_and_generated(
        self,
        generator: ColumnGenerator,
        column_name: str,
        active_records: list[dict],
        records_with_skip_status: list[tuple[bool, dict]],
    ) -> list[dict]:
        """Generate only for active (non-skipped) records and merge back with skipped ones."""
        if not active_records:
            return [record for _, record in records_with_skip_status]

        active_df = lazy.pd.DataFrame(strip_skip_metadata_from_records(active_records))
        result_records = generator.generate(active_df).to_dict(orient="records")
        if len(result_records) != len(active_records):
            raise DatasetGenerationError(
                f"Generator for '{column_name}' returned {len(result_records)} rows "
                f"but {len(active_records)} active (non-skipped) records were expected."
            )

        result_iter = iter(result_records)
        batch: list[dict] = []
        for skipped, record in records_with_skip_status:
            if skipped:
                batch.append(record)
                continue
            gen_result = next(result_iter)
            prior_skipped = record.get(SKIPPED_COLUMNS_RECORD_KEY)
            if prior_skipped is not None:
                gen_result[SKIPPED_COLUMNS_RECORD_KEY] = prior_skipped
            batch.append(gen_result)
        return batch

    def _run_model_health_check_if_needed(self) -> None:
        model_aliases: set[str] = set()
        for config in self.single_column_configs:
            model_aliases.update(config.get_model_aliases())

        if not model_aliases:
            return

        if DATA_DESIGNER_ASYNC_ENGINE:
            loop = ensure_async_engine_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._resource_provider.model_registry.arun_health_check(list(model_aliases)),
                loop,
            )
            try:
                future.result(timeout=180)
            except TimeoutError:
                future.cancel()
                raise
        else:
            self._resource_provider.model_registry.run_health_check(list(model_aliases))

    def _run_mcp_tool_check_if_needed(self) -> None:
        tool_aliases = sorted(
            {config.tool_alias for config in self.llm_generated_column_configs if getattr(config, "tool_alias", None)}
        )
        if not tool_aliases:
            return
        if self._resource_provider.mcp_registry is None:
            raise DatasetGenerationError(f"Tool alias(es) {tool_aliases!r} specified but no MCPRegistry configured.")
        self._resource_provider.mcp_registry.run_health_check(tool_aliases)

    def _setup_fan_out(
        self,
        generator: ColumnGeneratorWithModelRegistry,
        max_workers: int,
        progress_bar: StickyProgressBar | None = None,
    ) -> tuple[ProgressTracker, dict[str, Any]]:
        if generator.get_generation_strategy() != GenerationStrategy.CELL_BY_CELL:
            raise DatasetGenerationError(
                f"Generator {generator.name} is not a {GenerationStrategy.CELL_BY_CELL} "
                "generator so concurrent fan-out is not supported."
            )

        allow_resize = generator.config.allow_resize
        if allow_resize:
            self._cell_resize_results = [None] * self.batch_manager.num_records_batch
            self._cell_resize_mode = True
            self._current_column_display_name = self._column_display_name(generator.config)
        else:
            self._cell_resize_mode = False

        label = f"{generator.config.column_type} column '{generator.config.name}'"
        progress_tracker = ProgressTracker(
            total_records=self.batch_manager.num_records_batch,
            label=label,
            progress_bar=progress_bar,
            progress_bar_key=generator.config.name,
        )
        progress_tracker.log_start(max_workers)

        settings = self._resource_provider.run_config
        executor_kwargs: dict = {
            "column_name": generator.config.name,
            "result_callback": self._make_result_callback(progress_tracker),
            "error_callback": self._make_error_callback(progress_tracker),
            "shutdown_error_rate": settings.shutdown_error_rate,
            "shutdown_error_window": settings.shutdown_error_window,
            "disable_early_shutdown": settings.disable_early_shutdown,
        }

        return progress_tracker, executor_kwargs

    def _finalize_fan_out(self, progress_tracker: ProgressTracker) -> None:
        progress_tracker.log_final()

        if self._cell_resize_mode:
            # Flatten results in index order; skip indices in _records_to_drop (failed cells),
            # so those rows are omitted from the new buffer.
            new_records: list[dict] = []
            for i in range(len(self._cell_resize_results)):
                if i in self._records_to_drop:
                    continue
                r = self._cell_resize_results[i]
                if r is not None:
                    new_records.extend(r if isinstance(r, list) else [r])
            self._log_resize_if_changed(
                self._current_column_display_name,
                self.batch_manager.num_records_in_buffer,
                len(new_records),
                True,
            )
            self.batch_manager.replace_buffer(new_records, allow_resize=True)
            self._records_to_drop.clear()
            self._cell_resize_mode = False
            self._cell_resize_results = []
        elif len(self._records_to_drop) > 0:
            self._cleanup_dropped_record_images(self._records_to_drop)
            self.batch_manager.drop_records(self._records_to_drop)
            self._records_to_drop.clear()

    def _fan_out_with_async(self, generator: ColumnGeneratorWithModelRegistry, max_workers: int) -> None:
        if getattr(generator.config, "tool_alias", None):
            logger.info("🛠️ Tool calling enabled")
        bar = StickyProgressBar() if self._resource_provider.run_config.progress_bar else None
        can_skip = self._column_can_skip(generator.config.name)
        with bar or contextlib.nullcontext():
            progress_tracker, executor_kwargs = self._setup_fan_out(generator, max_workers, progress_bar=bar)
            executor = AsyncConcurrentExecutor(max_workers=max_workers, **executor_kwargs)
            work_items: list[tuple[Any, dict[str, Any]]] = []
            for i, record in self.batch_manager.iter_current_batch():
                if can_skip and self._should_skip_cell(generator.config.name, record):
                    self._write_skip_to_record(generator.config.name, record)
                    self.batch_manager.update_record(i, record)
                    progress_tracker.record_skipped()
                    continue
                work_items.append(
                    (
                        generator.agenerate(record),
                        {"index": i, "column_name": generator.config.name},
                    )
                )
            executor.run(work_items)
            self._finalize_fan_out(progress_tracker)

    def _fan_out_with_threads(self, generator: ColumnGeneratorWithModelRegistry, max_workers: int) -> None:
        if getattr(generator.config, "tool_alias", None):
            logger.info("🛠️ Tool calling enabled")
        bar = StickyProgressBar() if self._resource_provider.run_config.progress_bar else None
        can_skip = self._column_can_skip(generator.config.name)
        with bar or contextlib.nullcontext():
            progress_tracker, executor_kwargs = self._setup_fan_out(generator, max_workers, progress_bar=bar)
            with ConcurrentThreadExecutor(max_workers=max_workers, **executor_kwargs) as executor:
                for i, record in self.batch_manager.iter_current_batch():
                    if can_skip and self._should_skip_cell(generator.config.name, record):
                        self._write_skip_to_record(generator.config.name, record)
                        self.batch_manager.update_record(i, record)
                        progress_tracker.record_skipped()
                        continue
                    executor.submit(
                        lambda record: generator.generate(record),
                        record,
                        context={"index": i, "column_name": generator.config.name},
                    )
            self._finalize_fan_out(progress_tracker)

    def _make_result_callback(self, progress_tracker: ProgressTracker) -> Callable[[dict], None]:
        def callback(result: dict, *, context: dict | None = None) -> None:
            self._worker_result_callback(result, context=context)
            progress_tracker.record_success()

        return callback

    def _make_error_callback(self, progress_tracker: ProgressTracker) -> Callable[[Exception], None]:
        def callback(exc: Exception, *, context: dict | None = None) -> None:
            self._worker_error_callback(exc, context=context)
            progress_tracker.record_failure()

        return callback

    def _write_processed_batch(self, dataframe: pd.DataFrame) -> None:
        self.batch_manager.replace_buffer(dataframe.to_dict(orient="records"), allow_resize=False)
        self.batch_manager.write()

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

    def _cleanup_dropped_record_images(self, dropped_indices: set[int]) -> None:
        """Remove saved image files for records that will be dropped.

        When a record fails during generation, any images already saved to disk
        for that record in previous columns become dangling. This method deletes
        those files so they don't accumulate.
        """
        media_storage = self.artifact_storage.media_storage
        if not self._has_image_columns() or media_storage is None or media_storage.mode != StorageMode.DISK:
            return

        image_col_names = [
            col.name for col in self.single_column_configs if col.column_type == DataDesignerColumnType.IMAGE
        ]

        buffer = self.batch_manager.get_current_batch(as_dataframe=False)
        for idx in dropped_indices:
            if idx < 0 or idx >= len(buffer):
                continue
            for col_name in image_col_names:
                paths = buffer[idx].get(col_name, [])
                for path in [paths] if isinstance(paths, str) else paths:
                    media_storage.delete_image(path)

    @staticmethod
    def _extract_failure_detail(exc: Exception) -> str:
        detail = getattr(exc, "detail", None)
        if isinstance(detail, str):
            normalized_detail = " ".join(detail.split()).strip()
            if normalized_detail:
                return normalized_detail
        exc_str = str(exc).strip()
        for line in exc_str.splitlines():
            if "Cause:" in line:
                return " ".join(line.split("Cause:", maxsplit=1)[1].split()).strip()
        return " ".join(exc_str.split()).strip() or type(exc).__name__

    @classmethod
    def _classify_worker_failure(cls, exc: Exception) -> str:
        failure_kind = getattr(exc, "failure_kind", None)
        if isinstance(failure_kind, str) and failure_kind.strip():
            return failure_kind.replace("_", " ")

        detail = cls._extract_failure_detail(exc).lower()
        exc_name = type(exc).__name__.lower()

        if "timeout" in exc_name or "timed out" in detail:
            return "timeout"
        if "rate" in exc_name and "limit" in exc_name:
            return "rate limit"
        if "authentication" in exc_name:
            return "authentication"
        if "permission" in exc_name:
            return "permission denied"
        if "contextwindow" in exc_name or "context width" in detail:
            return "context window"
        if "response_schema" in detail or "schema" in detail:
            return "schema validation"
        if "validation" in exc_name or "validation" in detail:
            return "validation"
        return "generation error"

    @classmethod
    def _format_worker_failure_warning(cls, exc: Exception, *, context: dict | None = None) -> str:
        record_index = context["index"] if context is not None and "index" in context else "unknown"
        column_name = context.get("column_name") if context is not None else None
        context_label = f" in column {column_name!r}" if column_name else ""
        failure_kind = cls._classify_worker_failure(exc)
        failure_detail = cls._extract_failure_detail(exc)
        return (
            f"⚠️ Generation for record at index {record_index} failed{context_label} ({failure_kind}). "
            f"Will omit this record from the dataset. Detail: {failure_detail}"
        )

    def _worker_error_callback(self, exc: Exception, *, context: dict | None = None) -> None:
        """If a worker fails, we can handle the exception here."""
        logger.warning(self._format_worker_failure_warning(exc, context=context))
        if context is None or "index" not in context:
            raise RuntimeError("Worker error callback called without a valid context index.")
        self._records_to_drop.add(context["index"])

    def _worker_result_callback(self, result: dict | list[dict], *, context: dict | None = None) -> None:
        if self._cell_resize_mode:
            self._cell_resize_results[context["index"]] = result
        else:
            self.batch_manager.update_record(context["index"], result)

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
