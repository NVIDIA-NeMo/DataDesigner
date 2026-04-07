# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType
from data_designer.config.config_builder import BuilderConfig
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorType,
)
from data_designer.config.version import get_library_version
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorWithModel,
    GenerationStrategy,
)
from data_designer.engine.column_generators.utils.generator_classification import column_type_is_model_generated
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.engine.dataset_builders.utils.config_compiler import compile_dataset_builder_column_configs
from data_designer.engine.dataset_builders.utils.dataset_batch_manager import DatasetBatchManager
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.processor_runner import ProcessorRunner, ProcessorStage
from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.engine.dataset_builders.utils.skip_evaluator import evaluate_skip_when, should_skip_by_propagation
from data_designer.engine.dataset_builders.utils.skip_tracker import (
    SKIPPED_COLUMNS_RECORD_KEY,
    apply_skip_to_record,
    get_skipped_column_names,
    strip_skip_metadata_from_records,
)
from data_designer.engine.dataset_builders.utils.sticky_progress_bar import StickyProgressBar
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum, TelemetryHandler
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.storage.artifact_storage import SDG_CONFIG_FILENAME, ArtifactStorage
from data_designer.engine.storage.media_storage import StorageMode

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry
    from data_designer.engine.dataset_builders.utils.task_model import TaskTrace
    from data_designer.engine.models.usage import ModelUsageStats

logger = logging.getLogger(__name__)

DATA_DESIGNER_ASYNC_ENGINE = os.environ.get("DATA_DESIGNER_ASYNC_ENGINE", "0") == "1"

if DATA_DESIGNER_ASYNC_ENGINE:
    import asyncio
    import sys

    if sys.version_info < (3, 11):
        raise RuntimeError(
            "DATA_DESIGNER_ASYNC_ENGINE requires Python 3.11+ (asyncio.TaskGroup). "
            f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )
    from data_designer.engine.dataset_builders.async_scheduler import (
        DEFAULT_TASK_POOL_SIZE,
        LLM_WAIT_POOL_MULTIPLIER,
        AsyncTaskScheduler,
    )
    from data_designer.engine.dataset_builders.utils.async_concurrency import (
        AsyncConcurrentExecutor,
        ensure_async_engine_loop,
    )
    from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
    from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager


_CLIENT_VERSION: str = get_library_version()


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
    def processors(self) -> tuple[Processor, ...]:
        return self._processor_runner.processors

    @property
    def task_traces(self) -> list[TaskTrace]:
        return self._task_traces

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
    def llm_generated_column_configs(self) -> list[ColumnConfigT]:
        return [config for config in self.single_column_configs if column_type_is_model_generated(config.column_type)]

    def build(
        self,
        *,
        num_records: int,
        on_batch_complete: Callable[[Path], None] | None = None,
        save_multimedia_to_disk: bool = True,
    ) -> Path:
        """Build the dataset.

        Args:
            num_records: Number of records to generate.
            on_batch_complete: Optional callback function called when each batch completes.
            save_multimedia_to_disk: Whether to save generated multimedia (images, audio, video) to disk.
                If False, multimedia is stored directly in the DataFrame (e.g., images as base64).
                Default is True.

        Returns:
            Path to the generated dataset directory.
        """
        self._run_model_health_check_if_needed()
        self._run_mcp_tool_check_if_needed()
        self._write_builder_config()

        # Set media storage mode based on parameters
        if self._has_image_columns():
            mode = StorageMode.DISK if save_multimedia_to_disk else StorageMode.DATAFRAME
            self.artifact_storage.set_media_storage_mode(mode)

        generators = self._initialize_generators()
        start_time = time.perf_counter()
        buffer_size = self._resource_provider.run_config.buffer_size

        if DATA_DESIGNER_ASYNC_ENGINE:
            self._validate_async_compatibility()
            self._build_async(generators, num_records, buffer_size, on_batch_complete)
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

        self._processor_runner.run_after_generation(buffer_size)
        self._resource_provider.model_registry.log_model_usage(time.perf_counter() - start_time)

        return self.artifact_storage.final_dataset_path

    def build_preview(self, *, num_records: int) -> pd.DataFrame:
        self._run_model_health_check_if_needed()
        self._run_mcp_tool_check_if_needed()

        # Set media storage to DATAFRAME mode for preview - base64 stored directly in DataFrame
        if self._has_image_columns():
            self.artifact_storage.set_media_storage_mode(StorageMode.DATAFRAME)

        generators = self._initialize_generators()
        start_time = time.perf_counter()

        if DATA_DESIGNER_ASYNC_ENGINE:
            self._validate_async_compatibility()
            dataset = self._build_async_preview(generators, num_records)
        else:
            group_id = uuid.uuid4().hex
            self.batch_manager.start(num_records=num_records, buffer_size=num_records)
            self._run_batch(generators, batch_mode="preview", save_partial_results=False, group_id=group_id)
            dataset = self.batch_manager.get_current_batch(as_dataframe=True)
            self.batch_manager.reset()

        self._resource_provider.model_registry.log_model_usage(time.perf_counter() - start_time)

        return dataset

    def _build_async_preview(self, generators: list[ColumnGenerator], num_records: int) -> pd.DataFrame:
        """Async preview path - single row group, no disk writes, returns in-memory DataFrame."""
        logger.info("⚡ DATA_DESIGNER_ASYNC_ENGINE is enabled - using async task-queue preview")

        scheduler, buffer_manager = self._prepare_async_run(
            generators,
            num_records,
            buffer_size=num_records,
            run_post_batch_in_scheduler=False,
        )

        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(scheduler.run(), loop)
        future.result()

        self._task_traces = scheduler.traces

        if not buffer_manager.has_row_group(0):
            return lazy.pd.DataFrame()

        dataset = buffer_manager.get_dataframe(0)
        buffer_manager.free_row_group(0)
        return dataset

    def _validate_async_compatibility(self) -> None:
        """Raise if any column uses allow_resize=True with the async scheduler."""
        offending = [config.name for config in self.single_column_configs if getattr(config, "allow_resize", False)]
        if offending:
            raise DatasetGenerationError(
                f"allow_resize=True is not supported with DATA_DESIGNER_ASYNC_ENGINE=1. "
                f"Offending column(s): {offending}. Either remove allow_resize=True or "
                f"disable the async scheduler."
            )

    def _build_async(
        self,
        generators: list[ColumnGenerator],
        num_records: int,
        buffer_size: int,
        on_batch_complete: Callable[[Path], None] | None = None,
    ) -> None:
        """Async task-queue builder path - dispatches tasks based on dependency readiness."""
        logger.info("⚡ DATA_DESIGNER_ASYNC_ENGINE is enabled - using async task-queue builder")

        settings = self._resource_provider.run_config
        trace_enabled = settings.async_trace or os.environ.get("DATA_DESIGNER_ASYNC_TRACE", "0") == "1"

        def finalize_row_group(rg_id: int) -> None:
            def on_complete(final_path: Path | str | None) -> None:
                if final_path is not None and on_batch_complete:
                    on_batch_complete(final_path)

            buffer_manager.checkpoint_row_group(rg_id, on_complete=on_complete)

        scheduler, buffer_manager = self._prepare_async_run(
            generators,
            num_records,
            buffer_size,
            on_finalize_row_group=finalize_row_group,
            shutdown_error_rate=settings.shutdown_error_rate,
            shutdown_error_window=settings.shutdown_error_window,
            disable_early_shutdown=settings.disable_early_shutdown,
            trace=trace_enabled,
        )

        # Telemetry snapshot
        group_id = uuid.uuid4().hex
        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()

        # Run on background event loop
        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(scheduler.run(), loop)
        future.result()

        self._task_traces = scheduler.traces

        # Emit telemetry
        try:
            usage_deltas = self._resource_provider.model_registry.get_usage_deltas(pre_batch_snapshot)
            self._emit_batch_inference_events("batch", usage_deltas, group_id)
        except Exception:
            logger.debug("Failed to emit batch telemetry for async run", exc_info=True)

        # Write metadata
        buffer_manager.write_metadata(target_num_records=num_records, buffer_size=buffer_size)

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

        # Partition into row groups
        row_groups: list[tuple[int, int]] = []
        remaining = num_records
        rg_id = 0
        while remaining > 0:
            size = min(buffer_size, remaining)
            row_groups.append((rg_id, size))
            remaining -= size
            rg_id += 1

        tracker = CompletionTracker.with_graph(graph, row_groups)
        buffer_manager = RowGroupBufferManager(self.artifact_storage)

        # Pre-batch processor callback: runs after seed tasks complete for a row group.
        # If it raises, the scheduler drops all rows in the row group (skips it).
        def on_seeds_complete(rg_id: int, rg_size: int) -> None:
            df = buffer_manager.get_dataframe(rg_id)
            df = self._processor_runner.run_pre_batch_on_df(df)
            buffer_manager.replace_dataframe(rg_id, df)
            for ri in range(rg_size):
                if buffer_manager.is_dropped(rg_id, ri) and not tracker.is_dropped(rg_id, ri):
                    tracker.drop_row(rg_id, ri)

        # Post-batch processor callback: runs after all columns, before finalization.
        def on_before_checkpoint(rg_id: int, rg_size: int) -> None:
            df = buffer_manager.get_dataframe(rg_id)
            df = self._processor_runner.run_post_batch(df, current_batch_number=rg_id)
            buffer_manager.replace_dataframe(rg_id, df)

        # Coarse upper bound: sums all registered aliases, not just those used
        # in this build. Oversizing is harmless - ThrottleManager enforces
        # the real per-key limit; the semaphore is a memory-safety cap.
        aggregate = self._resource_provider.model_registry.get_aggregate_max_parallel_requests()

        scheduler = AsyncTaskScheduler(
            generators=gen_map,
            graph=graph,
            tracker=tracker,
            row_groups=row_groups,
            buffer_manager=buffer_manager,
            max_submitted_tasks=DEFAULT_TASK_POOL_SIZE,
            max_llm_wait_tasks=max(DEFAULT_TASK_POOL_SIZE, LLM_WAIT_POOL_MULTIPLIER * aggregate),
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
            progress_interval=self._resource_provider.run_config.progress_interval,
            progress_bar=self._resource_provider.run_config.progress_bar,
        )
        return scheduler, buffer_manager

    def process_preview(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = self._processor_runner.run_post_batch(dataset.copy(), current_batch_number=None)
        return self._processor_runner.run_after_generation_on_df(df)

    def _has_image_columns(self) -> bool:
        """Check if config has any image generation columns."""
        return any(col.column_type == DataDesignerColumnType.IMAGE for col in self.single_column_configs)

    def _initialize_generators(self) -> list[ColumnGenerator]:
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
        self._graph = ExecutionGraph.create(self._column_configs, strategies)
        return generators

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
    ) -> None:
        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()
        ran_pre_batch = False
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

    def _run_from_scratch_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate_from_scratch(self.batch_manager.num_records_batch)
        self.batch_manager.add_records(df.to_dict(orient="records"))

    def _run_cell_by_cell_generator(self, generator: ColumnGenerator) -> None:
        max_workers = self._resource_provider.run_config.non_inference_max_parallel_workers
        if isinstance(generator, ColumnGeneratorWithModel):
            max_workers = generator.inference_parameters.max_parallel_requests
        if DATA_DESIGNER_ASYNC_ENGINE:
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
        """Fast check: can *column_name* ever be skipped (expression gate or propagation)?"""
        graph = self._require_graph()
        if graph.get_skip_config(column_name) is not None:
            return True
        return graph.should_propagate_skip(column_name) and bool(graph.get_required_columns(column_name))

    def _should_skip_cell(self, column_name: str, record: dict) -> bool:
        """Decide whether a single cell should be skipped (propagation or expression gate)."""
        graph = self._require_graph()

        skipped_cols = get_skipped_column_names(record)

        if graph.should_propagate_skip(column_name):
            required = graph.get_required_columns(column_name)
            if should_skip_by_propagation(required, skipped_cols):
                return True

        skip_config = graph.get_skip_config(column_name)
        if skip_config is not None:
            return evaluate_skip_when(skip_config.when, record)

        return False

    def _write_skip_to_record(self, column_name: str, record: dict) -> None:
        """Write skip metadata and the skip value into *record* in-place."""
        graph = self._require_graph()
        skip_config = graph.get_skip_config(column_name)
        skip_value = skip_config.value if skip_config is not None else None
        apply_skip_to_record(
            record,
            column_name=column_name,
            cell_value=skip_value,
            side_effect_columns=graph.get_side_effect_columns(column_name),
        )

    def _run_full_column_generator(self, generator: ColumnGenerator) -> None:
        original_count = self.batch_manager.num_records_in_buffer
        allow_resize = getattr(generator.config, "allow_resize", False)

        if isinstance(generator.config, MultiColumnConfig):
            df = generator.generate(self.batch_manager.get_current_batch(as_dataframe=True))
            self._log_resize_if_changed(
                self._column_display_name(generator.config), original_count, len(df), allow_resize
            )
            self.batch_manager.replace_buffer(df.to_dict(orient="records"), allow_resize=allow_resize)
            return

        column_name = generator.config.name
        if not self._column_can_skip(column_name):
            df = generator.generate(self.batch_manager.get_current_batch(as_dataframe=True))
            self._log_resize_if_changed(
                self._column_display_name(generator.config), original_count, len(df), allow_resize
            )
            self.batch_manager.replace_buffer(df.to_dict(orient="records"), allow_resize=allow_resize)
            return

        has_skipped = False
        active_records: list[dict] = []
        records_with_skip_status: list[tuple[bool, dict]] = []
        for _, record in self.batch_manager.iter_current_batch():
            skipped = self._should_skip_cell(column_name, record)
            if skipped:
                has_skipped = True
                self._write_skip_to_record(column_name, record)
            else:
                active_records.append(record)
            records_with_skip_status.append((skipped, record))

        if has_skipped:
            batch: list[dict]
            if active_records:
                active_df = lazy.pd.DataFrame(strip_skip_metadata_from_records(active_records))
                result_records = generator.generate(active_df).to_dict(orient="records")
                if len(result_records) != len(active_records):
                    raise DatasetGenerationError(
                        f"Generator for '{column_name}' returned {len(result_records)} rows "
                        f"but {len(active_records)} active (non-skipped) records were expected."
                    )

                result_iter = iter(result_records)
                batch = []
                for skipped, record in records_with_skip_status:
                    if skipped:
                        batch.append(record)
                        continue
                    gen_result = next(result_iter)
                    prior_skipped = record.get(SKIPPED_COLUMNS_RECORD_KEY)
                    if prior_skipped is not None:
                        gen_result[SKIPPED_COLUMNS_RECORD_KEY] = prior_skipped
                    batch.append(gen_result)
            else:
                batch = [record for _, record in records_with_skip_status]

            self._log_resize_if_changed(
                self._column_display_name(generator.config), original_count, len(batch), allow_resize
            )
            self.batch_manager.replace_buffer(batch, allow_resize=allow_resize)
            return

        df = generator.generate(self.batch_manager.get_current_batch(as_dataframe=True))
        self._log_resize_if_changed(self._column_display_name(generator.config), original_count, len(df), allow_resize)
        self.batch_manager.replace_buffer(df.to_dict(orient="records"), allow_resize=allow_resize)

    def _run_model_health_check_if_needed(self) -> None:
        model_aliases: set[str] = set()
        for config in self.single_column_configs:
            if column_type_is_model_generated(config.column_type):
                model_aliases.add(config.model_alias)
            if isinstance(config, CustomColumnConfig) and config.model_aliases:
                model_aliases.update(config.model_aliases)

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
