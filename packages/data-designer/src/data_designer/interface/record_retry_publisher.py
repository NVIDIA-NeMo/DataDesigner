# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.engine.storage.artifact_storage import METADATA_FILENAME, ArtifactStorage, BatchStage, ResumeMode
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.record_retry_builders import RecordRetryBuilderFactory
from data_designer.interface.record_retry_state import (
    FINAL_COMPLETION_FILENAME,
    PUBLICATION_INPUT_PATH,
    FinalCompletion,
    RetryManifest,
    read_final_completion,
    write_json_atomic,
    write_parquet_atomic,
)
from data_designer.interface.record_retry_utils import (
    aggregate_model_usage,
    clear_ambiguous_finalization,
    coalesce_accepted,
    count_storage_records,
    empty_publication_dataframe,
    metadata_retry_summary,
    write_canonical_metadata,
    write_original_builder_config,
)
from data_designer.interface.results import DatasetCreationResults, _load_analysis_from_artifact_storage

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.dataset_builders.scheduling.task_model import TaskTrace
    from data_designer.interface.data_designer import DataDesigner


class RecordRetryPublisher:
    """Publish committed accepted partitions as one canonical dataset result."""

    def __init__(self, data_designer: DataDesigner) -> None:
        self.data_designer = data_designer

    def publish(
        self,
        *,
        builder_factory: RecordRetryBuilderFactory,
        stage_path: Path,
        artifact_path: Path,
        dataset_name: str,
        manifest: RetryManifest,
        base_df: pd.DataFrame,
        task_traces: list[TaskTrace],
    ) -> tuple[DatasetCreationResults, dict[str, dict[str, Any]]]:
        """Coalesce accepted rows and publish an empty or nonempty result."""
        coalesced = coalesce_accepted(stage_path, manifest, base_df)
        if len(coalesced) == 0:
            return self.publish_empty(
                builder_factory=builder_factory,
                artifact_path=artifact_path,
                dataset_name=dataset_name,
                manifest=manifest,
                coalesced=coalesced,
                task_traces=task_traces,
            )
        return self.publish_nonempty(
            builder_factory=builder_factory,
            stage_path=stage_path,
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            manifest=manifest,
            coalesced=coalesced,
            task_traces=task_traces,
        )

    def publish_nonempty(
        self,
        *,
        builder_factory: RecordRetryBuilderFactory,
        stage_path: Path,
        artifact_path: Path,
        dataset_name: str,
        manifest: RetryManifest,
        coalesced: pd.DataFrame,
        task_traces: list[TaskTrace],
    ) -> tuple[DatasetCreationResults, dict[str, dict[str, Any]]]:
        """Run terminal processors and profiling for a nonempty accepted cohort."""
        internal_columns = [manifest.slot_column, manifest.attempt_column]
        missing_internal = [column for column in internal_columns if column not in coalesced]
        if missing_internal:
            raise DataDesignerWorkflowError(
                f"Coalesced output is missing internal cohort identity columns: {missing_internal!r}."
            )
        publication_input_path = stage_path / PUBLICATION_INPUT_PATH
        write_parquet_atomic(coalesced.drop(columns=internal_columns), publication_input_path)

        completion_path = stage_path / FINAL_COMPLETION_FILENAME
        completion = read_final_completion(completion_path)
        current_task_traces = list(task_traces)
        if completion is None:
            clear_ambiguous_finalization(stage_path)
            final_resume = ResumeMode.ALWAYS if (stage_path / METADATA_FILENAME).exists() else ResumeMode.IF_POSSIBLE
            result = self.data_designer._create(
                builder_factory.build_final_builder(publication_input_path),
                num_records=len(coalesced),
                dataset_name=dataset_name,
                artifact_path=artifact_path,
                resume=final_resume,
                profiling_config_builder=builder_factory.build_original_builder(),
                profiling_target_num_records=manifest.target_records,
            )
            current_task_traces.extend(result.task_traces)
            actual_records = result.count_records()
            if actual_records != len(coalesced):
                raise DataDesignerWorkflowError(
                    "Final record-retry processors changed the accepted row count from "
                    f"{len(coalesced)} to {actual_records}; exact record retry requires row-count-preserving "
                    "terminal processors."
                )
            storage = result.artifact_storage
            analysis = result.load_analysis()
            completion = FinalCompletion(
                accepted_records=actual_records,
                model_usage=result.model_usage or {},
            )
            write_json_atomic(completion.model_dump(mode="json"), completion_path)
        else:
            if completion.accepted_records != len(coalesced):
                raise DataDesignerWorkflowError(
                    "Final completion marker does not match the coalesced accepted-record count."
                )
            storage = ArtifactStorage(
                artifact_path=artifact_path,
                dataset_name=dataset_name,
                resume=ResumeMode.ALWAYS,
            )
            if count_storage_records(storage) != completion.accepted_records:
                raise DataDesignerWorkflowError(
                    "Final dataset does not match its durable record-retry completion marker."
                )
            analysis = _load_analysis_from_artifact_storage(storage)
            if analysis is None:
                raise DataDesignerWorkflowError(
                    "Final record-retry profiling metadata is missing or invalid after durable completion."
                )

        final_model_usage = completion.model_usage
        publication_manifest = manifest.model_copy(update={"final_model_usage": final_model_usage})
        aggregate_usage = aggregate_model_usage(publication_manifest)
        original_builder = builder_factory.build_original_builder()
        write_original_builder_config(stage_path, builder_factory.original_config)
        write_canonical_metadata(
            storage=storage,
            original_config=builder_factory.original_config,
            manifest=publication_manifest,
            actual_records=completion.accepted_records,
            model_usage=aggregate_usage,
        )
        return (
            DatasetCreationResults(
                artifact_storage=storage,
                analysis=analysis,
                config_builder=original_builder,
                dataset_metadata=DatasetMetadata(seed_column_names=manifest.base_seed_column_names),
                task_traces=current_task_traces,
                model_usage=aggregate_usage,
            ),
            final_model_usage,
        )

    def publish_empty(
        self,
        *,
        builder_factory: RecordRetryBuilderFactory,
        artifact_path: Path,
        dataset_name: str,
        manifest: RetryManifest,
        coalesced: pd.DataFrame,
        task_traces: list[TaskTrace],
    ) -> tuple[DatasetCreationResults, dict[str, dict[str, Any]]]:
        """Publish a schema-bearing zero-row result without terminal processors."""
        storage = ArtifactStorage(
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            resume=ResumeMode.ALWAYS,
        )
        empty = empty_publication_dataframe(
            coalesced,
            original_config=builder_factory.original_config,
            internal_columns={manifest.slot_column, manifest.attempt_column},
        )
        write_parquet_atomic(empty, storage.create_batch_file_path(0, BatchStage.FINAL_RESULT))
        write_original_builder_config(storage.base_dataset_path, builder_factory.original_config)
        final_model_usage: dict[str, dict[str, Any]] = {}
        publication_manifest = manifest.model_copy(update={"final_model_usage": final_model_usage})
        model_usage = aggregate_model_usage(publication_manifest)
        storage.write_metadata(
            {
                **builder_factory.original_config.fingerprint(),
                "target_num_records": manifest.target_records,
                "original_target_num_records": manifest.target_records,
                "actual_num_records": 0,
                "record_retry": metadata_retry_summary(publication_manifest, model_usage),
            }
        )
        return (
            DatasetCreationResults(
                artifact_storage=storage,
                analysis=None,
                config_builder=builder_factory.build_original_builder(),
                dataset_metadata=DatasetMetadata(seed_column_names=manifest.base_seed_column_names),
                task_traces=list(task_traces),
                model_usage=model_usage,
            ),
            final_model_usage,
        )
