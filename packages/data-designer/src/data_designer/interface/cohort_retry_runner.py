# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.engine.storage.artifact_storage import METADATA_FILENAME, ArtifactStorage, BatchStage, ResumeMode
from data_designer.interface.cohort_retry import RetryExhaustion, RetryUntil
from data_designer.interface.cohort_retry_builders import CohortRetryBuilderProjection
from data_designer.interface.cohort_retry_state import (
    ATTEMPT_COMPLETION_FILENAME,
    BASE_COHORT_PATH,
    FINAL_COMPLETION_FILENAME,
    INTERNAL_ATTEMPT_COLUMN_BASENAME,
    INTERNAL_SLOT_COLUMN_BASENAME,
    PUBLICATION_INPUT_PATH,
    AttemptCompletion,
    AttemptManifest,
    FinalCompletion,
    RetryManifest,
    get_attempt_accepted_path,
    get_attempt_artifact_path,
    get_attempt_directory,
    get_attempt_input_path,
    get_attempt_name,
    load_retry_manifest,
    read_attempt_completion,
    read_final_completion,
    write_json_atomic,
    write_parquet_atomic,
    write_retry_manifest,
)
from data_designer.interface.cohort_retry_utils import (
    aggregate_model_usage,
    builder_from_projection,
    classify_attempt,
    clear_ambiguous_finalization,
    coalesce_accepted,
    copy_preserved_seed_media,
    count_storage_records,
    empty_attempt_output,
    empty_publication_dataframe,
    load_and_validate_base_cohort,
    load_committed_accepted_ids,
    load_completed_attempt_output,
    local_seed_media_root,
    metadata_retry_summary,
    package_accepted_media,
    read_accepted_slot_ids,
    retry_bounds_exhausted,
    unique_name,
    write_canonical_metadata,
    write_or_validate_attempt_input,
    write_original_builder_config,
)
from data_designer.interface.errors import CohortRetryExhaustedError, DataDesignerWorkflowError
from data_designer.interface.results import DatasetCreationResults, _load_analysis_from_artifact_storage

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.interface.data_designer import DataDesigner


class CohortRetryRunner:
    """Orchestrate ordinary Data Designer runs for one bounded retry cohort."""

    def __init__(self, data_designer: DataDesigner) -> None:
        self._data_designer = data_designer
        self._task_traces: list[Any] = []

    def run(
        self,
        *,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        artifact_path: Path,
        policy: RetryUntil,
        fingerprint: str,
        resume: ResumeMode,
        workflow_resume: ResumeMode,
    ) -> DatasetCreationResults:
        """Run or resume one cohort-retry stage and publish its canonical result."""
        if num_records < 1:
            raise DataDesignerWorkflowError("Cohort retry requires num_records to be at least 1.")
        if policy.max_candidate_records is not None and policy.max_candidate_records < num_records:
            raise DataDesignerWorkflowError(
                "retry_until.max_candidate_records must be at least the stage num_records so the first "
                "complete cohort attempt can run."
            )

        projection = CohortRetryBuilderProjection(config_builder, policy)
        stage_path = artifact_path / dataset_name
        stage_path.mkdir(parents=True, exist_ok=True)
        manifest = load_retry_manifest(
            stage_path=stage_path,
            fingerprint=fingerprint,
            policy=policy,
            resume=resume,
            workflow_resume=workflow_resume,
        )

        if manifest is None:
            base_df, manifest = self._materialize_base_cohort(
                projection=projection,
                stage_path=stage_path,
                num_records=num_records,
                fingerprint=fingerprint,
            )
        else:
            base_df = load_and_validate_base_cohort(stage_path, manifest)

        accepted_ids = load_committed_accepted_ids(stage_path, manifest)
        pending_ids = [slot_id for slot_id in range(num_records) if slot_id not in accepted_ids]

        while pending_ids and not retry_bounds_exhausted(policy, manifest, len(pending_ids)):
            attempt = self._run_attempt(
                projection=projection,
                stage_path=stage_path,
                manifest=manifest,
                base_df=base_df,
                pending_ids=pending_ids,
            )
            manifest.attempts.append(attempt)
            manifest.status = "running"
            write_retry_manifest(stage_path, manifest)
            attempt_index = len(manifest.attempts) - 1
            accepted_ids.update(
                read_accepted_slot_ids(
                    stage_path / get_attempt_accepted_path(attempt_index),
                    manifest.slot_column,
                    manifest.target_records,
                )
            )
            pending_ids = [slot_id for slot_id in pending_ids if slot_id not in accepted_ids]

        manifest.unresolved_slot_ids = list(pending_ids)
        if pending_ids:
            manifest.status = "exhausted"
            write_retry_manifest(stage_path, manifest)
            if policy.on_exhausted == RetryExhaustion.RAISE:
                raise CohortRetryExhaustedError(
                    target_records=num_records,
                    accepted_records=manifest.accepted_records,
                    candidate_records=manifest.candidate_records,
                    attempts=len(manifest.attempts),
                    unresolved_slot_ids=manifest.unresolved_slot_ids,
                )

        coalesced = coalesce_accepted(stage_path, manifest, base_df)
        result = self._finalize(
            projection=projection,
            stage_path=stage_path,
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            manifest=manifest,
            coalesced=coalesced,
        )
        manifest.status = "complete"
        write_retry_manifest(stage_path, manifest)
        return result

    def _materialize_base_cohort(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        stage_path: Path,
        num_records: int,
        fingerprint: str,
    ) -> tuple[pd.DataFrame, RetryManifest]:
        base_root = stage_path / BASE_COHORT_PATH.parent
        base_root.mkdir(parents=True, exist_ok=True)
        base_seed_column_names: list[str] = []
        base_model_usage: dict[str, dict[str, Any]] = {}

        if projection.requires_base_materialization:
            result = self._data_designer._create(
                projection.build_base_builder(),
                num_records=num_records,
                dataset_name="run",
                artifact_path=base_root,
                profile=False,
            )
            self._task_traces.extend(result.task_traces)
            base_df = result.load_dataset()
            if len(base_df) != num_records:
                raise DataDesignerWorkflowError(
                    f"Base cohort materialization produced {len(base_df)} of {num_records} required rows."
                )
            base_seed_column_names = list(result.dataset_metadata.seed_column_names)
            base_model_usage = result.model_usage or {}
            copy_preserved_seed_media(
                attempt_input=base_df,
                seed_column_names=base_seed_column_names,
                source_root=local_seed_media_root(projection.original_config),
                run_path=base_root,
            )
        else:
            base_df = lazy.pd.DataFrame(index=range(num_records))

        used_names = set(base_df.columns)
        used_names.update(
            output_name
            for column in projection.original_config.columns
            for output_name in (column.name, *column.side_effect_columns)
        )
        slot_column = unique_name(INTERNAL_SLOT_COLUMN_BASENAME, used_names)
        attempt_column = unique_name(INTERNAL_ATTEMPT_COLUMN_BASENAME, used_names | {slot_column})
        base_df.insert(0, slot_column, range(num_records))
        write_parquet_atomic(base_df, stage_path / BASE_COHORT_PATH)

        manifest = RetryManifest(
            fingerprint=fingerprint,
            target_records=num_records,
            policy=projection.retry_until.to_dict(),
            slot_column=slot_column,
            attempt_column=attempt_column,
            base_seed_column_names=base_seed_column_names,
            base_model_usage=base_model_usage,
        )
        write_retry_manifest(stage_path, manifest)
        return base_df, manifest

    def _run_attempt(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        stage_path: Path,
        manifest: RetryManifest,
        base_df: pd.DataFrame,
        pending_ids: list[int],
    ) -> AttemptManifest:
        attempt_index = len(manifest.attempts)
        attempt_dir = stage_path / get_attempt_directory(attempt_index)
        input_path = stage_path / get_attempt_input_path(attempt_index)
        run_path = stage_path / get_attempt_artifact_path(attempt_index)
        accepted_path = stage_path / get_attempt_accepted_path(attempt_index)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        attempt_input = base_df[base_df[manifest.slot_column].isin(pending_ids)].copy()
        attempt_input[manifest.attempt_column] = attempt_index
        if len(attempt_input) != len(pending_ids):
            raise DataDesignerWorkflowError("Pending slot projection did not produce exactly one input row per slot.")
        write_or_validate_attempt_input(attempt_input, input_path, manifest.slot_column)

        copy_preserved_seed_media(
            attempt_input=attempt_input,
            seed_column_names=manifest.base_seed_column_names,
            source_root=stage_path / BASE_COHORT_PATH.parent,
            run_path=run_path,
        )
        completion_path = attempt_dir / ATTEMPT_COMPLETION_FILENAME
        completion = read_attempt_completion(completion_path, pending_ids)
        if completion is None:
            attempt_resume = ResumeMode.ALWAYS if run_path.exists() and any(run_path.iterdir()) else ResumeMode.NEVER
            result = self._data_designer._create(
                projection.build_attempt_builder(input_path),
                num_records=len(attempt_input),
                dataset_name=run_path.name,
                artifact_path=attempt_dir,
                resume=attempt_resume,
                profile=False,
                allow_empty=True,
            )
            self._task_traces.extend(result.task_traces)
            output_records = result.count_records()
            output = result.load_dataset() if output_records else empty_attempt_output(attempt_input, projection)
            completion = AttemptCompletion(
                input_slot_ids=list(pending_ids),
                output_records=output_records,
                model_usage=result.model_usage or {},
            )
            write_json_atomic(completion.model_dump(mode="json"), completion_path)
        else:
            output = load_completed_attempt_output(
                run_path=run_path,
                completion=completion,
                attempt_input=attempt_input,
                projection=projection,
            )

        accepted, counts = classify_attempt(
            output=output,
            attempt_input=attempt_input,
            manifest=manifest,
            predicate_column=projection.predicate_column,
        )
        accepted = package_accepted_media(
            accepted,
            attempt_root=run_path,
            stage_path=stage_path,
            attempt_name=get_attempt_name(attempt_index),
        )
        write_parquet_atomic(accepted, accepted_path)
        return AttemptManifest(
            input_records=len(attempt_input),
            **counts,
            model_usage=completion.model_usage,
        )

    def _finalize(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        stage_path: Path,
        artifact_path: Path,
        dataset_name: str,
        manifest: RetryManifest,
        coalesced: pd.DataFrame,
    ) -> DatasetCreationResults:
        if len(coalesced) == 0:
            manifest.final_model_usage = {}
            return self._write_empty_final(
                projection=projection,
                artifact_path=artifact_path,
                dataset_name=dataset_name,
                manifest=manifest,
                coalesced=coalesced,
            )

        internal_columns = [manifest.slot_column, manifest.attempt_column]
        missing_internal = [column for column in internal_columns if column not in coalesced]
        if missing_internal:
            raise DataDesignerWorkflowError(
                f"Coalesced output is missing internal cohort identity columns: {missing_internal!r}."
            )
        publication_input_path = stage_path / PUBLICATION_INPUT_PATH
        write_parquet_atomic(coalesced.drop(columns=internal_columns), publication_input_path)

        manifest.status = "finalizing"
        write_retry_manifest(stage_path, manifest)
        completion_path = stage_path / FINAL_COMPLETION_FILENAME
        completion = read_final_completion(completion_path)
        if completion is None:
            clear_ambiguous_finalization(stage_path)
            final_resume = ResumeMode.ALWAYS if (stage_path / METADATA_FILENAME).exists() else ResumeMode.IF_POSSIBLE
            result = self._data_designer._create(
                projection.build_final_builder(publication_input_path),
                num_records=len(coalesced),
                dataset_name=dataset_name,
                artifact_path=artifact_path,
                resume=final_resume,
                profiling_config_builder=builder_from_projection(projection),
                profiling_target_num_records=manifest.target_records,
            )
            self._task_traces.extend(result.task_traces)
            actual_records = result.count_records()
            if actual_records != len(coalesced):
                raise DataDesignerWorkflowError(
                    "Final cohort-retry processors changed the accepted row count from "
                    f"{len(coalesced)} to {actual_records}; exact cohort retry requires row-count-preserving "
                    "terminal processors."
                )
            analysis = result._load_optional_analysis()
            if analysis is None:
                raise DataDesignerWorkflowError("Final cohort-retry publication completed without profiling analysis.")
            completion = FinalCompletion(
                accepted_records=actual_records,
                model_usage=result.model_usage or {},
            )
            write_json_atomic(completion.model_dump(mode="json"), completion_path)
            storage = result.artifact_storage
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
                    "Final dataset does not match its durable cohort-retry completion marker."
                )
            analysis = _load_analysis_from_artifact_storage(storage)
            if analysis is None:
                raise DataDesignerWorkflowError(
                    "Final cohort-retry profiling metadata is missing or invalid after durable completion."
                )

        manifest.final_model_usage = completion.model_usage
        aggregate_usage = aggregate_model_usage(manifest)
        original_builder = builder_from_projection(projection)
        write_original_builder_config(stage_path, projection)
        write_canonical_metadata(
            storage=storage,
            projection=projection,
            manifest=manifest,
            actual_records=completion.accepted_records,
            model_usage=aggregate_usage,
        )
        return DatasetCreationResults(
            artifact_storage=storage,
            analysis=analysis,
            config_builder=original_builder,
            dataset_metadata=DatasetMetadata(seed_column_names=manifest.base_seed_column_names),
            task_traces=list(self._task_traces),
            model_usage=aggregate_usage,
        )

    def _write_empty_final(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        artifact_path: Path,
        dataset_name: str,
        manifest: RetryManifest,
        coalesced: pd.DataFrame,
    ) -> DatasetCreationResults:
        storage = ArtifactStorage(
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            resume=ResumeMode.ALWAYS,
        )
        empty = empty_publication_dataframe(
            coalesced,
            projection=projection,
            internal_columns={manifest.slot_column, manifest.attempt_column},
        )
        write_parquet_atomic(empty, storage.create_batch_file_path(0, BatchStage.FINAL_RESULT))
        write_original_builder_config(storage.base_dataset_path, projection)
        model_usage = aggregate_model_usage(manifest)
        storage.write_metadata(
            {
                **projection.original_config.fingerprint(),
                "target_num_records": manifest.target_records,
                "original_target_num_records": manifest.target_records,
                "actual_num_records": 0,
                "cohort_retry": metadata_retry_summary(manifest, model_usage),
            }
        )
        return DatasetCreationResults(
            artifact_storage=storage,
            analysis=None,
            config_builder=builder_from_projection(projection),
            dataset_metadata=DatasetMetadata(seed_column_names=manifest.base_seed_column_names),
            task_traces=list(self._task_traces),
            model_usage=model_usage,
        )
