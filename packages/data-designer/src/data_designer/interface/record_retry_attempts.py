# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.record_retry_builders import RecordRetryBuilderFactory, unique_name
from data_designer.interface.record_retry_state import (
    ATTEMPT_COMPLETION_FILENAME,
    BASE_COHORT_PATH,
    INTERNAL_ATTEMPT_COLUMN_BASENAME,
    INTERNAL_SLOT_COLUMN_BASENAME,
    AttemptCompletion,
    AttemptManifest,
    RetryManifest,
    get_attempt_accepted_path,
    get_attempt_artifact_path,
    get_attempt_directory,
    get_attempt_input_path,
    get_attempt_name,
    read_attempt_completion,
    write_json_atomic,
    write_parquet_atomic,
)
from data_designer.interface.record_retry_utils import (
    classify_attempt,
    copy_preserved_seed_media,
    empty_attempt_output,
    load_completed_attempt_output,
    local_seed_media_root,
    package_accepted_media,
    write_or_validate_attempt_input,
)

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.dataset_builders.scheduling.task_model import TaskTrace
    from data_designer.interface.data_designer import DataDesigner


class RecordRetryAttemptRunner:
    """Execute base materialization and immutable candidate attempts."""

    def __init__(self, data_designer: DataDesigner) -> None:
        self.data_designer = data_designer
        self.task_traces: list[TaskTrace] = []

    def materialize_base(
        self,
        *,
        builder_factory: RecordRetryBuilderFactory,
        stage_path: Path,
        num_records: int,
        fingerprint: str,
    ) -> tuple[pd.DataFrame, RetryManifest]:
        """Materialize and persist the immutable retry cohort."""
        base_root = stage_path / BASE_COHORT_PATH.parent
        base_root.mkdir(parents=True, exist_ok=True)
        base_seed_column_names: list[str] = []
        base_model_usage: dict[str, dict[str, Any]] = {}

        if builder_factory.requires_base_materialization:
            result = self.data_designer._create(
                builder_factory.build_base_builder(),
                num_records=num_records,
                dataset_name="run",
                artifact_path=base_root,
                profile=False,
            )
            self.task_traces.extend(result.task_traces)
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
                source_root=local_seed_media_root(builder_factory.original_config),
                run_path=base_root,
            )
        else:
            base_df = lazy.pd.DataFrame(index=range(num_records))

        used_names = set(base_df.columns)
        used_names.update(
            output_name
            for column in builder_factory.original_config.columns
            for output_name in (column.name, *column.side_effect_columns)
        )
        slot_column = unique_name(INTERNAL_SLOT_COLUMN_BASENAME, used_names)
        attempt_column = unique_name(INTERNAL_ATTEMPT_COLUMN_BASENAME, used_names | {slot_column})
        base_df.insert(0, slot_column, range(num_records))
        write_parquet_atomic(base_df, stage_path / BASE_COHORT_PATH)

        manifest = RetryManifest(
            fingerprint=fingerprint,
            target_records=num_records,
            policy=builder_factory.retry_until,
            slot_column=slot_column,
            attempt_column=attempt_column,
            base_seed_column_names=base_seed_column_names,
            base_model_usage=base_model_usage,
        )
        return base_df, manifest

    def run_attempt(
        self,
        *,
        builder_factory: RecordRetryBuilderFactory,
        stage_path: Path,
        manifest: RetryManifest,
        base_df: pd.DataFrame,
        pending_ids: list[int],
    ) -> AttemptManifest:
        """Execute or recover one immutable pending-slot attempt."""
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
            result = self.data_designer._create(
                builder_factory.build_attempt_builder(input_path),
                num_records=len(attempt_input),
                dataset_name=run_path.name,
                artifact_path=attempt_dir,
                resume=attempt_resume,
                profile=False,
                allow_empty=True,
            )
            self.task_traces.extend(result.task_traces)
            output_records = result.count_records()
            output = (
                result.load_dataset()
                if output_records
                else empty_attempt_output(attempt_input, builder_factory.original_config)
            )
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
                original_config=builder_factory.original_config,
            )

        accepted, counts = classify_attempt(
            output=output,
            attempt_input=attempt_input,
            manifest=manifest,
            predicate_column=builder_factory.retry_until.predicate_column,
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
