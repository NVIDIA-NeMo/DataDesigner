# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface.errors import DataDesignerWorkflowError, RecordRetryExhaustedError
from data_designer.interface.record_retry import RetryExhaustion, RetryUntil
from data_designer.interface.record_retry_attempts import RecordRetryAttemptRunner
from data_designer.interface.record_retry_builders import RecordRetryBuilderFactory
from data_designer.interface.record_retry_publisher import RecordRetryPublisher
from data_designer.interface.record_retry_state import (
    get_attempt_accepted_path,
    load_retry_manifest,
    write_retry_manifest,
)
from data_designer.interface.record_retry_utils import (
    load_and_validate_base_cohort,
    load_committed_accepted_ids,
    read_accepted_slot_ids,
    retry_bounds_exhausted,
)
from data_designer.interface.results import DatasetCreationResults

if TYPE_CHECKING:
    from data_designer.interface.data_designer import DataDesigner


class RecordRetryRunner:
    """Coordinate one bounded record retry without owning phase execution."""

    def __init__(
        self,
        data_designer: DataDesigner,
        *,
        attempt_runner: RecordRetryAttemptRunner | None = None,
        publisher: RecordRetryPublisher | None = None,
    ) -> None:
        self.data_designer = data_designer
        self.attempt_runner = attempt_runner if attempt_runner is not None else RecordRetryAttemptRunner(data_designer)
        self.publisher = publisher if publisher is not None else RecordRetryPublisher(data_designer)

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
        """Run or resume one record-retry stage and publish its canonical result."""
        self.attempt_runner.task_traces.clear()
        if num_records < 1:
            raise DataDesignerWorkflowError("Record retry requires num_records to be at least 1.")
        if policy.max_candidate_records is not None and policy.max_candidate_records < num_records:
            raise DataDesignerWorkflowError(
                "retry_until.max_candidate_records must be at least the stage num_records so the first "
                "complete cohort attempt can run."
            )

        builder_factory = RecordRetryBuilderFactory(config_builder, policy)
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
            base_df, manifest = self.attempt_runner.materialize_base(
                builder_factory=builder_factory,
                stage_path=stage_path,
                num_records=num_records,
                fingerprint=fingerprint,
            )
            write_retry_manifest(stage_path, manifest)
        else:
            base_df = load_and_validate_base_cohort(stage_path, manifest)

        accepted_ids = load_committed_accepted_ids(stage_path, manifest)
        pending_ids = [slot_id for slot_id in range(num_records) if slot_id not in accepted_ids]

        while pending_ids and not retry_bounds_exhausted(policy, manifest, len(pending_ids)):
            attempt = self.attempt_runner.run_attempt(
                builder_factory=builder_factory,
                stage_path=stage_path,
                manifest=manifest,
                base_df=base_df,
                pending_ids=pending_ids,
            )
            manifest.attempts.append(attempt)
            manifest.status = "running"
            write_retry_manifest(stage_path, manifest)
            accepted_ids.update(
                read_accepted_slot_ids(
                    stage_path / get_attempt_accepted_path(len(manifest.attempts) - 1),
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
                raise RecordRetryExhaustedError(
                    target_records=num_records,
                    accepted_records=manifest.accepted_records,
                    candidate_records=manifest.candidate_records,
                    attempts=len(manifest.attempts),
                    unresolved_slot_ids=manifest.unresolved_slot_ids,
                )

        manifest.status = "finalizing"
        write_retry_manifest(stage_path, manifest)
        result, final_model_usage = self.publisher.publish(
            builder_factory=builder_factory,
            stage_path=stage_path,
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            manifest=manifest,
            base_df=base_df,
            task_traces=self.attempt_runner.task_traces,
        )
        manifest.final_model_usage = final_model_usage
        manifest.status = "complete"
        write_retry_manifest(stage_path, manifest)
        return result
